
from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from .config import RunConfig
from .instances import build_instance
from .qaoa import QAOAShapes, qaoa_qnode, qaoa_sample_qnode, qaoa_energy, qaoa_energy_cvar
from .classical import ClassicalProduct
from .meta_map import MetaPolyRBF
from .optim import es_train_driver
from .eval import time_to_target, aggregate_curves, safe_median
from .plotting import plot_curve


def _minmax_norm(nums):
    import numpy as np
    vals = np.array([float(x) for x in nums if x == x and np.isfinite(x)], dtype=float)
    if len(vals) == 0:
        return {"min": float('nan'), "max": float('nan'), "norm": []}
    mn, mx = float(np.min(vals)), float(np.max(vals))
    span = mx - mn
    def norm_one(x):
        if not np.isfinite(x):
            return float('nan')
        return 0.0 if span == 0 else (float(x) - mn) / span
    return {"min": mn, "max": mx, "norm": [norm_one(x) for x in nums]}


def export_metrics(res: pd.DataFrame, ttt: pd.DataFrame, outdir: str, target: float) -> str:
    algos = sorted(res["algo"].unique())
    res = res.copy()
    res["success"] = res["final_ratio"] >= target
    summary = {"target_ratio": float(target), "algos": {}}
    med_final = {}
    med_calls = {}

    for algo in algos:
        sub = res[res["algo"] == algo]
        final_ratio_median = safe_median(sub["final_ratio"].tolist())
        ratio0_median = safe_median(sub["ratio0"].tolist()) if "ratio0" in sub else float('nan')
        success_rate = float(np.mean(sub["success"].values)) if len(sub) else float('nan')
        succ = sub[sub["success"]]
        median_hit_calls = safe_median(succ["hit_calls"].tolist())
        median_hit_time = safe_median(succ["hit_walltime"].tolist())

        med_final[algo] = final_ratio_median
        med_calls[algo] = median_hit_calls if median_hit_calls == median_hit_calls else float('inf')

        by_split = {}
        for split in sorted(sub["split"].unique()):
            s2 = sub[sub["split"] == split]
            by_split[split] = {
                "final_ratio_median": safe_median(s2["final_ratio"].tolist()),
                "ratio0_median": safe_median(s2["ratio0"].tolist()) if "ratio0" in s2 else float('nan'),
                "success_rate": float(np.mean(s2["success"].values)) if len(s2) else float('nan'),
            }

        summary["algos"][algo] = {
            "final_ratio_median": float(final_ratio_median),
            "ratio0_median": float(ratio0_median),
            "success_rate": float(success_rate),
            "median_hit_calls": float(median_hit_calls) if median_hit_calls == median_hit_calls else None,
            "median_hit_walltime_s": float(median_hit_time) if median_hit_time == median_hit_time else None,
            "by_split": by_split,
        }

    norm_final = _minmax_norm([med_final[a] for a in algos])
    norm_calls = _minmax_norm([med_calls[a] for a in algos])

    for i, algo in enumerate(algos):
        nf = norm_final["norm"][i] if i < len(norm_final["norm"]) else float('nan')
        nc = norm_calls["norm"][i] if i < len(norm_calls["norm"]) else float('nan')
        sr = summary["algos"][algo]["success_rate"]
        nf = 0.5 if not (nf == nf) else nf
        nc = 0.5 if not (nc == nc) else nc
        sr = 0.0 if not (sr == sr) else sr
        composite = 0.5 * nf + 0.3 * sr + 0.2 * (1.0 - nc)
        summary["algos"][algo]["composite_score_0_1"] = float(composite)

    ranking = sorted(((a, summary["algos"][a]["composite_score_0_1"]) for a in algos),
                     key=lambda x: x[1], reverse=True)
    summary["ranking"] = [{"algo": a, "score": float(s)} for a, s in ranking]

    out_path = os.path.join(outdir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("METRICS:", json.dumps(summary["ranking"]))
    return out_path


def main(argv=None):
    cfg = RunConfig.from_cli(argv)

    os.makedirs(cfg.outdir, exist_ok=True)
    np.random.seed(cfg.seed)

    # Build splits
    train_seeds = list(range(cfg.seed, cfg.seed + cfg.num_train))
    test_seeds_id = list(range(cfg.seed + 10_000, cfg.seed + 10_000 + cfg.num_test_id))
    test_seeds_ood = list(range(cfg.seed + 20_000, cfg.seed + 20_000 + cfg.num_test_ood))

    def make_instances(seeds, p_graph):
        from .instances import build_instance
        return [build_instance(sd, n=cfg.N, p=p_graph, mode=cfg.weight_mode, eps_floor=cfg.eps_floor) for sd in seeds]

    train_insts   = make_instances(train_seeds, cfg.p)
    test_insts_id = make_instances(test_seeds_id, cfg.p)
    test_insts_ood= make_instances(test_seeds_ood, cfg.p_ood)

    # λ grids
    lams_ID  = np.linspace(-1.0, 1.0, num=5)
    lams_OOD = np.array([-1.6, -1.3, -1.2, 1.2, 1.3, 1.6])

    # QAOA machinery
    qshape = QAOAShapes(n=cfg.N, p=cfg.p_layers)
    zz_qnode, pairs = qaoa_qnode(cfg.N, cfg.p_layers, shots=cfg.shots)

    # CVaR sampling node
    sample_qnode = None
    if cfg.cvar_alpha < 1.0:
        if cfg.shots is None or int(cfg.shots) <= 0:
            raise ValueError("For CVaR set --shots > 0 (sampling required).")
        sample_qnode, _ = qaoa_sample_qnode(cfg.N, cfg.p_layers, shots=cfg.shots)

    # Meta models
    meta_qaoa = MetaPolyRBF(out_dim=qshape.size, deg=cfg.deg, R=cfg.R, seed=cfg.seed + 111)
    meta_class = MetaPolyRBF(out_dim=cfg.N,         deg=cfg.deg, R=cfg.R, seed=cfg.seed + 222)

    # Objectives
    def qaoa_objective_from_params(params_vec, inst, lam):
        if cfg.cvar_alpha < 1.0:
            return qaoa_energy_cvar(inst, lam, params_vec, qshape, sample_qnode, pairs, cfg.cvar_alpha)
        else:
            return qaoa_energy(inst, lam, params_vec, qshape, zz_qnode, pairs)

    classical = ClassicalProduct(n=cfg.N)
    def classical_objective_from_params(theta, inst, lam):
        rng = np.random.default_rng(42)
        shots_class = cfg.shots if (cfg.shots is not None and cfg.shots > 0) else cfg.classical_shots
        Es, calls = classical.energy_samples(inst, lam, theta, shots=shots_class, rng=rng)
        if cfg.cvar_alpha < 1.0:
            alpha = float(cfg.cvar_alpha)
            m = len(Es)
            k = max(1, int(np.ceil(alpha * m)))
            idx = np.argpartition(Es, -k)[-k:]
            return float(np.mean(Es[idx])), calls
        else:
            return float(np.mean(Es)), calls

    # Meta-Training
    def es_train_wrapper(model, train_insts, objective, name: str):
        return es_train_driver(model, train_insts, objective_from_params=objective,
                               lam_min=-1.0, lam_max=1.0,
                               M=cfg.M, batch=cfg.batch,
                               es_eps=cfg.es_eps, es_eta=cfg.es_eta,
                               inner_steps=1, a_spsa=cfg.a_meta, c_spsa=cfg.c_meta,
                               seed=cfg.seed + (101 if name == 'qaoa' else 202))

    import time as _time
    t0 = _time.perf_counter()
    meta_qaoa = es_train_wrapper(meta_qaoa, train_insts, qaoa_objective_from_params, name='qaoa')
    t_meta_qaoa = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    meta_class = es_train_wrapper(meta_class, train_insts, classical_objective_from_params, name='class')
    t_meta_class = _time.perf_counter() - t0

    # Evaluation
    def do_eval(insts, lams, tag):
        res_qaoa = time_to_target(insts, lams, target_ratio=cfg.target,
                                  algo_name="opt-meta-qaoa",
                                  theta_of_lambda=lambda L: meta_qaoa.g(L),
                                  objective_from_params=qaoa_objective_from_params,
                                  Kmax=cfg.K, a_spsa=cfg.a_spsa, c_spsa=cfg.c_spsa,
                                  seed=cfg.seed + 333)
        res_qaoa["split"] = tag
        res_qaoa_zero = res_qaoa.copy()
        res_qaoa_zero["algo"] = "meta-qaoa"
        res_qaoa_zero["final_ratio"] = res_qaoa_zero["ratio0"]

        res_class = time_to_target(insts, lams, target_ratio=cfg.target,
                                   algo_name="opt-meta-classical",
                                   theta_of_lambda=lambda L: meta_class.g(L),
                                   objective_from_params=classical_objective_from_params,
                                   Kmax=cfg.K, a_spsa=cfg.a_spsa, c_spsa=cfg.c_spsa,
                                   seed=cfg.seed + 444)
        res_class["split"] = tag
        res_class_zero = res_class.copy()
        res_class_zero["algo"] = "meta-classical"
        res_class_zero["final_ratio"] = res_class_zero["ratio0"]

        return pd.concat([res_qaoa, res_qaoa_zero, res_class, res_class_zero], ignore_index=True)

    res = pd.concat([
        do_eval(test_insts_id,  lams_ID,  "ID-graph / ID-λ"),
        do_eval(test_insts_id,  lams_OOD, "ID-graph / OOD-λ"),
        do_eval(test_insts_ood, lams_OOD, "OOD-graph / OOD-λ"),
    ], ignore_index=True)

    # Save raw results
    out_csv = os.path.join(cfg.outdir, "results_meta_qaoa_vs_classical.csv")
    res.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

    # TTT summary
    res_all = res.copy()
    res_all["success"] = res_all["final_ratio"] >= cfg.target

    def _nanquant(x, q):
        arr = np.asarray([v for v in x if v == v], dtype=float)
        return float(np.percentile(arr, q)) if len(arr) else float('nan')

    ttt = (res_all.groupby(["algo", "split"], dropna=False)
           .agg(median_calls=("hit_calls", lambda x: np.nanmedian(x)),
                q25=("hit_calls", lambda x: _nanquant(x, 25)),
                q75=("hit_calls", lambda x: _nanquant(x, 75)),
                median_time=("hit_walltime", lambda x: np.nanmedian(x)),
                success_rate=("success", "mean"))
           .reset_index())
    ttt_csv = os.path.join(cfg.outdir, "time_to_target.csv")
    ttt.to_csv(ttt_csv, index=False)
    print(f"[Saved] {ttt_csv}")

    # Print OOD summaries
    for split in ["ID-graph / OOD-λ", "OOD-graph / OOD-λ"]:
        print(f"\n=== Time-to-Target ({split}) ===")
        print(ttt[ttt["split"] == split].to_string(index=False))
        agg = (res_all[res_all["split"] == split]
               .groupby("algo")
               .agg(median_final_ratio=("final_ratio", "median"),
                    median_ratio0=("ratio0", "median"))
               .reset_index())
        print(f"\n--- Quality (Median) {split} ---")
        print(agg.to_string(index=False))

    # Curves
    for split in ["ID-graph / ID-λ", "ID-graph / OOD-λ", "OOD-graph / OOD-λ"]:
        df1 = aggregate_curves(res, "meta-qaoa", split, which="final_ratio")
        df2 = aggregate_curves(res, "opt-meta-qaoa", split, which="final_ratio")
        df3 = aggregate_curves(res, "meta-classical", split, which="final_ratio")
        df4 = aggregate_curves(res, "opt-meta-classical", split, which="final_ratio")
        plot_curve([("meta-qaoa", df1), ("opt-meta-qaoa", df2), ("meta-classical", df3), ("opt-meta-classical", df4)],
                   title=f"{split} — Few-Shot (K={cfg.K})",
                   out_png=os.path.join(cfg.outdir, f"curve_{split.replace(' ','_').replace('/','_')}.png"))

    # Fairness report
    with open(os.path.join(cfg.outdir, "fairness.txt"), "w", encoding="utf-8") as f:
        f.write(f"Objective: {'CVaR(alpha='+str(cfg.cvar_alpha)+')' if cfg.cvar_alpha < 1.0 else 'Expected energy'}\n")
        f.write(f"QAOA params: 2p={qshape.size}, Meta dims: {meta_qaoa.phi.size}\n")
        f.write(f"Classical params: n={cfg.N}, Meta dims: {meta_class.phi.size}\n")
        f.write(f"Meta-Train: M={cfg.M}, batch={cfg.batch}, es_eps={cfg.es_eps}, es_eta={cfg.es_eta}, inner=1\n")
        f.write(f"SPSA test: K={cfg.K}, a={cfg.a_spsa}, c={cfg.c_spsa}\n")
        f.write(f"Shots per energy eval: qaoa={cfg.shots if cfg.shots else 0} | classical={cfg.shots if cfg.shots else cfg.classical_shots}\n")
        f.write(f"Meta wall-time: qaoa={t_meta_qaoa:.3f}s | classical={t_meta_class:.3f}s\n")
        f.write(f"Graphs: train={cfg.num_train}@p={cfg.p}, test_id={cfg.num_test_id}, test_ood={cfg.num_test_ood}@p_ood={cfg.p_ood}\n")
        f.write(f"Weight mode: {cfg.weight_mode}, eps_floor={cfg.eps_floor}\n")
        f.write("Lambdas train: U[-1,1], test grids: ID=[-1,-0.5,0,0.5,1], OOD=[-1.6,-1.3,-1.2,1.2,1.3,1.6]\n")
    print(f"[Saved] {os.path.join(cfg.outdir, 'fairness.txt')}")

    # Metrics
    metrics_path = export_metrics(res, ttt, cfg.outdir, cfg.target)
    print(f"[Saved] {metrics_path}")
