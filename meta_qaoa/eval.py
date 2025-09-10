
from __future__ import annotations
from typing import Callable, Sequence
import math, time
import numpy as np
import pandas as pd
from .instances import Instance, oracle_best
from .optim import spsa_step


def time_to_target(insts: Sequence[Instance], lambdas: np.ndarray, target_ratio: float,
                   algo_name: str,
                   theta_of_lambda: Callable[[float], np.ndarray],
                   objective_from_params: Callable[[np.ndarray, Instance, float], tuple[float, int]],
                   Kmax: int, a_spsa: float, c_spsa: float,
                   seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for idx, inst in enumerate(insts):
        for lam in lambdas:
            lamf = float(lam)
            E_star = oracle_best(inst, lamf)
            theta = theta_of_lambda(lamf)
            calls_total = 0
            t0 = time.perf_counter()

            # Zero-Shot
            E0, calls = objective_from_params(theta, inst, lamf)
            calls_total += calls
            ratio0 = E0 / max(E_star, 1e-12)

            hit_k = -1
            hit_calls = math.nan
            hit_time = math.nan

            bestE = E0
            if (E_star > 0) and (E0 >= target_ratio * E_star):
                hit_k = 0
                hit_calls = calls_total
                hit_time = time.perf_counter() - t0

            # Few-Shot SPSA
            for k in range(1, Kmax + 1):
                def obj_th(th_vec):
                    return objective_from_params(th_vec, inst, lamf)
                theta, calls_k, _ = spsa_step(theta, obj_th, a=a_spsa, c=c_spsa,
                                               seed=rng.integers(1_000_000_000))
                calls_total += calls_k
                Ecur, calls_eval = objective_from_params(theta, inst, lamf)
                calls_total += calls_eval
                bestE = max(bestE, Ecur)
                if (hit_k < 0) and (E_star > 0) and (Ecur >= target_ratio * E_star):
                    hit_k = k
                    hit_calls = calls_total
                    hit_time = time.perf_counter() - t0

            rows.append({
                "algo": algo_name,
                "inst_id": idx,
                "lambda": float(lamf),
                "E_star": float(E_star),
                "E0": float(E0),
                "ratio0": float(ratio0),
                "final_ratio": float(bestE / max(E_star, 1e-12)),
                "hit_k": int(hit_k),
                "hit_calls": float(hit_calls) if hit_k >= 0 else math.nan,
                "hit_walltime": float(hit_time) if hit_k >= 0 else math.nan,
            })
    return pd.DataFrame(rows)


def aggregate_curves(res: pd.DataFrame, algo: str, split: str, which: str = "final_ratio") -> pd.DataFrame:
    sub = res[(res["algo"] == algo) & (res["split"] == split)]
    rows = []
    for L in sorted(sub["lambda"].unique()):
        vals = sub[sub["lambda"] == L][which].values
        if len(vals) == 0:
            continue
        rows.append({
            "lambda": float(L),
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "n": int(len(vals)),
        })
    return pd.DataFrame(rows)


def safe_median(xs) -> float:
    import numpy as np
    vals = [float(x) for x in xs if x == x and np.isfinite(x)]
    return float(np.median(vals)) if vals else float('nan')
