
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import argparse
import yaml


@dataclass
class RunConfig:
    # Problem / Graphs
    N: int = 8
    p: float = 0.4
    p_ood: float = 0.55
    num_train: int = 60
    num_test_id: int = 16
    num_test_ood: int = 16
    weight_mode: str = "abs_floor"  # abs_floor|softplus|signed_norm
    eps_floor: float = 0.02

    # QAOA
    p_layers: int = 2
    shots: Optional[int] = 2048  # None -> analytic expectations where possible

    # Meta models
    deg: int = 2
    R: int = 5

    # Meta-Training (ES)
    M: int = 18
    batch: int = 12
    es_eps: float = 0.08
    es_eta: float = 0.35
    a_meta: float = 0.08
    c_meta: float = 0.06

    # Few-Shot (Test/opt-meta)
    K: int = 20
    a_spsa: float = 0.08
    c_spsa: float = 0.06

    # Classical baseline shots (when shots is None for QAOA expectation)
    classical_shots: int = 2048

    # Eval target
    target: float = 0.98

    # CVaR
    cvar_alpha: float = 1.0

    # General
    seed: int = 1234
    outdir: str = "outputs"

    @staticmethod
    def _parse_shots(s: Optional[str]) -> Optional[int]:
        if s is None:
            return None
        low = s.strip().lower()
        if low in {"none", "null", "auto"}:
            return None
        return int(s)

    @classmethod
    def from_cli(cls, argv=None) -> "RunConfig":
        ap = argparse.ArgumentParser(description="Meta-QAOA vs. Classical Baseline (PennyLane)")
        ap.add_argument("--config", type=str, default=None, help="YAML config path")

        # Mirror dataclass fields for overrides
        ap.add_argument("--N", type=int)
        ap.add_argument("--p", type=float)
        ap.add_argument("--p-ood", dest="p_ood", type=float)
        ap.add_argument("--num-train", dest="num_train", type=int)
        ap.add_argument("--num-test-id", dest="num_test_id", type=int)
        ap.add_argument("--num-test-ood", dest="num_test_ood", type=int)
        ap.add_argument("--weight-mode", type=str, choices=["abs_floor", "softplus", "signed_norm"])
        ap.add_argument("--eps-floor", dest="eps_floor", type=float)

        ap.add_argument("--p-layers", dest="p_layers", type=int)
        ap.add_argument("--shots", dest="shots", type=str)  # parse to Optional[int]

        ap.add_argument("--deg", type=int)
        ap.add_argument("--R", type=int)

        ap.add_argument("--M", type=int)
        ap.add_argument("--batch", type=int)
        ap.add_argument("--es-eps", dest="es_eps", type=float)
        ap.add_argument("--es-eta", dest="es_eta", type=float)
        ap.add_argument("--a-meta", dest="a_meta", type=float)
        ap.add_argument("--c-meta", dest="c_meta", type=float)

        ap.add_argument("--K", type=int)
        ap.add_argument("--a-spsa", dest="a_spsa", type=float)
        ap.add_argument("--c-spsa", dest="c_spsa", type=float)

        ap.add_argument("--classical-shots", dest="classical_shots", type=int)

        ap.add_argument("--target", type=float)

        ap.add_argument("--cvar-alpha", dest="cvar_alpha", type=float)

        ap.add_argument("--seed", type=int)
        ap.add_argument("--outdir", type=str)

        args = ap.parse_args(argv)

        # Start with defaults or YAML
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = cls(**yaml.safe_load(f))
        else:
            cfg = cls()

        # Apply overrides if given
        for field in cfg.__dataclass_fields__:
            cli_name = field
            if field == "p_ood":
                cli_name = "p_ood"
            val = getattr(args, cli_name, None)
            if val is not None:
                if field == "shots":
                    setattr(cfg, field, cls._parse_shots(val))
                else:
                    setattr(cfg, field, val)

        return cfg
