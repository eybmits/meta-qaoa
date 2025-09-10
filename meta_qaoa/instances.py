
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import numpy as np


@dataclass
class Instance:
    n: int
    edges: List[Tuple[int, int]]
    w0: Dict[Tuple[int, int], float]
    mode: str
    eps_floor: float
    b: Dict[Tuple[int, int], float]
    c: Dict[Tuple[int, int], float]


def build_instance(seed: int, n: int = 8, p: float = 0.4, mode: str = "abs_floor", eps_floor: float = 0.02) -> Instance:
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    w0 = {e: float(rng.uniform(-1.0, 1.0)) for e in edges}
    b = {e: float(rng.uniform(-1.0, 1.0)) for e in edges}
    c = {e: float(rng.uniform(-1.0, 1.0)) for e in edges}
    return Instance(n=n, edges=edges, w0=w0, mode=mode, eps_floor=eps_floor, b=b, c=c)


def weight_ij(inst: Instance, lam: float, e: Tuple[int, int]) -> float:
    raw = inst.w0[e] + inst.b[e] * lam + inst.c[e] * (lam**2)
    if inst.mode == "abs_floor":
        return abs(raw) + inst.eps_floor
    elif inst.mode == "softplus":
        return float(np.log1p(np.exp(raw))) + inst.eps_floor
    elif inst.mode == "signed_norm":
        return raw
    else:
        raise ValueError("Unknown mode")


def norm_signed_sum(inst: Instance, lam: float) -> float:
    return max(1e-9, sum(abs(weight_ij(inst, lam, e)) for e in inst.edges))


def oracle_best(inst: Instance, lam: float) -> float:
    """Exact Max-Cut energy via brute force (small n)."""
    n = inst.n
    best = -math.inf
    norm = norm_signed_sum(inst, lam) if inst.mode == "signed_norm" else 1.0
    for x in range(1 << n):
        e = 0.0
        for (i, j) in inst.edges:
            si = 1.0 if ((x >> i) & 1) == 0 else -1.0
            sj = 1.0 if ((x >> j) & 1) == 0 else -1.0
            w = weight_ij(inst, lam, (i, j))
            if inst.mode == "signed_norm":
                w = w / norm
            e += 0.5 * w * (1.0 - si * sj)
        if e > best:
            best = e
    return float(best)
