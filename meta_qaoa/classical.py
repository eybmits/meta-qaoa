
from __future__ import annotations
import numpy as np
from typing import Tuple
from .instances import Instance, norm_signed_sum, weight_ij


class ClassicalProduct:
    """Independent-spin product baseline; p_i = sigmoid(theta_i)."""
    def __init__(self, n: int):
        self.n = n

    def energy_samples(self, inst: Instance, lam: float, theta: np.ndarray,
                       shots: int, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
        p = 1.0 / (1.0 + np.exp(-theta))
        u = rng.random((shots, self.n))
        spins = np.where(u < p, +1.0, -1.0)
        norm = norm_signed_sum(inst, lam) if inst.mode == "signed_norm" else 1.0
        E = np.zeros(shots, dtype=float)
        for k in range(shots):
            s = spins[k]
            e = 0.0
            for (i, j) in inst.edges:
                w = weight_ij(inst, lam, (i, j))
                if inst.mode == "signed_norm":
                    w = w / norm
                e += 0.5 * w * (1.0 - s[i] * s[j])
            E[k] = e
        return E, shots
