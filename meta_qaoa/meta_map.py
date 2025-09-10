
from __future__ import annotations
import numpy as np


class MetaPolyRBF:
    def __init__(self, out_dim: int, deg: int = 2, R: int = 5,
                 lam_min: float = -1.0, lam_max: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.out_dim = out_dim
        self.deg = deg
        self.R = R
        self.centers = np.linspace(lam_min, lam_max, R)
        d = np.median(np.diff(self.centers)) if R > 1 else 1.0
        self.sigma = max(1e-6, d)
        self.A = 0.1 * rng.standard_normal((out_dim, deg + 1))
        self.C = 0.1 * rng.standard_normal((out_dim, R))

    @property
    def phi(self) -> np.ndarray:
        return np.concatenate([self.A.ravel(), self.C.ravel()])

    @phi.setter
    def phi(self, vec: np.ndarray):
        D = self.deg + 1
        An = self.out_dim * D
        self.A = vec[:An].reshape(self.out_dim, D)
        self.C = vec[An:].reshape(self.out_dim, self.R)

    def g(self, lam: float) -> np.ndarray:
        poly = np.array([lam ** k for k in range(self.deg + 1)])
        rbf = np.exp(-0.5 * ((lam - self.centers) ** 2) / (self.sigma ** 2))
        return self.A @ poly + self.C @ rbf
