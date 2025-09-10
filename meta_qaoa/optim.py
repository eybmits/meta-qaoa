
from __future__ import annotations
from typing import Callable, Sequence, Tuple
import numpy as np
from .meta_map import MetaPolyRBF
from .instances import Instance


def spsa_step(theta: np.ndarray,
              objective: Callable[[np.ndarray], Tuple[float, int]],
              a: float, c: float, seed: int = 0) -> tuple[np.ndarray, int, float]:
    rng = np.random.default_rng(seed)
    Delta = rng.choice([-1.0, +1.0], size=theta.shape)
    E_plus, calls_p = objective(theta + c * Delta)
    E_minus, calls_m = objective(theta - c * Delta)
    ghat = (E_plus - E_minus) / (2.0 * c) * Delta
    new_theta = theta + a * ghat
    calls = calls_p + calls_m
    return new_theta, calls, float(max(E_plus, E_minus))


def es_train_driver(model: MetaPolyRBF, train_insts: Sequence[Instance],
                    objective_from_params: Callable[[np.ndarray, Instance, float], tuple[float, int]],
                    lam_min: float, lam_max: float,
                    M: int, batch: int,
                    es_eps: float, es_eta: float,
                    inner_steps: int, a_spsa: float, c_spsa: float,
                    seed: int) -> MetaPolyRBF:
    rng = np.random.default_rng(seed)
    phi = model.phi.copy()
    for _ in range(M):
        lams = rng.uniform(lam_min, lam_max, size=batch)
        insts = rng.choice(train_insts, size=batch, replace=True)
        tasks = list(zip(insts, lams))
        Delta = rng.choice([-1.0, +1.0], size=phi.shape)

        def meta_obj(phi_vec: np.ndarray) -> float:
            model.phi = phi_vec
            vals = []
            for inst, lam in tasks:
                theta0 = model.g(float(lam))
                theta = theta0.copy()
                if inner_steps > 0:
                    rng_loc = np.random.default_rng(rng.integers(1_000_000_000))
                    Delta_th = rng_loc.choice([-1.0, +1.0], size=theta.shape)
                    E_plus, _ = objective_from_params(theta + c_spsa * Delta_th, inst, float(lam))
                    E_minus, _ = objective_from_params(theta - c_spsa * Delta_th, inst, float(lam))
                    ghat = (E_plus - E_minus) / (2.0 * c_spsa) * Delta_th
                    theta = theta + a_spsa * ghat
                E_val, _ = objective_from_params(theta, inst, float(lam))
                vals.append(E_val)
            return float(np.mean(vals))

        Jp = meta_obj(phi + es_eps * Delta)
        Jm = meta_obj(phi - es_eps * Delta)
        grad = (Jp - Jm) / (2.0 * es_eps) * Delta
        phi = phi + es_eta * grad
        model.phi = phi
    return model
