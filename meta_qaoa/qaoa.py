
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pennylane as qml
from .instances import Instance, norm_signed_sum, weight_ij


def _total_shots_from_device(dev) -> int:
    try:
        from pennylane.measurements import Shots  # type: ignore
        sh = getattr(dev, "shots", None)
        if isinstance(sh, Shots):
            return int(sh.total_shots or 0)
        if sh is None:
            return 0
        return int(sh)
    except Exception:
        try:
            sh = getattr(dev, "shots", None)
            return int(sh) if sh is not None else 0
        except Exception:
            return 0


@dataclass
class QAOAShapes:
    n: int
    p: int
    @property
    def size(self) -> int:
        return 2 * self.p


def qaoa_qnode(n: int, p: int, shots: Optional[int]):
    dev = qml.device("default.qubit", wires=n, shots=shots)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    @qml.qnode(dev, interface=None)
    def zz_expectations(lam: float, params: np.ndarray, weighted_angles: np.ndarray):
        betas = params[:p]
        gammas = params[p:]
        for i in range(n):
            qml.Hadamard(wires=i)
        for l in range(p):
            kk = 0
            for (i, j) in pairs:
                theta = weighted_angles[l, kk]
                qml.IsingZZ(theta if np.isfinite(theta) else 0.0, wires=[i, j])
                kk += 1
            b = betas[l]
            for i in range(n):
                qml.RX(2.0 * b, wires=i)
        return [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for (i, j) in pairs]

    return zz_expectations, pairs


def qaoa_sample_qnode(n: int, p: int, shots: Optional[int]):
    if shots is None or shots <= 0:
        raise ValueError("Sampling QNode requires positive shots.")
    dev = qml.device("default.qubit", wires=n, shots=shots)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    @qml.qnode(dev, interface=None)
    def sample_bits(lam: float, params: np.ndarray, weighted_angles: np.ndarray):
        betas = params[:p]
        gammas = params[p:]
        for i in range(n):
            qml.Hadamard(wires=i)
        for l in range(p):
            kk = 0
            for (i, j) in pairs:
                theta = weighted_angles[l, kk]
                qml.IsingZZ(theta if np.isfinite(theta) else 0.0, wires=[i, j])
                kk += 1
            b = betas[l]
            for i in range(n):
                qml.RX(2.0 * b, wires=i)
        return qml.sample(wires=range(n))

    return sample_bits, pairs


def qaoa_energy(inst: Instance, lam: float, params: np.ndarray, pshape: QAOAShapes,
                zz_qnode, pairs: List[Tuple[int, int]]) -> tuple[float, int]:
    betas = params[:pshape.p]
    gammas = params[pshape.p:]
    _ = betas
    num_pairs = len(pairs)
    weighted = np.zeros((pshape.p, num_pairs), dtype=float)
    norm = norm_signed_sum(inst, lam) if inst.mode == "signed_norm" else 1.0
    ws = []
    for (i, j) in pairs:
        if (i, j) in inst.edges:
            w = weight_ij(inst, lam, (i, j))
            if inst.mode == "signed_norm":
                w = w / norm
        else:
            w = 0.0
        ws.append(w)
    ws = np.asarray(ws, dtype=float)
    for l in range(pshape.p):
        weighted[l, :] = -gammas[l] * ws
    zz_vals = np.array(zz_qnode(lam, np.asarray(params, dtype=float), weighted))
    E = 0.0
    for idx, (i, j) in enumerate(pairs):
        if (i, j) not in inst.edges:
            continue
        w = weight_ij(inst, lam, (i, j))
        if inst.mode == "signed_norm":
            w = w / norm
        vij = zz_vals[idx]
        E += 0.5 * w * (1.0 - vij)
    dev = zz_qnode.device
    shots = _total_shots_from_device(dev)
    return float(E), int(shots)


def qaoa_energy_cvar(inst: Instance, lam: float, params: np.ndarray, pshape: QAOAShapes,
                     sample_qnode, pairs: List[Tuple[int, int]], alpha: float) -> tuple[float, int]:
    gammas = params[pshape.p:]
    num_pairs = len(pairs)
    weighted = np.zeros((pshape.p, num_pairs), dtype=float)
    norm = norm_signed_sum(inst, lam) if inst.mode == "signed_norm" else 1.0
    ws = []
    for (i, j) in pairs:
        w = weight_ij(inst, lam, (i, j)) if (i, j) in inst.edges else 0.0
        ws.append(w / norm if inst.mode == "signed_norm" else w)
    ws = np.asarray(ws, dtype=float)
    for l in range(pshape.p):
        weighted[l, :] = -gammas[l] * ws
    bits = np.asarray(sample_qnode(lam, np.asarray(params, dtype=float), weighted))  # (shots, n)
    spins = 1.0 - 2.0 * bits
    shots = spins.shape[0]
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
    # CVaR (top-alpha fraction) or mean if alpha=1
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0,1]")
    if alpha < 1.0:
        m = len(E)
        k = max(1, int(np.ceil(alpha * m)))
        idx = np.argpartition(E, -k)[-k:]
        val = float(np.mean(E[idx]))
    else:
        val = float(np.mean(E))
    dev = sample_qnode.device
    from .qaoa import _total_shots_from_device as _shots  # avoid circular import confusion
    return float(val), _shots(dev)
