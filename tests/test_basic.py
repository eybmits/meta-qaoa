
from meta_qaoa.instances import build_instance, oracle_best, weight_ij
from meta_qaoa.qaoa import QAOAShapes, qaoa_qnode, qaoa_energy
import numpy as np


def test_instance_weights():
    inst = build_instance(seed=0, n=4, p=0.9, mode="abs_floor", eps_floor=0.02)
    lam = 0.3
    # weights are positive with abs_floor
    for e in inst.edges:
        assert weight_ij(inst, lam, e) > 0.0


def test_oracle_best_runs():
    inst = build_instance(seed=1, n=4, p=0.9, mode="abs_floor", eps_floor=0.02)
    lam = -0.2
    val = oracle_best(inst, lam)
    assert np.isfinite(val)


def test_qaoa_expectation_runs():
    n, p = 4, 1
    inst = build_instance(seed=2, n=n, p=1.0, mode="abs_floor", eps_floor=0.02)
    lam = 0.1
    shape = QAOAShapes(n=n, p=p)
    zz, pairs = qaoa_qnode(n, p, shots=None)  # analytic expectations
    params = np.zeros(shape.size, dtype=float)
    E, calls = qaoa_energy(inst, lam, params, shape, zz, pairs)
    assert np.isfinite(E)
    assert calls == 0  # analytic mode should report zero shots
