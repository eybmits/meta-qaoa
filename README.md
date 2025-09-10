
# Meta-QAOA (PennyLane) — Professional Refactor

This repository compares **Meta-QAOA** (with entanglement-capable circuits) against a simple
**classical product baseline** for a *per-edge parametrized* weighted Max-Cut family.
It supports **CVaR** evaluation via sampling and **Expected Energy** via analytic expectations.

## Features
- Per-edge λ-dependent weights: `w_ij(λ) = |w0 + b λ + c λ^2| + ε` (or softplus / signed_norm)
- QAOA(p) with IsingZZ cost and RX mixer
- Meta parameterization: Polynomial + RBF map `g(λ)`
- ES meta-training with optional inner SPSA step to encourage few-shot adaptation
- Evaluation on ID and OOD splits; Time-to-Target (TTT), Zero-/Few-Shot curves
- Plots and CSV exports; neutral **metrics.json** summary (no special wording)

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quick Start
```bash
meta-qaoa --help

# Example run
meta-qaoa   --N 8 --p 0.4 --p-ood 0.55   --num-train 60 --num-test-id 16 --num-test-ood 16   --p-layers 2 --shots 2048   --deg 2 --R 5   --M 18 --batch 12 --es-eps 0.08 --es-eta 0.35   --K 20 --a-spsa 0.08 --c-spsa 0.06   --target 0.98 --cvar-alpha 1.0   --weight-mode abs_floor --eps-floor 0.02   --seed 1234 --outdir outputs
```

## Config-driven runs
You can also use a YAML config. See [`configs/default.yaml`](configs/default.yaml).

```bash
meta-qaoa --config configs/default.yaml
```

## Tests
```bash
pytest
```

## Repository Layout
```
meta_qaoa/
  __init__.py
  config.py
  instances.py
  classical.py
  qaoa.py
  meta_map.py
  optim.py
  eval.py
  plotting.py
  cli.py
tests/
  test_basic.py
configs/
  default.yaml
```

## License
MIT
