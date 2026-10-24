# Causal Policy Learning & Counterfactual Decision System

Owner: Sanawar Batool. See Architecture Rev.2 and the Novatore brief for scope.

## Setup (Windows PowerShell)
```
py -3.11 -m venv .venv        # or: python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/demo_naive_failure.py
```

## Layout
- `src/cpl/simulator.py` - formal causal simulator (Arch. Sections 4-5)
- `configs/` - YAML configs (every experiment is config-driven, seeded)
- `tests/` - pytest suite
- `scripts/` - runnable entry points
- `results/` - JSON/CSV outputs (git-ignored)
- `AI_LOG.md` - AI-use log required by the brief

## Conventions
- Every random draw goes through `np.random.default_rng(seed)`.
- Models only ever see `SimData.observed`; `SimData.oracle` is for evaluation.
- Notation: `delta_a(x)` = logit-scale lift inside the simulator; `tau_a(x)` = revenue-scale causal effect (the estimand).
