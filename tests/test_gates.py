import numpy as np
from cpl.gates import check_gates, paired
from cpl.estimators import make_estimator, REGISTRY


def _res(shift):
    r = np.random.default_rng(0)
    base = lambda v: {"ate_err": r.normal(0, .05, (5, 3)), "spearman": r.normal(.5, .05, (5, 3)),
                      "value": v}
    x = np.array([.08, .09, .10, .07, .09])
    d = {s: {"X": base(x), "L": base(x + shift)} for s in ("rho0", "kappa0")}
    return d


def test_gate4_flags_a_clear_loser_and_passes_a_tie():
    ok = {g[0]: g[1] for g in check_gates(_res(+0.0), "L")}
    bad = {g[0]: g[1] for g in check_gates(_res(-0.05), "L")}
    assert all(v for k, v in ok.items() if k.startswith("G4"))
    assert not any(v for k, v in bad.items() if k.startswith("G4"))


def test_paired_and_registry():
    m, se = paired([1, 2, 3], [1, 1, 1]); assert abs(m - 1) < 1e-9 and se > 0
    assert set(REGISTRY) >= {"S", "T", "X", "DR-S", "Forest"}
    assert hasattr(make_estimator("DR-S", 1), "predict_tau")
