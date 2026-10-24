import numpy as np
from cpl.propensity import (fit_propensity, overlap_diagnostics,
                            clip_propensity, ess, N_ARMS)


def _toy(n=20000, rho=1.0, kappa=0.0, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 5)); U = r.normal(size=n)
    W = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 1, 0]], float)
    w = np.array([0, 1, -1, 0.5])
    logit = rho * (X @ W.T + kappa * np.outer(U, w))
    e = np.exp(logit - logit.max(1, keepdims=True)); e /= e.sum(1, keepdims=True)
    T = np.array([r.choice(4, p=p) for p in e])
    # e_x = propensity given X only (what a model could at best learn)
    return X, T, e


def test_rows_sum_to_one_and_shape():
    X, T, _ = _toy(4000)
    for kind in ("logistic", "lgbm"):
        e = fit_propensity(X, T, kind=kind)
        assert e.shape == (4000, N_ARMS)
        assert np.allclose(e.sum(1), 1) and (e > 0).all()


def test_logistic_recovers_known_propensity_when_kappa_zero():
    X, T, e_true = _toy(20000, kappa=0.0)
    e = fit_propensity(X, T, kind="logistic")
    d = overlap_diagnostics(e, T, e_true=e_true)
    assert all(d["arms"][a]["mae_vs_true"] < 0.02 for a in range(4))
    assert abs(d["log_loss"] - d["log_loss_true"]) < 0.02


def test_hidden_confounding_leaves_gap_vs_true_propensity():
    X, T, e_true = _toy(20000, kappa=2.0)
    e = fit_propensity(X, T, kind="logistic")
    d = overlap_diagnostics(e, T, e_true=e_true)
    # model cannot see U, so it cannot match e(a|X,U)
    assert max(d["arms"][a]["mae_vs_true"] for a in range(4)) > 0.03


def test_diagnostics_and_clip():
    X, T, _ = _toy(4000)
    e = fit_propensity(X, T)
    d = overlap_diagnostics(e, T)
    assert 0 < d["arms"][1]["ess_ratio"] <= 1
    c = clip_propensity(e, 0.05)
    assert np.allclose(c.sum(1), 1) and c.min() >= 0.05 / 1.2 - 1e-9
    assert abs(ess(np.ones(100)) - 100) < 1e-9


def test_deterministic():
    X, T, _ = _toy(3000)
    assert np.allclose(fit_propensity(X, T, "lgbm", seed=1), fit_propensity(X, T, "lgbm", seed=1))
