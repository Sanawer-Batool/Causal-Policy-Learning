import numpy as np
from scipy.stats import spearmanr
from cpl.dr_learner import dr_pseudo_outcomes, DRLearner


def _toy(n, seed):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 4))
    tau = np.column_stack([1.0 * (X[:, 0] > 0), 0.5 + X[:, 1], -0.5 + 2 * (X[:, 2] > 0.5)])
    logit = np.column_stack([np.zeros(n), 0.8 * X[:, 0], -0.8 * X[:, 1], 0.5 * X[:, 2]])
    e = np.exp(logit); e /= e.sum(1, keepdims=True)
    T = (e.cumsum(1) < r.random((n, 1))).sum(1)
    mu = np.column_stack([X[:, 3]] + [X[:, 3] + tau[:, k] for k in range(3)])
    Y = mu[np.arange(n), T] + r.normal(size=n)
    return X, T, Y, tau, mu, e


def _err(phi, tau):
    return np.abs(phi.mean(0) - tau.mean(0))


def test_pseudo_outcome_double_robustness():
    X, T, Y, tau, mu, e = _toy(60000, 0)
    wrong_mu, wrong_e = np.zeros_like(mu), np.full_like(e, 0.25)
    assert _err(dr_pseudo_outcomes(mu, e, T, Y), tau).max() < 0.1              # both right
    assert _err(dr_pseudo_outcomes(wrong_mu, e, T, Y), tau).max() < 0.15       # e right
    assert _err(dr_pseudo_outcomes(mu, wrong_e, T, Y), tau).max() < 0.1        # mu right
    assert _err(dr_pseudo_outcomes(wrong_mu, wrong_e, T, Y), tau).max() > 0.2  # both wrong


def test_fit_predict_recovers_ranking_and_ate():
    X, T, Y, _, _, _ = _toy(20000, 0)
    Xn, _, _, tau, _, _ = _toy(5000, 1)
    m = DRLearner(seed=0).fit(X, T, Y)
    th = m.predict_tau(Xn)
    assert th.shape == (5000, 3)
    assert np.abs(m.ate_aipw_ - tau.mean(0)).max() < 0.15
    for k in range(3):
        assert spearmanr(th[:, k], tau[:, k])[0] > 0.5


def test_deterministic_and_s_nuisance():
    X, T, Y, *_ = _toy(4000, 0)
    a = DRLearner(seed=2).fit(X, T, Y).predict_tau(X[:50])
    b = DRLearner(seed=2).fit(X, T, Y).predict_tau(X[:50])
    assert np.allclose(a, b)
    assert DRLearner(seed=2, nuisance="s").fit(X, T, Y).predict_tau(X[:50]).shape == (50, 3)
