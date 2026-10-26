import numpy as np
from scipy.stats import spearmanr
from cpl.learners import SLearner, TLearner, XLearner


def _toy(n, seed, kappa=0.0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 4)); U = r.normal(size=n)
    tau = np.column_stack([1.0 * (X[:, 0] > 0), 0.5 + X[:, 1], -0.5 + 2 * (X[:, 2] > 0.5)])
    logit = np.column_stack([np.zeros(n), 0.5 * X[:, 0], -0.5 * X[:, 1], 0.3 * X[:, 2]])
    logit[:, 1:] += kappa * U[:, None] * np.array([1.0, -1.0, 0.5])
    e = np.exp(logit); e /= e.sum(1, keepdims=True)
    T = (e.cumsum(1) < r.random((n, 1))).sum(1)
    mu0 = X[:, 3] + kappa * U
    Y = mu0 + np.where(T > 0, tau[np.arange(n), np.clip(T - 1, 0, 2)], 0) + r.normal(size=n)
    return X, T, Y, tau


def test_shapes_and_recovery_when_no_hidden_confounding():
    X, T, Y, _ = _toy(20000, 0)
    Xn, _, _, tau = _toy(5000, 1)
    for L in (SLearner, TLearner, XLearner):
        th = L(seed=0).fit(X, T, Y).predict_tau(Xn)
        assert th.shape == (5000, 3)
        for k in range(3):
            assert abs(th[:, k].mean() - tau[:, k].mean()) < 0.2, (L.__name__, k)
            assert spearmanr(th[:, k], tau[:, k])[0] > 0.5, (L.__name__, k)


def test_deterministic():
    X, T, Y, _ = _toy(4000, 0)
    a = XLearner(seed=3).fit(X, T, Y).predict_tau(X[:100])
    b = XLearner(seed=3).fit(X, T, Y).predict_tau(X[:100])
    assert np.allclose(a, b)
