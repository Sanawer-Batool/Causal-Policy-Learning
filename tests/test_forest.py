import numpy as np
from scipy.stats import spearmanr
from cpl.forest import CausalForest, EconmlDR
from test_dr import _toy


def test_forest_shapes_ranking_and_interval():
    X, T, Y, *_ = _toy(8000, 0)
    Xn, _, _, tau, _, _ = _toy(2000, 1)
    m = CausalForest(seed=0, n_estimators=200).fit(X, T, Y)
    th = m.predict_tau(Xn)
    lo, hi = m.predict_tau_interval(Xn)
    assert th.shape == lo.shape == hi.shape == (2000, 3)
    assert (lo <= th + 1e-9).all() and (th <= hi + 1e-9).all()
    for k in range(3):
        assert spearmanr(th[:, k], tau[:, k])[0] > 0.4


def test_econml_dr_matches_ours_roughly():
    from cpl.dr_learner import DRLearner
    X, T, Y, *_ = _toy(8000, 0)
    Xn, *_ = _toy(2000, 1)
    ours = DRLearner(seed=0, clip=0.02).fit(X, T, Y).predict_tau(Xn)
    theirs = EconmlDR(seed=0).fit(X, T, Y).predict_tau(Xn)
    assert theirs.shape == (2000, 3)
    for k in range(3):
        assert spearmanr(ours[:, k], theirs[:, k])[0] > 0.7
