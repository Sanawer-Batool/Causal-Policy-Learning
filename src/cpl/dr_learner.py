"""Phase 3c: DR-learner (Kennedy 2020), hand-written, 3 arms vs control.

Pseudo-outcome for arm a (a = 1, 2, 3):
  phi_a = mu_a(x) - mu_0(x) + 1{T=a}(Y - mu_a(x))/e_a(x) - 1{T=0}(Y - mu_0(x))/e_0(x)
Then regress phi_a on X. Nuisances mu (outcome) and e (propensity) are cross-fitted,
so no unit's phi uses a model that saw that unit. Propensities are clipped below.
It is consistent if EITHER mu or e is right, but it only adjusts for OBSERVED X.
Hidden U is not fixed by any amount of doubly-robust machinery.
"""
import numpy as np
from lightgbm import LGBMRegressor

from .learners import make_regressor, ARMS
from .propensity import fit_propensity, clip_propensity
from .splits import crossfit_predict


def dr_pseudo_outcomes(mu, e, T, Y):
    """mu, e: (n, 4) outcome predictions and propensities. Returns phi, shape (n, 3)."""
    n = len(T)
    idx = np.arange(n)
    c = (T == 0)
    ctrl = c * (Y - mu[:, 0]) / e[:, 0]
    cols = []
    for a in ARMS:
        treated = (T == a) * (Y - mu[:, a]) / e[:, a]
        cols.append(mu[:, a] - mu[:, 0] + treated - ctrl)
    return np.column_stack(cols)


class _TOutcome:
    """Per-arm outcome models. fit/predict take X with T appended as last column."""
    def __init__(self, seed, kw): self.seed, self.kw = seed, kw
    def fit(self, Xa, y):
        X, T = Xa[:, :-1], Xa[:, -1].astype(int)
        self.m_ = {a: make_regressor(self.seed, **self.kw).fit(X[T == a], y[T == a])
                   for a in range(4)}
        return self
    def predict(self, Xa):
        return np.column_stack([self.m_[a].predict(Xa[:, :-1]) for a in range(4)])


class _SOutcome:
    """Single outcome model on [X, arm dummies]; predicts all four arms."""
    def __init__(self, seed, kw): self.seed, self.kw = seed, kw
    def fit(self, Xa, y):
        X, T = Xa[:, :-1], Xa[:, -1].astype(int)
        self.m_ = make_regressor(self.seed, **self.kw).fit(
            np.hstack([X, np.eye(4)[T][:, 1:]]), y)
        return self
    def predict(self, Xa):
        X = Xa[:, :-1]
        return np.column_stack([
            self.m_.predict(np.hstack([X, np.tile(np.eye(4)[a][1:], (len(X), 1))]))
            for a in range(4)])


class DRLearner:
    def __init__(self, seed=0, K=5, clip=0.02, nuisance="t", propensity="logistic",
                 final_kw=None, nuisance_kw=None):
        self.seed, self.K, self.clip = seed, K, clip
        self.nuisance, self.propensity = nuisance, propensity
        self.final_kw = dict(min_child_samples=200, num_leaves=7)
        self.final_kw.update(final_kw or {})
        self.nuisance_kw = nuisance_kw or {}

    def fit(self, X, T, Y):
        cls = {"t": _TOutcome, "s": _SOutcome}[self.nuisance]
        Xa = np.column_stack([X, T])
        self.mu_ = crossfit_predict(lambda: cls(self.seed, self.nuisance_kw), Xa, Y,
                                    K=self.K, seed=self.seed)
        self.e_raw_ = fit_propensity(X, T, self.propensity, K=self.K, seed=self.seed)
        self.e_ = clip_propensity(self.e_raw_, self.clip)
        self.pseudo_ = dr_pseudo_outcomes(self.mu_, self.e_, T, Y)
        self.ate_aipw_ = self.pseudo_.mean(0)          # DR estimate of the ATE per arm
        self.final_ = [make_regressor(self.seed, **self.final_kw).fit(X, self.pseudo_[:, k])
                       for k in range(3)]
        return self

    def predict_tau(self, X):
        return np.column_stack([m.predict(X) for m in self.final_])
