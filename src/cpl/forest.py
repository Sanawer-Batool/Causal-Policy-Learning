"""Phase 3d: causal forest (econml CausalForestDML) with discrete multi-arm treatment,
plus econml's DRLearner as an independent cross-check of our hand-written DRLearner.
Same interface as the other learners: fit(X, T, Y); predict_tau(X) -> (n, 3).
The forest also gives intervals (used by Phase 4)."""
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor


def _reg(seed):
    return LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                         min_child_samples=50, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.8, random_state=seed, verbose=-1)


def _clf(seed):
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                          min_child_samples=50, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, verbose=-1)


class CausalForest:
    def __init__(self, seed=0, n_estimators=500, min_samples_leaf=50, cv=5, **kw):
        self.seed, self.n_estimators = seed, n_estimators
        self.min_samples_leaf, self.cv, self.kw = min_samples_leaf, cv, kw

    def fit(self, X, T, Y):
        from econml.dml import CausalForestDML
        self.model_ = CausalForestDML(
            model_y=_reg(self.seed), model_t=_clf(self.seed), discrete_treatment=True,
            categories=[0, 1, 2, 3], cv=self.cv, n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf, random_state=self.seed, **self.kw)
        self.model_.fit(Y, T, X=X)
        return self

    def predict_tau(self, X):
        return np.asarray(self.model_.const_marginal_effect(X)).reshape(len(X), 3)

    def predict_tau_interval(self, X, alpha=0.10):
        lo, hi = self.model_.const_marginal_effect_interval(X, alpha=alpha)
        return (np.asarray(lo).reshape(len(X), 3), np.asarray(hi).reshape(len(X), 3))


class EconmlDR:
    """econml's DRLearner with our nuisance choices: cross-check, not a competitor."""
    def __init__(self, seed=0, cv=5, min_propensity=0.02):
        self.seed, self.cv, self.min_propensity = seed, cv, min_propensity

    def fit(self, X, T, Y):
        from econml.dr import DRLearner
        self.model_ = DRLearner(
            model_propensity=_clf(self.seed), model_regression=_reg(self.seed),
            model_final=LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                      min_child_samples=200, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, random_state=self.seed, verbose=-1),
            min_propensity=self.min_propensity, cv=self.cv, categories=[0, 1, 2, 3],
            random_state=self.seed)
        self.model_.fit(Y, T, X=X)
        return self

    def predict_tau(self, X):
        return np.asarray(self.model_.const_marginal_effect(X)).reshape(len(X), 3)
