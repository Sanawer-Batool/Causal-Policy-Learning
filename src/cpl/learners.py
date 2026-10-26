"""Phase 3b: hand-written S / T / X meta-learners for 3 treatment arms vs control.

Every learner: fit(X, T, Y) then predict_tau(X) -> (n, 3), columns = arms [1, 2, 3],
tau_a(x) = E[Y(a) - Y(0) | x]. Arm 0 is control. Fit on D_dev, score on fresh data.
"""
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ARMS = (1, 2, 3)


def make_regressor(seed=0, **kw):
    params = dict(n_estimators=300, learning_rate=0.03, num_leaves=15,
                  min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, random_state=seed, verbose=-1)
    params.update(kw)
    return LGBMRegressor(**params)


class SLearner:
    """One model mu(x, arm). tau_a = mu(x, a) - mu(x, 0)."""

    def __init__(self, seed=0, **kw):
        self.seed, self.kw = seed, kw

    @staticmethod
    def _design(X, arm):
        dummies = np.zeros((len(X), len(ARMS)))
        if arm > 0:
            dummies[:, arm - 1] = 1.0
        return np.hstack([X, dummies])

    def fit(self, X, T, Y):
        D = np.hstack([X, np.eye(len(ARMS) + 1)[T][:, 1:]])
        self.model_ = make_regressor(self.seed, **self.kw).fit(D, Y)
        return self

    def predict_tau(self, X):
        mu0 = self.model_.predict(self._design(X, 0))
        return np.column_stack([self.model_.predict(self._design(X, a)) - mu0
                                for a in ARMS])


class TLearner:
    """One outcome model per arm. tau_a = mu_a(x) - mu_0(x)."""

    def __init__(self, seed=0, **kw):
        self.seed, self.kw = seed, kw

    def fit(self, X, T, Y):
        self.mu_ = {a: make_regressor(self.seed, **self.kw).fit(X[T == a], Y[T == a])
                    for a in (0,) + ARMS}
        return self

    def predict_tau(self, X):
        mu0 = self.mu_[0].predict(X)
        return np.column_stack([self.mu_[a].predict(X) - mu0 for a in ARMS])


class XLearner:
    """Kunzel et al., run pairwise (arm a vs control).
    1. mu_0, mu_a outcome models.  2. Impute effects:
       treated:  D1 = Y - mu_0(X);   control: D0 = mu_a(X) - Y.
    3. tau1(x) <- D1 on treated X; tau0(x) <- D0 on control X.
    4. tau_a(x) = g(x) * tau0(x) + (1 - g(x)) * tau1(x), g = P(T=a | x, T in {0,a})."""

    def __init__(self, seed=0, **kw):
        self.seed, self.kw = seed, kw

    def fit(self, X, T, Y):
        s, kw = self.seed, self.kw
        c = T == 0
        mu0 = make_regressor(s, **kw).fit(X[c], Y[c])
        self.parts_ = {}
        for a in ARMS:
            t = T == a
            mua = make_regressor(s, **kw).fit(X[t], Y[t])
            d1 = Y[t] - mu0.predict(X[t])
            d0 = mua.predict(X[c]) - Y[c]
            tau1 = make_regressor(s, **kw).fit(X[t], d1)
            tau0 = make_regressor(s, **kw).fit(X[c], d0)
            pair = t | c
            g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            g.fit(X[pair], t[pair].astype(int))
            self.parts_[a] = (tau0, tau1, g)
        return self

    def predict_tau(self, X):
        cols = []
        for a in ARMS:
            tau0, tau1, g = self.parts_[a]
            w = g.predict_proba(X)[:, 1]
            cols.append(w * tau0.predict(X) + (1 - w) * tau1.predict(X))
        return np.column_stack(cols)
