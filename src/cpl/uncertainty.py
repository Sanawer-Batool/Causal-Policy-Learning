from __future__ import annotations

import copy

from joblib import Parallel, delayed
import numpy as np


def _fit_member(base, X, T, Y, seed, scheme):
    rng = np.random.default_rng(seed)
    n = len(X)
    if scheme == "bootstrap":
        idx = rng.integers(0, n, size=n)
    else:
        idx = rng.choice(n, size=n // 2, replace=False)
    member = copy.deepcopy(base)
    member.fit(X[idx], T[idx], Y[idx])
    return member


class BootstrapEnsemble:
    """Row-bootstrap ensemble around any estimator with fit(X,T,Y)/predict_tau(X).

    Intervals reflect sampling VARIANCE only. They cannot see confounding bias.
    """

    def __init__(
        self,
        base,
        B: int = 30,
        seed: int = 0,
        scheme: str = "bootstrap",
        n_jobs: int = 1,
    ):
        if scheme not in ("bootstrap", "subsample"):
            raise ValueError(scheme)
        self.base, self.B, self.seed, self.scheme = base, B, seed, scheme
        self.n_jobs = n_jobs
        self.members_: list = []

    def fit(self, X, T, Y):
        self.members_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_member)(
                self.base,
                X,
                T,
                Y,
                self.seed + 1000 * b,
                self.scheme,
            )
            for b in range(self.B)
        )
        return self

    def predict_draws(self, X):
        return np.stack([m.predict_tau(X) for m in self.members_])

    def predict_tau(self, X):
        return self.predict_draws(X).mean(axis=0)

    def predict_tau_interval(self, X, level: float = 0.90):
        d = self.predict_draws(X)
        a = (1 - level) / 2
        lo, hi = np.quantile(d, [a, 1 - a], axis=0)
        return lo, hi


def coverage_report(tau_true, lo, hi, levels_label="90%"):
    """Per-arm coverage and width, plus coverage by |tau| quartile."""
    inside = (tau_true >= lo) & (tau_true <= hi)
    out = {
        "coverage": inside.mean(axis=0).tolist(),
        "mean_width": (hi - lo).mean(axis=0).tolist(),
        "median_width": np.median(hi - lo, axis=0).tolist(),
    }
    q = np.quantile(np.abs(tau_true), [0.25, 0.5, 0.75], axis=0)
    by_q = []
    for a in range(tau_true.shape[1]):
        edges = np.concatenate([[-np.inf], q[:, a], [np.inf]])
        a_abs = np.abs(tau_true[:, a])
        by_q.append(
            [
                float(
                    inside[(a_abs > edges[k]) & (a_abs <= edges[k + 1]), a].mean()
                )
                for k in range(4)
            ]
        )
    out["coverage_by_abs_tau_quartile"] = by_q
    return out


def reliability_curve(tau_true, draws, levels=(0.5, 0.7, 0.8, 0.9)):
    """Nominal vs empirical coverage from bootstrap draws (B, n, 3)."""
    res = {}
    for L in levels:
        a = (1 - L) / 2
        lo, hi = np.quantile(draws, [a, 1 - a], axis=0)
        res[L] = ((tau_true >= lo) & (tau_true <= hi)).mean(axis=0).tolist()
    return res
