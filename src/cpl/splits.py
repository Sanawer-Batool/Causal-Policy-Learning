from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold


class SealedTest:
    """D_test: can be opened exactly once per process."""

    def __init__(self, data):
        self._data = data
        self._opened = False

    def open(self):
        if self._opened:
            raise RuntimeError("D_test already opened. Final numbers are computed once.")
        self._opened = True
        return self._data


@dataclass
class Splits:
    dev: object
    ope: object
    test: SealedTest


def make_sim_splits(sim, n, seed):
    ss = np.random.SeedSequence(seed)
    s_dev, s_ope, s_test = [int(child.generate_state(1)[0]) for child in ss.spawn(3)]
    n_dev, n_ope = n // 2, n // 4
    return Splits(
        sim.sample(n_dev, s_dev),
        sim.sample(n_ope, s_ope),
        SealedTest(sim.sample(n - n_dev - n_ope, s_test)),
    )


def make_split_indices(n, seed, frac=(0.5, 0.25, 0.25)):
    """Return sorted, disjoint indices for real-data development splits."""
    if len(frac) != 3 or not np.isclose(sum(frac), 1.0):
        raise ValueError("frac must contain three values summing to 1")
    perm = np.random.default_rng(seed).permutation(n)
    a = int(round(frac[0] * n))
    b = a + int(round(frac[1] * n))
    return np.sort(perm[:a]), np.sort(perm[a:b]), np.sort(perm[b:])


def crossfit_predict(make_model, X, y, predict=None, K=5, seed=0):
    """Return out-of-fold predictions from K independently fitted models."""
    if K < 2 or K > len(X):
        raise ValueError("K must be between 2 and the number of samples")
    predict = predict or (lambda model, features: model.predict(features))
    out = None
    for train, valid in KFold(K, shuffle=True, random_state=seed).split(X):
        model = make_model()
        model.fit(X[train], y[train])
        prediction = np.asarray(predict(model, X[valid]))
        if out is None:
            out = np.zeros((len(X),) + prediction.shape[1:], dtype=prediction.dtype)
        out[valid] = prediction
    return out
