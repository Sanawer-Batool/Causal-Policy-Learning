import numpy as np

from cpl.uncertainty import BootstrapEnsemble, coverage_report


class _Mean:
    """Toy estimator: tau_hat = arm mean difference."""

    def fit(self, X, T, Y):
        self.t = np.array(
            [Y[T == a].mean() - Y[T == 0].mean() for a in (1, 2, 3)]
        )
        return self

    def predict_tau(self, X):
        return np.tile(self.t, (len(X), 1))


def test_interval_ordering_and_reproducibility():
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 2))
    T = rng.integers(0, 4, n)
    Y = rng.normal(size=n) + 0.3 * (T == 1)
    for scheme in ("bootstrap", "subsample"):
        a = (
            BootstrapEnsemble(_Mean(), B=20, seed=1, scheme=scheme)
            .fit(X, T, Y)
            .predict_tau_interval(X)
        )
        b = (
            BootstrapEnsemble(_Mean(), B=20, seed=1, scheme=scheme)
            .fit(X, T, Y)
            .predict_tau_interval(X)
        )
        assert np.all(a[0] <= a[1]) and np.allclose(a[0], b[0])


def test_known_truth_coverage_near_nominal():
    hits = 0
    for s in range(60):
        rng = np.random.default_rng(s)
        n = 3000
        X = rng.normal(size=(n, 1))
        T = rng.integers(0, 4, n)
        Y = rng.normal(size=n) + 0.3 * (T == 1)
        lo, hi = (
            BootstrapEnsemble(_Mean(), B=40, seed=s)
            .fit(X, T, Y)
            .predict_tau_interval(X[:1])
        )
        hits += lo[0, 0] <= 0.3 <= hi[0, 0]
    assert 0.80 <= hits / 60 <= 0.98


def test_parallel_reproducibility():
    rng = np.random.default_rng(2)
    n = 400
    X = rng.normal(size=(n, 2))
    T = rng.integers(0, 4, n)
    Y = rng.normal(size=n)
    for scheme in ("bootstrap", "subsample"):
        serial = BootstrapEnsemble(
            _Mean(), B=6, seed=3, scheme=scheme, n_jobs=1
        ).fit(X, T, Y)
        parallel = BootstrapEnsemble(
            _Mean(), B=6, seed=3, scheme=scheme, n_jobs=2
        ).fit(X, T, Y)
        serial_interval = serial.predict_tau_interval(X)
        parallel_interval = parallel.predict_tau_interval(X)
        assert np.allclose(serial_interval[0], parallel_interval[0])
        assert np.allclose(serial_interval[1], parallel_interval[1])
