import numpy as np

from cpl.sensitivity import (
    coverage_vs_gamma,
    decisiveness,
    decisiveness_split,
    gamma_star,
    required_gamma,
)


def test_required_gamma_known_shift():
    tau_true = np.array([[1.0, -0.5, 0.2], [0.3, 0.7, -1.2]])
    lo = tau_true + 0.5
    hi = tau_true + 1.0
    s = 2.0
    gamma = required_gamma(tau_true=tau_true, lo=lo, hi=hi, s=s)
    assert np.allclose(gamma, 0.25)


def test_gamma_star_and_coverage_vs_gamma():
    tau_true = np.zeros((4, 3))
    lo = np.array(
        [
            [0.5, 0.0, -0.1],
            [0.5, 0.2, -0.2],
            [0.5, -0.1, 0.1],
            [0.5, 0.1, 0.0],
        ]
    )
    hi = lo + 0.2
    s = 1.0
    g = required_gamma(tau_true=tau_true, lo=lo, hi=hi, s=s)
    assert np.allclose(g[:, 0], 0.5)
    star = gamma_star(tau_true=tau_true, lo=lo, hi=hi, s=s, target=0.9)
    assert np.all(star >= 0)
    cov = coverage_vs_gamma(tau_true=tau_true, lo=lo, hi=hi, s=s, grid=(0, 0.5))
    assert cov[0.0][0] == 0.0
    assert cov[0.5][0] == 1.0


def test_decisiveness_vector_gamma():
    lo = np.array([[1.0, -1.0, 0.0], [0.4, -0.2, 2.0]])
    hi = np.array([[1.2, -0.4, 0.5], [0.8, 0.2, 2.5]])
    gamma = np.array([0.0, 0.5, 0.0])
    s = 1.0
    cost = np.array([0.7, 0.0, 1.5])
    d = decisiveness(lo=lo, hi=hi, s=s, gamma=gamma, cost=cost)
    assert d.shape == (3,)
    assert np.all((0 <= d) & (d <= 1))


def test_decisiveness_split_reports_treatment_precision():
    tau_true = np.array([[1.0, 0.0, 2.0], [0.2, 0.0, 0.5]])
    lo = np.array([[0.8, -1.0, 1.8], [0.1, -1.0, 0.4]])
    hi = np.array([[1.2, 1.0, 2.2], [0.3, 1.0, 0.6]])
    out = decisiveness_split(
        tau_true=tau_true,
        lo=lo,
        hi=hi,
        s=1.0,
        gamma=0.0,
        cost=np.array([0.7, 0.0, 1.5]),
    )
    assert np.allclose(out["treat_sure"], [0.5, 0.0, 0.5])
    assert np.allclose(out["skip_sure"], [0.5, 0.0, 0.5])
    assert np.allclose(out["precision_of_treat_sure"], [1.0, np.nan, 1.0], equal_nan=True)
