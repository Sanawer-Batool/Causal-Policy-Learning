from __future__ import annotations

import numpy as np


def required_gamma(
    tau_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    s: float,
) -> np.ndarray:
    """Per unit/arm: smallest Gamma whose widened interval covers the truth."""
    if s <= 0:
        raise ValueError("s must be positive")
    miss_low = lo - tau_true
    miss_high = tau_true - hi
    return np.maximum.reduce([miss_low, miss_high, np.zeros_like(lo)]) / s


def gamma_star(
    tau_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    s: float,
    target: float = 0.90,
) -> np.ndarray:
    if not 0 < target <= 1:
        raise ValueError("target must be in (0, 1]")
    g = required_gamma(tau_true, lo, hi, s)
    return np.quantile(g, target, axis=0)


def decisiveness(
    lo: np.ndarray,
    hi: np.ndarray,
    s: float,
    gamma: np.ndarray | float,
    cost: np.ndarray | list[float],
) -> np.ndarray:
    """Fraction whose widened interval excludes break-even cost per arm."""
    if s <= 0:
        raise ValueError("s must be positive")
    gamma_arr = np.asarray(gamma, dtype=float)
    if np.any(gamma_arr < 0):
        raise ValueError("gamma must be non-negative")
    cost_arr = np.asarray(cost, dtype=float)
    lo_w = lo - gamma_arr * s
    hi_w = hi + gamma_arr * s
    return ((lo_w > cost_arr) | (hi_w < cost_arr)).mean(axis=0)


def decisiveness_split(
    tau_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    s: float,
    gamma: np.ndarray | float,
    cost: np.ndarray | list[float],
) -> dict[str, list[float]]:
    """Split confident treatment and skip decisions, including treatment precision."""
    if s <= 0:
        raise ValueError("s must be positive")
    gamma_arr = np.asarray(gamma, dtype=float)
    if np.any(gamma_arr < 0):
        raise ValueError("gamma must be non-negative")
    cost_arr = np.asarray(cost, dtype=float)
    lo_w = lo - gamma_arr * s
    hi_w = hi + gamma_arr * s
    treat_sure = lo_w > cost_arr
    skip_sure = hi_w < cost_arr
    precision = np.array(
        [
            (tau_true[treat_sure[:, arm], arm] > cost_arr[arm]).mean()
            if treat_sure[:, arm].any()
            else np.nan
            for arm in range(len(cost_arr))
        ]
    )
    return {
        "treat_sure": treat_sure.mean(axis=0).tolist(),
        "skip_sure": skip_sure.mean(axis=0).tolist(),
        "precision_of_treat_sure": precision.tolist(),
    }


def coverage_vs_gamma(
    tau_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    s: float,
    grid=(0, 0.25, 0.5, 1, 2, 4),
) -> dict[float, list[float]]:
    g = required_gamma(tau_true, lo, hi, s)
    return {float(G): (g <= G).mean(axis=0).tolist() for G in grid}
