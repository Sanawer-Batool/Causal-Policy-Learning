import numpy as np


def pehe(tau_hat, tau_true):
    """Return overall and per-arm PEHE plus median absolute error."""
    error = np.asarray(tau_hat) - np.asarray(tau_true)
    return {
        "pehe": float(np.sqrt(np.mean(error**2))),
        "pehe_per_arm": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "median_abs_err": float(np.median(np.abs(error))),
    }


def ate_error(tau_hat, tau_true):
    difference = np.mean(tau_hat, axis=0) - np.mean(tau_true, axis=0)
    return {
        "ate_err_per_arm": difference.tolist(),
        "ate_abs_err_mean": float(np.mean(np.abs(difference))),
    }


def regret(v_oracle, v_policy):
    return float(v_oracle - v_policy)


def cost_per_incremental(value, spend):
    """Spend per unit of gross incremental revenue."""
    gross = value + spend
    return float(spend / gross) if gross > 1e-12 else float("nan")
