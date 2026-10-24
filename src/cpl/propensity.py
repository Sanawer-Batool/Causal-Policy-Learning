"""Phase 3a: cross-fitted multinomial propensity model + overlap diagnostics.

The model sees X only, never U. When kappa > 0 it is therefore misspecified;
that gap is what the hidden-confounding stress tests measure.
Arm order: 0 = no action, 1 = email, 2 = voucher, 3 = phone.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from .splits import crossfit_predict

N_ARMS = 4


def _make_model(kind, seed):
    if kind == "logistic":
        return lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if kind == "lgbm":
        return lambda: LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=15,
            min_child_samples=50, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, verbose=-1)
    raise ValueError(f"unknown kind {kind!r}")


def _predict_full(model, X):
    """predict_proba padded to all N_ARMS columns (a fold may miss a class)."""
    p = model.predict_proba(X)
    classes = np.asarray(model.classes_, int)
    full = np.zeros((len(X), N_ARMS))
    full[:, classes] = p
    return full


def fit_propensity(X, T, kind="logistic", K=5, seed=0, floor=1e-6):
    """Out-of-fold P(T=a | X), shape (n, 4), rows sum to 1."""
    e = crossfit_predict(_make_model(kind, seed), np.asarray(X), np.asarray(T, int),
                         predict=_predict_full, K=K, seed=seed)
    e = np.clip(e, floor, None)
    return e / e.sum(axis=1, keepdims=True)


def clip_propensity(e, lo=0.01):
    """Floor propensities and renormalize. Used inside 1/e terms (DR, IPS)."""
    e = np.clip(e, lo, None)
    return e / e.sum(axis=1, keepdims=True)


def ess(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def overlap_diagnostics(e_hat, T, thresh=0.01, e_true=None):
    """Per-arm overlap summary. Inverse-propensity weights are taken over the
    units that actually received the arm, which is what DR/IPS will use.
    If e_true (known in the simulator) is given, also report estimation error."""
    T = np.asarray(T, int)
    out = {"thresh": thresh, "arms": {}}
    for a in range(N_ARMS):
        p, got = e_hat[:, a], T == a
        w = 1.0 / np.clip(e_hat[got, a], 1e-12, None)
        arm = {
            "n_treated": int(got.sum()),
            "min_e": float(p.min()),
            "q01_e": float(np.quantile(p, 0.01)),
            "mean_e": float(p.mean()),
            "frac_below_thresh": float((p < thresh).mean()),
            "ess_treated": ess(w),
            "ess_ratio": ess(w) / max(int(got.sum()), 1),
            "max_weight": float(w.max()) if len(w) else float("nan"),
        }
        if e_true is not None:
            arm["mae_vs_true"] = float(np.abs(p - e_true[:, a]).mean())
            arm["corr_vs_true"] = float(np.corrcoef(p, e_true[:, a])[0, 1])
        out["arms"][a] = arm
    out["log_loss"] = float(-np.log(e_hat[np.arange(len(T)), T]).mean())
    if e_true is not None:
        out["log_loss_true"] = float(-np.log(
            np.clip(e_true[np.arange(len(T)), T], 1e-12, None)).mean())
    return out


def plot_propensity(e_hat, T, path):
    """Per-arm histogram of e_hat, treated vs. all. Overlap = the two overlap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T = np.asarray(T, int)
    fig, axes = plt.subplots(1, N_ARMS, figsize=(14, 3))
    for a, ax in enumerate(axes):
        ax.hist(e_hat[:, a], bins=50, alpha=0.5, density=True, label="all")
        ax.hist(e_hat[T == a, a], bins=50, alpha=0.5, density=True, label="got arm")
        ax.set_title(f"arm {a}"); ax.set_xlabel("e_hat")
    axes[0].legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
