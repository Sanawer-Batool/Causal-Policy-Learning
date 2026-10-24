"""Formal causal simulator (Architecture Rev.2, Sections 4 and 5).

Generative story for each customer i, in order:
    X ~ P(X)                       observed covariates (standardised)
    S ~ Bernoulli(pi_S(X))         sensitive attribute, correlated with X
    U ~ N(0, 1)                    HIDDEN loyalty / price sensitivity
    T ~ Categorical(e(.|X,U))      logging policy (nature's assignment)
    V ~ Uniform(0, 1)              shared "luck" so Z_a are coupled across arms
    Z_a = 1[V < sigmoid(f(X) + delta_a(X) + eta*U)]     conversion under a
    B ~ LogNormal(mu(X), nu^2)     basket value if converted
    Y(a) = Z_a * B                 potential outcome (revenue)
    Y = Y(T)                       what we observe

Naming note: the architecture doc writes the logit-scale lift as tau_a(X)
inside the generative model, but defines the *estimand* tau_a(x) =
E[Y(a) - Y(0) | X=x] on the revenue scale. These are different objects. Here
the logit-scale lift is called `delta_a` and `tau_a` is always the revenue-scale
estimand, computed by `true_tau`.

Information boundary: models may only see `SimData.observed`. Everything in
`SimData.oracle` is ground truth for evaluation only.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, softmax

FEATURES = ["recency", "frequency", "monetary", "tenure", "channel_pref"]
N_ACTIONS = 4  # 0: none, 1: email discount, 2: direct-mail voucher, 3: phone call
ACTION_NAMES = ["no_action", "email_discount", "mail_voucher", "phone_call"]

# Correlation of the latent standardised covariates (r, f, m, t, c).
_CORR = np.array(
    [
        [1.0, -0.3, -0.1, 0.0, 0.0],
        [-0.3, 1.0, 0.4, 0.2, 0.0],
        [-0.1, 0.4, 1.0, 0.1, 0.0],
        [0.0, 0.2, 0.1, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)
_CHOL = np.linalg.cholesky(_CORR)

# How strongly the hidden U pushes each action's logging score (action 0 = 0).
_W = np.array([0.0, 1.0, 0.6, 1.2])


@dataclass(frozen=True)
class SimConfig:
    kappa: float = 1.0   # U -> T strength
    eta: float = 1.0     # U -> Y strength
    rho: float = 1.0     # logging-policy sharpness (temperature inverse)
    eps: float = 0.02    # propensity floor
    nu: float = 0.5      # std-dev of log basket value
    costs: tuple = (0.0, 0.3, 0.8, 1.5)
    budget_levels: dict = field(
        default_factory=lambda: {"tight": 0.08, "medium": 0.18, "generous": 0.30}
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SimConfig":
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        raw["costs"] = tuple(raw["costs"])
        return cls(**raw)

    def with_(self, **kw) -> "SimConfig":
        """Return a copy with some knobs changed (used by stress-test sweeps)."""
        return replace(self, **kw)


@dataclass
class ObservedData:
    """What a model is allowed to see."""
    X: pd.DataFrame
    S: np.ndarray            # sensitive attribute (for constraints/evaluation only)
    T: np.ndarray            # action actually taken
    Y: np.ndarray            # observed revenue
    logging_propensity: np.ndarray  # e(T_i | X_i, U_i), known in the simulator


@dataclass
class OracleData:
    """Ground truth, never given to any model."""
    U: np.ndarray
    Y_pot: np.ndarray        # (n, 4) all potential outcomes
    p_pot: np.ndarray        # (n, 4) conversion probabilities given (X, U)
    e_full: np.ndarray       # (n, 4) full logging distribution given (X, U)


@dataclass
class SimData:
    observed: ObservedData
    oracle: OracleData

    def __len__(self) -> int:
        return len(self.observed.Y)


# --------------------------------------------------------------------------
# Structural functions of X. Columns of z: r, f, m, t, c (all ~ N(0,1)).
# --------------------------------------------------------------------------
def _z(X: pd.DataFrame) -> np.ndarray:
    return X[FEATURES].to_numpy(dtype=float)


def baseline_logit(z: np.ndarray) -> np.ndarray:
    """f(X): baseline conversion logit if nobody is contacted."""
    r, f, m, t, _ = z.T
    return -2.2 + 0.5 * f + 0.4 * m - 0.5 * r + 0.3 * t + 0.2 * np.tanh(f * m)


def log_basket_mean(z: np.ndarray) -> np.ndarray:
    """mu(X): mean of log basket value."""
    _, _, m, t, _ = z.T
    return 2.5 + 0.35 * m + 0.15 * t


def logit_lift(z: np.ndarray) -> np.ndarray:
    """delta_a(X), shape (n, 4). Nonlinear, action-specific, sometimes negative.

    1 email discount : helps price-sensitive, low-frequency customers; mildly
                       annoys frequent buyers (negative lift).
    2 mail voucher   : helps long-tenure customers whose recency is mid-range.
    3 phone call     : helps high-value, lapsed (high recency) customers most.
    """
    r, f, m, t, c = z.T
    d1 = -0.5 + 1.8 * expit(-2.0 * f) * expit(-c)
    d2 = -0.4 + 1.5 * expit(1.5 * t) * np.exp(-0.5 * (r - 0.5) ** 2)
    d3 = -0.6 + 2.4 * expit(2.0 * m) * expit(2.0 * (r - 0.3))
    return np.column_stack([np.zeros_like(r), d1, d2, d3])


def logging_scores(z: np.ndarray) -> np.ndarray:
    """g_a(X), shape (n, 4): who the historical business tended to contact.

    Note the deliberate trap: action 1 is aimed at FREQUENT customers (high f),
    exactly where its true effect is smallest or negative.
    """
    r, f, m, t, c = z.T
    g1 = -0.2 + 0.8 * f + 0.2 * c
    g2 = -0.6 + 0.6 * t - 0.2 * r
    g3 = -1.0 + 0.7 * m - 0.4 * r
    return np.column_stack([np.zeros_like(r), g1, g2, g3])


class CausalSimulator:
    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()
        self.costs = np.asarray(self.config.costs, dtype=float)
        if len(self.costs) != N_ACTIONS or self.costs[0] != 0.0:
            raise ValueError("costs must have 4 entries with c_0 = 0")

    # ------------------------------------------------------------------ draw
    def _draw_X(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        z = rng.standard_normal((n, 5)) @ _CHOL.T
        return pd.DataFrame(z, columns=FEATURES)

    def _logging_distribution(self, z: np.ndarray, U: np.ndarray) -> np.ndarray:
        """e(a | X, U) = floor_eps( softmax_a( rho * [g_a + kappa * w_a * U] ) )."""
        cfg = self.config
        scores = cfg.rho * (logging_scores(z) + cfg.kappa * _W[None, :] * U[:, None])
        e = softmax(scores, axis=1)
        e = np.clip(e, cfg.eps, None)
        return e / e.sum(axis=1, keepdims=True)

    def sample(self, n: int, seed: int) -> SimData:
        rng = np.random.default_rng(seed)
        cfg = self.config

        X = self._draw_X(n, rng)
        z = _z(X)
        S = rng.binomial(1, expit(0.8 * z[:, 3] - 0.5 * z[:, 4] + 0.3 * z[:, 1])).astype(int)
        U = rng.standard_normal(n)

        e_full = self._logging_distribution(z, U)
        cum = np.cumsum(e_full, axis=1)
        T = np.minimum((rng.random(n)[:, None] > cum).sum(axis=1), N_ACTIONS - 1)

        p_pot = expit(baseline_logit(z)[:, None] + logit_lift(z) + cfg.eta * U[:, None])
        V = rng.random(n)
        Z = (V[:, None] < p_pot).astype(float)
        B = np.exp(log_basket_mean(z) + cfg.nu * rng.standard_normal(n))
        Y_pot = Z * B[:, None]

        idx = np.arange(n)
        observed = ObservedData(X=X, S=S, T=T, Y=Y_pot[idx, T], logging_propensity=e_full[idx, T])
        oracle = OracleData(U=U, Y_pot=Y_pot, p_pot=p_pot, e_full=e_full)
        return SimData(observed=observed, oracle=oracle)

    # ------------------------------------------------------------ exact truth
    def true_mean_outcomes(self, X: pd.DataFrame, n_nodes: int = 64, chunk: int = 20000) -> np.ndarray:
        """E[Y(a) | X=x] for a=0..3, shape (n, 4).

        B and (U, V) are independent given X, so
            E[Y(a)|x] = E[B|x] * E_U[ sigmoid(f + delta_a + eta*U) ].
        E[B|x] = exp(mu + nu^2/2) is analytic; the U-expectation is computed with
        Gauss-Hermite quadrature, so there is NO Monte-Carlo noise in the truth.
        """
        nodes, weights = np.polynomial.hermite_e.hermegauss(n_nodes)
        weights = weights / np.sqrt(2 * np.pi)  # probabilists' Hermite -> N(0,1) expectation
        z_all = _z(X)
        out = np.empty((len(z_all), N_ACTIONS))
        for s in range(0, len(z_all), chunk):
            z = z_all[s : s + chunk]
            logit = baseline_logit(z)[:, None, None] + logit_lift(z)[:, :, None] \
                + self.config.eta * nodes[None, None, :]
            p_bar = (expit(logit) * weights[None, None, :]).sum(axis=2)
            eb = np.exp(log_basket_mean(z) + 0.5 * self.config.nu**2)
            out[s : s + chunk] = p_bar * eb[:, None]
        return out

    def true_tau(self, X: pd.DataFrame) -> np.ndarray:
        """tau_a(x) = E[Y(a) - Y(0) | X=x] for a=1,2,3, shape (n, 3)."""
        m = self.true_mean_outcomes(X)
        return m[:, 1:] - m[:, [0]]

    def net_value(self, X: pd.DataFrame) -> np.ndarray:
        """m_a(x) = tau_a(x) - c_a for a=0..3 (column 0 is identically 0)."""
        tau = np.column_stack([np.zeros(len(X)), self.true_tau(X)])
        return tau - self.costs[None, :]

    def oracle_actions(self, X: pd.DataFrame) -> np.ndarray:
        """Unconstrained optimal action a*(x) = argmax_a m_a(x)."""
        return self.net_value(X).argmax(axis=1)

    def policy_value(self, X: pd.DataFrame, actions: np.ndarray) -> float:
        """True V(pi) = mean_x [ tau_{pi(x)}(x) - c_{pi(x)} ] for deterministic pi,
        or a soft policy if `actions` is an (n, 4) matrix of probabilities."""
        m = self.net_value(X)
        actions = np.asarray(actions)
        if actions.ndim == 1:
            return float(m[np.arange(len(m)), actions].mean())
        return float((actions * m).sum(axis=1).mean())

    def policy_cost(self, actions: np.ndarray) -> float:
        actions = np.asarray(actions)
        if actions.ndim == 1:
            return float(self.costs[actions].mean())
        return float((actions * self.costs[None, :]).sum(axis=1).mean())
