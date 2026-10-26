"""The only module that knows about simulator internals."""
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np


@dataclass
class Obs:
    X: np.ndarray
    S: np.ndarray
    T: np.ndarray
    Y: np.ndarray
    e_log: np.ndarray


def _field(observed, name: str):
    if isinstance(observed, Mapping):
        return observed[name]
    if hasattr(observed, name):
        return getattr(observed, name)
    return observed[name]


def unpack(data) -> Obs:
    observed = data.observed
    propensity = "logging_propensity"
    if isinstance(observed, Mapping) and propensity not in observed:
        propensity = "e_log"
    return Obs(
        np.asarray(_field(observed, "X")),
        np.asarray(_field(observed, "S"), int),
        np.asarray(_field(observed, "T"), int),
        np.asarray(_field(observed, "Y"), float),
        np.asarray(_field(observed, propensity), float),
    )


def costs4(sim) -> np.ndarray:
    cfg = getattr(sim, "cfg", None) or sim.config
    costs = np.asarray(cfg.costs, float)
    return costs if len(costs) == 4 else np.r_[0.0, costs]


def _x(data):
    return data.observed.X


def true_tau(sim, data) -> np.ndarray:
    return sim.true_tau(_x(data))


def oracle_actions(sim, data) -> np.ndarray:
    return sim.oracle_actions(_x(data))


def true_value(sim, data, actions) -> float:
    return float(sim.policy_value(_x(data), actions))


def true_cost(sim, data, actions) -> float:
    return float(sim.policy_cost(actions))
