import json

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from cpl.baselines import budget_cut
from cpl.harness import run_baselines
from cpl.metrics import ate_error, pehe
from cpl.simulator import CausalSimulator, SimConfig
from cpl.splits import SealedTest, crossfit_predict, make_split_indices


def test_split_partition():
    development, ope, test = make_split_indices(1000, 0)
    assert (len(development), len(ope), len(test)) == (500, 250, 250)
    assert len(set(development) | set(ope) | set(test)) == 1000


def test_test_set_opens_once():
    sealed = SealedTest("data")
    sealed.open()
    with pytest.raises(RuntimeError):
        sealed.open()


def test_crossfit_has_no_leakage():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(500, 3)), rng.normal(size=500)
    oof = crossfit_predict(lambda: DecisionTreeRegressor(), X, y)
    insample = DecisionTreeRegressor().fit(X, y).predict(X)
    assert np.corrcoef(insample, y)[0, 1] > 0.99
    assert abs(np.corrcoef(oof, y)[0, 1]) < 0.2


def test_metrics_zero_when_perfect():
    truth = np.random.default_rng(0).normal(size=(100, 3))
    assert pehe(truth, truth)["pehe"] == 0
    assert ate_error(truth, truth)["ate_abs_err_mean"] == 0


def test_budget_cut_respects_budget():
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 4, 1000)
    costs = np.array([0, 0.1, 0.3, 0.6])
    output = budget_cut(actions, rng.random(1000), costs, 0.15)
    assert costs[output].mean() <= 0.15 + 1e-9


def test_baselines_are_reproducible_except_runtime():
    sim = CausalSimulator(SimConfig())
    first = run_baselines(sim, 800, 7, 0.18)
    second = run_baselines(sim, 800, 7, 0.18)
    first.pop("runtime_s")
    second.pop("runtime_s")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
