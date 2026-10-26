import json
import platform
import subprocess
import time

import numpy as np

from .adapters import costs4, oracle_actions, true_cost, true_value
from .baselines import ALL
from .metrics import cost_per_incremental, regret
from .splits import make_sim_splits


def score_policy(sim, data, actions):
    value = true_value(sim, data, actions)
    spend = true_cost(sim, data, actions)
    oracle_value = true_value(sim, data, oracle_actions(sim, data))
    return {
        "value": value,
        "spend": spend,
        "oracle_value_unconstrained": oracle_value,
        "regret_vs_unconstrained_oracle": regret(oracle_value, value),
        "cost_per_incr_revenue": cost_per_incremental(value, spend),
        "treat_rate": float(np.mean(np.asarray(actions) > 0)),
        "action_share": np.bincount(actions, minlength=4).tolist(),
    }


def run_baselines(sim, n, seed, budget):
    start = time.time()
    costs = costs4(sim)
    splits = make_sim_splits(sim, n, seed)
    results = {}
    for baseline_class in ALL:
        policy = baseline_class(costs, seed).fit(splits.dev)
        results[baseline_class.name] = score_policy(
            sim, splits.ope, policy.act(splits.ope, budget)
        )
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    return {
        "seed": seed,
        "n": n,
        "budget": budget,
        "split": "D_ope",
        "git_sha": git_sha,
        "python": platform.python_version(),
        "runtime_s": round(time.time() - start, 2),
        "results": results,
    }


def save(result, path):
    with open(path, "w") as output:
        json.dump(result, output, indent=2)
