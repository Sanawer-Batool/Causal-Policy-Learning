"""Phase 3b noise-floor experiment across confounding and overlap settings."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cpl.adapters import costs4, unpack
from cpl.baselines import NaiveWhatIf, budget_cut
from cpl.learners import SLearner, TLearner, XLearner
from cpl.simulator import CausalSimulator, SimConfig
from cpl.splits import make_sim_splits


LEARNERS = {
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
}
ARM_NAMES = ["none", "email", "voucher", "phone"]
BUDGET = 0.18


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def _summary(values) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": np.mean(array, axis=0).tolist()
        if array.ndim > 1
        else float(np.mean(array)),
        "sd": np.std(array, axis=0, ddof=1).tolist()
        if array.ndim > 1
        else float(np.std(array, ddof=1)),
    }


def _budgeted_oracle(sim, X, costs, true_tau):
    net = np.column_stack([np.zeros(len(X)), true_tau]) - costs[None, :]
    actions = budget_cut(net.argmax(axis=1), net.max(axis=1), costs, BUDGET)
    return actions, float(sim.policy_value(X, actions))


def run(
    n: int = 50_000,
    seeds=range(5),
    config_path: str = "configs/sim_default.yaml",
) -> dict:
    """Run all requested settings and aggregate results over ``seeds``."""
    seeds = list(seeds)
    base = SimConfig.from_yaml(config_path)
    settings = {
        "kappa1_rho_default": {"kappa": base.kappa, "rho": base.rho},
        "kappa0": {"kappa": 0.0, "rho": base.rho},
        "rho0": {"kappa": base.kappa, "rho": 0.0},
    }
    costs = np.asarray(base.costs, dtype=float)
    output = {
        "experiment": "phase3b_noise_floor",
        "n_total": n,
        "seeds": list(seeds),
        "budget": BUDGET,
        "split": "D_dev fit, D_ope evaluation",
        "settings": {},
    }
    allocation_inputs = {name: [] for name in (*LEARNERS, "oracle")}
    oracle_values = []

    for setting_name, changes in settings.items():
        config = base.with_(**changes)
        per_learner = {name: {"ate_error": [], "spearman": [], "policy_value": []}
                       for name in LEARNERS}
        if setting_name in {"kappa0", "rho0"}:
            per_learner["naive_what_if"] = {
                "ate_error": [], "spearman": [], "policy_value": []
            }

        for seed in seeds:
            sim = CausalSimulator(config)
            splits = make_sim_splits(sim, n, seed)
            development = unpack(splits.dev)
            evaluation_X = splits.ope.observed.X
            true_tau = sim.true_tau(evaluation_X)

            if setting_name == "kappa1_rho_default":
                oracle_actions, oracle_value = _budgeted_oracle(
                    sim, evaluation_X, costs, true_tau
                )
                oracle_values.append(oracle_value)
                allocation_inputs["oracle"].append(
                    np.bincount(oracle_actions, minlength=4) / len(oracle_actions)
                )

            for name, learner_class in LEARNERS.items():
                tau_hat = learner_class(seed=seed).fit(
                    development.X, development.T, development.Y
                ).predict_tau(evaluation_X)
                actions, value = _budgeted_oracle(
                    sim,
                    evaluation_X,
                    costs,
                    tau_hat,
                )
                per_learner[name]["ate_error"].append(
                    (tau_hat.mean(axis=0) - true_tau.mean(axis=0)).tolist()
                )
                per_learner[name]["spearman"].append([
                    _spearman(tau_hat[:, arm], true_tau[:, arm])
                    for arm in range(true_tau.shape[1])
                ])
                per_learner[name]["policy_value"].append(value)
                if setting_name == "kappa1_rho_default":
                    allocation_inputs[name].append(
                        np.bincount(actions, minlength=4) / len(actions)
                    )

            if setting_name in {"kappa0", "rho0"}:
                policy = NaiveWhatIf(costs, seed=seed).fit(splits.dev)
                actions = policy.act(splits.ope, BUDGET)
                tau_hat = np.column_stack([
                    policy.models[arm].predict(evaluation_X)
                    - policy.models[0].predict(evaluation_X)
                    for arm in range(1, 4)
                ])
                per_learner["naive_what_if"]["ate_error"].append(
                    (tau_hat.mean(axis=0) - true_tau.mean(axis=0)).tolist()
                )
                per_learner["naive_what_if"]["spearman"].append([
                    _spearman(tau_hat[:, arm], true_tau[:, arm])
                    for arm in range(true_tau.shape[1])
                ])
                per_learner["naive_what_if"]["policy_value"].append(
                    float(sim.policy_value(evaluation_X, actions))
                )

        output["settings"][setting_name] = {
            "config": {
                "kappa": config.kappa,
                "eta": config.eta,
                "rho": config.rho,
            },
            "learners": {
                name: {metric: _summary(values) for metric, values in metrics.items()}
                for name, metrics in per_learner.items()
            },
        }

    output["allocation_shares_kappa1"] = {
        name: {
            "arms": ARM_NAMES,
            **{metric: value for metric, value in _summary(values).items()},
        }
        for name, values in allocation_inputs.items()
    }
    output["oracle_budget_value_kappa1"] = _summary(oracle_values)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--config", default=str(
        Path(__file__).parents[1] / "configs" / "sim_default.yaml"
    ))
    parser.add_argument("--output", type=Path,
                        default=Path("results") / "phase3b_noise_floor.json")
    args = parser.parse_args()

    result = run(args.n, range(5), args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
