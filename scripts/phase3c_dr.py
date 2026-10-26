"""Phase 3c: doubly robust learners across confounding and overlap settings."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cpl.adapters import unpack
from cpl.baselines import budget_cut
from cpl.dr_learner import DRLearner
from cpl.learners import SLearner, XLearner
from cpl.simulator import CausalSimulator, SimConfig
from cpl.splits import make_sim_splits


DR_VARIANTS = {
    "dr_t_clip02": {"nuisance": "t", "clip": 0.02},
    "dr_t_clip10": {"nuisance": "t", "clip": 0.10},
    "dr_s_clip02": {"nuisance": "s", "clip": 0.02},
    "dr_s_clip10": {"nuisance": "s", "clip": 0.10},
}
COMPARATORS = {
    "s_learner": SLearner,
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
    if array.ndim > 1:
        return {
            "mean": np.mean(array, axis=0).tolist(),
            "sd": np.std(array, axis=0, ddof=1).tolist(),
        }
    return {
        "mean": float(np.mean(array)),
        "sd": float(np.std(array, ddof=1)),
    }


def _policy_from_tau(sim, X, costs, tau_hat):
    net = np.column_stack([np.zeros(len(X)), tau_hat]) - costs[None, :]
    actions = budget_cut(net.argmax(axis=1), net.max(axis=1), costs, BUDGET)
    return actions, float(sim.policy_value(X, actions))


def run(
    n: int = 50_000,
    seeds=range(5),
    config_path: str = "configs/sim_default.yaml",
) -> dict:
    """Run DR variants and S/X comparators, aggregating results over ``seeds``."""
    seeds = list(seeds)
    base = SimConfig.from_yaml(config_path)
    settings = {
        "kappa1_rho_default": {"kappa": base.kappa, "rho": base.rho},
        "kappa0": {"kappa": 0.0, "rho": base.rho},
        "rho0": {"kappa": base.kappa, "rho": 0.0},
    }
    costs = np.asarray(base.costs, dtype=float)
    output = {
        "experiment": "phase3c_dr",
        "n_total": n,
        "seeds": seeds,
        "budget": BUDGET,
        "split": "D_dev fit, D_ope evaluation",
        "dr_variants": DR_VARIANTS,
        "settings": {},
    }
    allocation_inputs = {
        name: [] for name in (*COMPARATORS, *DR_VARIANTS, "oracle")
    }
    oracle_values = []

    for setting_name, changes in settings.items():
        config = base.with_(**changes)
        metric_names = ("ate_error", "spearman", "policy_value")
        per_learner = {
            name: {metric: [] for metric in metric_names}
            for name in (*COMPARATORS, *DR_VARIANTS)
        }
        for name in DR_VARIANTS:
            per_learner[name]["ate_aipw_error"] = []
        policy_diffs = {
            name: {
                "vs_s_learner": [],
                "vs_x_learner": [],
            }
            for name in DR_VARIANTS
        }

        for seed in seeds:
            sim = CausalSimulator(config)
            splits = make_sim_splits(sim, n, seed)
            development = unpack(splits.dev)
            evaluation_X = splits.ope.observed.X
            true_tau = sim.true_tau(evaluation_X)
            fitted = {}

            if setting_name == "kappa1_rho_default":
                oracle_tau = true_tau
                oracle_actions, oracle_value = _policy_from_tau(
                    sim, evaluation_X, costs, oracle_tau
                )
                oracle_values.append(oracle_value)
                allocation_inputs["oracle"].append(
                    np.bincount(oracle_actions, minlength=4) / len(oracle_actions)
                )

            for name, learner_class in COMPARATORS.items():
                tau_hat = learner_class(seed=seed).fit(
                    development.X, development.T, development.Y
                ).predict_tau(evaluation_X)
                fitted[name] = _policy_from_tau(
                    sim, evaluation_X, costs, tau_hat
                )
                actions, value = fitted[name]
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

            for name, variant in DR_VARIANTS.items():
                learner = DRLearner(seed=seed, **variant).fit(
                    development.X, development.T, development.Y
                )
                tau_hat = learner.predict_tau(evaluation_X)
                actions, value = _policy_from_tau(
                    sim, evaluation_X, costs, tau_hat
                )
                fitted[name] = (actions, value)
                per_learner[name]["ate_error"].append(
                    (tau_hat.mean(axis=0) - true_tau.mean(axis=0)).tolist()
                )
                per_learner[name]["ate_aipw_error"].append(
                    (learner.ate_aipw_ - true_tau.mean(axis=0)).tolist()
                )
                per_learner[name]["spearman"].append([
                    _spearman(tau_hat[:, arm], true_tau[:, arm])
                    for arm in range(true_tau.shape[1])
                ])
                per_learner[name]["policy_value"].append(value)
                policy_diffs[name]["vs_s_learner"].append(
                    value - fitted["s_learner"][1]
                )
                policy_diffs[name]["vs_x_learner"].append(
                    value - fitted["x_learner"][1]
                )
                if setting_name == "kappa1_rho_default":
                    allocation_inputs[name].append(
                        np.bincount(actions, minlength=4) / len(actions)
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
            "policy_value_differences_per_seed": policy_diffs,
        }

    output["allocation_shares_kappa1"] = {
        name: {
            "arms": ARM_NAMES,
            **_summary(values),
        }
        for name, values in allocation_inputs.items()
    }
    output["oracle_budget_value_kappa1"] = _summary(oracle_values)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parents[1] / "configs" / "sim_default.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "phase3c_dr.json",
    )
    args = parser.parse_args()

    result = run(args.n, range(5), args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
