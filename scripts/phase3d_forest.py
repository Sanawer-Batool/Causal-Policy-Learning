"""Phase 3d: causal forest and econml DR cross-check."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cpl.adapters import unpack
from cpl.baselines import budget_cut
from cpl.dr_learner import DRLearner
from cpl.forest import CausalForest, EconmlDR
from cpl.simulator import CausalSimulator, SimConfig
from cpl.splits import make_sim_splits


LEARNERS = ("causal_forest", "econml_dr", "dr_s_clip10")
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


def _policy_from_tau(sim, X, costs, tau_hat):
    net = np.column_stack([np.zeros(len(X)), tau_hat]) - costs[None, :]
    actions = budget_cut(net.argmax(axis=1), net.max(axis=1), costs, BUDGET)
    return actions, float(sim.policy_value(X, actions))


def _setting_config(base: SimConfig) -> dict:
    return {
        "rho0": {"kappa": base.kappa, "rho": 0.0},
        "kappa0": {"kappa": 0.0, "rho": base.rho},
        "kappa1_rho_default": {"kappa": base.kappa, "rho": base.rho},
    }


def run(
    n: int = 50_000,
    seeds=range(5),
    config_path: str = "configs/sim_default.yaml",
    n_estimators: int = 500,
) -> dict:
    """Run all forest/econml DR settings and aggregate over ``seeds``."""
    seeds = list(seeds)
    base = SimConfig.from_yaml(config_path)
    costs = np.asarray(base.costs, dtype=float)
    output = {
        "experiment": "phase3d_forest",
        "n_total": n,
        "seeds": seeds,
        "budget": BUDGET,
        "forest_n_estimators": n_estimators,
        "split": "D_dev fit, D_ope evaluation",
        "settings": {},
    }
    allocation_inputs = {name: [] for name in (*LEARNERS, "oracle")}
    oracle_values = []

    for setting_name, changes in _setting_config(base).items():
        config = base.with_(**changes)
        per_learner = {
            name: {
                "ate_error": [],
                "spearman": [],
                "policy_value": [],
            }
            for name in LEARNERS
        }
        per_learner["dr_s_clip10"]["ate_aipw_error"] = []
        crosscheck = []
        coverage = []
        width = []

        for seed in seeds:
            sim = CausalSimulator(config)
            splits = make_sim_splits(sim, n, seed)
            development = unpack(splits.dev)
            evaluation_X = splits.ope.observed.X
            true_tau = sim.true_tau(evaluation_X)
            fitted = {}

            if setting_name == "kappa1_rho_default":
                oracle_actions, oracle_value = _policy_from_tau(
                    sim, evaluation_X, costs, true_tau
                )
                oracle_values.append(oracle_value)
                allocation_inputs["oracle"].append(
                    np.bincount(oracle_actions, minlength=4) / len(oracle_actions)
                )

            forest = CausalForest(
                seed=seed, n_estimators=n_estimators
            ).fit(development.X, development.T, development.Y)
            forest_tau = forest.predict_tau(evaluation_X)
            lo, hi = forest.predict_tau_interval(evaluation_X)
            coverage.append([
                np.mean((lo[:, arm] <= true_tau[:, arm])
                        & (true_tau[:, arm] <= hi[:, arm]))
                for arm in range(3)
            ])
            width.append(np.mean(hi - lo, axis=0))
            fitted["causal_forest"] = _policy_from_tau(
                sim, evaluation_X, costs, forest_tau
            )

            econml = EconmlDR(seed=seed).fit(
                development.X, development.T, development.Y
            )
            econml_tau = econml.predict_tau(evaluation_X)
            fitted["econml_dr"] = _policy_from_tau(
                sim, evaluation_X, costs, econml_tau
            )

            ours = DRLearner(seed=seed, nuisance="s", clip=0.10).fit(
                development.X, development.T, development.Y
            )
            ours_tau = ours.predict_tau(evaluation_X)
            fitted["dr_s_clip10"] = _policy_from_tau(
                sim, evaluation_X, costs, ours_tau
            )
            crosscheck.append([
                _spearman(ours_tau[:, arm], econml_tau[:, arm])
                for arm in range(3)
            ])

            for name, tau_hat in (
                ("causal_forest", forest_tau),
                ("econml_dr", econml_tau),
                ("dr_s_clip10", ours_tau),
            ):
                actions, value = fitted[name]
                per_learner[name]["ate_error"].append(
                    (tau_hat.mean(axis=0) - true_tau.mean(axis=0)).tolist()
                )
                per_learner[name]["spearman"].append([
                    _spearman(tau_hat[:, arm], true_tau[:, arm])
                    for arm in range(3)
                ])
                per_learner[name]["policy_value"].append(value)
                if setting_name == "kappa1_rho_default":
                    allocation_inputs[name].append(
                        np.bincount(actions, minlength=4) / len(actions)
                    )
            per_learner["dr_s_clip10"]["ate_aipw_error"].append(
                (ours.ate_aipw_ - true_tau.mean(axis=0)).tolist()
            )

        output["settings"][setting_name] = {
            "config": {
                "kappa": config.kappa,
                "eta": config.eta,
                "rho": config.rho,
            },
            "learners": {
                name: {
                    metric: _summary(values)
                    for metric, values in metrics.items()
                }
                for name, metrics in per_learner.items()
            },
            "dr_vs_econml_spearman": _summary(crosscheck),
            "forest_interval_coverage": _summary(coverage),
            "forest_interval_width": _summary(width),
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


def runtime_check(
    n: int,
    n_estimators: int,
    config_path: str,
    seed: int = 0,
) -> dict:
    """Time one forest fit on a development split of size ``n // 2``."""
    config = SimConfig.from_yaml(config_path)
    sim = CausalSimulator(config)
    development = unpack(make_sim_splits(sim, n, seed).dev)
    started = time.perf_counter()
    CausalForest(seed=seed, n_estimators=n_estimators).fit(
        development.X, development.T, development.Y
    )
    elapsed = time.perf_counter() - started
    return {
        "n_total": n,
        "n_fit": len(development.X),
        "n_estimators": n_estimators,
        "seconds": elapsed,
        "minutes": elapsed / 60.0,
    }


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
        default=Path("results") / "phase3d_forest.json",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--runtime-check", action="store_true")
    parser.add_argument("--runtime-n", type=int, default=25_000)
    args = parser.parse_args()

    if args.runtime_check:
        print(json.dumps(runtime_check(
            args.runtime_n, args.n_estimators, args.config
        ), indent=2))
        return

    result = run(args.n, range(5), args.config, args.n_estimators)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
