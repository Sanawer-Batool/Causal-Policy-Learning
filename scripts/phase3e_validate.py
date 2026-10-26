"""Phase 3e: frozen estimator validation on fresh confirmation seeds."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cpl.adapters import unpack
from cpl.baselines import NaiveWhatIf, budget_cut
from cpl.dr_learner import DRLearner
from cpl.estimators import make_estimator
from cpl.gates import check_gates, paired
from cpl.simulator import CausalSimulator, SimConfig
from cpl.splits import make_sim_splits


LEARNERS = ("naive_what_if", "S", "T", "X", "DR-S", "Forest")
CALIBRATION_LEARNERS = ("S", "T", "X", "DR-S", "Forest")
ARM_NAMES = ["email", "voucher", "phone"]
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


def _fit_naive(development_data, evaluation_X, costs, seed):
    learner = NaiveWhatIf(costs, seed=seed).fit(development_data)
    means = np.column_stack([
        model.predict(evaluation_X) for model in learner.models
    ])
    tau_hat = means[:, 1:] - means[:, [0]]
    return learner, tau_hat


def _fit_estimator(name, development, evaluation_X, seed):
    learner = make_estimator(name, seed=seed).fit(
        development.X, development.T, development.Y
    )
    return learner, learner.predict_tau(evaluation_X)


def _calibration(tau_hat, true_tau):
    return {
        "slope_true_on_hat": [
            float(np.polyfit(tau_hat[:, arm], true_tau[:, arm], 1)[0])
            for arm in range(3)
        ],
        "std_hat_over_std_true": [
            float(np.std(tau_hat[:, arm]) / np.std(true_tau[:, arm]))
            for arm in range(3)
        ],
        "top15_mean_true_tau": [
            float(np.mean(
                true_tau[np.argsort(tau_hat[:, arm])[-max(1, int(0.15 * len(tau_hat))):], arm]
            ))
            for arm in range(3)
        ],
    }


def _gate_report(raw_results):
    report = {}
    for learner in LEARNERS:
        rows = check_gates(raw_results, learner, ref="X")
        formatted = []
        for label, passed, values in rows:
            row = {
                "gate": label,
                "passed": passed,
                "values": values,
            }
            if label.startswith("G4"):
                setting = label.split("[", 1)[1].rstrip("]")
                mean, se = paired(
                    raw_results[setting][learner]["value"],
                    raw_results[setting]["X"]["value"],
                )
                row["strict_mean_diff_ge_0"] = bool(mean >= 0)
                row["paired_mean_diff_ge_minus_2se"] = bool(mean >= -2 * se)
                row["mean_diff"] = mean
                row["se"] = se
            formatted.append(row)
        report[learner] = formatted
    return report


def run(
    n: int = 50_000,
    seeds=range(10, 15),
    config_path: str = "configs/sim_default.yaml",
) -> dict:
    """Run the frozen registry and baseline on identical splits per seed."""
    seeds = list(seeds)
    base = SimConfig.from_yaml(config_path)
    costs = np.asarray(base.costs, dtype=float)
    setting_changes = {
        "rho0": {"kappa": base.kappa, "rho": 0.0},
        "kappa0": {"kappa": 0.0, "rho": base.rho},
        "kappa1": {"kappa": base.kappa, "rho": base.rho},
    }
    output = {
        "experiment": "phase3e_validation",
        "n_total": n,
        "seeds": seeds,
        "budget": BUDGET,
        "split": "D_dev fit, D_ope evaluation; D_test sealed",
        "settings": {},
    }
    raw_results = {}
    paired_policy_diffs = {}

    for setting_name, changes in setting_changes.items():
        config = base.with_(**changes)
        raw_results[setting_name] = {}
        per_learner = {
            name: {"ate_err": [], "spearman": [], "value": []}
            for name in LEARNERS
        }
        calibration = {name: [] for name in CALIBRATION_LEARNERS}

        for seed in seeds:
            sim = CausalSimulator(config)
            splits = make_sim_splits(sim, n, seed)
            development = unpack(splits.dev)
            evaluation_X = splits.ope.observed.X
            true_tau = sim.true_tau(evaluation_X)

            fitted = {}
            naive, tau_hat = _fit_naive(
                splits.dev, evaluation_X, costs, seed
            )
            fitted["naive_what_if"] = (
                tau_hat,
                _policy_from_tau(sim, evaluation_X, costs, tau_hat),
            )
            for name in ("S", "T", "X", "DR-S", "Forest"):
                learner, tau_hat = _fit_estimator(
                    name, development, evaluation_X, seed
                )
                fitted[name] = (
                    tau_hat,
                    _policy_from_tau(sim, evaluation_X, costs, tau_hat),
                )
                if setting_name in {"rho0", "kappa0"}:
                    calibration[name].append(
                        _calibration(tau_hat, true_tau)
                    )

            for name in LEARNERS:
                tau_hat, (actions, value) = fitted[name]
                per_learner[name]["ate_err"].append(
                    (tau_hat.mean(axis=0) - true_tau.mean(axis=0)).tolist()
                )
                per_learner[name]["spearman"].append([
                    _spearman(tau_hat[:, arm], true_tau[:, arm])
                    for arm in range(3)
                ])
                per_learner[name]["value"].append(value)

        raw_results[setting_name] = per_learner
        output["settings"][setting_name] = {
            "config": {
                "kappa": config.kappa,
                "eta": config.eta,
                "rho": config.rho,
            },
            "per_seed": {
                name: [
                    {
                        "seed": seed,
                        "ate_err": metrics["ate_err"][index],
                        "spearman": metrics["spearman"][index],
                        "policy_value": metrics["value"][index],
                    }
                    for index, seed in enumerate(seeds)
                ]
                for name, metrics in per_learner.items()
            },
            "learners": {
                name: {
                    metric: _summary(values)
                    for metric, values in metrics.items()
                }
                for name, metrics in per_learner.items()
            },
        }
        if setting_name in {"rho0", "kappa0"}:
            output["settings"][setting_name]["calibration"] = {
                name: {
                    metric: _summary([
                        row[metric] for row in rows
                    ])
                    for metric in rows[0]
                }
                for name, rows in calibration.items()
            }

    for setting_name in ("rho0", "kappa0", "kappa1"):
        paired_policy_diffs[setting_name] = {
            "Forest-S": (
                np.asarray(raw_results[setting_name]["Forest"]["value"])
                - np.asarray(raw_results[setting_name]["S"]["value"])
            ).tolist(),
            "Forest-X": (
                np.asarray(raw_results[setting_name]["Forest"]["value"])
                - np.asarray(raw_results[setting_name]["X"]["value"])
            ).tolist(),
            "Forest-DR-S": (
                np.asarray(raw_results[setting_name]["Forest"]["value"])
                - np.asarray(raw_results[setting_name]["DR-S"]["value"])
            ).tolist(),
        }
    output["gates"] = _gate_report(raw_results)
    output["paired_policy_value_differences"] = paired_policy_diffs
    return output


def _fmt(value):
    if isinstance(value, list):
        return "[" + ", ".join(f"{x:.4f}" for x in value) + "]"
    return f"{value:.4f}"


def print_report(result):
    order = list(LEARNERS)
    for setting, block in result["settings"].items():
        print(f"\n## {setting}")
        for metric in ("ate_err", "spearman", "value"):
            print(f"\n### {metric}")
            print("| learner | mean | SD |")
            print("|---|---|---|")
            for name in order:
                summary = block["learners"][name][metric]
                print(
                    f"| {name} | {_fmt(summary['mean'])} | "
                    f"{_fmt(summary['sd'])} |"
                )
        if "calibration" in block:
            print("\n### calibration")
            for name, metrics in block["calibration"].items():
                print(f"\n{ name }")
                for metric, summary in metrics.items():
                    print(
                        f"- {metric}: {_fmt(summary['mean'])} "
                        f"(SD {_fmt(summary['sd'])})"
                    )

    print("\n## paired policy-value differences")
    for setting, differences in result["paired_policy_value_differences"].items():
        print(f"\n### {setting} (seeds 10..14)")
        for label, values in differences.items():
            print(f"- {label}: {_fmt(values)}")

    print("\n## gates")
    for learner, rows in result["gates"].items():
        print(f"\n### {learner}")
        for row in rows:
            print(
                f"- {row['gate']}: {'PASS' if row['passed'] else 'FAIL'} "
                f"{row['values']}"
            )
            if "strict_mean_diff_ge_0" in row:
                print(
                    f"  strict G4: "
                    f"{'PASS' if row['strict_mean_diff_ge_0'] else 'FAIL'}; "
                    f"paired G4: "
                    f"{'PASS' if row['paired_mean_diff_ge_minus_2se'] else 'FAIL'}"
                )


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
        default=Path("results") / "phase3e_validation.json",
    )
    args = parser.parse_args()

    result = run(args.n, range(10, 15), args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print_report(result)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
