"""Evaluate S-, T-, and X-learners on simulator holdout data."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cpl.adapters import costs4, unpack
from cpl.baselines import budget_cut
from cpl.learners import SLearner, TLearner, XLearner
from cpl.simulator import CausalSimulator, SimConfig
from cpl.splits import make_sim_splits


LEARNERS = {
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
}


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    correlation = spearmanr(x, y).statistic
    return None if not np.isfinite(correlation) else float(correlation)


def run(
    n: int = 50_000,
    seed: int = 0,
    config_path: str = "configs/sim_default.yaml",
    kappa: float | None = None,
) -> dict:
    """Fit learners on D_dev and evaluate effects and policy value on D_ope."""
    config = SimConfig.from_yaml(config_path)
    if kappa is not None:
        config = config.with_(kappa=kappa)
    sim = CausalSimulator(config)
    splits = make_sim_splits(sim, n, seed)

    development = unpack(splits.dev)
    evaluation = unpack(splits.ope)
    true_tau = sim.true_tau(splits.ope.observed.X)
    costs = costs4(sim)
    results = {}

    for name, learner_class in LEARNERS.items():
        tau_hat = learner_class(seed=seed).fit(
            development.X, development.T, development.Y
        ).predict_tau(evaluation.X)
        net = np.column_stack(
            [np.zeros(len(tau_hat)), tau_hat - costs[1:][None, :]]
        )
        actions = net.argmax(axis=1)
        actions = budget_cut(actions, net.max(axis=1), costs, 0.18)

        results[name] = {
            "ate_error": (tau_hat.mean(axis=0) - true_tau.mean(axis=0)).tolist(),
            "median_absolute_error": np.median(
                np.abs(tau_hat - true_tau), axis=0
            ).tolist(),
            "spearman_correlation": [
                _spearman(tau_hat[:, arm], true_tau[:, arm])
                for arm in range(tau_hat.shape[1])
            ],
            "policy_value": float(sim.policy_value(splits.ope.observed.X, actions)),
        }

    return {
        "n_total": n,
        "n_dev": len(splits.dev),
        "n_ope": len(splits.ope),
        "seed": seed,
        "split": "D_ope",
        "budget": 0.18,
        "config": {
            "kappa": config.kappa,
            "eta": config.eta,
            "rho": config.rho,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parents[1] / "configs" / "sim_default.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for kappa, label in ((0.0, "kappa0"), (None, "kappa1")):
        result = run(args.n, args.seed, args.config, kappa)
        output = args.output_dir / f"phase3b_learners_{label}.json"
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
