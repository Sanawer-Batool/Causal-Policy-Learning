"""Run Phase 3a propensity estimation on the simulator development split."""
import argparse
import json
from pathlib import Path

import numpy as np

from cpl.adapters import unpack
from cpl.propensity import fit_propensity, overlap_diagnostics
from cpl.simulator import CausalSimulator, SimConfig
from cpl.splits import make_sim_splits


def run(n: int, seed: int, config_path: str, kappa: float | None = None) -> dict:
    config = SimConfig.from_yaml(config_path)
    if kappa is not None:
        config = config.with_(kappa=kappa)
    sim = CausalSimulator(config)
    splits = make_sim_splits(sim, n, seed)
    o = unpack(splits.dev)
    e_true_full = splits.dev.oracle.e_full
    X = o.X

    results = {}
    for kind in ("logistic", "lgbm"):
        e_hat = fit_propensity(X, o.T, kind=kind, seed=seed)
        diagnostics = overlap_diagnostics(e_hat, o.T, e_true=e_true_full)
        realized = e_hat[np.arange(len(o.T)), o.T]
        diagnostics["corr_realized_vs_e_log"] = float(
            np.corrcoef(realized, o.e_log)[0, 1]
        )
        results[kind] = diagnostics
    return {
        "n_total": n,
        "n_dev": len(o.T),
        "seed": seed,
        "config": {
            "kappa": config.kappa,
            "eta": config.eta,
            "rho": config.rho,
        },
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parents[1] / "configs" / "sim_default.yaml"),
    )
    args = parser.parse_args()
    output = run(args.n, args.seed, args.config, args.kappa)
    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
