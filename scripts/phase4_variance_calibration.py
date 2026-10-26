"""Phase 4 variance calibration against independent simulated datasets."""

import argparse
import json

import numpy as np

from cpl.estimators import make_estimator
from cpl.scenarios import load_scenario
from cpl.uncertainty import BootstrapEnsemble


def calibrate(name, setting="kappa0", K=30, n_eval=2000, B=30, n_jobs=4):
    Xo = load_scenario(setting, 0)[3][:n_eval]
    preds = [
        make_estimator(name, seed=100 + k)
        .fit(*load_scenario(setting, 100 + k)[:3])
        .predict_tau(Xo)
        for k in range(K)
    ]
    true_sd = np.std(preds, axis=0, ddof=1)
    X, T, Y = load_scenario(setting, 0)[:3]
    out = {}
    for scheme in ("bootstrap", "subsample"):
        ensemble = BootstrapEnsemble(
            make_estimator(name),
            B=B,
            seed=0,
            scheme=scheme,
            n_jobs=n_jobs,
        ).fit(X, T, Y)
        ensemble_sd = ensemble.predict_draws(Xo).std(axis=0, ddof=1)
        out[scheme] = (
            np.median(ensemble_sd, axis=0) / np.median(true_sd, axis=0)
        ).tolist()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["S", "DR-S"])
    parser.add_argument("--setting", default="kappa0")
    parser.add_argument("--K", type=int, default=30)
    parser.add_argument("--n_eval", type=int, default=2000)
    parser.add_argument("--B", type=int, default=30)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--out", default="results/phase4_variance_calibration.json")
    args = parser.parse_args()
    results = {
        name: calibrate(
            name,
            setting=args.setting,
            K=args.K,
            n_eval=args.n_eval,
            B=args.B,
            n_jobs=args.n_jobs,
        )
        for name in args.methods
    }
    with open(args.out, "w") as output:
        json.dump(results, output, indent=1)
    for name, result in results.items():
        print(name, result, flush=True)
