"""Phase 4d: sensitivity widening and non-circular Gamma selection check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cpl.adapters import unpack
from cpl.estimators import make_estimator
from cpl.scenarios import SETTINGS
from cpl.sensitivity import (
    coverage_vs_gamma,
    decisiveness,
    decisiveness_split,
    gamma_star,
    required_gamma,
)
from cpl.simulator import CausalSimulator, SimConfig
from cpl.splits import make_sim_splits
from cpl.uncertainty import BootstrapEnsemble


def _mean_vector(rows, key):
    return np.mean([np.asarray(row[key], dtype=float) for row in rows], axis=0).tolist()


def _mean_nested_vector(rows, key, nested_key):
    return np.mean(
        [np.asarray(row[key][nested_key], dtype=float) for row in rows], axis=0
    ).tolist()


def _interval_arrays(row):
    if "intervals" not in row:
        raise ValueError(
            "Input rows must include intervals. Re-run scripts/phase4_audit.py "
            "with --save_intervals."
        )
    block = row["intervals"]
    return (
        np.asarray(block["tau_true"], dtype=float),
        np.asarray(block["lo"], dtype=float),
        np.asarray(block["hi"], dtype=float),
    )


def _row_metrics(row, costs, target, grid):
    tau_true, lo, hi = _interval_arrays(row)
    s = float(row["y_sd_dev"])
    center = 0.5 * (lo + hi)
    costs = np.asarray(costs, dtype=float)[1:]
    g_star = gamma_star(tau_true, lo, hi, s=s, target=target)
    out = {
        "seed": int(row["seed"]),
        "coverage_vs_gamma": coverage_vs_gamma(tau_true, lo, hi, s=s, grid=grid),
        "gamma_star": g_star.tolist(),
        "decisiveness_at_gamma_star": decisiveness(
            lo=lo, hi=hi, s=s, gamma=g_star, cost=costs
        ).tolist(),
        "decisiveness_split_gamma0": decisiveness_split(
            tau_true=tau_true, lo=lo, hi=hi, s=s, gamma=0, cost=costs
        ),
        "decisiveness_split_at_gamma_star": decisiveness_split(
            tau_true=tau_true, lo=lo, hi=hi, s=s, gamma=g_star, cost=costs
        ),
        "bias_ratio_true": (np.abs((center - tau_true).mean(axis=0)) / s).tolist(),
        "bias_ratio_proxy": (np.abs(center.mean(axis=0)) / s).tolist(),
    }
    return out


def _summarize_group(rows, costs, target, grid):
    per_seed = [_row_metrics(row, costs, target, grid) for row in rows]
    gamma_keys = [float(g) for g in grid]
    coverage_summary = {
        str(g): np.mean(
            [np.asarray(item["coverage_vs_gamma"][g], dtype=float) for item in per_seed],
            axis=0,
        ).tolist()
        for g in gamma_keys
    }
    return {
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
        "summary": {
            "coverage_vs_gamma": coverage_summary,
            "gamma_star": _mean_vector(per_seed, "gamma_star"),
            "decisiveness_at_gamma_star": _mean_vector(
                per_seed, "decisiveness_at_gamma_star"
            ),
            "decisiveness_split_gamma0": {
                key: _mean_nested_vector(per_seed, "decisiveness_split_gamma0", key)
                for key in ("treat_sure", "skip_sure", "precision_of_treat_sure")
            },
            "decisiveness_split_at_gamma_star": {
                key: _mean_nested_vector(
                    per_seed, "decisiveness_split_at_gamma_star", key
                )
                for key in ("treat_sure", "skip_sure", "precision_of_treat_sure")
            },
            "bias_ratio_true": _mean_vector(per_seed, "bias_ratio_true"),
            "bias_ratio_proxy": _mean_vector(per_seed, "bias_ratio_proxy"),
        },
    }


def sensitivity_from_audit(rows, costs, target, grid):
    by_method_setting = {}
    for row in rows:
        method = row["method"]
        setting = row["setting"]
        by_method_setting.setdefault(method, {}).setdefault(setting, []).append(row)
    return {
        method: {
            setting: _summarize_group(group, costs, target, grid)
            for setting, group in sorted(settings.items())
        }
        for method, settings in sorted(by_method_setting.items())
    }


def _intervals_for_method(method, X, T, Y, Xo, seed, B, n_jobs, s_scheme):
    if method == "Forest":
        est = make_estimator("Forest", seed=seed).fit(X, T, Y)
        lo, hi = est.predict_tau_interval(Xo, alpha=0.10)
        return lo, hi
    if method in {"S", "DR-S"}:
        est = BootstrapEnsemble(
            make_estimator(method, seed=seed),
            B=B,
            seed=seed,
            scheme=s_scheme,
            n_jobs=n_jobs,
        ).fit(X, T, Y)
        lo, hi = est.predict_tau_interval(Xo, level=0.90)
        return lo, hi
    raise ValueError(f"Unsupported method for kappa-map: {method}")


def build_kappa_map(
    base_config_path,
    kappas,
    seeds,
    n_total,
    B,
    n_jobs,
    s_scheme,
    costs,
    target,
    methods,
):
    base = SimConfig.from_yaml(base_config_path)
    rows = []
    for kappa in kappas:
        config = base.with_(kappa=float(kappa), eta=1.0)
        for seed in seeds:
            sim = CausalSimulator(config)
            splits = make_sim_splits(sim, n_total, seed)
            development = unpack(splits.dev)
            Xo = splits.ope.observed.X
            tau_true = sim.true_tau(Xo)
            s = float(np.std(development.Y, ddof=1))
            for method in methods:
                lo, hi = _intervals_for_method(
                    method=method,
                    X=development.X,
                    T=development.T,
                    Y=development.Y,
                    Xo=Xo,
                    seed=seed,
                    B=B,
                    n_jobs=n_jobs,
                    s_scheme=s_scheme,
                )
                center = 0.5 * (lo + hi)
                g_star = gamma_star(tau_true=tau_true, lo=lo, hi=hi, s=s, target=target)
                rows.append(
                    {
                        "method": method,
                        "kappa": float(kappa),
                        "seed": int(seed),
                        "gamma_star": g_star.tolist(),
                        "true_bias_ratio": (
                            np.abs((center - tau_true).mean(axis=0)) / s
                        ).tolist(),
                        "proxy_bias_ratio": (np.abs(center.mean(axis=0)) / s).tolist(),
                        "coverage_gamma0": (
                            ((tau_true >= lo) & (tau_true <= hi)).mean(axis=0).tolist()
                        ),
                        "decisiveness_at_gamma_star": decisiveness(
                            lo=lo, hi=hi, s=s, gamma=g_star, cost=costs[1:]
                        ).tolist(),
                    }
                )
    return rows


def _nearest_gamma(points, ratio):
    points = sorted(points, key=lambda p: p["proxy_bias_ratio"])
    chosen = min(points, key=lambda p: abs(p["proxy_bias_ratio"] - ratio))
    return chosen["gamma_star"]


def out_of_sample_map_test(
    rows,
    map_rows,
    map_seed_set,
    test_seed_set,
    map_methods=("Forest", "DR-S"),
):
    map_points = {}
    for row in map_rows:
        for arm in range(3):
            map_points.setdefault((row["method"], arm), []).append(
                {
                    "proxy_bias_ratio": float(row["proxy_bias_ratio"][arm]),
                    "gamma_star": float(row["gamma_star"][arm]),
                }
            )
    test_rows = []
    for row in rows:
        method = row["method"]
        seed = int(row["seed"])
        if method not in map_methods or seed in map_seed_set or seed not in test_seed_set:
            continue
        tau_true, lo, hi = _interval_arrays(row)
        s = float(row["y_sd_dev"])
        center = 0.5 * (lo + hi)
        proxy = np.abs(center.mean(axis=0)) / s
        chosen_gamma = np.array(
            [
                _nearest_gamma(map_points[(method, arm)], float(proxy[arm]))
                for arm in range(3)
            ],
            dtype=float,
        )
        required = required_gamma(tau_true=tau_true, lo=lo, hi=hi, s=s)
        restored = (required <= chosen_gamma).mean(axis=0)
        baseline = (required <= 0).mean(axis=0)
        test_rows.append(
            {
                "method": method,
                "setting": row["setting"],
                "seed": seed,
                "chosen_gamma": chosen_gamma.tolist(),
                "coverage_baseline_gamma0": baseline.tolist(),
                "coverage_with_chosen_gamma": restored.tolist(),
                "proxy_bias_ratio": proxy.tolist(),
            }
        )
    summary = {}
    for method in map_methods:
        method_rows = [row for row in test_rows if row["method"] == method]
        if not method_rows:
            continue
        for setting in sorted({row["setting"] for row in method_rows}):
            subset = [row for row in method_rows if row["setting"] == setting]
            summary.setdefault(method, {})[setting] = {
                "n_rows": len(subset),
                "mean_chosen_gamma": _mean_vector(subset, "chosen_gamma"),
                "mean_coverage_baseline_gamma0": _mean_vector(
                    subset, "coverage_baseline_gamma0"
                ),
                "mean_coverage_with_chosen_gamma": _mean_vector(
                    subset, "coverage_with_chosen_gamma"
                ),
            }
    return {"per_seed": test_rows, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default="results/phase4_dev.json")
    parser.add_argument("--out", default="results/phase4d_sensitivity.json")
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--grid", type=float, nargs="+", default=[0, 0.25, 0.5, 1, 2, 4])
    parser.add_argument("--map_kappas", type=float, nargs="+", default=[0, 0.5, 1, 2])
    parser.add_argument("--map_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--test_seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14])
    parser.add_argument("--n_total", type=int, default=50_000)
    parser.add_argument("--B", type=int, default=30)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument(
        "--s_scheme",
        choices=("bootstrap", "subsample"),
        default="subsample",
    )
    parser.add_argument("--config", default="configs/sim_default.yaml")
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["kappa0", "kappa1"],
        choices=list(SETTINGS),
    )
    parser.add_argument("--methods", nargs="+", default=["Forest", "DR-S"])
    args = parser.parse_args()

    data = json.loads(Path(args.infile).read_text())
    filtered = [
        row
        for row in data
        if row["setting"] in set(args.settings) and row["method"] in set(args.methods)
    ]
    if not filtered:
        raise ValueError("No rows matched --methods/--settings in the input file.")
    for row in filtered:
        if "intervals" not in row or "y_sd_dev" not in row:
            raise ValueError(
                "Found row without saved intervals. Re-run phase4_audit.py "
                "with --save_intervals."
            )

    costs = np.asarray(SimConfig.from_yaml(args.config).costs, dtype=float)
    sensitivity_report = sensitivity_from_audit(
        rows=filtered,
        costs=costs,
        target=args.target,
        grid=tuple(args.grid),
    )
    map_rows = build_kappa_map(
        base_config_path=args.config,
        kappas=args.map_kappas,
        seeds=args.map_seeds,
        n_total=args.n_total,
        B=args.B,
        n_jobs=args.n_jobs,
        s_scheme=args.s_scheme,
        costs=costs,
        target=args.target,
        methods=args.methods,
    )
    out_of_sample = out_of_sample_map_test(
        rows=filtered,
        map_rows=map_rows,
        map_seed_set=set(args.map_seeds),
        test_seed_set=set(args.test_seeds),
        map_methods=tuple(args.methods),
    )

    output = {
        "experiment": "phase4d_sensitivity",
        "input_file": args.infile,
        "target": args.target,
        "grid": [float(g) for g in args.grid],
        "costs": costs.tolist(),
        "methods": args.methods,
        "settings": args.settings,
        "sensitivity": sensitivity_report,
        "kappa_map": {
            "kappas": [float(kappa) for kappa in args.map_kappas],
            "seeds": args.map_seeds,
            "s_scheme": args.s_scheme,
            "rows": map_rows,
        },
        "out_of_sample_map_test": out_of_sample,
    }
    Path(args.out).write_text(json.dumps(output, indent=1))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
