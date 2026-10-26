"""Phase 4a/4b: interval coverage audit. D_test is never opened."""

import argparse
import json
import time

import numpy as np

from cpl.estimators import make_estimator
from cpl.scenarios import SETTINGS, load_scenario
from cpl.uncertainty import (
    BootstrapEnsemble,
    coverage_report,
    reliability_curve,
)


def audit_one(name, setting, seed, B, scheme, n_jobs=1):
    X, T, Y, Xo, tau = load_scenario(setting, seed)
    t0 = time.time()
    if name == "Forest":
        est = make_estimator("Forest", seed=seed).fit(X, T, Y)
        lo, hi = est.predict_tau_interval(Xo, alpha=0.10)
        draws = None
    else:
        est = BootstrapEnsemble(
            make_estimator(name, seed=seed),
            B=B,
            seed=seed,
            scheme=scheme,
            n_jobs=n_jobs,
        ).fit(X, T, Y)
        lo, hi = est.predict_tau_interval(Xo, level=0.90)
        draws = est.predict_draws(Xo)
    out = coverage_report(tau, lo, hi)
    out["center_bias_mean"] = (est.predict_tau(Xo) - tau).mean(axis=0).tolist()
    if draws is not None:
        out["reliability"] = reliability_curve(tau, draws)
    out.update(
        method=name,
        setting=setting,
        seed=seed,
        scheme=scheme,
        B=B,
        runtime_s=round(time.time() - t0, 1),
    )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--methods", nargs="+", default=["Forest", "S", "DR-S", "X"])
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--schemes", nargs="+", default=["bootstrap"])
    ap.add_argument("--B", type=int, default=30)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--out", default="results/phase4_audit.json")
    a = ap.parse_args()
    rows = []
    for s in a.seeds:
        for st in a.settings:
            for m in a.methods:
                for sch in (["native"] if m == "Forest" else a.schemes):
                    rows.append(audit_one(m, st, s, a.B, sch, a.n_jobs))
                    print(
                        m,
                        st,
                        s,
                        sch,
                        np.round(rows[-1]["coverage"], 2),
                        flush=True,
                    )
    with open(a.out, "w") as output:
        json.dump(rows, output, indent=1)
