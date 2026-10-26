"""The agreed Phase 3 tolerances as code. Inputs are per-seed arrays, same seeds for
every learner (so differences can be paired):
  res[setting][learner] = {"ate_err": (S,3), "spearman": (S,3), "value": (S,)}
settings: "rho0", "kappa0", "kappa1". kappa1 is recorded, never gated."""
import numpy as np

ATE_TOL = 0.15


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def check_gates(res, learner, ref="X"):
    out = []
    for s in ("rho0", "kappa0"):
        r, x = res[s][learner], res[s][ref]
        m = np.abs(np.asarray(r["ate_err"]).mean(0))
        out.append((f"G{1 if s == 'rho0' else 2} |ATE err| <= {ATE_TOL} [{s}]",
                    bool((m <= ATE_TOL).all()), np.round(m, 3).tolist()))
        sp, spx = np.asarray(r["spearman"]), np.asarray(x["spearman"])
        ok = sp.mean(0) >= spx.mean(0) - spx.std(0, ddof=1)
        out.append((f"G3 Spearman >= {ref} - 1sd [{s}]", bool(ok.all()),
                    np.round(sp.mean(0), 3).tolist()))
        if learner != ref:
            m, se = paired(r["value"], x["value"])
            out.append((f"G4 value not sig. below {ref} (diff >= -2 SE) [{s}]",
                        bool(m >= -2 * se), [round(m, 4), round(se, 4)]))
    return out
