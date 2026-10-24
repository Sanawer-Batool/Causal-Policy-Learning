"""Phase 1 demo: naive predictive targeting fails on confounded data.

Three things are shown, all scored against exact simulator truth:
  1. The naive observational contrast is biased vs. the true effect.
  2. 'Rank by predicted revenue' targeting is worse than random at equal spend.
  3. A predictive what-if model (fit Y ~ X,T, plug in each action) is worse than
     the oracle by a wide margin and gets worse as hidden confounding (kappa) grows.
"""
import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from cpl.simulator import CausalSimulator, SimConfig, FEATURES


def run(kappa: float, n: int, seed: int) -> dict:
    sim = CausalSimulator(SimConfig().with_(kappa=kappa, eta=1.0))
    train = sim.sample(n, seed)
    o = train.observed
    test_X = sim.sample(20_000, seed + 1000).observed.X   # fresh oracle sample
    rng = np.random.default_rng(seed)

    # --- strategy A: rank by predicted revenue, give everyone chosen the action 1
    resp = HistGradientBoostingRegressor(random_state=seed).fit(o.X[FEATURES], o.Y)
    score = resp.predict(test_X[FEATURES])
    oracle_a = sim.oracle_actions(test_X)
    spend_target = sim.policy_cost(oracle_a)
    q = spend_target / sim.costs[1]                  # fraction treatable at that spend
    a_rank = np.where(score >= np.quantile(score, 1 - q), 1, 0)

    # --- strategy B: predictive what-if (fit Y ~ X,T; treat if predicted lift > cost)
    Xt = o.X[FEATURES].assign(T=o.T)
    m = HistGradientBoostingRegressor(random_state=seed).fit(Xt, o.Y)
    pred = np.column_stack([m.predict(test_X[FEATURES].assign(T=a)) for a in range(4)])
    net = (pred - pred[:, [0]]) - sim.costs[None, :]
    a_whatif = net.argmax(axis=1)

    # --- reference: random policy with matched spend
    a_rand = np.where(rng.random(len(test_X)) < q, 1, 0)

    # --- naive observational ATE vs truth (effect on the treated)
    treated = o.T > 0
    naive = o.Y[treated].mean() - o.Y[~treated].mean()
    true_att = (train.oracle.Y_pot[np.arange(n), o.T][treated] - train.oracle.Y_pot[treated, 0]).mean()

    return {
        "kappa": kappa,
        "naive_contrast": naive, "true_effect_on_treated": true_att,
        "V_oracle": sim.policy_value(test_X, oracle_a),
        "V_whatif": sim.policy_value(test_X, a_whatif),
        "V_rank_by_predicted_revenue": sim.policy_value(test_X, a_rank),
        "V_random_matched_spend": sim.policy_value(test_X, a_rand),
        "spend_whatif": sim.policy_cost(a_whatif),
        "spend_oracle": spend_target,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for kappa in [0.0, 1.0, 2.0]:
        r = run(kappa, args.n, args.seed)
        print(f"\n=== kappa = {kappa} ===")
        for k, v in r.items():
            if k != "kappa":
                print(f"  {k:32s} {v:8.3f}")
