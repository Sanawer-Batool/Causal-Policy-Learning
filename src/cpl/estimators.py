"""Frozen estimator registry. Configs here are the ones confirmed in Phase 3e.
Change a config only with a logged reason; never tune on fresh confirmation seeds."""
from .learners import SLearner, TLearner, XLearner
from .dr_learner import DRLearner
from .forest import CausalForest, EconmlDR

REGISTRY = {
    "S": lambda seed: SLearner(seed=seed),
    "T": lambda seed: TLearner(seed=seed),
    "X": lambda seed: XLearner(seed=seed),
    "DR-S": lambda seed: DRLearner(seed=seed, nuisance="s", clip=0.10),   # frozen in 3c
    "Forest": lambda seed: CausalForest(seed=seed, n_estimators=500),
    "EconmlDR": lambda seed: EconmlDR(seed=seed),                          # cross-check only
}


def make_estimator(name, seed=0):
    return REGISTRY[name](seed)
