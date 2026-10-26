import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .adapters import unpack


def _lgb(seed):
    """Construct the configured tree regressor without requiring an extra backend."""
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=50, random_state=seed,
    )


def budget_cut(actions, score, costs, budget):
    """Keep highest-score treated units until mean spend reaches ``budget``."""
    actions = np.asarray(actions, int)
    score = np.asarray(score, float)
    costs = np.asarray(costs, float)
    n = len(actions)
    candidates = np.where(actions > 0)[0]
    order = candidates[np.argsort(-score[candidates], kind="stable")]
    keep = order[np.cumsum(costs[actions[order]]) <= budget * n]
    output = np.zeros(n, dtype=int)
    output[keep] = actions[keep]
    return output


def naive_best_arm(o, costs):
    base = o.Y[o.T == 0].mean()
    gain = [o.Y[o.T == arm].mean() - base - costs[arm] for arm in (1, 2, 3)]
    best = int(np.argmax(gain))
    return best + 1 if gain[best] > 0 else 0


class Baseline:
    name = "base"

    def __init__(self, costs, seed=0):
        self.costs = np.asarray(costs, float)
        self.seed = seed

    def fit(self, data):
        return self

    def act(self, data, budget):
        raise NotImplementedError


class RandomMatched(Baseline):
    name = "random_matched"

    def act(self, data, budget):
        n = len(unpack(data).X)
        rng = np.random.default_rng(self.seed)
        return budget_cut(rng.integers(1, 4, n), rng.random(n), self.costs, budget)


class GlobalBestArm(Baseline):
    name = "global_best_arm"

    def fit(self, data):
        self.arm = naive_best_arm(unpack(data), self.costs)
        return self

    def act(self, data, budget):
        n = len(unpack(data).X)
        rng = np.random.default_rng(self.seed)
        return budget_cut(np.full(n, self.arm), rng.random(n), self.costs, budget)


class ResponseRanking(Baseline):
    name = "response_ranking"

    def fit(self, data):
        observed = unpack(data)
        self.arm = naive_best_arm(observed, self.costs)
        self.model = _lgb(self.seed).fit(observed.X, observed.Y)
        return self

    def act(self, data, budget):
        observed = unpack(data)
        return budget_cut(
            np.full(len(observed.X), self.arm),
            self.model.predict(observed.X),
            self.costs,
            budget,
        )


class NaiveWhatIf(Baseline):
    name = "naive_what_if"

    def fit(self, data):
        observed = unpack(data)
        self.models = [
            _lgb(self.seed).fit(observed.X[observed.T == arm], observed.Y[observed.T == arm])
            for arm in range(4)
        ]
        return self

    def act(self, data, budget):
        observed = unpack(data)
        means = np.column_stack([model.predict(observed.X) for model in self.models])
        net = np.column_stack(
            [np.zeros(len(observed.X)), means[:, 1:] - means[:, [0]] - self.costs[1:]]
        )
        actions = net.argmax(axis=1)
        return budget_cut(actions, net.max(axis=1), self.costs, budget)


ALL = [RandomMatched, GlobalBestArm, ResponseRanking, NaiveWhatIf]
