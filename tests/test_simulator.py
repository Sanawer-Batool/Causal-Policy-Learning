import numpy as np
import pytest

from cpl.simulator import CausalSimulator, SimConfig, N_ACTIONS

N = 60_000


def make(**kw):
    return CausalSimulator(SimConfig().with_(**kw))


def test_shapes_and_observed_outcome_consistency():
    d = make().sample(2000, seed=1)
    o, g = d.observed, d.oracle
    assert g.Y_pot.shape == (2000, N_ACTIONS)
    # Y_obs must equal the potential outcome of the action actually taken
    assert np.allclose(o.Y, g.Y_pot[np.arange(2000), o.T])
    # logged propensity is the probability of the action actually taken
    assert np.allclose(o.logging_propensity, g.e_full[np.arange(2000), o.T])
    assert np.allclose(g.e_full.sum(axis=1), 1.0)


def test_reproducible_with_seed_and_different_across_seeds():
    sim = make()
    a, b, c = sim.sample(500, 7), sim.sample(500, 7), sim.sample(500, 8)
    assert np.array_equal(a.observed.Y, b.observed.Y)
    assert np.array_equal(a.observed.T, b.observed.T)
    assert not np.array_equal(a.observed.Y, c.observed.Y)


def test_propensity_floor_respected():
    cfg = SimConfig().with_(eps=0.05, rho=3.0)
    d = CausalSimulator(cfg).sample(20_000, 0)
    # clip-then-renormalise can shave the floor slightly, never by much
    assert d.oracle.e_full.min() >= 0.05 / (1 + 4 * 0.05) - 1e-12


def test_true_tau_matches_monte_carlo():
    """Gauss-Hermite truth must agree with brute-force averaging of potential
    outcomes in a big sample, within a few standard errors, in coarse X-bins."""
    sim = make()
    d = sim.sample(400_000, seed=3)
    tau = sim.true_tau(d.observed.X)
    mc = d.oracle.Y_pot[:, 1:] - d.oracle.Y_pot[:, [0]]
    # compare overall mean effect per action
    se = mc.std(axis=0) / np.sqrt(len(mc))
    assert np.all(np.abs(mc.mean(axis=0) - tau.mean(axis=0)) < 4 * se)
    # and within a bin of one covariate
    mask = d.observed.X["monetary"].to_numpy() > 1.0
    se_b = mc[mask].std(axis=0) / np.sqrt(mask.sum())
    assert np.all(np.abs(mc[mask].mean(axis=0) - tau[mask].mean(axis=0)) < 4 * se_b)


def test_tau_zero_for_action_zero_and_net_value_column_zero():
    sim = make()
    X = sim.sample(300, 0).observed.X
    nv = sim.net_value(X)
    assert nv.shape == (300, 4)
    assert np.all(nv[:, 0] == 0.0)


def test_effects_are_heterogeneous_and_sometimes_negative():
    sim = make()
    tau = sim.true_tau(sim.sample(20_000, 0).observed.X)
    assert np.all(tau.std(axis=0) > 0.3)           # varies across customers
    assert np.all((tau < 0).mean(axis=0) > 0.05)   # reject option is meaningful
    # different actions are best for different customers
    shares = np.bincount(sim.oracle_actions(sim.sample(20_000, 0).observed.X), minlength=4)
    assert (shares > 0).all()


def test_kappa_increases_dependence_of_assignment_on_hidden_U():
    """Raising kappa must make T depend more on the hidden U."""
    strengths = []
    for kappa in [0.0, 0.5, 1.0, 2.0]:
        d = make(kappa=kappa).sample(N, seed=5)
        U, T = d.oracle.U, d.observed.T
        strengths.append(abs(np.corrcoef(U, (T == 3).astype(float))[0, 1])
                         + abs(np.corrcoef(U, (T == 1).astype(float))[0, 1]))
    assert strengths[0] < 0.02                      # kappa=0 -> T independent of U
    assert all(b > a for a, b in zip(strengths, strengths[1:]))


def test_kappa_increases_naive_bias():
    """Naive treated-vs-untreated contrast should drift further from the true
    effect on the treated as hidden confounding (kappa) grows."""
    biases = []
    for kappa in [0.0, 1.0, 2.0]:
        d = make(kappa=kappa, eta=1.0).sample(N, seed=11)
        o, g = d.observed, d.oracle
        treated = o.T > 0
        naive = o.Y[treated].mean() - o.Y[~treated].mean()
        truth = (g.Y_pot[np.arange(N), o.T][treated] - g.Y_pot[treated, 0]).mean()
        biases.append(naive - truth)
    assert biases[0] < biases[1] < biases[2]


def test_eta_zero_means_no_hidden_outcome_confounding_in_true_tau():
    """With eta=0, U does not affect Y so tau is unchanged by kappa."""
    X = make().sample(1000, 0).observed.X
    t1 = make(eta=0.0, kappa=0.0).true_tau(X)
    t2 = make(eta=0.0, kappa=2.0).true_tau(X)
    assert np.allclose(t1, t2)


def test_policy_value_ordering_oracle_beats_random_and_none():
    sim = make()
    X = sim.sample(20_000, 0).observed.X
    rng = np.random.default_rng(0)
    v_oracle = sim.policy_value(X, sim.oracle_actions(X))
    v_none = sim.policy_value(X, np.zeros(len(X), dtype=int))
    v_rand = sim.policy_value(X, rng.integers(0, 4, len(X)))
    assert v_none == 0.0
    assert v_oracle > v_rand
    assert v_oracle > 0


def test_stronger_rho_reduces_overlap():
    def min_support(rho):
        e = make(rho=rho, eps=1e-6).sample(20_000, 0).oracle.e_full
        return (e < 0.01).mean()
    assert min_support(0.5) < min_support(1.0) < min_support(3.0)
