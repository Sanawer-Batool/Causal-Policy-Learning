from __future__ import annotations

from cpl.adapters import unpack
from cpl.simulator import CausalSimulator, SimConfig
from cpl.splits import make_sim_splits


SETTINGS = {
    "rho0": dict(kappa=0.0, eta=0.0, rho=0.0),
    "kappa0": dict(kappa=0.0, eta=1.0),
    "kappa1": dict(kappa=1.0, eta=1.0),
}


def load_scenario(setting: str, seed: int):
    """Return D_dev arrays, D_ope features, and true D_ope effects."""
    base = SimConfig.from_yaml("configs/sim_default.yaml")
    config = base.with_(**SETTINGS[setting])
    sim = CausalSimulator(config)
    splits = make_sim_splits(sim, 50_000, seed)
    development = unpack(splits.dev)
    evaluation_X = splits.ope.observed.X
    tau_true = sim.true_tau(evaluation_X)
    return (
        development.X,
        development.T,
        development.Y,
        evaluation_X,
        tau_true,
    )
