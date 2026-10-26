import argparse
from pathlib import Path

from cpl.harness import run_baselines, save
from cpl.simulator import SimConfig, CausalSimulator


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/sim_default.yaml")
parser.add_argument("--n", type=int, default=50_000)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--budget", type=float, default=0.18)
args = parser.parse_args()

sim = CausalSimulator(SimConfig.from_yaml(args.config))
Path("results").mkdir(exist_ok=True)
for seed in args.seeds:
    result = run_baselines(sim, args.n, seed, args.budget)
    save(result, f"results/phase2_baselines_seed{seed}.json")
    for name, metrics in result["results"].items():
        print(f"seed {seed} {name:18s} value={metrics['value']:+.3f} spend={metrics['spend']:.3f}")
