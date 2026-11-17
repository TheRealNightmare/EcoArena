"""Train EcoPPO (both species), the learner built to keep the ecosystem alive. See ecoarena/trainers/ecoppo.py.

    python scripts/train_ecoppo.py --name ecoppo-seed0 --seed 0
    # ablations: switch one part off
    python scripts/train_ecoppo.py --no-lineage | --resets | --balance 0 | --worlds S1 S1 S1 S1 S1 S1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecoarena.trainers.ecoppo import DEFAULT_WORLDS, EcoPPOConfig, EcoPPOTrainer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", nargs="+", default=list(DEFAULT_WORLDS),
                   help="training worlds: scenario or scenario@drag, e.g. S1 S2@0.1")
    p.add_argument("--worlds-each", type=int, default=1)
    p.add_argument("--base", default="s1_large.json")
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--iters", type=int, default=100_000, help="upper bound; training stops at --world-steps")
    p.add_argument("--world-steps", type=float, default=150_000, help="same budget as the benchmark")
    p.add_argument("--minibatch", type=int, default=16384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--no-lineage", action="store_true", help="ablation: benchmark energy reward")
    p.add_argument("--resets", action="store_true", help="ablation: restart a world when a species dies out")
    p.add_argument("--founders", type=int, default=10)
    p.add_argument("--migrant-every", type=int, default=10)
    p.add_argument("--balance", type=float, default=1.0, help="0 = ablation: no automatic balancing")
    p.add_argument("--balance-power", type=float, default=2.0)
    p.add_argument("--balance-mode", choices=["opponent", "ratio"], default="opponent")
    p.add_argument("--collapse-frac", type=float, default=0.0, help="count a collapse below this share of the start")
    p.add_argument("--extinction-penalty", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-worlds", type=int, default=0, help="kept for study.py; EcoPPO is tested by study.py eval")
    p.add_argument("--learn", nargs="+", default=["prey", "predator"], help="always both; accepted for study.py")
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="results/ecoppo")
    a = p.parse_args()

    name = a.name or f"ecoppo-seed{a.seed}"
    out = Path(a.out) / name
    cfg = EcoPPOConfig(worlds=tuple(a.worlds), worlds_each=a.worlds_each, base=a.base, rollout_steps=a.rollout,
                       iterations=a.iters, world_step_budget=int(a.world_steps), minibatch=a.minibatch, lr=a.lr,
                       gamma=a.gamma, lineage=not a.no_lineage, no_reset=not a.resets, founders=a.founders,
                       migrant_every=a.migrant_every, balance=a.balance, balance_power=a.balance_power,
                       balance_mode=a.balance_mode, collapse_frac=a.collapse_frac,
                       extinction_penalty=a.extinction_penalty, seed=a.seed, device=a.device)
    print(f"training {name} on {a.device}: {a.world_steps:,.0f} world-steps over {len(cfg.worlds)} worlds "
          f"({', '.join(cfg.worlds)})", flush=True)
    trainer = EcoPPOTrainer(cfg, out)
    trainer.train()
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1, default=str))
    (out / "eval.json").write_text(json.dumps({"learn": ["prey", "predator"], "results": {}}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
