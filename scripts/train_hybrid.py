"""Train the hybrid (M5): evolve first, then fine-tune the evolved champion with PPO.

    python scripts/train_hybrid.py --learn prey
    python scripts/train_hybrid.py --learn predator
    python scripts/train_hybrid.py --learn prey predator

Phase 1 runs GA (or ES, --evo es) for --evo-frac of the world-step budget; phase 2 starts PPO's actor from each
species' champion (fresh critic, value-only warm-up for --critic-warmup iterations) and trains for the rest.
Total budget is the same 150k world-steps every other method gets. Writes results/hybrid/<name>/ with the
evolution phase in evo/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_ga
import train_ppo

from ecoarena.env import PREDATOR, PREY, load_config
from ecoarena.eval.quick import evaluate
from ecoarena.trainers.es import ESConfig, ESTrainer
from ecoarena.trainers.ga import GAConfig, GATrainer
from ecoarena.trainers.ppo import PPOConfig, PPOTrainer, load_policy

NAMES = {"prey": PREY, "predator": PREDATOR}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--learn", nargs="+", choices=list(NAMES), default=["prey"])
    p.add_argument("--opponent", default="heuristic", choices=["heuristic", "random"])
    p.add_argument("--base", default="s1_large.json")
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--world-steps", type=float, default=150_000)
    p.add_argument("--evo", default="ga", choices=["ga", "es"])
    p.add_argument("--evo-frac", type=float, default=0.5, help="share of the budget spent evolving")
    p.add_argument("--population", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--gen-steps", type=int, default=128)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--minibatch", type=int, default=16384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--critic-warmup", type=int, default=20)
    p.add_argument("--init-log-std", type=float, default=-1.0, help="PPO action noise at hand-over (std 0.37)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-worlds", type=int, default=4)
    p.add_argument("--eval-steps", type=int, default=3000)
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="results/hybrid")
    a = p.parse_args()

    learn = [NAMES[n] for n in a.learn]
    name = a.name or f"{'+'.join(a.learn)}-{a.evo}-ppo-seed{a.seed}"
    out = Path(a.out) / name
    wcfg = load_config("S1", base=a.base)
    evo_steps = int(a.evo_frac * a.world_steps)

    # ---- phase 1: evolution
    common = {"learn": tuple(learn), "opponent": a.opponent, "n_worlds": a.worlds, "population": a.population,
              "sigma": a.sigma, "gen_steps": a.gen_steps, "world_step_budget": evo_steps, "seed": a.seed,
              "device": a.device}
    ecfg, Trainer = (GAConfig(**common), GATrainer) if a.evo == "ga" else (ESConfig(**common), ESTrainer)
    print(f"hybrid {name}: phase 1 {a.evo} for {evo_steps:,} world-steps", flush=True)
    evo = Trainer(wcfg, ecfg, out / "evo")
    hist1 = evo.train()
    train_ga.plot_curves(hist1, a.learn, out / "evo" / "curves.png")
    phase1_world_steps = hist1[-1]["world_steps"]

    # ---- phase 2: PPO from the champions
    ppo_steps = max(int(a.world_steps) - phase1_world_steps, a.rollout * a.worlds)  # 0 would mean 'no limit'
    pcfg = PPOConfig(learn=tuple(learn), opponent=a.opponent, n_worlds=a.worlds, rollout_steps=a.rollout,
                     iterations=100_000, world_step_budget=ppo_steps, minibatch=a.minibatch, lr=a.lr,
                     critic_warmup=a.critic_warmup, seed=a.seed, device=a.device)
    print(f"hybrid {name}: phase 2 PPO for {ppo_steps:,} world-steps from the {a.evo} champions", flush=True)
    ppo = PPOTrainer(wcfg, pcfg, out)
    for n, sp in NAMES.items():
        if sp in learn:
            ppo.init_actor(sp, out / "evo" / f"{n}.pt", a.init_log_std)
    hist2 = ppo.train()
    train_ppo.plot_curves(hist2, a.learn, out / "curves.png")

    # ---- evaluation: hybrid, the evolved champion alone (phase 1 only), and the baselines
    print("evaluating...", flush=True)
    ev_seed = 10_000 + a.seed
    other = {sp: a.opponent for sp in (PREY, PREDATOR) if sp not in learn}
    final = {sp: load_policy(out / f"{n}.pt", a.device) for n, sp in NAMES.items() if sp in learn}
    evolved = {sp: load_policy(out / "evo" / f"{n}.pt", a.device) for n, sp in NAMES.items() if sp in learn}
    results = {}
    for label, pols in [
        ("hybrid", {**final, **other}),
        (f"{a.evo}_phase1", {**evolved, **other}),
        ("heuristic", {sp: "heuristic" for sp in learn} | other),
        ("random", {sp: "random" for sp in learn} | other),
    ]:
        results[label] = evaluate(wcfg, pols, a.eval_worlds, a.eval_steps, ev_seed, a.device)
        print(label, json.dumps(results[label]), flush=True)
    (out / "eval.json").write_text(json.dumps({
        "method": "hybrid", "evo": a.evo, "evo_frac": a.evo_frac, "learn": a.learn, "opponent": a.opponent,
        "base": a.base, "world_steps": {"evo": phase1_world_steps, "ppo": hist2[-1]["world_steps"]},
        "eval_worlds": a.eval_worlds, "eval_steps": a.eval_steps, "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
