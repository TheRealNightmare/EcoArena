"""Train PPO (M1) and evaluate it against the baselines.

Stages:
    python scripts/train_ppo.py --learn prey                 # prey learn vs heuristic predators
    python scripts/train_ppo.py --learn predator             # predators learn vs heuristic prey
    python scripts/train_ppo.py --learn prey predator        # both learn together

Writes results/ppo/<name>/: prey.pt / predator.pt, history.json, curves.png, eval.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecoarena.env import PREDATOR, PREY, load_config
from ecoarena.eval.quick import evaluate
from ecoarena.trainers.ppo import PPOConfig, PPOTrainer, load_policy

NAMES = {"prey": PREY, "predator": PREDATOR}


def plot_curves(history: list[dict], learn: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    it = [h["iter"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for name in learn:
        steps = np.cumsum([0] + [0] * (len(history) - 1))
        steps = [h["agent_steps"][name] / 1e6 for h in history]
        r = np.array([h.get(f"{name}/reward_per_agent_step", np.nan) for h in history])
        k = max(1, len(r) // 20)
        smooth = np.convolve(r, np.ones(k) / k, mode="valid")
        axes[0].plot(steps[k - 1 :], smooth, label=name)
        axes[1].plot(steps, [h.get(f"{name}/entropy", np.nan) for h in history], label=name)
    axes[0].set_title("reward per agent-step (smoothed)")
    axes[0].set_xlabel("agent-steps (millions)")
    axes[1].set_title("policy entropy")
    axes[1].set_xlabel("agent-steps (millions)")
    axes[2].plot(it, [h["pop_prey"] for h in history], color="#2f7d4f", label="prey")
    axes[2].plot(it, [h["pop_pred"] for h in history], color="#b8452f", label="predators")
    axes[2].set_title("population per world at end of each rollout")
    axes[2].set_xlabel("iteration")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--learn", nargs="+", choices=list(NAMES), default=["prey"])
    p.add_argument("--opponent", default="heuristic", choices=["heuristic", "random"])
    p.add_argument("--base", default="s1_large.json")
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--iters", type=int, default=3000, help="upper bound; training stops at --agent-steps")
    p.add_argument("--world-steps", type=float, default=150_000,
                   help="fairness budget: world-steps summed over parallel worlds; all learners train throughout")
    p.add_argument("--agent-steps", type=float, default=0,
                   help="optional: freeze each species at this many agent-steps (off by default; see ppo.py)")
    p.add_argument("--minibatch", type=int, default=16384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--birth-bonus", type=float, default=0.0, help="benchmark uses 0 for every method")
    p.add_argument("--eco-lambda", type=float, default=0.0, help="study 2: shared ecosystem-health reward")
    p.add_argument("--prey-every", type=int, default=1, help="study 2: update prey every k-th step; 0 = frozen")
    p.add_argument("--pred-every", type=int, default=1, help="study 2: update predators every k-th step; 0 = frozen")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-worlds", type=int, default=4)
    p.add_argument("--eval-steps", type=int, default=3000)
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="results/ppo")
    a = p.parse_args()

    learn = [NAMES[n] for n in a.learn]
    name = a.name or f"{'+'.join(a.learn)}-vs-{a.opponent}-{Path(a.base).stem}-seed{a.seed}"
    out = Path(a.out) / name
    wcfg = load_config("S1", base=a.base)
    pcfg = PPOConfig(learn=tuple(learn), opponent=a.opponent, n_worlds=a.worlds, rollout_steps=a.rollout,
                     iterations=a.iters, agent_step_budget=int(a.agent_steps), world_step_budget=int(a.world_steps), minibatch=a.minibatch, lr=a.lr, birth_bonus=a.birth_bonus, seed=a.seed, device=a.device,
                     gamma=a.gamma, gae_lambda=a.gae_lambda, eco_lambda=a.eco_lambda,
                     prey_every=a.prey_every, pred_every=a.pred_every)
    print(f"training {name} on {a.device}: {a.world_steps:,.0f} world-steps, "
          f"{a.rollout} steps x {a.worlds} worlds per iteration", flush=True)
    trainer = PPOTrainer(wcfg, pcfg, out)
    history = trainer.train()
    plot_curves(history, a.learn, out / "curves.png")

    # Evaluation on fresh seeds: trained species vs the same opponent, compared with both baselines.
    if a.eval_worlds <= 0:
        (out / "eval.json").write_text(json.dumps({"learn": a.learn, "results": {}}))
        print(f"wrote {out} (no quick evaluation)")
        return
    print("evaluating...", flush=True)
    ev_seed = 10_000 + a.seed
    trained = {sp: load_policy(out / f"{n}.pt", a.device) for n, sp in NAMES.items() if sp in learn}
    other = {sp: a.opponent for sp in (PREY, PREDATOR) if sp not in learn}
    results = {}
    for label, pols in [
        ("ppo", {**trained, **other}),
        ("heuristic", {sp: "heuristic" for sp in learn} | other),
        ("random", {sp: "random" for sp in learn} | other),
    ]:
        results[label] = evaluate(wcfg, pols, a.eval_worlds, a.eval_steps, ev_seed, a.device)
        print(label, json.dumps(results[label]), flush=True)
    (out / "eval.json").write_text(json.dumps({"learn": a.learn, "opponent": a.opponent, "base": a.base,
                                               "eval_worlds": a.eval_worlds, "eval_steps": a.eval_steps,
                                               "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
