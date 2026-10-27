"""Train the genetic algorithm (M2) and evaluate its champions against the baselines.

    python scripts/train_ga.py --learn prey                  # prey evolve vs heuristic predators
    python scripts/train_ga.py --learn predator              # predators evolve vs heuristic prey
    python scripts/train_ga.py --learn prey predator         # both co-evolve

Same world-step budget (default 150k), network and reward as PPO. Writes results/ga/<name>/.
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
from ecoarena.trainers.ga import GAConfig, GATrainer
from ecoarena.trainers.ppo import load_policy

NAMES = {"prey": PREY, "predator": PREDATOR}


def plot_curves(history: list[dict], learn: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ws = [h["world_steps"] / 1e3 for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for name in learn:
        for key, style in (("best", "-"), ("mean", ":")):
            y = np.array([h.get(f"{name}/{key}", np.nan) for h in history], dtype=float)
            k = max(1, len(y) // 25)
            sm = np.convolve(np.nan_to_num(y), np.ones(k) / k, mode="valid")
            axes[0].plot(ws[k - 1 :], sm, style, label=f"{name} {key}")
    axes[0].set_title("fitness = reward per agent-step (smoothed)")
    axes[0].set_xlabel("world-steps (thousands)")
    axes[1].plot(ws, [h["pop_prey"] for h in history], color="#2f7d4f", label="prey")
    axes[1].plot(ws, [h["pop_pred"] for h in history], color="#b8452f", label="predators")
    axes[1].set_title("population per world at end of each generation")
    axes[1].set_xlabel("world-steps (thousands)")
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
    p.add_argument("--population", type=int, default=50)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--gen-steps", type=int, default=128)
    p.add_argument("--world-steps", type=float, default=150_000)
    p.add_argument("--elite-frac", type=float, default=0.1)
    p.add_argument("--elite-score", default="last", choices=["last", "mean"])
    p.add_argument("--birth-bonus", type=float, default=0.0, help="0 = attributable fitness (see ga.py)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-worlds", type=int, default=4)
    p.add_argument("--eval-steps", type=int, default=3000)
    p.add_argument("--name", default=None)
    p.add_argument("--out", default="results/ga")
    a = p.parse_args()

    learn = [NAMES[n] for n in a.learn]
    name = a.name or f"{'+'.join(a.learn)}-vs-{a.opponent}-{Path(a.base).stem}-seed{a.seed}"
    out = Path(a.out) / name
    wcfg = load_config("S1", base=a.base)
    gcfg = GAConfig(learn=tuple(learn), opponent=a.opponent, n_worlds=a.worlds, population=a.population,
                    sigma=a.sigma, gen_steps=a.gen_steps, birth_bonus=a.birth_bonus,
                    elite_frac=a.elite_frac, elite_score=a.elite_score, world_step_budget=int(a.world_steps), seed=a.seed,
                    device=a.device)
    print(f"evolving {name} on {a.device}: population {a.population}, {a.gen_steps} steps x {a.worlds} worlds "
          f"per generation, {a.world_steps:,.0f} world-steps", flush=True)
    trainer = GATrainer(wcfg, gcfg, out)
    history = trainer.train()
    plot_curves(history, a.learn, out / "curves.png")

    print("evaluating champions...", flush=True)
    ev_seed = 10_000 + a.seed
    champs = {sp: load_policy(out / f"{n}.pt", a.device) for n, sp in NAMES.items() if sp in learn}
    other = {sp: a.opponent for sp in (PREY, PREDATOR) if sp not in learn}
    results = {}
    for label, pols in [
        ("ga", {**champs, **other}),
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
