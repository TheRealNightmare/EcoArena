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
from ecoarena.env import PREDATOR, PREY, EcoWorld, load_config
from ecoarena.policies import make_policy
from ecoarena.trainers.ppo import PPOConfig, PPOTrainer, load_policy

NAMES = {"prey": PREY, "predator": PREDATOR}


@torch.no_grad()
def evaluate(cfg, policies: dict, worlds: int, steps: int, seed: int, device: str) -> dict:
    """Run fixed policies and report agent-level and ecosystem-level numbers."""
    w = EcoWorld(cfg, n_worlds=worlds, seed=seed, device=device)
    pol = {sp: (p if callable(p) else make_policy(p, w, sp)) for sp, p in policies.items()}
    food = [0.0, 0.0]
    alive_steps = [0, 0]
    deaths = [0, 0]
    births = [0, 0]
    eaten = 0
    encounter_mass = 0.0  # sum over steps and worlds of prey x predators (Lotka-Volterra interaction term)
    pops = np.zeros((steps, worlds, 2))
    for t in range(steps):
        o = w.observe()
        e0 = [w.species[sp].energy.clone() for sp in (PREY, PREDATOR)]
        a0 = [w.species[sp].alive.clone() for sp in (PREY, PREDATOR)]
        info = w.step(pol[PREY](o[PREY]), pol[PREDATOR](o[PREDATOR]))
        for sp in (PREY, PREDATOR):
            alive_steps[sp] += int(a0[sp].sum())
            deaths[sp] += int(info["deaths"][sp].sum())
            births[sp] += int(info["births"][sp].sum())
            gained = (w.species[sp].energy - e0[sp]).clamp(min=0)
            food[sp] += float(gained[a0[sp] & ~info["died"][sp]].sum())
        eaten += int(info["eaten"].sum())
        encounter_mass += float((a0[PREY].sum(1).float() * a0[PREDATOR].sum(1).float()).sum())
        pops[t] = info["populations"].cpu().numpy()
    alive_both = (pops > 0).all(axis=2)
    return {
        "worlds_survived": int(alive_both[-1].sum()),
        "mean_pop": pops.mean(axis=(0, 1)).round(1).tolist(),
        "prey_mean_lifespan": round(alive_steps[PREY] / max(deaths[PREY], 1), 1),
        "pred_mean_lifespan": round(alive_steps[PREDATOR] / max(deaths[PREDATOR], 1), 1),
        "prey_share_of_deaths_eaten": round(eaten / max(deaths[PREY], 1), 3),
        "prey_births_per_1k_agent_steps": round(1000 * births[PREY] / max(alive_steps[PREY], 1), 2),
        "pred_births_per_1k_agent_steps": round(1000 * births[PREDATOR] / max(alive_steps[PREDATOR], 1), 2),
        "pred_catches_per_1k_agent_steps": round(1000 * eaten / max(alive_steps[PREDATOR], 1), 2),
        # captures per (prey x predator x step) x 1e6: the Lotka-Volterra attack rate. Independent of how many
        # animals there are, so it compares hunting/escaping skill across ecosystem states. Lower = better prey.
        "attack_rate_e6": round(1e6 * eaten / max(encounter_mass, 1.0), 3),
    }


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
    p.add_argument("--agent-steps", type=float, default=20e6, help="per learning species (fairness budget)")
    p.add_argument("--minibatch", type=int, default=16384)
    p.add_argument("--lr", type=float, default=3e-4)
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
                     iterations=a.iters, agent_step_budget=int(a.agent_steps), minibatch=a.minibatch, lr=a.lr, seed=a.seed, device=a.device)
    print(f"training {name} on {a.device}: up to {a.agent_steps:,.0f} agent-steps per learning species, "
          f"{a.rollout} steps x {a.worlds} worlds per iteration", flush=True)
    trainer = PPOTrainer(wcfg, pcfg, out)
    history = trainer.train()
    plot_curves(history, a.learn, out / "curves.png")

    # Evaluation on fresh seeds: trained species vs the same opponent, compared with both baselines.
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
