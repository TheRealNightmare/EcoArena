"""Run a baseline policy in W parallel worlds and report survival + population curves.

Each world is an independent replicate (the RNG key includes the world index), so
`--worlds 5` gives the 5 seeds the survivability gate asks for.

    python scripts/run_baseline.py --policy heuristic --worlds 5 --steps 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecoarena.env import EcoWorld, load_config
from ecoarena.policies import make_policy


def run(policy: str, scenario: str, worlds: int, steps: int, seed: int, device: str, overrides: dict):
    cfg = load_config(scenario, **overrides)
    world = EcoWorld(cfg, n_worlds=worlds, seed=seed, device=device)
    prey_pol, pred_pol = make_policy(policy, world, 0), make_policy(policy, world, 1)
    pops = np.zeros((steps + 1, worlds, 2), dtype=np.int32)
    pops[0] = world.populations().cpu().numpy()
    captures = np.zeros(worlds, dtype=np.int64)
    t0 = time.time()
    for t in range(steps):
        o_prey, o_pred = world.observe()
        info = world.step(prey_pol(o_prey), pred_pol(o_pred))
        pops[t + 1] = info["populations"].cpu().numpy()
        captures += info["captures"].cpu().numpy()
        if (pops[t + 1] == 0).any(axis=1).all():
            pops[t + 2 :] = pops[t + 1]
            break
    secs = time.time() - t0
    return cfg, pops, captures, secs


def summarize(pops: np.ndarray) -> dict:
    alive_both = (pops > 0).all(axis=2)  # [T, W]
    extinct_at = [int(np.argmin(col)) if not col.all() else None for col in alive_both.T]
    survived = sum(e is None for e in extinct_at)
    tail = pops[len(pops) // 2 :]
    return {
        "worlds_survived": survived,
        "worlds": pops.shape[1],
        "extinct_at_step": extinct_at,
        "mean_pop_second_half": tail.mean(axis=(0, 1)).round(1).tolist(),
        "cv_second_half": (tail.std(axis=0) / np.maximum(tail.mean(axis=0), 1e-9)).mean(axis=0).round(3).tolist(),
    }


def plot(pops: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W = pops.shape[1]
    fig, axes = plt.subplots(W, 1, figsize=(9, 1.9 * W), sharex=True, squeeze=False)
    for w in range(W):
        ax = axes[w, 0]
        ax.plot(pops[:, w, 0], color="#2f7d4f", lw=1, label="prey")
        ax.plot(pops[:, w, 1], color="#b8452f", lw=1, label="predators")
        ax.set_ylabel(f"seed {w}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(loc="upper right", fontsize=7, frameon=False)
    axes[0, 0].set_title(title, fontsize=10)
    axes[-1, 0].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="heuristic", choices=["heuristic", "random"])
    p.add_argument("--scenario", default="S1")
    p.add_argument("--worlds", type=int, default=5)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. predator.max_speed=0.7")
    p.add_argument("--out", default="results/baselines")
    a = p.parse_args()

    overrides = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
    _cfg, pops, capt, secs = run(a.policy, a.scenario, a.worlds, a.steps, a.seed, a.device, overrides)
    s = summarize(pops)
    agent_steps = int(pops[1:].sum())
    s.update({
        "policy": a.policy, "scenario": a.scenario, "steps": a.steps, "device": a.device,
        "wall_seconds": round(secs, 1),
        "world_steps_per_sec": round(a.steps * a.worlds / secs, 1),
        "agent_steps_per_sec": round(agent_steps / secs, 1),
        "captures_per_world": capt.tolist(),
        "overrides": overrides,
    })
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.policy}-{a.scenario}-seed{a.seed}"
    np.save(out / f"{stem}-pops.npy", pops)
    (out / f"{stem}-summary.json").write_text(json.dumps(s, indent=2))
    plot(pops, out / f"{stem}.png", f"{a.policy} · {a.scenario} · {s['worlds_survived']}/{a.worlds} worlds keep both species")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
