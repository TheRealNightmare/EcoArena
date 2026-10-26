"""Throughput test: agent-steps per second for 1, 16 and 64 parallel worlds.

Run on the GPU machine:
    python scripts/timing.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecoarena.env import EcoWorld, load_config
from ecoarena.policies import make_policy


def bench(worlds: int, steps: int, device: str, base: str) -> tuple[float, float]:
    world = EcoWorld(load_config("S1", base=base), n_worlds=worlds, seed=0, device=device)
    pp, qp = make_policy("heuristic", world, 0), make_policy("heuristic", world, 1)
    for _ in range(10):  # warm-up
        o1, o2 = world.observe()
        world.step(pp(o1), qp(o2))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    agent_steps = 0
    t0 = time.perf_counter()
    for _ in range(steps):
        o1, o2 = world.observe()
        info = world.step(pp(o1), qp(o2))
        agent_steps += int(info["populations"].sum())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return agent_steps / dt, steps * worlds / dt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--worlds", type=int, nargs="*", default=[1, 16, 64])
    p.add_argument("--base", default="s1_balanced.json", help="e.g. s1_large.json")
    a = p.parse_args()
    print(f"device={a.device} torch={torch.__version__} base={a.base}")
    if a.device.startswith("cuda"):
        print(torch.cuda.get_device_name(0))
    for w in a.worlds:
        aps, wps = bench(w, a.steps, a.device, a.base)
        hours_20m = 20e6 / aps / 3600
        print(f"worlds={w:>3}  agent-steps/s={aps:>12,.0f}  world-steps/s={wps:>9,.0f}  "
              f"20M agent-steps ≈ {hours_20m:5.2f} h (env only, no learning)")


if __name__ == "__main__":
    main()
