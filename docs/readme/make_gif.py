"""Render the README animation: a top-down view of the real benchmark world (Python simulator) with trained brains.

    python docs/readme/make_gif.py [--brain es] [--seed 0] [--frames 150] [--every 3]

Writes docs/readme/world.gif. Prey are sand dots, predators red arrows (pointing where they head), refuges grey,
grass shaded by how much there is. CPU is fine (a few minutes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ecoarena.env import PREDATOR, PREY, EcoWorld, load_config
from ecoarena.trainers.ppo import load_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="es")
    ap.add_argument("--run-seed", type=int, default=0, help="which benchmark training seed's brains")
    ap.add_argument("--seed", type=int, default=20_000, help="world seed")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--every", type=int, default=3, help="world steps per frame")
    ap.add_argument("--warmup", type=int, default=400, help="steps simulated before recording starts")
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--view", type=float, default=90, help="side of the square window shown (world units)")
    ap.add_argument("--x0", type=float, default=40)
    ap.add_argument("--y0", type=float, default=60)
    ap.add_argument("--out", default=str(Path(__file__).with_name("world.gif")))
    a = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    cfg = load_config("S1", base="s1_large.json")
    w = EcoWorld(cfg, n_worlds=1, seed=a.seed, device="cpu")
    d = ROOT / "results" / "bench" / "train" / a.brain / f"{a.brain}-seed{a.run_seed}"
    pol = {PREY: load_policy(d / "prey.pt"), PREDATOR: load_policy(d / "predator.pt")}
    G, L = cfg.grass_grid, cfg.world_size
    refuge = w.refuge[0].reshape(G, G).numpy().T
    cmap = LinearSegmentedColormap.from_list("g", ["#3b2f1c", "#4d6b2a", "#6f9a3a", "#9cc755"])

    def advance(n: int) -> None:
        with torch.no_grad():
            for _ in range(n):
                o = w.observe()
                w.step(pol[PREY](o[PREY]), pol[PREDATOR](o[PREDATOR]))

    advance(a.warmup)
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="ecoarena-gif-"))
    fig = plt.figure(figsize=(4.6, 4.6), dpi=100)
    x0, y0, V = a.x0, a.y0, a.view
    for k in range(a.frames):
        fig.clf()
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(x0, x0 + V)
        ax.set_ylim(y0, y0 + V)
        ax.axis("off")
        grass = w.grass[0].reshape(G, G).numpy().T
        ax.imshow(grass, origin="lower", extent=(0, L, 0, L), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~refuge, refuge), origin="lower", extent=(0, L, 0, L),
                  cmap=LinearSegmentedColormap.from_list("r", ["#8f96a3", "#8f96a3"]), alpha=0.85)
        prey, pred = w.species[PREY], w.species[PREDATOR]
        pa, qa = prey.alive[0].numpy(), pred.alive[0].numpy()
        pp = prey.pos[0].numpy()[pa]
        ax.scatter(pp[:, 0], pp[:, 1], s=24, c="#f1d39a", edgecolors="#4a3510", linewidths=0.7, zorder=3)
        qp, qh = pred.pos[0].numpy()[qa], pred.heading[0].numpy()[qa]
        ax.quiver(qp[:, 0], qp[:, 1], np.cos(qh), np.sin(qh), color="#e2532f", scale=26, width=0.011,
                  headwidth=2.6, headlength=2.8, headaxislength=2.5, pivot="middle", zorder=4,
                  edgecolor="#2a0c05", linewidth=0.4)
        n_prey, n_pred = int(pa.sum()), int(qa.sum())
        ax.text(x0 + 2.5, y0 + V - 3, f"step {w.t:,}", color="#fffaf0", fontsize=8.5, family="monospace", va="top",
                bbox={"boxstyle": "round,pad=0.35", "fc": (0.07, 0.1, 0.07, 0.72), "ec": "none"})
        ax.text(x0 + V - 2.5, y0 + V - 3, f"prey {n_prey}  predators {n_pred} (whole world)", color="#fffaf0", fontsize=8.5,
                family="monospace", va="top", ha="right",
                bbox={"boxstyle": "round,pad=0.35", "fc": (0.07, 0.1, 0.07, 0.72), "ec": "none"})
        fig.canvas.draw()
        Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).save(tmp / f"f{k:04d}.png")
        advance(a.every)
        if k % 25 == 0:
            print(f"frame {k}/{a.frames}  step {w.t}  prey {n_prey}  predators {n_pred}", flush=True)
    # ffmpeg: one shared palette, and unchanged pixels between frames stored as transparent (much smaller file)
    import subprocess

    pal = tmp / "palette.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(tmp / "f%04d.png"), "-vf", "palettegen=max_colors=32:stats_mode=diff", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(a.fps), "-i", str(tmp / "f%04d.png"), "-i", str(pal),
                    "-lavfi", "paletteuse=dither=none:diff_mode=rectangle", "-loop", "0", a.out], check=True)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
