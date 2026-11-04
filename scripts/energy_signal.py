"""Energy-leak diagnosis: can selection in GA's shared-world evaluation even see the energy a predator saves by
coasting instead of jittering?

    python scripts/energy_signal.py [--windows 32] [--worlds 2] [--device cpu]

All 50 genome columns run the SAME predator brain (a trained GA champion), so any ranking between them is pure
luck of catches and deaths. Then columns 0-24 are credited exactly the energy that coasting saves:
    move_cost * (jitter thrust^2 - coasting thrust^2) / energy_cap per living agent-step
(measured thrust^2: GA ~0.56, PPO ~0.04). If selection could see that saving, the 5 elites would come from the
credited half far more often than the 50% expected by chance. Repeated for fitness windows of 128 (what GA,
ES and NEAT use), 512 and 1,024 steps. Writes results/study/energy_signal.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecoarena.env import PREDATOR, PREY, EcoWorld, load_config
from ecoarena.trainers.common import step_rewards
from ecoarena.trainers.ppo import load_policy

P = 50  # genome columns, as in the benchmark


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=32, help="number of 128-step windows to record")
    ap.add_argument("--worlds", type=int, default=2)
    ap.add_argument("--brain", default="ga-seed0")
    ap.add_argument("--jitter-th2", type=float, default=0.56)
    ap.add_argument("--coast-th2", type=float, default=0.04)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    cfg = load_config("S1", base="s1_large.json")
    m = a.brain.split("-")[0]
    d = ROOT / "results" / "bench" / "train" / m / a.brain
    pol = {PREY: load_policy(d / "prey.pt", a.device), PREDATOR: load_policy(d / "predator.pt", a.device)}
    qc = cfg.predator
    saving = qc.move_cost * (a.jitter_th2 - a.coast_th2) / qc.energy_cap  # per living predator-step

    seed = 31_000
    w = EcoWorld(cfg, n_worlds=a.worlds, seed=seed, device=a.device)
    rsum = np.zeros((a.windows, P))
    nsum = np.zeros((a.windows, P))
    with torch.no_grad():
        for k in range(a.windows):
            if int(w.populations()[:, 1].min()) < 30:  # keep enough predators: start fresh worlds
                seed += 1
                w = EcoWorld(cfg, n_worlds=a.worlds, seed=seed, device=a.device)
            for _ in range(128):
                o = w.observe()
                e0, a0 = w.species[PREDATOR].energy.clone(), w.species[PREDATOR].alive.clone()
                info = w.step(pol[PREY](o[PREY]), pol[PREDATOR](o[PREDATOR]))
                r = step_rewards(w, info, PREDATOR, e0, a0)
                Wn, N = r.shape
                rsum[k] += r.reshape(Wn, N // P, P).sum((0, 1)).cpu().numpy()
                nsum[k] += a0.reshape(Wn, N // P, P).sum((0, 1)).float().cpu().numpy()
            print(f"window {k + 1}/{a.windows}  predators {w.populations()[:, 1].tolist()}", flush=True)

    out = {"brain": a.brain, "saving_per_agent_step": saving, "windows": {}}
    rng = np.random.default_rng(0)
    for L in (1, 4, 8):  # 128, 512, 1024 steps
        n = a.windows // L
        R = rsum[: n * L].reshape(n, L, P).sum(1)
        Nn = nsum[: n * L].reshape(n, L, P).sum(1)
        fit = R / np.maximum(Nn, 1)
        noise_sd = float(fit.std(axis=1).mean())  # spread between identical genomes = pure luck
        credited = fit.copy()
        credited[:, :25] += saving
        top = np.argsort(-credited, axis=1)[:, :5]
        share = float((top < 25).mean())
        # the same test with random halves, to show 50% is what luck alone gives
        perm_share = float(np.mean([(np.argsort(-(fit + saving * (rng.permutation(P) < 25)), axis=1)[:, :5] < 25).mean()
                                    for _ in range(1)]))
        out["windows"][str(128 * L)] = {"generations": n, "noise_sd": noise_sd, "signal_to_noise": saving / noise_sd,
                                        "elites_from_credited_half": share, "check_random_half": perm_share,
                                        "samples_per_genome": float(Nn.mean())}
        print(f"window {128 * L:5d} steps: luck spread {noise_sd:.5f}, saving {saving:.5f} "
              f"(signal/noise {saving / noise_sd:.2f}); elites from the credited half {100 * share:.0f}% (chance 50%)")
    f = ROOT / "results" / "study" / "energy_signal.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
