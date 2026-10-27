"""Can GA fitness tell a good prey policy from a bad one? Signal-to-noise check for the GA evaluation.

Runs known prey policies side by side in the same worlds, exactly as the GA does: slot s belongs to "genome"
s % P, and genome j runs policy j % K. Per 128-step window it records each genome's fitness (mean reward per
agent-step, same rule as the GA). Signal = gap between policy means; noise = spread between genomes running the
SAME policy in the same window. If noise >> signal, truncation selection on one window picks winners by luck.

    python scripts/ga_signal.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecoarena.env import PREDATOR, PREY, EcoWorld, load_config
from ecoarena.policies import make_policy
from ecoarena.trainers.common import step_rewards
from ecoarena.trainers.ppo import load_policy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="s1_large.json")
    p.add_argument("--worlds", type=int, default=4)
    p.add_argument("--genomes", type=int, default=50)
    p.add_argument("--window", type=int, default=128)
    p.add_argument("--windows", type=int, default=20)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--policies", nargs="+", default=["ppo:results/ppo/prey-v2/prey.pt", "heuristic", "random",
                                                     "ga:results/ga/ga-prey-nb-g128-s02/prey.pt",
                                                     "ga-old:results/ga/ga-prey-g128-s02/prey.pt"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/ga/signal.json")
    a = p.parse_args()

    cfg = load_config("S1", base=a.base)
    w = EcoWorld(cfg, n_worlds=a.worlds, seed=4242, device=a.device)
    names, pols = [], []
    for spec in a.policies:
        name, _, path = spec.partition(":")
        names.append(name)
        pols.append(load_policy(path, a.device) if path else make_policy(name, w, PREY))
    pred = make_policy("heuristic", w, PREDATOR)
    K, P = len(pols), a.genomes
    N = w.species[PREY].alive.shape[1]
    genome_of_slot = torch.arange(N, device=a.device) % P
    policy_of_slot = genome_of_slot % K

    def act(obs):
        out = torch.zeros(obs.shape[0], obs.shape[1], 2, device=a.device)
        for k, pol in enumerate(pols):
            m = policy_of_slot == k
            out[:, m] = pol(obs[:, m])
        return out

    rows = {bb: [] for bb in (0.0, 1.0)}  # birth bonus -> list of [P] fitness per window
    for t in range(a.warmup + a.window * a.windows):
        obs = w.observe()
        e0, a0 = w.species[PREY].energy.clone(), w.species[PREY].alive.clone()
        info = w.step(act(obs[PREY]), pred(obs[PREDATOR]))
        if bool((w.populations() == 0).any()):
            sys.exit(f"extinction at step {t}; rerun with another seed")
        if t < a.warmup:
            continue
        if (t - a.warmup) % a.window == 0:
            acc = {bb: [torch.zeros(P, device=a.device), torch.zeros(P, device=a.device)] for bb in rows}
        for bb in rows:
            r = step_rewards(w, info, PREY, e0, a0, bb, 1.0)
            acc[bb][0].index_add_(0, genome_of_slot, r.sum(0))
            acc[bb][1].index_add_(0, genome_of_slot, a0.sum(0).float())
        if (t - a.warmup) % a.window == a.window - 1:
            for bb, lst in rows.items():
                lst.append((acc[bb][0] / acc[bb][1].clamp(min=1)).cpu())

    report = {"genomes": P, "window": a.window, "windows": a.windows, "worlds": a.worlds,
              "agent_steps_per_genome_per_window": float(acc[0.0][1].mean()), "by_birth_bonus": {}}
    gpol = torch.arange(P) % K
    for bb, lst in rows.items():
        F = torch.stack(lst)  # [windows, P]
        per_pol = {names[k]: F[:, gpol == k] for k in range(K)}
        within = torch.stack([v.std(1) for v in per_pol.values()]).mean().item()  # same-policy spread, one window
        means = {n: v.mean().item() for n, v in per_pol.items()}
        # how often would one window's top-10% genomes all be from the best policy?
        best = max(means, key=means.get)
        top = F.argsort(1, descending=True)[:, : max(1, P // 10)]
        hit = (gpol[top] == names.index(best)).float().mean().item()
        report["by_birth_bonus"][str(bb)] = {
            "policy_mean_fitness": means, "same_policy_std_one_window": within,
            "best_policy": best, "share_of_top10pct_from_best_policy": hit, "chance_level": 1 / K,
        }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
