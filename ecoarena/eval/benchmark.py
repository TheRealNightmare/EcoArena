"""Benchmark evaluation: fixed policies on fresh worlds, every number kept per world.

`quick.evaluate` pools everything over worlds, which is right for a training-time check. The paper needs the
spread across worlds and the time of extinction (for survival curves), so this records each world separately
and returns the population time series too.
"""

from __future__ import annotations

import numpy as np
import torch

from ..env import PREDATOR, PREY, EcoWorld
from ..policies import make_policy
from ..trainers.common import step_rewards


@torch.no_grad()
def evaluate_worlds(cfg, policies: dict, worlds: int, steps: int, seed: int, device: str,
                    record_every: int = 10) -> dict:
    """policies: {species: callable | "random" | "heuristic"}. Returns per-world metrics, a pooled summary and
    the population series sampled every `record_every` steps."""
    w = EcoWorld(cfg, n_worlds=worlds, seed=seed, device=device)
    pol = {sp: (p if callable(p) else make_policy(p, w, sp)) for sp, p in policies.items()}
    z = lambda: torch.zeros(worlds, device=device, dtype=torch.float64)
    alive_steps, deaths, births, reward = ([z(), z()] for _ in range(4))
    eaten, encounter = z(), z()
    extinct_at = np.full((worlds, 2), -1, dtype=np.int64)
    pops = np.zeros((steps, worlds, 2), dtype=np.int32)
    for t in range(steps):
        o = w.observe()
        e0 = [w.species[sp].energy.clone() for sp in (PREY, PREDATOR)]
        a0 = [w.species[sp].alive.clone() for sp in (PREY, PREDATOR)]
        info = w.step(pol[PREY](o[PREY]), pol[PREDATOR](o[PREDATOR]))
        for sp in (PREY, PREDATOR):
            alive_steps[sp] += a0[sp].sum(1)
            deaths[sp] += info["deaths"][sp]
            births[sp] += info["births"][sp]
            reward[sp] += step_rewards(w, info, sp, e0[sp], a0[sp], birth_bonus=0.0).sum(1).double()
        eaten += info["eaten"].sum(1)
        encounter += a0[PREY].sum(1).double() * a0[PREDATOR].sum(1).double()
        p = info["populations"].cpu().numpy()
        pops[t] = p
        newly = (p == 0) & (extinct_at < 0)
        extinct_at[newly] = t + 1

    c = lambda x: x.cpu().numpy()
    A = [c(x) for x in alive_steps]
    Dth = [c(x) for x in deaths]
    B = [c(x) for x in births]
    R = [c(x) for x in reward]
    E, M = c(eaten), c(encounter)
    half = pops[steps // 2 :].astype(np.float64)
    per_world = []
    for i in range(worlds):
        rec = {
            "survived": bool((pops[-1, i] > 0).all()),
            "extinct_at": {"prey": int(extinct_at[i, 0]) if extinct_at[i, 0] > 0 else None,
                           "predator": int(extinct_at[i, 1]) if extinct_at[i, 1] > 0 else None},
            "mean_pop_2nd_half": half[:, i].mean(0).round(1).tolist(),
            # coefficient of variation of each population over the second half: lower = steadier ecosystem
            "cv_2nd_half": [round(float(half[:, i, s].std() / max(half[:, i, s].mean(), 1e-9)), 3) for s in (0, 1)],
            "attack_rate_e6": round(1e6 * E[i] / max(M[i], 1.0), 3),
            "pred_catches_per_1k_agent_steps": round(1000 * E[i] / max(A[1][i], 1), 3),
            "prey_mean_lifespan": round(A[0][i] / max(Dth[0][i], 1), 1),
            "pred_mean_lifespan": round(A[1][i] / max(Dth[1][i], 1), 1),
            "prey_births_per_1k_agent_steps": round(1000 * B[0][i] / max(A[0][i], 1), 3),
            "pred_births_per_1k_agent_steps": round(1000 * B[1][i] / max(A[1][i], 1), 3),
            "prey_reward_per_1k_agent_steps": round(1000 * R[0][i] / max(A[0][i], 1), 3),
            "pred_reward_per_1k_agent_steps": round(1000 * R[1][i] / max(A[1][i], 1), 3),
        }
        per_world.append(rec)
    keys = [k for k, v in per_world[0].items() if isinstance(v, float)]
    summary = {k: round(float(np.mean([r[k] for r in per_world])), 3) for k in keys}
    summary["worlds_survived"] = int(sum(r["survived"] for r in per_world))
    summary["worlds"] = worlds
    return {"summary": summary, "per_world": per_world, "pops": pops[::record_every]}
