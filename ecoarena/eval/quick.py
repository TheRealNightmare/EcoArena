"""Quick evaluation shared by every trainer: fixed policies on fresh worlds, agent- and ecosystem-level numbers."""

from __future__ import annotations

import numpy as np
import torch

from ..env import PREDATOR, PREY, EcoWorld
from ..policies import make_policy
from ..trainers.common import step_rewards


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
    reward = [0.0, 0.0]  # shared training reward without the birth bonus (= GA fitness)
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
            reward[sp] += float(step_rewards(w, info, sp, e0[sp], a0[sp], birth_bonus=0.0).sum())
        eaten += int(info["eaten"].sum())
        encounter_mass += float((a0[PREY].sum(1).float() * a0[PREDATOR].sum(1).float()).sum())
        pops[t] = info["populations"].cpu().numpy()
    alive_both = (pops > 0).all(axis=2)
    return {
        "worlds_survived": int(alive_both[-1].sum()),
        "mean_pop": pops.mean(axis=(0, 1)).round(1).tolist(),
        # energy in, no dying: what every learner is trained on. Read it next to the attack rate: random prey
        # get caught no more often than trained ones per encounter, yet score far lower here (they starve).
        "prey_reward_per_1k_agent_steps": round(1000 * reward[PREY] / max(alive_steps[PREY], 1), 3),
        "pred_reward_per_1k_agent_steps": round(1000 * reward[PREDATOR] / max(alive_steps[PREDATOR], 1), 3),
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
