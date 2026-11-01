"""Pieces every training method shares, so the comparison stays fair."""

from __future__ import annotations

import torch

from ..env import EcoWorld


def ecosystem_health(world: EcoWorld) -> torch.Tensor:
    """[W, 1] in [0, 1]: geometric mean of each species' population relative to its starting count, capped at 1.
    1 = both species at or above their starting numbers; 0 = a species is extinct."""
    pops = world.populations().float()  # [W, 2]
    init = torch.tensor([float(world.species_cfg[s].init_count) for s in (0, 1)], device=pops.device)
    rel = (pops / init).clamp(max=1.0)
    return (rel[:, 0] * rel[:, 1]).sqrt().unsqueeze(1)


def step_rewards(world: EcoWorld, info: dict, sp: int, energy_before: torch.Tensor, alive_before: torch.Tensor,
                 birth_bonus: float = 0.0, death_penalty: float = 1.0, eco_lambda: float = 0.0) -> torch.Tensor:
    """Per-slot reward for one step, the same rule for every method and both species.

        r = Δenergy / energy_cap
          + (child_energy / energy_cap + birth_bonus) per offspring   (refunds the energy given to the child)
          - death_penalty                                             on death (starved, old age, eaten)

    Slots that held no living agent at the start of the step get 0.

    birth_bonus is 0 for every method in the benchmark (2026-10-29). PPO was first trained with +1; the control
    runs prey-nobonus / predator-nobonus showed that bonus was what made PPO look worse than evolution on
    predators (catches/1k 5.30 -> 7.45) and it also held PPO prey back (attack rate 11.4 -> 8.2).

    eco_lambda (study 2, default 0): adds a shared reward for the world's ecosystem health; see ecosystem_health.
    """
    sc = world.species_cfg[sp]
    st = world.species[sp]
    died = info["died"][sp]
    r = (st.energy - energy_before) / sc.energy_cap
    wi, _child, par = info["birth_pairs"][sp]
    if wi.numel():
        r[wi, par] += sc.child_energy / sc.energy_cap + birth_bonus
    if eco_lambda:
        # "care" term for the study: every living animal also earns eco_lambda / 1000 per step times its world's
        # ecosystem health, so over a ~1000-step life lambda = 1 is worth about one death penalty
        r = r + (eco_lambda / 1000.0) * ecosystem_health(world)
    r = torch.where(died, torch.full_like(r, -death_penalty), r)
    return torch.where(alive_before, r, torch.zeros_like(r))
