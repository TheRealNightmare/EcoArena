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


def homeostat_gate(world: EcoWorld, sp: int, threshold: float) -> bool:
    """Homeostatic Co-Training: species `sp` should pause learning when the other species' population, averaged
    over the worlds, is below `threshold` times its starting count. threshold 0 = never pause."""
    if threshold <= 0:
        return False
    other = 1 - sp
    pops = world.populations()[:, other].float().mean()
    return bool(pops < threshold * world.species_cfg[other].init_count)


def step_rewards(world: EcoWorld, info: dict, sp: int, energy_before: torch.Tensor, alive_before: torch.Tensor,
                 birth_bonus: float = 0.0, death_penalty: float = 1.0, eco_lambda: float = 0.0,
                 cost_weight: float = 1.0, lineage: bool = False) -> torch.Tensor:
    """Per-slot reward for one step, the same rule for every method and both species.

        r = Δenergy / energy_cap
          + (child_energy / energy_cap + birth_bonus) per offspring   (refunds the energy given to the child)
          - death_penalty                                             on death (starved, old age, eaten)

    Slots that held no living agent at the start of the step get 0.

    birth_bonus is 0 for every method in the benchmark (2026-10-29). PPO was first trained with +1; the control
    runs prey-nobonus / predator-nobonus showed that bonus was what made PPO look worse than evolution on
    predators (catches/1k 5.30 -> 7.45) and it also held PPO prey back (attack rate 11.4 -> 8.2).

    eco_lambda (study 2, default 0): adds a shared reward for the world's ecosystem health; see ecosystem_health.
    cost_weight (study 2, default 1): multiplies the movement-cost part of the reward (see world.last_thrust).
    lineage (EcoPPO, default False): count the reward in children instead of energy, and charge every step the
        surplus an animal must bank to replace itself once in a lifetime:
            r = (Δenergy + child refund) / child_energy - 1 / max_age,   -death_penalty on death
        An animal that only breaks even on energy now loses about one child's worth over a full life, the same as
        dying, so the learner is pushed to R0 >= 1 (scripts/leak_model.py, Eq. 1 of the paper), not just to survive.
    """
    sc = world.species_cfg[sp]
    st = world.species[sp]
    died = info["died"][sp]
    unit = sc.child_energy if lineage else sc.energy_cap
    r = (st.energy - energy_before) / unit
    wi, _child, par = info["birth_pairs"][sp]
    if wi.numel():
        r[wi, par] += sc.child_energy / unit + birth_bonus
    if lineage:
        r = r - 1.0 / sc.max_age
    if cost_weight != 1.0:
        # energy-leak fix (study 2, E4): count the movement cost cost_weight times in the reward. The world itself
        # is unchanged; only how strongly a learner feels the cost of thrust.
        th = world.last_thrust[sp]
        r = r - (cost_weight - 1.0) * sc.move_cost * th * th / unit
    if eco_lambda:
        # "care" term for the study: every living animal also earns eco_lambda / 1000 per step times its world's
        # ecosystem health, so over a ~1000-step life lambda = 1 is worth about one death penalty
        r = r + (eco_lambda / 1000.0) * ecosystem_health(world)
    r = torch.where(died, torch.full_like(r, -death_penalty), r)
    return torch.where(alive_before, r, torch.zeros_like(r))
