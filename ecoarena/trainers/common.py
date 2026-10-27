"""Pieces every training method shares, so the comparison stays fair."""

from __future__ import annotations

import torch

from ..env import EcoWorld


def step_rewards(world: EcoWorld, info: dict, sp: int, energy_before: torch.Tensor, alive_before: torch.Tensor,
                 birth_bonus: float = 1.0, death_penalty: float = 1.0) -> torch.Tensor:
    """Per-slot reward for one step, the same rule for every method and both species.

        r = Δenergy / energy_cap
          + (child_energy / energy_cap + birth_bonus) per offspring   (refunds the energy given to the child)
          - death_penalty                                             on death (starved, old age, eaten)

    Slots that held no living agent at the start of the step get 0.
    """
    sc = world.species_cfg[sp]
    st = world.species[sp]
    died = info["died"][sp]
    r = (st.energy - energy_before) / sc.energy_cap
    wi, _child, par = info["birth_pairs"][sp]
    if wi.numel():
        r[wi, par] += sc.child_energy / sc.energy_cap + birth_bonus
    r = torch.where(died, torch.full_like(r, -death_penalty), r)
    return torch.where(alive_before, r, torch.zeros_like(r))
