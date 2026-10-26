"""B0 (random) and B1 (hand-coded heuristic) policies.

Both act only on the same observation vector a learned policy sees, so the comparison is fair.
Observation layout per agent: n_rays × [dist, grass, prey, predator, refuge, none], then
[energy, speed, age, grass_under_self, in_refuge].
"""

from __future__ import annotations

import math

import torch

from ..env import rng
from ..env.world import PREDATOR, PREY, EcoWorld

D, GRASS, PREY_HIT, PRED_HIT, REFUGE, NONE = 0, 1, 2, 3, 4, 5
PER_SECTOR = 6
OWN_GRASS, OWN_REFUGE = -2, -1


def _rays(obs: torch.Tensor, n_rays: int) -> torch.Tensor:
    return obs[..., : n_rays * PER_SECTOR].reshape(*obs.shape[:-1], n_rays, PER_SECTOR)


def _nearest_ray(rays: torch.Tensor, channel: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Index of the closest ray that hit `channel`, and whether any did."""
    d = torch.where(rays[..., channel] > 0.5, rays[..., D], torch.full_like(rays[..., D], 9.0))
    dmin, j = d.min(dim=-1)
    return j, dmin < 9.0


class RandomPolicy:
    """B0: uniform random actions."""

    def __init__(self, world: EcoWorld, species: int):
        self.world, self.sp = world, species

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        W, N, _ = obs.shape
        w, s = rng.grid(W, N, obs.device)
        st = rng.S_POLICY + self.sp * rng.SPECIES_STREAM_OFFSET
        a = torch.stack([
            rng.uniform(self.world.seed, self.world.t, st, w, s),
            rng.uniform(self.world.seed, self.world.t, st + 1, w, s),
        ], -1)
        return a * 2 - 1


class HeuristicPolicy:
    """B1: prey that see a predator hide in a visible refuge (or hold still inside one), else flee;
    otherwise they graze or steer to grass. Predators chase the nearest visible prey, else wander."""

    def __init__(self, world: EcoWorld, species: int):
        self.world, self.sp = world, species
        self.offsets = world.ray_offsets[species]
        self.max_turn = world.species_cfg[species].max_turn
        self.R = world.cfg.n_rays

    def _turn_toward(self, angle: torch.Tensor) -> torch.Tensor:
        return torch.clamp(angle / self.max_turn, -1.0, 1.0)

    def _wander(self, shape, device) -> torch.Tensor:
        W, N = shape
        w, s = rng.grid(W, N, device)
        st = rng.S_POLICY + 2 + self.sp * rng.SPECIES_STREAM_OFFSET
        return (rng.uniform(self.world.seed, self.world.t, st, w, s) * 2 - 1) * 0.3

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        rays = _rays(obs, self.R)
        W, N = obs.shape[:2]
        turn = self._wander((W, N), obs.device)
        thrust = torch.full((W, N), 0.5, device=obs.device)

        if self.sp == PREY:
            jg, see_grass = _nearest_ray(rays, GRASS)
            turn = torch.where(see_grass, self._turn_toward(self.offsets[jg]), turn)
            grazing = obs[..., OWN_GRASS] > 0.25
            thrust = torch.where(grazing, torch.full_like(thrust, -1.0), thrust)
            turn = torch.where(grazing, torch.zeros_like(turn), turn)
            jp, see_pred = _nearest_ray(rays, PRED_HIT)
            away = torch.remainder(self.offsets[jp] + 2 * math.pi, 2 * math.pi) - math.pi
            jr, see_refuge = _nearest_ray(rays, REFUGE)
            inside = obs[..., OWN_REFUGE] > 0.5
            flee_turn = torch.where(see_refuge, self._turn_toward(self.offsets[jr]), self._turn_toward(away))
            turn = torch.where(see_pred, torch.where(inside, torch.zeros_like(turn), flee_turn), turn)
            thrust = torch.where(see_pred, torch.where(inside, -torch.ones_like(thrust), torch.ones_like(thrust)), thrust)
        else:
            jp, see_prey = _nearest_ray(rays, PREY_HIT)
            turn = torch.where(see_prey, self._turn_toward(self.offsets[jp]), turn)
            thrust = torch.where(see_prey, torch.ones_like(thrust), torch.full_like(thrust, 0.6))
        return torch.stack([turn, thrust], -1)


def make_policy(name: str, world: EcoWorld, species: int):
    return {"random": RandomPolicy, "heuristic": HeuristicPolicy}[name](world, species)


__all__ = ["PREDATOR", "PREY", "HeuristicPolicy", "RandomPolicy", "make_policy"]
