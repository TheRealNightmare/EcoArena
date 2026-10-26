"""EcoArena world: a batched, non-episodic predator–prey ecosystem in PyTorch.

Layout
------
Every agent lives in a fixed slot. Each species has `capacity` slots per world, and tensors are
shaped [W, N] (W parallel worlds, N slots). A slot is either alive or free; births reuse free
slots. Nothing is a Python object per agent, so W worlds step together on the GPU.

Step order (the TypeScript port must follow it exactly)
------------------------------------------------------
1. apply actions (turn, then thrust -> speed)
2. move + wrap
3. prey eat grass
4. captures (each prey caught at most once; each predator eats at most one prey per step;
   prey standing in a refuge cell cannot be caught; digesting predators cannot catch)
5. energy drain + ageing + cooldowns
6. deaths (energy <= 0, too old, or eaten)
7. births into free slots
8. grass regrowth
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import rng
from .config import SpeciesConfig, WorldConfig

PREY, PREDATOR = 0, 1


@dataclass
class SpeciesState:
    pos: torch.Tensor  # [W, N, 2] float32
    heading: torch.Tensor  # [W, N] radians in [-pi, pi)
    speed: torch.Tensor  # [W, N]
    energy: torch.Tensor  # [W, N]
    age: torch.Tensor  # [W, N] int64
    cooldown: torch.Tensor  # [W, N] int64, steps until the agent may breed again
    digest: torch.Tensor  # [W, N] int64, steps until a predator may catch again
    alive: torch.Tensor  # [W, N] bool

    def clone(self) -> SpeciesState:
        return SpeciesState(**{k: v.clone() for k, v in self.__dict__.items()})


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


class EcoWorld:
    def __init__(self, cfg: WorldConfig, n_worlds: int = 1, seed: int = 0, device: str = "cpu"):
        self.cfg = cfg
        self.W = n_worlds
        self.seed = int(seed)
        self.device = torch.device(device)
        self.t = 0
        self.species_cfg = (cfg.prey, cfg.predator)
        # ray angle offsets relative to heading, per species: [R]
        self.ray_offsets = tuple(
            self._ray_offsets(sc.vision_cone_deg) for sc in self.species_cfg
        )
        self.reset()

    # ------------------------------------------------------------------ setup
    def _ray_offsets(self, cone_deg: float) -> torch.Tensor:
        """Centre angle of each vision sector, relative to heading."""
        R = self.cfg.n_rays
        width = math.radians(cone_deg) / R
        return (torch.arange(R, device=self.device, dtype=torch.float32) + 0.5) * width - math.radians(cone_deg) / 2

    def _stream(self, base: int, species: int) -> int:
        return base + species * rng.SPECIES_STREAM_OFFSET

    def _init_species(self, sp: int) -> SpeciesState:
        sc: SpeciesConfig = self.species_cfg[sp]
        W, N, L = self.W, sc.capacity, self.cfg.world_size
        w, s = rng.grid(W, N, self.device)
        u = lambda stream: rng.uniform(self.seed, 0, self._stream(stream, sp), w, s)
        pos = torch.stack([u(rng.S_INIT_X) * L, u(rng.S_INIT_Y) * L], dim=-1)
        heading = u(rng.S_INIT_HEADING) * (2 * math.pi) - math.pi
        # random starting ages so the founding generation does not all die of old age at once
        age = (u(rng.S_INIT_AGE) * (sc.max_age * 0.5)).to(torch.int64)
        alive = (s < sc.init_count).expand(W, N).clone()
        # random starting energy in [0.5, 1.5] x energy_init so founders don't all breed in lockstep
        energy0 = torch.clamp(sc.energy_init * (0.5 + u(rng.S_INIT_ENERGY)), max=sc.energy_cap)
        z = torch.zeros(W, N, device=self.device)
        return SpeciesState(
            pos=pos,
            heading=heading,
            speed=z.clone(),
            energy=torch.where(alive, energy0, z),
            age=torch.where(alive, age, torch.zeros_like(age)),
            cooldown=torch.zeros(W, N, dtype=torch.int64, device=self.device),
            digest=torch.zeros(W, N, dtype=torch.int64, device=self.device),
            alive=alive,
        )

    def _make_refuges(self) -> torch.Tensor:
        """Bool [1, G*G]: cells inside one of `refuge_count` circular patches.

        The layout depends on the seed only, so every parallel world (replicate) shares one map.
        """
        cfg, G, L = self.cfg, self.cfg.grass_grid, self.cfg.world_size
        if cfg.refuge_count == 0:
            return torch.zeros(1, G * G, dtype=torch.bool, device=self.device)
        zero = torch.zeros(1, 1, dtype=torch.int64, device=self.device)
        k = torch.arange(cfg.refuge_count, dtype=torch.int64, device=self.device).unsqueeze(0)
        centres = torch.stack([
            rng.uniform(self.seed, 0, rng.S_REFUGE_X, zero, k) * L,
            rng.uniform(self.seed, 0, rng.S_REFUGE_Y, zero, k) * L,
        ], -1)[0]  # [K, 2]
        ij = torch.arange(G, device=self.device, dtype=torch.float32)
        cx, cy = torch.meshgrid((ij + 0.5) * cfg.cell_size, (ij + 0.5) * cfg.cell_size, indexing="ij")
        cells = torch.stack([cx.reshape(-1), cy.reshape(-1)], -1)  # [G*G, 2] in cell_index order
        d = self.min_image(cells.unsqueeze(1) - centres.unsqueeze(0)).norm(dim=-1)  # [G*G, K]
        return (d <= cfg.refuge_radius).any(dim=1).unsqueeze(0)

    def _cap_refuge_grass(self) -> None:
        cap = torch.full_like(self.grass, self.cfg.refuge_grass_cap)
        self.grass = torch.where(self.refuge, torch.minimum(self.grass, cap), self.grass)

    def reset(self) -> None:
        self.t = 0
        self.species = [self._init_species(PREY), self._init_species(PREDATOR)]
        G = self.cfg.grass_grid
        w, c = rng.grid(self.W, G * G, self.device)
        g = rng.uniform(self.seed, 0, rng.S_GRASS_INIT, w, c)
        self.grass = self.cfg.grass_init_min + g * (self.cfg.grass_init_max - self.cfg.grass_init_min)
        self.refuge = self._make_refuges()  # [1, G*G], broadcasts over worlds
        self._cap_refuge_grass()

    # ------------------------------------------------------------------ helpers
    def min_image(self, d: torch.Tensor) -> torch.Tensor:
        L = self.cfg.world_size
        return d - L * torch.round(d / L)

    def cell_index(self, pos: torch.Tensor) -> torch.Tensor:
        """Flat grass-cell index for positions [..., 2] -> [...] int64."""
        G, cs = self.cfg.grass_grid, self.cfg.cell_size
        ij = torch.clamp((pos / cs).to(torch.int64), 0, G - 1)
        return ij[..., 0] * G + ij[..., 1]

    def in_refuge(self, pos: torch.Tensor) -> torch.Tensor:
        """Bool [W, N] for positions [W, N, 2]."""
        return torch.gather(self.refuge.expand(self.W, -1), 1, self.cell_index(pos))

    def populations(self) -> torch.Tensor:
        """[W, 2] alive counts (prey, predator)."""
        return torch.stack([s.alive.sum(1) for s in self.species], dim=1)

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def step(self, prey_action: torch.Tensor, pred_action: torch.Tensor) -> dict:
        """Advance every world by one step. Actions are [W, N, 2] in [-1, 1]: (turn, thrust).

        Returns an info dict with per-world event counts and birth (world, child, parent) indices,
        which neuroevolution uses to copy a parent's brain into its child.
        """
        cfg, L = self.cfg, self.cfg.world_size
        actions = (prey_action, pred_action)
        thrusts = []

        # 1–2. actions and movement
        for sp, (st, sc, a) in enumerate(zip(self.species, self.species_cfg, actions)):
            a = torch.clamp(a.to(torch.float32), -1.0, 1.0)
            turn, thrust = a[..., 0], a[..., 1]
            alive_f = st.alive.float()
            st.heading = wrap_angle(st.heading + turn * sc.max_turn * alive_f)
            st.speed = torch.clamp(st.speed + thrust * sc.max_accel, 0.0, sc.max_speed) * alive_f
            vel = torch.stack([torch.cos(st.heading), torch.sin(st.heading)], -1) * st.speed.unsqueeze(-1)
            st.pos = torch.remainder(st.pos + vel, L)
            thrusts.append(thrust * alive_f)

        prey = self.species[PREY]
        pc = cfg.prey

        # 3. prey eat grass (cell shared equally by all prey standing on it)
        ci = self.cell_index(prey.pos)
        counts = torch.zeros_like(self.grass).scatter_add_(1, ci, prey.alive.float())
        consumed = torch.minimum(self.grass, counts * cfg.grass_eat_per_step)
        per_prey = torch.where(
            prey.alive, consumed.gather(1, ci) / counts.gather(1, ci).clamp(min=1.0), 0.0
        )
        prey.energy = torch.clamp(prey.energy + per_prey * cfg.grass_energy_per_unit, max=pc.energy_cap)
        self.grass = self.grass - consumed

        # 4. captures
        eaten, n_capt = self._captures()
        # 5. drain, ageing
        for st, sc, th in zip(self.species, self.species_cfg, thrusts):
            drain = sc.base_drain + sc.move_cost * th * th
            st.energy = torch.where(st.alive, st.energy - drain, st.energy)
            st.age = st.age + st.alive.long()
            st.cooldown = torch.clamp(st.cooldown - 1, min=0)
            st.digest = torch.clamp(st.digest - 1, min=0)

        # 6. deaths
        deaths = []
        for sp, (st, sc) in enumerate(zip(self.species, self.species_cfg)):
            dead = st.alive & ((st.energy <= 0) | (st.age >= sc.max_age))
            if sp == PREY:
                dead = dead | eaten
            st.alive = st.alive & ~dead
            st.speed = torch.where(st.alive, st.speed, 0.0)
            deaths.append(dead.sum(1))

        # 7. births
        births = [self._births(sp) for sp in (PREY, PREDATOR)]

        # 8. grass regrowth (logistic + small floor so bare cells recover)
        g = self.grass
        self.grass = torch.clamp(g + cfg.grass_regrow_rate * g * (1 - g) + cfg.grass_regrow_floor, 0.0, 1.0)
        self._cap_refuge_grass()

        self.t += 1
        return {
            "t": self.t,
            "captures": n_capt,
            "deaths": deaths,
            "births": [b["count"] for b in births],
            "birth_pairs": [b["pairs"] for b in births],
            "populations": self.populations(),
        }

    def _captures(self) -> tuple[torch.Tensor, torch.Tensor]:
        prey, pred = self.species
        qc = self.cfg.predator
        rel = self.min_image(prey.pos.unsqueeze(1) - pred.pos.unsqueeze(2))  # [W, Nq, Np, 2]
        d2 = (rel * rel).sum(-1)
        hunting = pred.alive & (pred.digest <= 0)
        exposed = prey.alive & ~self.in_refuge(prey.pos)
        ok = hunting.unsqueeze(2) & exposed.unsqueeze(1) & (d2 <= qc.catch_radius**2)
        inf = torch.tensor(float("inf"), device=self.device)
        d2 = torch.where(ok, d2, inf)
        # each prey picks its nearest predator
        dmin, qarg = d2.min(dim=1)  # [W, Np]
        # each predator keeps only its nearest candidate prey
        M = torch.full_like(d2, float("inf"))
        M.scatter_(1, qarg.unsqueeze(1), dmin.unsqueeze(1))
        dsel, parg = M.min(dim=2)  # [W, Nq]
        pred_eats = torch.isfinite(dsel)
        # index assignment (not scatter_) so non-eating predators can't overwrite a True
        eaten = torch.zeros_like(prey.alive)
        w_idx = torch.arange(self.W, device=self.device).unsqueeze(1).expand_as(parg)
        eaten[w_idx[pred_eats], parg[pred_eats]] = True
        pred.energy = torch.where(
            pred_eats, torch.clamp(pred.energy + qc.energy_per_catch, max=qc.energy_cap), pred.energy
        )
        # +1 because step 5 decrements every agent in this same step
        pred.digest = torch.where(pred_eats, torch.full_like(pred.digest, qc.digest_steps + 1), pred.digest)
        return eaten, pred_eats.sum(1)

    def _births(self, sp: int) -> dict:
        st, sc = self.species[sp], self.species_cfg[sp]
        W, N = st.alive.shape
        idx = torch.arange(N, device=self.device).unsqueeze(0).expand(W, N)
        parent = st.alive & (st.energy >= sc.repro_threshold) & (st.cooldown <= 0)
        free = ~st.alive
        # stable "k-th parent -> k-th free slot" matching, in slot order
        par_order = torch.argsort(torch.where(parent, idx, idx + N), dim=1)
        free_order = torch.argsort(torch.where(free, idx, idx + N), dim=1)
        k = torch.minimum(parent.sum(1), free.sum(1)).unsqueeze(1)
        valid = idx < k
        wi = torch.arange(W, device=self.device).unsqueeze(1).expand(W, N)[valid]
        child = free_order[valid]
        par = par_order[valid]
        if wi.numel():
            u = lambda stream: rng.uniform(self.seed, self.t, self._stream(stream, sp), wi, child)
            ang = u(rng.S_BIRTH_ANGLE) * (2 * math.pi)
            off = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (2.0 * sc.radius)
            st.pos[wi, child] = torch.remainder(st.pos[wi, par] + off, self.cfg.world_size)
            st.heading[wi, child] = u(rng.S_BIRTH_HEADING) * (2 * math.pi) - math.pi
            st.speed[wi, child] = 0.0
            st.energy[wi, child] = sc.child_energy
            st.energy[wi, par] = st.energy[wi, par] - sc.child_energy
            st.age[wi, child] = 0
            st.cooldown[wi, child] = sc.repro_cooldown
            st.cooldown[wi, par] = sc.repro_cooldown
            st.digest[wi, child] = 0
            st.alive[wi, child] = True
        return {"count": k.squeeze(1), "pairs": (wi, child, par)}

    # ------------------------------------------------------------------ observations
    @torch.no_grad()
    def observe(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Ray-cast observations for (prey, predator), each [W, N, obs_dim]; zeros for free slots."""
        prey, pred = self.species
        all_pos = torch.cat([prey.pos, pred.pos], 1)
        all_alive = torch.cat([prey.alive, pred.alive], 1)
        Np, Nq = prey.alive.shape[1], pred.alive.shape[1]
        radius = torch.cat([
            torch.full((Np,), self.cfg.prey.radius, device=self.device),
            torch.full((Nq,), self.cfg.predator.radius, device=self.device),
        ])
        kind = torch.cat([
            torch.full((Np,), PREY, device=self.device, dtype=torch.int64),
            torch.full((Nq,), PREDATOR, device=self.device, dtype=torch.int64),
        ])
        return (
            self._observe_species(PREY, all_pos, all_alive, radius, kind, self_offset=0),
            self._observe_species(PREDATOR, all_pos, all_alive, radius, kind, self_offset=Np),
        )

    def _observe_species(self, sp, all_pos, all_alive, radius, kind, self_offset) -> torch.Tensor:
        cfg = self.cfg
        st, sc = self.species[sp], self.species_cfg[sp]
        W, No = st.alive.shape
        Nall = all_pos.shape[1]
        R, K, rng_ = cfg.n_rays, min(cfg.ray_neighbors, Nall - 1), sc.vision_range
        inf = float("inf")

        # candidate neighbours: K nearest alive agents within reach
        rel = self.min_image(all_pos.unsqueeze(1) - st.pos.unsqueeze(2))  # [W, No, Nall, 2]
        d2 = (rel * rel).sum(-1)
        reach = (rng_ + float(radius.max())) ** 2
        ok = all_alive.unsqueeze(1) & (d2 <= reach)
        self_idx = torch.arange(No, device=self.device) + self_offset
        ok[:, torch.arange(No, device=self.device), self_idx] = False
        d2 = torch.where(ok, d2, inf)
        d2k, nbr = torch.topk(d2, K, dim=2, largest=False)  # [W, No, K]
        relk = torch.gather(rel, 2, nbr.unsqueeze(-1).expand(W, No, K, 2))
        valid = torch.isfinite(d2k)
        rk = radius[nbr]
        kk = kind[nbr]

        # Vision sectors: the cone is split into R equal slices. Each neighbour lands in the slice
        # its bearing falls in, at distance (centre distance - body radius). Unlike thin rays,
        # nothing can hide between two rays.
        half = math.radians(sc.vision_cone_deg) / 2
        width = 2 * half / R
        bearing = wrap_angle(torch.atan2(relk[..., 1], relk[..., 0]) - st.heading.unsqueeze(-1))
        sector = torch.clamp(((bearing + half) / width).floor().to(torch.int64), 0, R - 1)
        surf = torch.clamp(torch.sqrt(torch.where(valid, d2k, 0.0)) - rk, min=0.0)
        seen = valid & (bearing.abs() <= half) & (surf <= rng_)
        per_kind = []
        for k in (PREY, PREDATOR):
            dk = torch.where(seen & (kk == k), surf, inf)
            per_kind.append(
                torch.full((W, No, R), inf, device=self.device).scatter_reduce(2, sector, dk, "amin")
            )
        agent_d = torch.minimum(per_kind[0], per_kind[1])
        agent_kind = torch.where(per_kind[PREDATOR] < per_kind[PREY], PREDATOR, PREY)

        # grass and refuges: sampled along each sector's centre line, first visible cell wins
        ang = st.heading.unsqueeze(-1) + self.ray_offsets[sp]  # [W, No, R]
        u = torch.stack([torch.cos(ang), torch.sin(ang)], -1)  # [W, No, R, 2]
        S = cfg.ray_grass_samples
        steps = (torch.arange(S, device=self.device, dtype=torch.float32) + 0.5) * (rng_ / S)
        pts = st.pos.unsqueeze(2).unsqueeze(3) + u.unsqueeze(3) * steps.view(1, 1, 1, S, 1)
        pts = torch.remainder(pts, cfg.world_size)
        cells = self.cell_index(pts).reshape(W, -1)
        g = torch.gather(self.grass, 1, cells).reshape(W, No, R, S)
        ref = torch.gather(self.refuge.expand(W, -1), 1, cells).reshape(W, No, R, S)
        stepv = steps.view(1, 1, 1, S)
        grass_d = torch.where((g > cfg.grass_visible_threshold) & ~ref, stepv, inf).min(dim=3).values
        refuge_d = torch.where(ref, stepv, inf).min(dim=3).values  # [W, No, R]

        # nearest thing per sector; ties go agent > refuge > grass
        is_agent = torch.isfinite(agent_d) & (agent_d <= refuge_d) & (agent_d <= grass_d)
        is_refuge = torch.isfinite(refuge_d) & ~is_agent & (refuge_d <= grass_d)
        is_grass = torch.isfinite(grass_d) & ~is_agent & ~is_refuge
        none = ~(is_agent | is_refuge | is_grass)
        dist = torch.minimum(torch.minimum(agent_d, refuge_d), grass_d)
        onehot = torch.stack([
            is_grass,
            is_agent & (agent_kind == PREY),
            is_agent & (agent_kind == PREDATOR),
            is_refuge,
            none,
        ], -1).float()
        dnorm = torch.where(none, 1.0, dist / rng_).unsqueeze(-1)
        rays = torch.cat([dnorm, onehot], -1).reshape(W, No, R * 6)

        own = torch.stack([
            st.energy / sc.energy_cap,
            st.speed / sc.max_speed,
            st.age.float() / sc.max_age,
            torch.gather(self.grass, 1, self.cell_index(st.pos)),
            self.in_refuge(st.pos).float(),
        ], -1)
        obs = torch.cat([rays, own], -1)
        return obs * st.alive.unsqueeze(-1).float()
