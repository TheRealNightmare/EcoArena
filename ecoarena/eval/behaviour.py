"""What the animals actually do: tactic measurements for the paper's behaviour analysis.

Runs fixed policies on fresh worlds (like `benchmark.evaluate_worlds`) and every `every` steps measures:

  prey_in_refuge        share of living prey standing in a refuge (safe from capture)
  prey_refuge_dist      mean distance of living prey to the nearest refuge cell (0 inside)
  pred_refuge_dist      the same for predators
  ambush_index          (refuge distance of a random point) / (predator refuge distance): > 1 = predators stay
                        closer to refuges than chance, i.e. wait where prey must come out to eat
  catch_refuge_dist     distance from the nearest refuge of each prey when it was caught
  edge_catch_share      share of catches within 3 units of a refuge
  prey_speed, pred_speed  mean speed as a share of the species' top speed
  pursuit_share         predators with prey in vision range that head toward the nearest prey (cos > 0.8)
  flee_alignment        prey with a predator within 12 units: mean cos between heading and "away from it"
  prey_clustering       (nearest-neighbour distance expected for uniform prey) / (observed): > 1 = herding
  pred_sit_and_wait     predators with no prey within 12 units that move at under 10% of top speed
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..env import PREDATOR, PREY, EcoWorld
from ..policies import make_policy


def refuge_distance_map(w: EcoWorld) -> torch.Tensor:
    """[G*G] distance from each grass cell's centre to the nearest refuge cell centre (0 for refuge cells)."""
    cfg = w.cfg
    G = cfg.grass_grid
    ij = torch.arange(G, device=w.device, dtype=torch.float32)
    cx, cy = torch.meshgrid((ij + 0.5) * cfg.cell_size, (ij + 0.5) * cfg.cell_size, indexing="ij")
    cells = torch.stack([cx.reshape(-1), cy.reshape(-1)], -1)
    ref = cells[w.refuge[0]]
    if ref.shape[0] == 0:
        return torch.full((G * G,), float("inf"), device=w.device)
    out = torch.empty(G * G, device=w.device)
    for i in range(0, G * G, 2048):  # chunked to bound memory
        d = w.min_image(cells[i : i + 2048].unsqueeze(1) - ref.unsqueeze(0)).norm(dim=-1)
        out[i : i + 2048] = d.min(1).values
    return out


def _nearest(w: EcoWorld, src_pos, src_alive, dst_pos, dst_alive, exclude_self: bool = False):
    """Vector to and distance of the nearest living dst for every src. [W, Ns, 2], [W, Ns]."""
    rel = w.min_image(dst_pos.unsqueeze(1) - src_pos.unsqueeze(2))  # [W, Ns, Nd, 2]
    d2 = (rel * rel).sum(-1)
    mask = src_alive.unsqueeze(2) & dst_alive.unsqueeze(1)
    if exclude_self:
        mask = mask & ~torch.eye(src_pos.shape[1], dtype=torch.bool, device=w.device).unsqueeze(0)
    d2 = torch.where(mask, d2, torch.full_like(d2, float("inf")))
    dmin, arg = d2.min(2)
    vec = torch.gather(rel, 2, arg[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    return vec, dmin.sqrt()


@torch.no_grad()
def measure_tactics(cfg, policies: dict, worlds: int, steps: int, seed: int, device: str, every: int = 10) -> dict:
    w = EcoWorld(cfg, n_worlds=worlds, seed=seed, device=device)
    pol = {sp: (p if callable(p) else make_policy(p, w, sp)) for sp, p in policies.items()}
    dmap = refuge_distance_map(w)
    chance_dist = float(dmap.mean())
    acc: dict[str, list[float]] = {k: [] for k in (
        "prey_in_refuge", "prey_refuge_dist", "pred_refuge_dist", "prey_speed", "pred_speed", "pursuit_share",
        "flee_alignment", "prey_clustering", "pred_sit_and_wait")}
    catch_d: list[float] = []
    qc, pc = cfg.predator, cfg.prey
    area = cfg.world_size**2
    for t in range(steps):
        prey, pred = w.species[PREY], w.species[PREDATOR]
        prey_pos0 = prey.pos.clone()
        o = w.observe()
        info = w.step(pol[PREY](o[PREY]), pol[PREDATOR](o[PREDATOR]))
        if info["eaten"].any():  # where catches happen, from the prey's position before the step
            cells = w.cell_index(prey_pos0[info["eaten"]])
            catch_d += dmap[cells].tolist()
        if t % every:
            continue
        prey, pred = w.species[PREY], w.species[PREDATOR]
        pa, qa = prey.alive, pred.alive
        if not bool(pa.any()) or not bool(qa.any()):
            continue
        pdist = dmap[w.cell_index(prey.pos)]
        qdist = dmap[w.cell_index(pred.pos)]
        acc["prey_in_refuge"].append(float(w.in_refuge(prey.pos)[pa].float().mean()))
        acc["prey_refuge_dist"].append(float(pdist[pa].mean()))
        acc["pred_refuge_dist"].append(float(qdist[qa].mean()))
        acc["prey_speed"].append(float(prey.speed[pa].mean()) / pc.max_speed)
        acc["pred_speed"].append(float(pred.speed[qa].mean()) / qc.max_speed)
        # predators: pursuit and sit-and-wait
        vec, d = _nearest(w, pred.pos, qa, prey.pos, pa)
        head = torch.stack([torch.cos(pred.heading), torch.sin(pred.heading)], -1)
        cos = (head * vec).sum(-1) / d.clamp(min=1e-6)
        sees = qa & (d <= qc.vision_range)
        if bool(sees.any()):
            acc["pursuit_share"].append(float((cos[sees] > 0.8).float().mean()))
        idle_ctx = qa & (d > 12.0)
        if bool(idle_ctx.any()):
            acc["pred_sit_and_wait"].append(float((pred.speed[idle_ctx] < 0.1 * qc.max_speed).float().mean()))
        # prey: fleeing and grouping
        vec, d = _nearest(w, prey.pos, pa, pred.pos, qa)
        head = torch.stack([torch.cos(prey.heading), torch.sin(prey.heading)], -1)
        threatened = pa & (d <= 12.0)
        if bool(threatened.any()):
            away = -(head * vec).sum(-1) / d.clamp(min=1e-6)
            acc["flee_alignment"].append(float(away[threatened].mean()))
        _, nn_d = _nearest(w, prey.pos, pa, prey.pos, pa, exclude_self=True)
        n = pa.sum(1, keepdim=True).float().expand_as(nn_d)
        expected = 0.5 / torch.sqrt(n / area)  # mean nearest-neighbour distance of a uniform (Poisson) pattern
        ok = pa & torch.isfinite(nn_d)
        if bool(ok.any()):
            acc["prey_clustering"].append(float((expected[ok] / nn_d[ok].clamp(min=1e-3)).median()))
    out = {k: (float(np.mean(v)) if v else math.nan) for k, v in acc.items()}
    out["ambush_index"] = chance_dist / out["pred_refuge_dist"] if out["pred_refuge_dist"] else math.nan
    out["prey_refuge_index"] = chance_dist / out["prey_refuge_dist"] if out["prey_refuge_dist"] else math.nan
    out["catches"] = len(catch_d)
    out["catch_refuge_dist"] = float(np.mean(catch_d)) if catch_d else math.nan
    out["edge_catch_share"] = float(np.mean(np.array(catch_d) <= 3.0)) if catch_d else math.nan
    out["chance_refuge_dist"] = chance_dist
    return out
