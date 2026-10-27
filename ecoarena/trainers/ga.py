"""Genetic algorithm (M2): deep GA over the weights of the shared 64-64 actor (Such et al., 2017).

How genomes are evaluated in a shared, never-resetting world
------------------------------------------------------------
Each species has a population of P genomes. For one generation (`gen_steps` world-steps) every animal slot is
assigned a genome: slot s gets genome perm[s % P], with a fresh random permutation each generation. So all P
genomes live in the same worlds at the same time and face the same ecosystem state, which keeps their
comparison fair, and P networks are evaluated in one batched matrix multiply (no per-genome episodes).

A genome's fitness is the mean reward per agent-step of the animals carrying it, using the same reward rule
as PPO (`common.step_rewards`). A newborn in a slot carries that slot's genome for the rest of the generation.

Selection: the top `elite_frac` genomes (with at least `min_samples` agent-steps) survive unchanged; the rest
of the next population are copies of random elites plus Gaussian noise (sigma). Elites are re-evaluated every
generation, and the champion is the elite with the best mean fitness over its evaluations.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from ..env import PREDATOR, PREY, EcoWorld, WorldConfig
from ..policies import make_policy
from .common import step_rewards
from .ppo import ActorCritic

SPECIES_NAME = {PREY: "prey", PREDATOR: "predator"}
HIDDEN = 64


@dataclass
class GAConfig:
    learn: tuple[int, ...] = (PREY,)
    opponent: str = "heuristic"
    n_worlds: int = 4
    population: int = 50  # must divide every learning species' slot capacity
    elite_frac: float = 0.1
    sigma: float = 0.02
    gen_steps: int = 128
    generations: int = 100_000  # upper bound; the world-step budget normally stops training
    world_step_budget: int = 150_000
    min_samples: int = 50  # agent-steps a genome needs before it can be an elite
    init_out_std: float = 0.5  # final-layer init scale (PPO uses 0.01; GA needs varied starting behaviour)
    birth_bonus: float = 1.0
    death_penalty: float = 1.0
    seed: int = 0
    device: str = "cpu"
    log_every: int = 5


# ---------------------------------------------------------------------------------------- genome <-> network
def param_shapes(obs_dim: int) -> list[tuple[str, tuple[int, ...]]]:
    return [
        ("w1", (obs_dim, HIDDEN)), ("b1", (HIDDEN,)),
        ("w2", (HIDDEN, HIDDEN)), ("b2", (HIDDEN,)),
        ("w3", (HIDDEN, 2)), ("b3", (2,)),
    ]


def n_params(obs_dim: int) -> int:
    return sum(math.prod(s) for _, s in param_shapes(obs_dim))


def unflatten(flat: torch.Tensor, obs_dim: int) -> dict[str, torch.Tensor]:
    """[P, n_params] -> dict of [P, *shape]."""
    out, i = {}, 0
    for name, shape in param_shapes(obs_dim):
        k = math.prod(shape)
        out[name] = flat[:, i : i + k].reshape(flat.shape[0], *shape)
        i += k
    return out


def init_population(P: int, obs_dim: int, out_std: float, gen: torch.Generator, device) -> torch.Tensor:
    """Orthogonal init like PPO's actor (gain sqrt(2) hidden), with a larger final layer for diversity."""
    rows = []
    for _ in range(P):
        parts = []
        for name, shape in param_shapes(obs_dim):
            if name.startswith("b"):
                parts.append(torch.zeros(math.prod(shape)))
            else:
                w = torch.empty(shape[1], shape[0])  # nn.Linear layout [out, in]
                gain = out_std if name == "w3" else math.sqrt(2)
                nn.init.orthogonal_(w, gain, generator=gen)
                parts.append(w.t().reshape(-1))
        rows.append(torch.cat(parts))
    return torch.stack(rows).to(device)


def to_actor_critic(flat_row: torch.Tensor, obs_dim: int) -> ActorCritic:
    """Pack one genome into the shared ActorCritic format so evaluation and export treat all methods alike."""
    p = unflatten(flat_row.unsqueeze(0), obs_dim)
    m = ActorCritic(obs_dim)
    with torch.no_grad():
        m.actor[0].weight.copy_(p["w1"][0].t())
        m.actor[0].bias.copy_(p["b1"][0])
        m.actor[2].weight.copy_(p["w2"][0].t())
        m.actor[2].bias.copy_(p["b2"][0])
        m.actor[4].weight.copy_(p["w3"][0].t())
        m.actor[4].bias.copy_(p["b3"][0])
    return m


def batched_actions(flat: torch.Tensor, obs: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Actions for every slot. obs [W, N, D]; slot s uses genome perm[s % P]. Deterministic (mean action)."""
    P = flat.shape[0]
    W, N, D = obs.shape
    p = unflatten(flat[perm], D)  # row j = genome used by slots with s % P == j
    x = obs.reshape(W, N // P, P, D).permute(2, 0, 1, 3).reshape(P, W * (N // P), D)
    h = torch.tanh(torch.bmm(x, p["w1"]) + p["b1"].unsqueeze(1))
    h = torch.tanh(torch.bmm(h, p["w2"]) + p["b2"].unsqueeze(1))
    a = torch.bmm(h, p["w3"]) + p["b3"].unsqueeze(1)  # [P, W*N/P, 2]
    a = a.reshape(P, W, N // P, 2).permute(1, 2, 0, 3).reshape(W, N, 2)
    return torch.clamp(a, -1.0, 1.0)


# ---------------------------------------------------------------------------------------- trainer
class GATrainer:
    def __init__(self, world_cfg: WorldConfig, cfg: GAConfig, out_dir: str | Path):
        self.wcfg, self.cfg = world_cfg, cfg
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device)
        for sp in cfg.learn:
            cap = (world_cfg.prey, world_cfg.predator)[sp].capacity
            if cap % cfg.population:
                raise ValueError(f"population {cfg.population} must divide {SPECIES_NAME[sp]} capacity {cap}")
        self.gen = torch.Generator().manual_seed(cfg.seed)
        self.world_seed = cfg.seed * 1000
        self.world = EcoWorld(world_cfg, n_worlds=cfg.n_worlds, seed=self.world_seed, device=cfg.device)
        D = world_cfg.obs_dim
        self.pop = {sp: init_population(cfg.population, D, cfg.init_out_std, self.gen, self.device)
                    for sp in cfg.learn}
        self.ids = {sp: list(range(cfg.population)) for sp in cfg.learn}  # lineage ids for fitness history
        self.next_id = cfg.population
        self.fit_hist: dict[int, dict[int, list[float]]] = {sp: {} for sp in cfg.learn}
        self.fixed = {sp: make_policy(cfg.opponent, self.world, sp) for sp in (PREY, PREDATOR) if sp not in cfg.learn}
        self.resets = 0
        self.history: list[dict] = []

    def _reset_world(self) -> None:
        self.resets += 1
        self.world_seed += 1
        self.world = EcoWorld(self.wcfg, n_worlds=self.cfg.n_worlds, seed=self.world_seed, device=self.cfg.device)
        self.fixed = {sp: make_policy(self.cfg.opponent, self.world, sp) for sp in self.fixed}

    # ------------------------------------------------------------------ one generation
    @torch.no_grad()
    def evaluate_generation(self) -> tuple[dict, dict, int]:
        cfg, w = self.cfg, self.world
        P = cfg.population
        perm = {sp: torch.randperm(P, generator=self.gen).to(self.device) for sp in cfg.learn}
        rsum = {sp: torch.zeros(P, device=self.device) for sp in cfg.learn}
        nsteps = {sp: torch.zeros(P, device=self.device) for sp in cfg.learn}
        stats = {"captures": 0, "resets": 0}
        steps_done = 0
        for _ in range(cfg.gen_steps):
            obs = w.observe()
            acts, e0, a0 = [None, None], [None, None], [None, None]
            for sp in (PREY, PREDATOR):
                e0[sp] = w.species[sp].energy.clone()
                a0[sp] = w.species[sp].alive.clone()
                acts[sp] = (batched_actions(self.pop[sp], obs[sp], perm[sp]) if sp in cfg.learn
                            else self.fixed[sp](obs[sp]))
            info = w.step(acts[PREY], acts[PREDATOR])
            steps_done += 1
            stats["captures"] += int(info["captures"].sum())
            for sp in cfg.learn:
                r = step_rewards(w, info, sp, e0[sp], a0[sp], cfg.birth_bonus, cfg.death_penalty)
                Wn, N = r.shape
                # slot s -> column s % P after reshaping [W, N/P, P]; column j holds genome perm[j]
                col_r = r.reshape(Wn, N // P, P).sum((0, 1))
                col_n = a0[sp].reshape(Wn, N // P, P).sum((0, 1)).float()
                rsum[sp].index_add_(0, perm[sp], col_r)
                nsteps[sp].index_add_(0, perm[sp], col_n)
            if bool((w.populations() == 0).any()):
                self._reset_world()
                stats["resets"] += 1
                break
        fitness = {sp: torch.where(nsteps[sp] > 0, rsum[sp] / nsteps[sp].clamp(min=1), torch.full_like(rsum[sp], float("nan")))
                   for sp in cfg.learn}
        return fitness, {**stats, "samples": {SPECIES_NAME[sp]: nsteps[sp].tolist() for sp in cfg.learn}}, steps_done

    # ------------------------------------------------------------------ selection + mutation
    def next_population(self, sp: int, fitness: torch.Tensor, samples: torch.Tensor) -> dict:
        cfg = self.cfg
        P = cfg.population
        n_elite = max(1, round(cfg.elite_frac * P))
        for j, gid in enumerate(self.ids[sp]):
            if samples[j] > 0 and not torch.isnan(fitness[j]):
                self.fit_hist[sp].setdefault(gid, []).append(float(fitness[j]))
        score = torch.where((samples >= cfg.min_samples) & ~torch.isnan(fitness), fitness,
                            torch.full_like(fitness, -float("inf")))
        order = torch.argsort(score.cpu(), descending=True)
        elite_idx = order[:n_elite]
        elites = self.pop[sp][elite_idx.to(self.device)]
        elite_ids = [self.ids[sp][int(i)] for i in elite_idx]
        n_child = P - n_elite
        parents = torch.randint(0, n_elite, (n_child,), generator=self.gen)
        noise = torch.randn(n_child, elites.shape[1], generator=self.gen).to(self.device) * cfg.sigma
        children = elites[parents.to(self.device)] + noise
        self.pop[sp] = torch.cat([elites, children])
        self.ids[sp] = elite_ids + list(range(self.next_id, self.next_id + n_child))
        self.next_id += n_child
        finite = score[torch.isfinite(score)]
        return {
            "best": float(finite.max()) if finite.numel() else float("nan"),
            "mean": float(finite.mean()) if finite.numel() else float("nan"),
            "elite_mean": float(score[elite_idx.to(score.device)].mean()) if finite.numel() else float("nan"),
            "evaluated": int(torch.isfinite(score).sum()),
        }

    def champion(self, sp: int) -> torch.Tensor:
        """Elite with the best mean fitness over all its evaluations (elites persist, so they have several)."""
        n_elite = max(1, round(self.cfg.elite_frac * self.cfg.population))
        best_j, best_v = 0, -float("inf")
        for j in range(n_elite):
            h = self.fit_hist[sp].get(self.ids[sp][j], [])
            v = sum(h) / len(h) if h else -float("inf")
            if v > best_v:
                best_j, best_v = j, v
        return self.pop[sp][best_j]

    # ------------------------------------------------------------------ main loop
    def train(self) -> list[dict]:
        cfg = self.cfg
        t0 = time.time()
        world_steps = 0
        for g in range(1, cfg.generations + 1):
            fitness, stats, steps = self.evaluate_generation()
            world_steps += steps * cfg.n_worlds
            row = {"gen": g, "secs": round(time.time() - t0, 1), "world_steps": world_steps,
                   "resets_total": self.resets, "captures": stats["captures"]}
            pops = self.world.populations().float().mean(0).tolist()
            row["pop_prey"], row["pop_pred"] = round(pops[0], 1), round(pops[1], 1)
            for sp in cfg.learn:
                samples = torch.tensor(stats["samples"][SPECIES_NAME[sp]], device=self.device)
                summ = self.next_population(sp, fitness[sp], samples)
                for k, v in summ.items():
                    row[f"{SPECIES_NAME[sp]}/{k}"] = round(v, 6) if isinstance(v, float) else v
            self.history.append(row)
            if g % cfg.log_every == 0 or g == 1:
                msg = " ".join(f"{SPECIES_NAME[sp]} best={row[f'{SPECIES_NAME[sp]}/best']:+.4f} "
                               f"mean={row[f'{SPECIES_NAME[sp]}/mean']:+.4f}" for sp in cfg.learn)
                print(f"gen {g:4d}  t={row['secs']:7.1f}s  ws={world_steps:,}  pop=({row['pop_prey']:.0f},"
                      f"{row['pop_pred']:.0f})  resets={self.resets}  {msg}", flush=True)
            done = world_steps >= cfg.world_step_budget
            if g % 25 == 0 or done:
                self.save()
            if done:
                print(f"budget reached at generation {g} ({world_steps:,} world-steps)", flush=True)
                break
        return self.history

    def save(self) -> None:
        D = self.wcfg.obs_dim
        for sp in self.cfg.learn:
            m = to_actor_critic(self.champion(sp).cpu(), D)
            torch.save({"state_dict": m.state_dict(), "obs_dim": D, "species": SPECIES_NAME[sp],
                        "method": "ga", "ga_config": asdict(self.cfg)}, self.out / f"{SPECIES_NAME[sp]}.pt")
        (self.out / "history.json").write_text(json.dumps(self.history))
