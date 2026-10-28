"""Evolution strategies (M3): OpenAI-ES (Salimans et al., 2017) over the weights of the shared 64-64 actor.

ES keeps one parameter vector theta per species. Each generation it evaluates P = 2K perturbed copies
theta +/- sigma * eps_k (antithetic pairs) and moves theta along the fitness-weighted average of the eps:

    g = 1 / (P * sigma) * sum_i u_i * eps_i,     u_i = centred rank of fitness_i in [-0.5, 0.5]
    theta <- Adam(theta, g - weight_decay * theta)

Evaluation is exactly the GA's (same worlds, same slot -> genome assignment, same fitness, same world-step
budget), so GA and ES differ only in how they turn fitness into the next population: GA copies the best few
genomes, ES uses every genome's score to estimate a gradient. The champion is theta itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..env import WorldConfig
from .ga import SPECIES_NAME, GAConfig, GATrainer, init_population


@dataclass
class ESConfig(GAConfig):
    sigma: float = 0.02
    lr: float = 0.01
    weight_decay: float = 0.005
    beta1: float = 0.9
    beta2: float = 0.999


def centred_ranks(x: torch.Tensor) -> torch.Tensor:
    """Ranks scaled to [-0.5, 0.5] (Salimans et al.): robust to outliers and to the ecosystem's overall state."""
    n = x.numel()
    r = torch.empty(n, device=x.device)
    r[torch.argsort(x)] = torch.arange(n, device=x.device, dtype=torch.float32)
    return r / max(n - 1, 1) - 0.5


class ESTrainer(GATrainer):
    def __init__(self, world_cfg: WorldConfig, cfg: ESConfig, out_dir: str | Path):
        if cfg.population % 2:
            raise ValueError("ES population must be even (antithetic pairs)")
        super().__init__(world_cfg, cfg, out_dir)
        D = world_cfg.obs_dim
        self.theta = {sp: init_population(1, D, cfg.init_out_std, self.gen, self.device)[0] for sp in cfg.learn}
        self.m = {sp: torch.zeros_like(t) for sp, t in self.theta.items()}
        self.v = {sp: torch.zeros_like(t) for sp, t in self.theta.items()}
        self.t = {sp: 0 for sp in cfg.learn}
        self.eps: dict[int, torch.Tensor] = {}
        for sp in cfg.learn:
            self._sample(sp)

    def _sample(self, sp: int) -> None:
        K = self.cfg.population // 2
        e = torch.randn(K, self.theta[sp].numel(), generator=self.gen).to(self.device)
        self.eps[sp] = torch.cat([e, -e])  # row i and row i + K are a pair
        self.pop[sp] = self.theta[sp].unsqueeze(0) + self.cfg.sigma * self.eps[sp]

    def next_population(self, sp: int, fitness: torch.Tensor, samples: torch.Tensor) -> dict:
        cfg = self.cfg
        K = cfg.population // 2
        ok = (samples >= cfg.min_samples) & ~torch.isnan(fitness)
        pair_ok = ok[:K] & ok[K:]  # use a pair only if both halves were measured
        keep = torch.cat([pair_ok, pair_ok])
        stats = {"best": float("nan"), "mean": float("nan"), "evaluated": int(keep.sum())}
        if int(pair_ok.sum()) >= 2:
            f = fitness[keep]
            u = centred_ranks(f)
            g = (u @ self.eps[sp][keep]) / (keep.sum() * cfg.sigma)
            g = g - cfg.weight_decay * self.theta[sp]
            self.t[sp] += 1
            b1, b2 = cfg.beta1, cfg.beta2
            self.m[sp] = b1 * self.m[sp] + (1 - b1) * g
            self.v[sp] = b2 * self.v[sp] + (1 - b2) * g * g
            mh = self.m[sp] / (1 - b1 ** self.t[sp])
            vh = self.v[sp] / (1 - b2 ** self.t[sp])
            step = cfg.lr * mh / (vh.sqrt() + 1e-8)
            self.theta[sp] = self.theta[sp] + step  # ascent: higher fitness is better
            stats.update(best=float(f.max()), mean=float(f.mean()), step_norm=float(step.norm()),
                         theta_norm=float(self.theta[sp].norm()))
        self._sample(sp)
        return stats

    def champion(self, sp: int) -> torch.Tensor:
        return self.theta[sp]

    def save(self) -> None:
        super().save()
        for sp in self.cfg.learn:  # mark the checkpoints as ES
            path = self.out / f"{SPECIES_NAME[sp]}.pt"
            ck = torch.load(path, weights_only=False)
            ck["method"] = "es"
            ck["es_config"] = ck.pop("ga_config")
            torch.save(ck, path)
