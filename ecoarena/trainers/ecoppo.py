"""EcoPPO: a PPO learner built to keep the whole ecosystem alive, from the causes of collapse we measured.

PPO trains both species. Four parts change how it trains, each switchable for the ablation:

1. Lineage reward (`lineage`). Reward is counted in children, and every step costs the surplus an animal must bank to
   replace itself once in its life (common.step_rewards). Breaking even on energy is no longer enough: the learner
   is pushed towards R0 >= 1, the condition our replacement model shows predators need (paper Eq. 1).
2. No-reset worlds (`no_reset`). The benchmark silently restarts a world when a species dies out, so no learner
   ever lives with the collapse it caused. Here worlds never restart: an extinct species is recolonised by a few
   migrants (`founders`, one every `migrant_every` steps), the other species keeps the state it is in, and every
   learner alive in that world pays `extinction_penalty` once per extinction.
3. Automatic balancing (`balance`). Each species' learning rate is scaled by how its opponent is doing:
       lr_s = lr * clip((rho_other / balance) ** balance_power, balance_min, 1)
   where rho_other is the opponent's population relative to its starting count, averaged over the training worlds.
   A species that is winning too hard slows down; with balance = 1 and power 2, an opponent at half its starting
   numbers gives a quarter of the learning rate, the fixed "prey 4x slower" of study 2 (E3), but only while needed.
4. Many worlds (`worlds`). Training worlds are drawn from several scenarios and friction levels at once
   ("S2" = scenario S2, "S2@0.1" = S2 with drag 0.1), so no single world's physics can be exploited.

Defaults also look ~1,000 steps ahead (gamma 0.999, from E1). The network, PPO update and world-step budget are
those of the benchmark PPO, so every difference comes from the four parts above.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import torch

from ..env import PREDATOR, PREY, EcoWorld, load_config
from .common import step_rewards
from .ppo import SPECIES_NAME, PPOConfig, PPOTrainer

DEFAULT_WORLDS = ("S1", "S2", "S4", "S1@0.1", "S2@0.1", "S4@0.1")


def parse_world(spec: str) -> tuple[str, float]:
    """'S2' -> ('S2', 0.0); 'S4@0.1' -> ('S4', 0.1)."""
    sc, _, drag = spec.partition("@")
    return sc, float(drag) if drag else 0.0


@dataclass
class EcoPPOConfig(PPOConfig):
    worlds: tuple[str, ...] = DEFAULT_WORLDS
    worlds_each: int = 1  # parallel copies of each training world
    base: str = "s1_large.json"
    lineage: bool = True
    no_reset: bool = True
    founders: int = 10  # immigration floor while training without resets
    migrant_every: int = 10
    balance: float = 1.0  # 0 = off
    balance_power: float = 2.0
    balance_min: float = 0.05
    gamma: float = 0.999
    extinction_penalty: float = 5.0


class EcoPPOTrainer(PPOTrainer):
    def __init__(self, cfg: EcoPPOConfig, out_dir):
        cfg.learn = (PREY, PREDATOR)
        cfg.n_worlds = cfg.worlds_each
        self.envs = []
        for i, spec in enumerate(cfg.worlds):
            sc, drag = parse_world(spec)
            over = {}
            if drag:
                over |= {"prey.drag": drag, "predator.drag": drag}
            if cfg.no_reset:
                over |= {"immigration_floor": cfg.founders, "immigration_interval": cfg.migrant_every}
            self.envs.append({"spec": spec, "cfg": load_config(sc, base=cfg.base, **over),
                              "seed": cfg.seed * 1000 + 100 * i, "resets": 0, "extinctions": 0})
        super().__init__(self.envs[0]["cfg"], cfg, out_dir)
        for e in self.envs:
            e["world"] = EcoWorld(e["cfg"], n_worlds=cfg.worlds_each, seed=e["seed"], device=cfg.device)
        self.lr_scale = {PREY: 1.0, PREDATOR: 1.0}

    # ------------------------------------------------------------------ helpers
    def relative_pops(self) -> torch.Tensor:
        """[2]: each species' population relative to its starting count, averaged over every training world."""
        rel = []
        for e in self.envs:
            init = torch.tensor([float(e["cfg"].prey.init_count), float(e["cfg"].predator.init_count)],
                                device=self.device)
            rel.append((e["world"].populations().float() / init).mean(0))
        return torch.stack(rel).mean(0)

    def _set_lr(self) -> None:
        cfg = self.cfg
        if cfg.balance > 0:
            rho = self.relative_pops()
            for sp in (PREY, PREDATOR):
                other = float(rho[1 - sp])
                self.lr_scale[sp] = min(1.0, max(cfg.balance_min, (other / cfg.balance) ** cfg.balance_power))
        for sp, opt in self.opts.items():
            for g in opt.param_groups:
                g["lr"] = cfg.lr * self.lr_scale[sp]

    # ------------------------------------------------------------------ rollout over every training world
    @torch.no_grad()
    def _rollout_env(self, e: dict) -> tuple[dict, dict]:
        cfg, w = self.cfg, e["world"]
        T = cfg.rollout_steps
        D = e["cfg"].obs_dim
        buf = {}
        for sp in (PREY, PREDATOR):
            W, N = w.species[sp].alive.shape
            buf[sp] = {k: torch.zeros(T, W, N, *s, device=self.device)
                       for k, s in (("obs", (D,)), ("act", (2,)), ("logp", ()), ("val", ()), ("rew", ()), ("done", ()))}
            buf[sp]["valid"] = torch.zeros(T, W, N, dtype=torch.bool, device=self.device)
        stats = {"captures": 0, "births": [0, 0], "deaths": [0, 0], "extinctions": 0, "resets": 0}
        t_cut = T
        for t in range(T):
            obs = w.observe()
            acts, e0, alive0 = [None, None], [], []
            pops0 = w.populations()
            for sp in (PREY, PREDATOR):
                st = w.species[sp]
                e0.append(st.energy.clone())
                alive0.append(st.alive.clone())
                a, logp, v = self.models[sp].act(obs[sp])
                b = buf[sp]
                b["obs"][t], b["act"][t], b["logp"][t], b["val"][t] = obs[sp], a, logp, v
                b["valid"][t] = st.alive
                acts[sp] = a
            info = w.step(acts[PREY], acts[PREDATOR])
            stats["captures"] += int(info["captures"].sum())
            for sp in (PREY, PREDATOR):
                stats["births"][sp] += int(info["births"][sp].sum())
                stats["deaths"][sp] += int(info["deaths"][sp].sum())
                buf[sp]["rew"][t] = step_rewards(w, info, sp, e0[sp], alive0[sp], cfg.birth_bonus, cfg.death_penalty,
                                                 cfg.eco_lambda, cfg.cost_weight, lineage=cfg.lineage)
                buf[sp]["done"][t] = info["died"][sp].float()
            # an extinction happens when the last animal of a species dies this step (migrants may refill the
            # world in the same step, so count before immigration)
            before_mig = info["populations"] - torch.stack(info["migrants"], 1)
            gone = ((pops0 > 0) & (before_mig == 0)).any(1)  # [W]
            if bool(gone.any()):
                stats["extinctions"] += int(gone.sum())
                e["extinctions"] += int(gone.sum())
                if cfg.extinction_penalty:
                    for sp in (PREY, PREDATOR):
                        hit = buf[sp]["valid"][t] & gone.unsqueeze(1)
                        buf[sp]["rew"][t] = buf[sp]["rew"][t] - cfg.extinction_penalty * hit.float()
                if not cfg.no_reset:  # benchmark behaviour: restart the world
                    e["resets"] += 1
                    e["seed"] += 1
                    e["world"] = EcoWorld(e["cfg"], n_worlds=cfg.worlds_each, seed=e["seed"], device=cfg.device)
                    stats["resets"] += 1
                    t_cut = t + 1
                    break
        last_val = {}
        if t_cut == T:
            obs = w.observe()
            for sp in (PREY, PREDATOR):
                last_val[sp] = torch.where(w.species[sp].alive, self.models[sp].value(obs[sp]), 0.0)
        else:
            for sp in (PREY, PREDATOR):
                buf[sp]["done"][t_cut - 1] = torch.ones_like(buf[sp]["done"][t_cut - 1])
                last_val[sp] = torch.zeros_like(buf[sp]["val"][0])
        for sp in (PREY, PREDATOR):
            for k in buf[sp]:
                buf[sp][k] = buf[sp][k][:t_cut]
        self._gae(buf, last_val)
        stats["world_steps"] = t_cut * cfg.worlds_each
        return buf, stats

    def rollout(self) -> tuple[dict, dict]:
        """Roll out every training world, then pool the samples of each species (flattened to the living ones)."""
        parts, total = [], {"captures": 0, "births": [0, 0], "deaths": [0, 0], "extinctions": 0, "resets": 0,
                            "world_steps": 0}
        for e in self.envs:
            buf, st = self._rollout_env(e)
            parts.append(buf)
            for k in ("captures", "extinctions", "resets", "world_steps"):
                total[k] += st[k]
            for sp in (PREY, PREDATOR):
                total["births"][sp] += st["births"][sp]
                total["deaths"][sp] += st["deaths"][sp]
        pooled = {}
        for sp in (PREY, PREDATOR):
            pooled[sp] = {}
            for k in ("obs", "act", "logp", "val", "rew", "adv", "ret"):
                pooled[sp][k] = torch.cat([p[sp][k][p[sp]["valid"]] for p in parts])
            pooled[sp]["valid"] = torch.ones(pooled[sp]["rew"].shape[0], dtype=torch.bool, device=self.device)
        return pooled, total

    # ------------------------------------------------------------------ main loop
    def train(self) -> list[dict]:
        cfg = self.cfg
        t0 = time.time()
        world_steps = 0
        for it in range(1, cfg.iterations + 1):
            self.iteration = it
            buf, stats = self.rollout()
            world_steps += stats["world_steps"]
            self._set_lr()
            upd = self.update(buf)
            rho = self.relative_pops().tolist()
            row = {
                "iter": it,
                "secs": round(time.time() - t0, 1),
                "rel_prey": round(rho[0], 3),
                "rel_pred": round(rho[1], 3),
                "lr_scale_prey": round(self.lr_scale[PREY], 3),
                "lr_scale_pred": round(self.lr_scale[PREDATOR], 3),
                "captures": stats["captures"],
                "births": stats["births"],
                "deaths": stats["deaths"],
                "extinctions": stats["extinctions"],
                "extinctions_total": sum(e["extinctions"] for e in self.envs),
                "resets_total": sum(e["resets"] for e in self.envs),
                "world_steps": world_steps,
                **{f"{k}/{kk}": round(vv, 5) if isinstance(vv, float) else vv
                   for k, v in upd.items() for kk, vv in v.items()},
            }
            self.history.append(row)
            if it % cfg.log_every == 0 or it == 1:
                print(f"it {it:4d}  t={row['secs']:7.1f}s  prey {rho[0]:.2f} pred {rho[1]:.2f} of start  "
                      f"lr x({self.lr_scale[PREY]:.2f}, {self.lr_scale[PREDATOR]:.2f})  "
                      f"extinctions={row['extinctions_total']}  "
                      + " ".join(f"{SPECIES_NAME[sp]} r={row.get(SPECIES_NAME[sp] + '/reward_per_agent_step', float('nan')):+.4f}"
                                 for sp in (PREY, PREDATOR)), flush=True)
            done = bool(cfg.world_step_budget) and world_steps >= cfg.world_step_budget
            if it % 25 == 0 or it == cfg.iterations or done:
                self.save()
            if done:
                print(f"budget reached at iteration {it} ({world_steps:,} world-steps)", flush=True)
                break
        (self.out / "worlds.json").write_text(json.dumps(
            [{k: e[k] for k in ("spec", "seed", "resets", "extinctions")} for e in self.envs], indent=1))
        return self.history
