"""PPO (M1) for EcoArena: one shared network per species, many agents, births and deaths mid-run.

How the variable population is handled
--------------------------------------
Agents live in fixed slots. Each rollout step stores, per slot, the observation, action, log-prob,
value and reward, plus two masks:

- `valid`: the slot held a living agent when it acted (only these steps are trained on);
- `done`:  that agent died during the step, so its return stops there. A newborn that later fills
  the same slot starts a new trajectory, because `done` cuts the advantage recursion.

Reward (identical rule for both species, so methods compare fairly)
-------------------------------------------------------------------
    r = Δenergy / energy_cap                      (food in, movement and metabolism out)
      + (child_energy / energy_cap + birth_bonus) per offspring   (refunds the energy given to the child)
      - death_penalty                             on death (starved, old age, eaten)

The world never resets during normal training; it is reset only when a species goes extinct (or every
`reset_every` steps if set), so learners see the full range of ecosystem states.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from ..env import PREDATOR, PREY, EcoWorld, WorldConfig
from ..policies import make_policy

SPECIES_NAME = {PREY: "prey", PREDATOR: "predator"}


# ---------------------------------------------------------------------------------------- model
def _layer(i: int, o: int, std: float = math.sqrt(2)) -> nn.Linear:
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, std)
    nn.init.zeros_(lin.bias)
    return lin


class ActorCritic(nn.Module):
    """MLP 64-64 tanh actor and critic (the architecture every method in the benchmark shares)."""

    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, act_dim, 0.01)
        )
        self.critic = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(), _layer(hidden, 1, 1.0)
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    min_log_std = -2.0

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mu = self.actor(obs)
        log_std = torch.clamp(self.log_std, min=self.min_log_std)
        return torch.distributions.Normal(mu, log_std.exp().expand_as(mu))

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)


class PPOPolicy:
    """Wraps a trained network so it plugs in wherever a baseline policy does: policy(obs) -> actions."""

    def __init__(self, model: ActorCritic, deterministic: bool = True):
        self.model, self.deterministic = model, deterministic

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        a, _, _ = self.model.act(obs, self.deterministic)
        return torch.clamp(a, -1.0, 1.0)


def load_policy(path: str | Path, device: str = "cpu", deterministic: bool = True) -> PPOPolicy:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ActorCritic(ck["obs_dim"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return PPOPolicy(m, deterministic)


# ---------------------------------------------------------------------------------------- config
@dataclass
class PPOConfig:
    learn: tuple[int, ...] = (PREY,)  # species that learn; the others use `opponent`
    opponent: str = "heuristic"
    n_worlds: int = 4
    rollout_steps: int = 128
    iterations: int = 200  # upper bound on iterations
    agent_step_budget: int = 0  # 0 = off; else stop once every learning species has this many agent-steps
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 4096
    ent_coef: float = 0.003  # small bonus: stage-1 run on GPU collapsed entropy to 0 without it
    min_log_std: float = -2.0  # action std never below ~0.14, so the policy keeps exploring
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    birth_bonus: float = 1.0
    death_penalty: float = 1.0
    reset_every: int = 0  # 0 = only reset on extinction
    seed: int = 0
    device: str = "cpu"
    log_every: int = 5
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------- trainer
class PPOTrainer:
    def __init__(self, world_cfg: WorldConfig, cfg: PPOConfig, out_dir: str | Path):
        self.wcfg, self.cfg = world_cfg, cfg
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.world_seed = cfg.seed * 1000
        self.world = EcoWorld(world_cfg, n_worlds=cfg.n_worlds, seed=self.world_seed, device=cfg.device)
        self.models = {sp: ActorCritic(world_cfg.obs_dim).to(self.device) for sp in cfg.learn}
        for m in self.models.values():
            m.min_log_std = cfg.min_log_std
        self.opts = {sp: torch.optim.Adam(m.parameters(), lr=cfg.lr, eps=1e-5) for sp, m in self.models.items()}
        self.fixed = {sp: make_policy(cfg.opponent, self.world, sp) for sp in (PREY, PREDATOR) if sp not in cfg.learn}
        self.resets = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ world management
    def _reset_world(self) -> None:
        self.resets += 1
        self.world_seed += 1
        self.world = EcoWorld(self.wcfg, n_worlds=self.cfg.n_worlds, seed=self.world_seed, device=self.cfg.device)
        self.fixed = {sp: make_policy(self.cfg.opponent, self.world, sp) for sp in self.fixed}

    def _maybe_reset(self) -> bool:
        extinct = bool((self.world.populations() == 0).any())
        periodic = self.cfg.reset_every and self.world.t >= self.cfg.reset_every
        if extinct or periodic:
            self._reset_world()
            return True
        return False

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def rollout(self) -> tuple[dict, dict]:
        cfg, w = self.cfg, self.world
        T = cfg.rollout_steps
        buf = {}
        for sp in cfg.learn:
            W, N = w.species[sp].alive.shape
            D = self.wcfg.obs_dim
            buf[sp] = {
                "obs": torch.zeros(T, W, N, D, device=self.device),
                "act": torch.zeros(T, W, N, 2, device=self.device),
                "logp": torch.zeros(T, W, N, device=self.device),
                "val": torch.zeros(T, W, N, device=self.device),
                "rew": torch.zeros(T, W, N, device=self.device),
                "done": torch.zeros(T, W, N, device=self.device),
                "valid": torch.zeros(T, W, N, dtype=torch.bool, device=self.device),
            }
        stats = {"captures": 0, "births": [0, 0], "deaths": [0, 0], "resets": 0}
        t_cut = T
        for t in range(T):
            obs = w.observe()
            acts, e0, alive0 = [None, None], [], []
            for sp in (PREY, PREDATOR):
                st = w.species[sp]
                e0.append(st.energy.clone())
                alive0.append(st.alive.clone())
                if sp in cfg.learn:
                    a, logp, v = self.models[sp].act(obs[sp])
                    b = buf[sp]
                    b["obs"][t], b["act"][t], b["logp"][t], b["val"][t] = obs[sp], a, logp, v
                    b["valid"][t] = st.alive
                    acts[sp] = a
                else:
                    acts[sp] = self.fixed[sp](obs[sp])
            info = w.step(acts[PREY], acts[PREDATOR])
            stats["captures"] += int(info["captures"].sum())
            for sp in (PREY, PREDATOR):
                stats["births"][sp] += int(info["births"][sp].sum())
                stats["deaths"][sp] += int(info["deaths"][sp].sum())
            for sp in cfg.learn:
                sc = w.species_cfg[sp]
                st = w.species[sp]
                died = info["died"][sp]
                r = (st.energy - e0[sp]) / sc.energy_cap
                wi, _child, par = info["birth_pairs"][sp]
                if wi.numel():
                    r[wi, par] += sc.child_energy / sc.energy_cap + cfg.birth_bonus
                r = torch.where(died, torch.full_like(r, -cfg.death_penalty), r)
                r = torch.where(alive0[sp], r, torch.zeros_like(r))
                buf[sp]["rew"][t] = r
                buf[sp]["done"][t] = died.float()
            if self._maybe_reset():
                stats["resets"] += 1
                t_cut = t + 1
                break
        # bootstrap values for agents still alive at the cut (zero if the world was just reset)
        last_val = {}
        if t_cut == T:
            obs = w.observe()
            for sp in cfg.learn:
                last_val[sp] = torch.where(w.species[sp].alive, self.models[sp].value(obs[sp]), 0.0)
        else:
            for sp in cfg.learn:
                b = buf[sp]
                b["done"][t_cut - 1] = torch.ones_like(b["done"][t_cut - 1])
                last_val[sp] = torch.zeros_like(b["val"][0])
        for sp in cfg.learn:
            for k in buf[sp]:
                buf[sp][k] = buf[sp][k][:t_cut]
        self._gae(buf, last_val)
        return buf, stats

    def _gae(self, buf: dict, last_val: dict) -> None:
        cfg = self.cfg
        for sp, b in buf.items():
            T = b["rew"].shape[0]
            adv = torch.zeros_like(b["rew"])
            gae = torch.zeros_like(b["rew"][0])
            for t in reversed(range(T)):
                # an agent alive at t that did not die is the same agent at t+1, so `done` alone
                # decides whether to bootstrap; a newborn in a freed slot starts after the cut
                nonterm = 1.0 - b["done"][t]
                next_v = last_val[sp] if t == T - 1 else b["val"][t + 1]
                delta = b["rew"][t] + cfg.gamma * next_v * nonterm - b["val"][t]
                gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
                gae = torch.where(b["valid"][t], gae, torch.zeros_like(gae))
                adv[t] = gae
            b["adv"] = adv
            b["ret"] = adv + b["val"]

    # ------------------------------------------------------------------ update
    def update(self, buf: dict) -> dict:
        cfg = self.cfg
        out = {}
        for sp, b in buf.items():
            m, opt = self.models[sp], self.opts[sp]
            mask = b["valid"]
            obs, act = b["obs"][mask], b["act"][mask]
            logp_old, adv, ret, val_old = b["logp"][mask], b["adv"][mask], b["ret"][mask], b["val"][mask]
            n = obs.shape[0]
            if n == 0:
                continue
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            clipfracs, pls, vls, ents, kls = [], [], [], [], []
            for _ in range(cfg.epochs):
                perm = torch.randperm(n, device=self.device)
                for i in range(0, n, cfg.minibatch):
                    idx = perm[i : i + cfg.minibatch]
                    d = m.dist(obs[idx])
                    logp = d.log_prob(act[idx]).sum(-1)
                    ratio = (logp - logp_old[idx]).exp()
                    pg = torch.max(-adv[idx] * ratio, -adv[idx] * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                    v = m.value(obs[idx])
                    v_clip = val_old[idx] + (v - val_old[idx]).clamp(-cfg.clip, cfg.clip)
                    vl = 0.5 * torch.max((v - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()
                    ent = d.entropy().sum(-1).mean()
                    loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(m.parameters(), cfg.max_grad_norm)
                    opt.step()
                    with torch.no_grad():
                        kls.append(((ratio - 1) - (logp - logp_old[idx])).mean().item())
                        clipfracs.append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                    pls.append(pg.item())
                    vls.append(vl.item())
                    ents.append(ent.item())
            out[SPECIES_NAME[sp]] = {
                "samples": n,
                "policy_loss": sum(pls) / len(pls),
                "value_loss": sum(vls) / len(vls),
                "entropy": sum(ents) / len(ents),
                "approx_kl": sum(kls) / len(kls),
                "clipfrac": sum(clipfracs) / len(clipfracs),
                "reward_per_agent_step": b["rew"][b["valid"]].mean().item(),
                "action_std": m.log_std.clamp(min=m.min_log_std).exp().mean().item(),
            }
        return out

    # ------------------------------------------------------------------ main loop
    def train(self) -> list[dict]:
        cfg = self.cfg
        t0 = time.time()
        agent_steps = {SPECIES_NAME[sp]: 0 for sp in cfg.learn}
        for it in range(1, cfg.iterations + 1):
            buf, stats = self.rollout()
            for sp in cfg.learn:
                agent_steps[SPECIES_NAME[sp]] += int(buf[sp]["valid"].sum())
            upd = self.update(buf)
            pops = self.world.populations().float().mean(0).tolist()
            row = {
                "iter": it,
                "secs": round(time.time() - t0, 1),
                "world_t": self.world.t,
                "pop_prey": round(pops[0], 1),
                "pop_pred": round(pops[1], 1),
                "captures": stats["captures"],
                "births": stats["births"],
                "deaths": stats["deaths"],
                "resets_total": self.resets,
                "agent_steps": dict(agent_steps),
                **{f"{k}/{kk}": round(vv, 5) if isinstance(vv, float) else vv
                   for k, v in upd.items() for kk, vv in v.items()},
            }
            self.history.append(row)
            if it % cfg.log_every == 0 or it == 1 or it == cfg.iterations:
                msg = " ".join(
                    f"{SPECIES_NAME[sp]} r={row.get(SPECIES_NAME[sp] + '/reward_per_agent_step', float('nan')):+.4f}"
                    f" kl={row.get(SPECIES_NAME[sp] + '/approx_kl', float('nan')):.4f}"
                    for sp in cfg.learn
                )
                print(f"it {it:4d}  t={row['secs']:7.1f}s  pop=({row['pop_prey']:.0f},{row['pop_pred']:.0f})  "
                      f"resets={self.resets}  {msg}", flush=True)
            budget_done = cfg.agent_step_budget and min(agent_steps.values()) >= cfg.agent_step_budget
            if it % 25 == 0 or it == cfg.iterations or budget_done:
                self.save()
            if budget_done:
                print(f"agent-step budget {cfg.agent_step_budget:,} reached at iteration {it}", flush=True)
                break
        return self.history

    def save(self) -> None:
        for sp, m in self.models.items():
            torch.save({"state_dict": m.state_dict(), "obs_dim": self.wcfg.obs_dim,
                        "species": SPECIES_NAME[sp], "ppo_config": asdict(self.cfg)},
                       self.out / f"{SPECIES_NAME[sp]}.pt")
        (self.out / "history.json").write_text(json.dumps(self.history))
