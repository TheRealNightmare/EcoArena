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
from .common import homeostat_gate, step_rewards

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


def load_policy(path: str | Path, device: str = "cpu", deterministic: bool = True):
    """Load any method's checkpoint as policy(obs) -> actions. PPO, GA and ES share the 64-64 actor; NEAT
    checkpoints hold their own evolved graph."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("method") == "neat":
        from .neat import NEATPolicy

        return NEATPolicy.from_checkpoint(ck, device)
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
    world_step_budget: int = 0  # 0 = off; else stop after this many world-steps (summed over parallel worlds)
    # Freezing a species at an agent-step budget is kept as an option but is off by default: in run both-v3
    # prey reached it at iteration 123 while predators were still untrained, and predators then over-exploited
    # the frozen prey (1/4 eval worlds kept both species). A shared world-step budget avoids that.
    agent_step_budget: int = 0
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
    birth_bonus: float = 0.0  # same reward as GA/ES/NEAT; see common.step_rewards
    death_penalty: float = 1.0
    reset_every: int = 0  # 0 = only reset on extinction
    # Hybrid (M5) starts the actor from an evolved genome. A fresh critic gives junk advantages at first, so the
    # first `critic_warmup` iterations train the value function only and leave the inherited policy untouched.
    critic_warmup: int = 0
    # ---- study 2 options (defaults reproduce the benchmark) ----
    eco_lambda: float = 0.0  # shared ecosystem-health reward, see common.step_rewards
    prey_every: int = 1  # learning-speed asymmetry: update this species every k-th iteration; 0 = never
    pred_every: int = 1
    cost_weight: float = 1.0  # energy-leak fix (E4), see common.step_rewards
    homeostat: float = 0.0  # Homeostatic Co-Training (E5), see common.homeostat_gate
    # E5 ablation: when a species dies out in a training world, every learner alive in that world gets this extra
    # penalty before the world is reset, so collapse is no longer free (the benchmark resets silently)
    extinction_penalty: float = 0.0
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
        self.iteration = 0
        self.history: list[dict] = []

    def init_actor(self, sp: int, path: str | Path, log_std: float | None = None) -> None:
        """Copy the actor (policy mean) of a saved 64-64 checkpoint (PPO, GA or ES) into species sp; the critic
        stays freshly initialised. Optionally set the starting action noise."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        actor = {k.removeprefix("actor."): v for k, v in ck["state_dict"].items() if k.startswith("actor.")}
        m = self.models[sp]
        m.actor.load_state_dict(actor)
        if log_std is not None:
            with torch.no_grad():
                m.log_std.fill_(log_std)
        self.opts[sp] = torch.optim.Adam(m.parameters(), lr=self.cfg.lr, eps=1e-5)

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
                buf[sp]["rew"][t] = step_rewards(w, info, sp, e0[sp], alive0[sp], cfg.birth_bonus, cfg.death_penalty,
                                                 cfg.eco_lambda, cfg.cost_weight)
                buf[sp]["done"][t] = info["died"][sp].float()
            if cfg.extinction_penalty:
                gone = (w.populations() == 0).any(1)  # [W] worlds where a species just died out
                if bool(gone.any()):
                    for sp in cfg.learn:
                        hit = buf[sp]["valid"][t] & gone.unsqueeze(1)
                        buf[sp]["rew"][t] = buf[sp]["rew"][t] - cfg.extinction_penalty * hit.float()
                    stats["extinctions"] = stats.get("extinctions", 0) + int(gone.sum())
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
    def update(self, buf: dict, skip: set | None = None) -> dict:
        cfg = self.cfg
        out = {}
        for sp, b in buf.items():
            if skip and sp in skip:
                continue
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
                    if cfg.critic_warmup and self.iteration <= cfg.critic_warmup:
                        loss = cfg.vf_coef * vl
                    else:
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
        world_steps = 0
        # A species that reaches the agent-step budget is frozen: it keeps acting with its trained
        # brain but is no longer updated, so every species gets the same training budget.
        self.frozen: set[int] = set()
        for it in range(1, cfg.iterations + 1):
            self.iteration = it
            buf, stats = self.rollout()
            world_steps += buf[cfg.learn[0]]["valid"].shape[0] * cfg.n_worlds
            for sp in cfg.learn:
                if sp not in self.frozen:
                    agent_steps[SPECIES_NAME[sp]] += int(buf[sp]["valid"].sum())
            every = {PREY: cfg.prey_every, PREDATOR: cfg.pred_every}
            resting = {sp for sp in cfg.learn if every[sp] <= 0 or it % every[sp]}
            gated = {sp for sp in cfg.learn if homeostat_gate(self.world, sp, cfg.homeostat)}
            resting |= gated
            upd = self.update(buf, skip=self.frozen | resting)
            if cfg.agent_step_budget:
                for sp in cfg.learn:
                    if sp not in self.frozen and agent_steps[SPECIES_NAME[sp]] >= cfg.agent_step_budget:
                        self.frozen.add(sp)
                        print(f"{SPECIES_NAME[sp]}: budget {cfg.agent_step_budget:,} reached at iteration {it}; "
                              f"frozen", flush=True)
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
                "world_steps": world_steps,
                "agent_steps": dict(agent_steps),
                "frozen": sorted(SPECIES_NAME[sp] for sp in self.frozen),
                "gated": sorted(SPECIES_NAME[sp] for sp in gated),
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
            budget_done = (bool(cfg.agent_step_budget) and len(self.frozen) == len(cfg.learn)) or (
                bool(cfg.world_step_budget) and world_steps >= cfg.world_step_budget)
            if it % 25 == 0 or it == cfg.iterations or budget_done:
                self.save()
            if budget_done:
                print(f"budget reached at iteration {it} ({world_steps:,} world-steps)", flush=True)
                break
        return self.history

    def save(self) -> None:
        for sp, m in self.models.items():
            torch.save({"state_dict": m.state_dict(), "obs_dim": self.wcfg.obs_dim,
                        "species": SPECIES_NAME[sp], "ppo_config": asdict(self.cfg)},
                       self.out / f"{SPECIES_NAME[sp]}.pt")
        (self.out / "history.json").write_text(json.dumps(self.history))
