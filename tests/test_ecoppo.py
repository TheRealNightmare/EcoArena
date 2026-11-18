"""EcoPPO: the lineage reward, extinction counting without resets, and automatic balancing."""

import torch

from ecoarena.env import PREDATOR, PREY, EcoWorld, load_config
from ecoarena.trainers.common import step_rewards
from ecoarena.trainers.ecoppo import EcoPPOConfig, EcoPPOTrainer, parse_world


def _zero(w):
    return (torch.zeros(w.W, w.species[PREY].alive.shape[1], 2), torch.zeros(w.W, w.species[PREDATOR].alive.shape[1], 2))


def test_parse_world():
    assert parse_world("S2") == ("S2", 0.0)
    assert parse_world("S4@0.1") == ("S4", 0.1)


def test_lineage_reward_counts_children_and_charges_replacement():
    cfg = load_config("S1", **{"prey.init_count": 4, "predator.init_count": 2, "refuge_count": 0})
    w = EcoWorld(cfg, n_worlds=1, seed=3)
    st, sc = w.species[PREDATOR], w.species_cfg[PREDATOR]
    e0, a0 = st.energy.clone(), st.alive.clone()
    info = w.step(*_zero(w))
    plain = step_rewards(w, info, PREDATOR, e0, a0)
    lin = step_rewards(w, info, PREDATOR, e0, a0, lineage=True)
    alive = a0[0] & ~info["died"][PREDATOR][0]
    # same energy change, measured in children instead of energy caps, minus the per-step replacement charge
    expect = plain[0, alive] * sc.energy_cap / sc.child_energy - 1.0 / sc.max_age
    assert torch.allclose(lin[0, alive], expect, atol=1e-6)


def _trainer(tmp_path, **kw):
    cfg = EcoPPOConfig(worlds=("S1", "S1@0.1"), base="s1_balanced.json", rollout_steps=8, iterations=2,
                       world_step_budget=10_000, minibatch=512, device="cpu", **kw)
    return EcoPPOTrainer(cfg, tmp_path)


def test_no_reset_counts_extinction_once_and_recolonises(tmp_path):
    tr = _trainer(tmp_path, founders=3, migrant_every=1)
    e = tr.envs[0]
    w = e["world"]
    w.species[PREDATOR].alive[:] = False  # wipe out the predators
    w.species[PREDATOR].alive[0, 0] = True
    w.species[PREDATOR].energy[0, 0] = 0.01  # the last one starves this step
    buf, st = tr._rollout_env(e)
    assert st["extinctions"] == 1 and st["resets"] == 0
    assert int(w.populations()[0, PREDATOR]) > 0  # migrants arrived; the world was not restarted
    assert buf[PREY]["rew"].shape[0] == tr.cfg.rollout_steps


def test_resets_ablation_restarts_world(tmp_path):
    tr = _trainer(tmp_path, no_reset=False)
    e = tr.envs[0]
    w = e["world"]
    w.species[PREDATOR].alive[:] = False
    w.species[PREDATOR].alive[0, 0] = True
    w.species[PREDATOR].energy[0, 0] = 0.01
    _, st = tr._rollout_env(e)
    assert st["resets"] == 1 and e["world"] is not w


def test_balance_slows_the_winning_side(tmp_path):
    tr = _trainer(tmp_path)
    for e in tr.envs:
        alive = e["world"].species[PREDATOR].alive
        idx = alive[0].nonzero().squeeze(1)
        alive[0, idx[len(idx) // 2:]] = False  # predators at about half their starting numbers
    tr._set_lr()
    assert tr.lr_scale[PREY] < 0.4 and tr.lr_scale[PREDATOR] == 1.0
    assert abs(tr.opts[PREY].param_groups[0]["lr"] - tr.cfg.lr * tr.lr_scale[PREY]) < 1e-12


def test_short_training_run(tmp_path):
    tr = _trainer(tmp_path)
    hist = tr.train()
    assert len(hist) == 2 and (tmp_path / "prey.pt").exists() and (tmp_path / "worlds.json").exists()


def test_ratio_balance_slows_only_the_leader(tmp_path):
    tr = _trainer(tmp_path, balance_mode="ratio")
    for e in tr.envs:
        alive = e["world"].species[PREDATOR].alive
        idx = alive[0].nonzero().squeeze(1)
        alive[0, idx[len(idx) // 2:]] = False  # predators at about half their start, prey at their full start
    tr._set_lr()
    assert tr.lr_scale[PREY] < 0.4 and tr.lr_scale[PREDATOR] == 1.0


def test_collapse_below_floor_counts_without_zero(tmp_path):
    tr = _trainer(tmp_path, collapse_frac=0.9, founders=0)
    e = tr.envs[0]
    w = e["world"]
    alive = w.species[PREDATOR].alive
    idx = alive[0].nonzero().squeeze(1)
    keep = int(e["floor"][1]) + 1  # one above the floor; the next death is a collapse
    alive[0, idx[keep:]] = False
    w.species[PREDATOR].energy[0, idx[0]] = 0.01  # dies this step
    _, st = tr._rollout_env(e)
    assert st["extinctions"] >= 1 and int(w.populations()[0, PREDATOR]) > 0


def test_head_start_freezes_prey_then_releases(tmp_path):
    tr = _trainer(tmp_path, head_start=0.5)
    tr.world_steps = 0
    tr._set_lr()
    assert tr.opts[PREY].param_groups[0]["lr"] == 0.0 and tr.opts[PREDATOR].param_groups[0]["lr"] == tr.cfg.lr
    tr.world_steps = tr.cfg.world_step_budget
    tr._set_lr()
    assert tr.opts[PREY].param_groups[0]["lr"] > 0.0
