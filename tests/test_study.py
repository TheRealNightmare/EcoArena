"""Study 2 options: ecosystem-health reward, episodic worlds, group assignment, learning-speed asymmetry."""

import pytest
import torch

from ecoarena.env import PREDATOR, PREY, EcoWorld, load_config
from ecoarena.trainers.common import ecosystem_health
from ecoarena.trainers.es import ESConfig, ESTrainer
from ecoarena.trainers.ga import GAConfig, GATrainer, batched_actions, world_actions
from ecoarena.trainers.ppo import PPOConfig, PPOTrainer


def wcfg():
    return load_config("S1", refuge_count=0)


def test_ecosystem_health_bounds():
    w = EcoWorld(wcfg(), n_worlds=2, seed=0)
    assert torch.allclose(ecosystem_health(w), torch.ones(2, 1))  # starting populations -> 1
    w.species[PREDATOR].alive[1] = False  # predators extinct in world 1
    h = ecosystem_health(w)
    assert h[0, 0] == 1.0 and h[1, 0] == 0.0
    alive_idx = w.species[PREY].alive[0].nonzero().squeeze(1)
    w.species[PREY].alive[0, alive_idx[: len(alive_idx) * 3 // 4]] = False  # about a quarter of the prey left
    left = int(w.species[PREY].alive[0].sum()) / w.cfg.prey.init_count
    assert abs(float(ecosystem_health(w)[0, 0]) - left**0.5) < 1e-6


def test_world_actions_uses_one_genome_per_world():
    torch.manual_seed(0)
    flat = torch.randn(3, 102 * 64 + 64 + 64 * 64 + 64 + 64 * 2 + 2) * 0.1
    obs = torch.randn(3, 10, 102)
    wperm = torch.tensor([2, 0, 1])
    a = world_actions(flat, obs, wperm)
    for w in range(3):  # same as running genome wperm[w] alone on world w's animals
        ref = batched_actions(flat[wperm[w] : wperm[w] + 1], obs[w : w + 1], torch.tensor([0]))
        assert torch.allclose(a[w], ref[0], atol=1e-6)


def test_group_needs_worlds_multiple_of_population(tmp_path):
    with pytest.raises(ValueError):
        GATrainer(wcfg(), GAConfig(n_worlds=3, population=2, assign="group"), tmp_path)


@pytest.mark.parametrize("Trainer,Config", [(GATrainer, GAConfig), (ESTrainer, ESConfig)])
def test_episodic_group_generation(tmp_path, Trainer, Config):
    cfg = Config(learn=(PREY, PREDATOR), n_worlds=4, population=4, gen_steps=6, world_step_budget=48,
                 min_samples=1, episodic=True, assign="group", eco_lambda=1.0, elite_frac=0.25)
    tr = Trainer(wcfg(), cfg, tmp_path)
    seed0 = tr.world_seed
    hist = tr.train()
    assert len(hist) == 2 and tr.world_seed == seed0 + 2  # fresh worlds every generation
    assert tr.resets == 0 and "extinct_worlds" in hist[0]
    assert (tmp_path / "prey.pt").exists() and (tmp_path / "predator.pt").exists()


def test_group_fitness_is_per_world(tmp_path):
    """With one world per genome, a genome's fitness is the mean reward of the animals in its world."""
    cfg = GAConfig(learn=(PREY,), n_worlds=2, population=2, gen_steps=5, min_samples=1, episodic=True,
                   assign="group")
    tr = GATrainer(wcfg(), cfg, tmp_path)
    fit, stats, steps = tr.evaluate_generation()
    assert steps == 5 and torch.isfinite(fit[PREY]).all()
    assert all(n > 0 for n in stats["samples"]["prey"])


def test_ga_asymmetry_freezes_and_slows(tmp_path):
    cfg = GAConfig(learn=(PREY, PREDATOR), n_worlds=1, population=50, gen_steps=4, world_step_budget=16,
                   min_samples=1, prey_every=0, pred_every=2)
    tr = GATrainer(wcfg(), cfg, tmp_path)
    prey0, pred0 = tr.pop[PREY].clone(), tr.pop[PREDATOR].clone()
    hist = tr.train()
    assert torch.equal(tr.pop[PREY], prey0)  # frozen at initialisation
    assert not torch.equal(tr.pop[PREDATOR], pred0)
    assert "prey/best" not in hist[0] and "predator/best" not in hist[0] and "predator/best" in hist[1]


def test_ppo_asymmetry_and_eco(tmp_path):
    cfg = PPOConfig(learn=(PREY, PREDATOR), n_worlds=1, rollout_steps=4, iterations=2, minibatch=512,
                    prey_every=0, pred_every=1, eco_lambda=2.0)
    tr = PPOTrainer(wcfg(), cfg, tmp_path)
    prey0 = {k: v.clone() for k, v in tr.models[PREY].state_dict().items()}
    pred0 = {k: v.clone() for k, v in tr.models[PREDATOR].state_dict().items()}
    hist = tr.train()
    assert all(torch.equal(v, tr.models[PREY].state_dict()[k]) for k, v in prey0.items())
    assert any(not torch.equal(v, tr.models[PREDATOR].state_dict()[k]) for k, v in pred0.items())
    assert "prey/policy_loss" not in hist[-1] and "predator/policy_loss" in hist[-1]


def test_eco_lambda_raises_reward_of_healthy_world(tmp_path):
    from ecoarena.trainers.common import step_rewards

    w = EcoWorld(wcfg(), n_worlds=1, seed=1)
    sp = PREY
    e0, a0 = w.species[sp].energy.clone(), w.species[sp].alive.clone()
    info = w.step(torch.zeros(1, w.species[PREY].alive.shape[1], 2), torch.zeros(1, w.species[PREDATOR].alive.shape[1], 2))
    r0 = step_rewards(w, info, sp, e0, a0)
    r1 = step_rewards(w, info, sp, e0, a0, eco_lambda=3.0)
    live = a0 & ~info["died"][sp]
    diff = (r1 - r0)[live]
    assert torch.allclose(diff, torch.full_like(diff, 3.0 / 1000 * float(ecosystem_health(w)[0, 0])))
    assert torch.equal(r1[~a0], r0[~a0])
