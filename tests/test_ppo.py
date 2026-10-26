import torch

from ecoarena.env import PREDATOR, PREY, load_config
from ecoarena.trainers.ppo import ActorCritic, PPOConfig, PPOTrainer, load_policy


def tiny_trainer(tmp_path, **kw):
    wcfg = load_config("S1", refuge_count=0)
    cfg = PPOConfig(n_worlds=1, rollout_steps=8, iterations=1, minibatch=512, **kw)
    return PPOTrainer(wcfg, cfg, tmp_path)


def test_gae_matches_hand_computation(tmp_path):
    tr = tiny_trainer(tmp_path, gamma=0.5, gae_lambda=1.0)
    T = 3
    b = {
        "rew": torch.tensor([1.0, 2.0, 4.0]).view(T, 1, 1),
        "val": torch.zeros(T, 1, 1),
        "done": torch.tensor([0.0, 0.0, 0.0]).view(T, 1, 1),
        "valid": torch.ones(T, 1, 1, dtype=torch.bool),
    }
    tr._gae({PREY: b}, {PREY: torch.tensor([[8.0]])})
    # returns with gamma 0.5 and bootstrap 8: 4+0.5*8=8, 2+0.5*8=6, 1+0.5*6=4
    assert torch.allclose(b["ret"].view(-1), torch.tensor([4.0, 6.0, 8.0]))


def test_gae_stops_at_death_and_newborn_starts_fresh(tmp_path):
    tr = tiny_trainer(tmp_path, gamma=0.5, gae_lambda=1.0)
    T = 3
    # agent dies at t=1; a newborn fills the slot at t=2 with reward 4
    b = {
        "rew": torch.tensor([1.0, -1.0, 4.0]).view(T, 1, 1),
        "val": torch.zeros(T, 1, 1),
        "done": torch.tensor([0.0, 1.0, 0.0]).view(T, 1, 1),
        "valid": torch.ones(T, 1, 1, dtype=torch.bool),
    }
    tr._gae({PREY: b}, {PREY: torch.tensor([[0.0]])})
    # the dead agent's return ignores the newborn: t1 = -1, t0 = 1 + 0.5*(-1)
    assert torch.allclose(b["ret"].view(-1), torch.tensor([0.5, -1.0, 4.0]))


def test_invalid_steps_get_no_advantage(tmp_path):
    tr = tiny_trainer(tmp_path)
    T = 2
    b = {
        "rew": torch.tensor([5.0, 5.0]).view(T, 1, 1),
        "val": torch.zeros(T, 1, 1),
        "done": torch.zeros(T, 1, 1),
        "valid": torch.tensor([False, False]).view(T, 1, 1),
    }
    tr._gae({PREY: b}, {PREY: torch.zeros(1, 1)})
    assert (b["adv"] == 0).all()


def test_rollout_rewards_birth_bonus_and_death_penalty(tmp_path):
    tr = tiny_trainer(tmp_path, birth_bonus=1.0, death_penalty=1.0)
    w = tr.world
    prey = w.species[PREY]
    prey.energy[0, 0] = 95.0  # will breed on step 1
    prey.energy[0, 1] = 0.05  # will starve on step 1
    w.grass[:] = 0.0  # no food anywhere, so nobody refuels
    buf, _ = tr.rollout()
    r0 = float(buf[PREY]["rew"][0, 0, 0])
    r1 = float(buf[PREY]["rew"][0, 0, 1])
    assert r1 == -1.0
    assert buf[PREY]["done"][0, 0, 1] == 1
    # breeder: the child's energy is refunded, so the reward is the bonus minus one step of metabolism
    assert 0.99 < r0 < 1.0


def test_only_learning_species_is_trained(tmp_path):
    tr = tiny_trainer(tmp_path, learn=(PREDATOR,))
    assert set(tr.models) == {PREDATOR}
    assert PREY in tr.fixed


def test_one_iteration_runs_and_checkpoint_loads(tmp_path):
    tr = tiny_trainer(tmp_path)
    tr.train()
    pol = load_policy(tmp_path / "prey.pt")
    obs = torch.zeros(1, 5, tr.wcfg.obs_dim)
    a = pol(obs)
    assert a.shape == (1, 5, 2) and (a.abs() <= 1).all()


def test_actor_critic_shapes():
    m = ActorCritic(102)
    a, logp, v = m.act(torch.zeros(3, 7, 102))
    assert a.shape == (3, 7, 2) and logp.shape == (3, 7) and v.shape == (3, 7)
