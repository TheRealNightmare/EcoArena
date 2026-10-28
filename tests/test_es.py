import pytest
import torch

from ecoarena.env import PREDATOR, PREY, load_config
from ecoarena.trainers.es import ESConfig, ESTrainer, centred_ranks
from ecoarena.trainers.ppo import load_policy

D = 102


def tiny(tmp_path, **kw):
    wcfg = load_config("S1", refuge_count=0)
    cfg = ESConfig(**{"n_worlds": 1, "population": 50, "gen_steps": 8, "world_step_budget": 24,
                      "min_samples": 1, **kw})
    return ESTrainer(wcfg, cfg, tmp_path)


def test_centred_ranks():
    u = centred_ranks(torch.tensor([3.0, -1.0, 10.0, 0.0]))
    assert torch.allclose(u, torch.tensor([1 / 6, -0.5, 0.5, -1 / 6]))


def test_population_is_antithetic_around_theta(tmp_path):
    tr = tiny(tmp_path)
    pop, th = tr.pop[PREY], tr.theta[PREY]
    assert torch.allclose((pop[:25] + pop[25:]) / 2, th.expand(25, -1), atol=1e-6)


def test_odd_population_rejected(tmp_path):
    with pytest.raises(ValueError):
        tiny(tmp_path, population=25)  # divides 600 and 150 but is odd


def test_update_moves_theta_toward_higher_fitness(tmp_path):
    tr = tiny(tmp_path, weight_decay=0.0)
    d = torch.randn(tr.theta[PREY].numel())
    d /= d.norm()
    th0 = tr.theta[PREY].clone()
    for _ in range(5):
        fit = tr.pop[PREY] @ d  # fitness grows along d
        tr.next_population(PREY, fit, torch.full((50,), 100.0))
    delta = tr.theta[PREY] - th0
    # 25 pairs in ~11k dimensions give a noisy estimate, but the step must lean along d far more than a random
    # direction would (whose projection is about |delta| / sqrt(n))
    assert float(delta @ d) > 5 * float(delta.norm()) / d.numel() ** 0.5


def test_unmeasured_pairs_are_dropped(tmp_path):
    tr = tiny(tmp_path)
    fit = torch.randn(50)
    fit[3] = float("nan")
    st = tr.next_population(PREY, fit, torch.full((50,), 100.0))
    assert st["evaluated"] == 48


def test_train_saves_theta_as_champion(tmp_path):
    tr = tiny(tmp_path, learn=(PREY, PREDATOR))
    hist = tr.train()
    assert hist[-1]["world_steps"] == 24
    pol = load_policy(tmp_path / "prey.pt")
    ck = torch.load(tmp_path / "prey.pt", weights_only=False)
    assert ck["method"] == "es"
    obs = torch.randn(1, 4, D)
    from ecoarena.trainers.ga import to_actor_critic

    expect = torch.clamp(to_actor_critic(tr.theta[PREY].cpu(), D).actor(obs), -1, 1)
    assert torch.allclose(pol(obs), expect, atol=1e-5)
