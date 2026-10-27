import pytest
import torch

from ecoarena.env import PREDATOR, PREY, load_config
from ecoarena.trainers.ga import (
    GAConfig,
    GATrainer,
    batched_actions,
    init_population,
    n_params,
    to_actor_critic,
)
from ecoarena.trainers.ppo import load_policy

D = 102


def test_batched_actions_match_each_genomes_own_network():
    P, W, N = 5, 2, 15
    pop = init_population(P, D, 0.5, torch.Generator().manual_seed(0), "cpu")
    obs = torch.randn(W, N, D)
    perm = torch.tensor([3, 0, 4, 1, 2])
    a = batched_actions(pop, obs, perm)
    for s in range(N):
        g = int(perm[s % P])
        expect = torch.clamp(to_actor_critic(pop[g], D).actor(obs[:, s]), -1, 1)
        assert torch.allclose(a[:, s], expect, atol=1e-5), f"slot {s}"


def test_population_shape_and_diversity():
    pop = init_population(4, D, 0.5, torch.Generator().manual_seed(1), "cpu")
    assert pop.shape == (4, n_params(D))
    assert not torch.allclose(pop[0], pop[1])


def tiny(tmp_path, **kw):
    wcfg = load_config("S1", refuge_count=0)  # small world: prey cap 600, predator cap 150
    cfg = GAConfig(**{"n_worlds": 1, "population": 50, "gen_steps": 8, "world_step_budget": 24,
                      "min_samples": 1, **kw})
    return GATrainer(wcfg, cfg, tmp_path)


def test_population_must_divide_capacity(tmp_path):
    with pytest.raises(ValueError):
        tiny(tmp_path, population=7)


def test_fitness_is_credited_to_the_right_genome(tmp_path):
    tr = tiny(tmp_path, gen_steps=1)
    w = tr.world
    prey = w.species[PREY]
    w.grass[:] = 0.0
    prey.energy[:] = 50.0
    # make exactly one prey (slot 7) starve: its genome must be the only one with a -1 death in it
    prey.energy[0, 7] = 0.01
    torch.manual_seed(0)
    tr.gen.manual_seed(123)
    expected_perm = torch.randperm(50, generator=torch.Generator().manual_seed(123))
    fitness, _stats, _ = tr.evaluate_generation()
    f = fitness[PREY]
    starving_genome = int(expected_perm[7 % 50])
    assert f[starving_genome] == torch.nanquantile(f, 0.0)  # the lowest fitness of all
    assert f[starving_genome] < -0.2


def test_elites_survive_unchanged_and_children_differ(tmp_path):
    tr = tiny(tmp_path)
    fit = torch.arange(50, dtype=torch.float32)  # genome 49 best
    samples = torch.full((50,), 100.0)
    before = tr.pop[PREY].clone()
    tr.next_population(PREY, fit, samples)
    n_elite = 5
    assert torch.equal(tr.pop[PREY][0], before[49])
    assert torch.equal(tr.pop[PREY][:n_elite], before[[49, 48, 47, 46, 45]])
    assert not any(torch.equal(tr.pop[PREY][i], before[j]) for i in range(n_elite, 50) for j in range(50))


def test_under_sampled_genomes_cannot_be_elites(tmp_path):
    tr = tiny(tmp_path, min_samples=10)
    fit = torch.zeros(50)
    fit[3] = 99.0  # great score but only 2 samples
    samples = torch.full((50,), 100.0)
    samples[3] = 2
    before = tr.pop[PREY].clone()
    tr.next_population(PREY, fit, samples)
    assert not any(torch.equal(tr.pop[PREY][i], before[3]) for i in range(5))


def test_train_stops_at_budget_and_saves_loadable_champions(tmp_path):
    tr = tiny(tmp_path, learn=(PREY, PREDATOR))
    hist = tr.train()
    assert hist[-1]["world_steps"] == 24
    for name in ("prey", "predator"):
        pol = load_policy(tmp_path / f"{name}.pt")
        assert pol(torch.zeros(1, 3, D)).shape == (1, 3, 2)
