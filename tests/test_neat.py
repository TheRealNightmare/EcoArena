import random

import torch

from ecoarena.env import PREDATOR, PREY, load_config
from ecoarena.trainers.neat import (
    Genome,
    Innovations,
    NEATConfig,
    NEATTrainer,
    compile_population,
    depth_of,
    propagate,
)
from ecoarena.trainers.ppo import load_policy

D = 102


def reference_forward(g: Genome, x: torch.Tensor) -> torch.Tensor:
    """Plain recursive evaluation of one genome, to check the batched dense version against."""
    cache = {}

    def val(n):
        if n < D:
            return x[..., n]
        if n not in cache:
            z = torch.full(x.shape[:-1], g.bias.get(n, 0.0))
            for src, dst, w, en in g.conns.values():
                if en and dst == n:
                    z = z + w * val(src)
            cache[n] = z if n in (D, D + 1) else torch.tanh(z)
        return cache[n]

    return torch.clamp(torch.stack([val(D), val(D + 1)], -1), -1, 1)


def tiny(tmp_path, **kw):
    wcfg = load_config("S1", refuge_count=0)
    cfg = NEATConfig(**{"n_worlds": 1, "population": 50, "gen_steps": 8, "world_step_budget": 24,
                        "min_samples": 1, **kw})
    return NEATTrainer(wcfg, cfg, tmp_path)


def grown_genomes(tmp_path, n_mut=60):
    tr = tiny(tmp_path, p_add_node=0.5, p_add_conn=0.8, p_toggle=0.1)
    gs = [g.copy(i) for i, g in enumerate(tr.genomes[PREY][:5])]
    for g in gs:
        for _ in range(n_mut):
            tr._mutate(g)
    return tr, gs


def test_batched_propagation_matches_reference(tmp_path):
    _tr, gs = grown_genomes(tmp_path)
    assert any(len(g.hidden) > 3 for g in gs)
    Win, Wh, b, depth = compile_population(gs, D, 48, "cpu")
    x = torch.rand(5, 7, D)
    out = propagate(x, Win, Wh, b, depth)
    for p, g in enumerate(gs):
        assert torch.allclose(out[p], reference_forward(g, x[p]), atol=1e-5), f"genome {p}"


def test_mutations_keep_graph_acyclic(tmp_path):
    _tr, gs = grown_genomes(tmp_path, n_mut=120)
    for g in gs:
        # Kahn's algorithm over ALL edges (disabled too): every node must get removed
        nodes = {n for c in g.conns.values() for n in c[:2]}
        indeg = dict.fromkeys(nodes, 0)
        for c in g.conns.values():
            indeg[c[1]] += 1
        ready = [n for n in nodes if indeg[n] == 0]
        seen = 0
        while ready:
            n = ready.pop()
            seen += 1
            for c in g.conns.values():
                if c[0] == n:
                    indeg[c[1]] -= 1
                    if indeg[c[1]] == 0:
                        ready.append(c[1])
        assert seen == len(nodes)


def test_depth_counts_longest_path():
    g = Genome(1)
    inn = Innovations(D + 2)
    g.hidden = [200, 201]
    g.bias = {200: 0.0, 201: 0.0, D: 0.0, D + 1: 0.0}
    for src, dst in [(0, 200), (200, 201), (201, D), (1, D + 1)]:
        g.conns[inn.conn(src, dst)] = [src, dst, 1.0, True]
    assert depth_of(g, D) == 3
    g.conns[inn.conn(200, 201)][3] = False
    assert depth_of(g, D) == 1


def test_crossover_takes_structure_from_fitter_parent(tmp_path):
    tr, gs = grown_genomes(tmp_path)
    a, b = gs[0], gs[1]
    c = tr._crossover(a, b)
    assert set(c.conns) == set(a.conns) and c.hidden == a.hidden


def test_speciation_population_size_and_champions(tmp_path):
    tr = tiny(tmp_path)
    random.seed(0)
    fit = torch.randn(50)
    before = {g.gid for g in tr.genomes[PREY]}
    st = tr.next_population(PREY, fit, torch.full((50,), 100.0))
    assert len(tr.genomes[PREY]) == 50
    assert st["species"] >= 1
    best_gid = tr.genomes[PREY][0].gid  # some species champion survives with its id
    assert best_gid in before


def test_train_and_load_champion(tmp_path):
    tr = tiny(tmp_path, learn=(PREY, PREDATOR), p_add_node=0.5)
    hist = tr.train()
    assert hist[-1]["world_steps"] == 24
    for name, sp in (("prey", PREY), ("predator", PREDATOR)):
        pol = load_policy(tmp_path / f"{name}.pt")
        obs = torch.rand(2, 3, D)
        g = tr.champion_genome(sp)
        assert torch.allclose(pol(obs), reference_forward(g, obs), atol=1e-5)
