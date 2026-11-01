"""NEAT (M4): NeuroEvolution of Augmenting Topologies (Stanley & Miikkulainen, 2002), evaluated on the GPU.

What is evolved
---------------
Each genome is a feed-forward graph: 102 input nodes, 2 output nodes (turn, thrust; linear, then clamped to
[-1, 1] like every other method), and up to `max_hidden` tanh hidden nodes. Genomes start minimal (every input
wired straight to both outputs, no hidden nodes, i.e. a linear policy) and grow by mutation:

    weight mutation   perturb each weight with N(0, weight_sigma), or rarely redraw it
    add connection    new edge between two unconnected nodes (never creating a cycle)
    add node          split an enabled edge a->b into a->new->b (old edge disabled)
    toggle            enable/disable an edge

Every structural gene carries an innovation number, so crossover lines genomes up gene by gene, and the
compatibility distance  delta = c_disjoint * (non-matching genes)/N + c_weight * mean|w diff over matching genes|
groups genomes into species. Fitness sharing (offspring per species proportional to its mean fitness) protects
new structure while its weights are still being tuned. The species threshold adapts to keep about
`target_species` species. Stagnant species (no improvement for `stagnation` generations) are dropped, except
the best two.

How it is evaluated (same as GA and ES)
---------------------------------------
Evaluation, fitness, slot assignment and the world-step budget are inherited from the GA unchanged. To run 50
different graphs in one batched pass, each genome is compiled to dense matrices over a fixed node layout
[inputs | hidden slots | outputs]; unused hidden slots have zero weights. Node values are computed by repeated
propagation: after k rounds every node whose longest path from the inputs has k edges or fewer is exact, so
`depth` rounds (the longest path in any genome) give the exact feed-forward output.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..env import WorldConfig
from .ga import SPECIES_NAME, GAConfig, GATrainer


@dataclass
class NEATConfig(GAConfig):
    max_hidden: int = 48
    init_weight_std: float = 0.3
    weight_mutate_rate: float = 0.8  # per gene, when the genome's weights are mutated
    weight_sigma: float = 0.05
    weight_replace_rate: float = 0.01
    bias_sigma: float = 0.05
    p_mutate_weights: float = 0.8
    p_add_conn: float = 0.08
    p_add_node: float = 0.04
    p_toggle: float = 0.01
    p_crossover: float = 0.75
    disabled_inherit: float = 0.75  # child gene disabled if disabled in either parent
    c_disjoint: float = 1.0
    c_weight: float = 0.5
    threshold: float = 0.6  # initial compatibility threshold; adapted each generation
    target_species: int = 5
    survival_frac: float = 0.2  # top fraction of each species allowed to breed
    stagnation: int = 20


# ---------------------------------------------------------------------------------------- genome
@dataclass
class Genome:
    gid: int
    conns: dict[int, list] = field(default_factory=dict)  # innovation -> [src, dst, weight, enabled]
    hidden: list[int] = field(default_factory=list)  # hidden node ids, in creation order
    bias: dict[int, float] = field(default_factory=dict)  # hidden + output node -> bias

    def copy(self, gid: int) -> Genome:
        return Genome(gid, {k: list(v) for k, v in self.conns.items()}, list(self.hidden), dict(self.bias))

    def n_enabled(self) -> int:
        return sum(1 for c in self.conns.values() if c[3])


class Innovations:
    """Global innovation numbers. The same new edge (src, dst) gets the same number in every genome, and the same
    edge split within one generation gets the same new node id, so crossover and speciation line genes up."""

    def __init__(self, first_node: int):
        self.edge: dict[tuple[int, int], int] = {}
        self.split: dict[tuple[int, int], int] = {}
        self.next_node = first_node

    def conn(self, src: int, dst: int) -> int:
        return self.edge.setdefault((src, dst), len(self.edge))

    def node(self, src: int, dst: int, genome: Genome) -> int:
        n = self.split.get((src, dst))
        if n is None or n in genome.hidden:  # a genome may split the same edge twice (after re-enabling it)
            n = self.next_node
            self.next_node += 1
            self.split[(src, dst)] = n
        return n

    def new_generation(self) -> None:
        self.split.clear()


def depth_of(g: Genome, D: int) -> int:
    """Longest path (in edges) from any input to any node, over enabled edges."""
    succ: dict[int, list[int]] = {}
    indeg: dict[int, int] = {n: 0 for n in [*g.hidden, D, D + 1]}
    for src, dst, _w, en in g.conns.values():
        if en:
            succ.setdefault(src, []).append(dst)
            if src >= D:
                indeg[dst] += 1
    depth = {n: (1 if any(c[3] and c[1] == n and c[0] < D for c in g.conns.values()) else 0) for n in indeg}
    ready = [n for n, k in indeg.items() if k == 0]
    while ready:
        n = ready.pop()
        for m in succ.get(n, []):
            depth[m] = max(depth[m], depth[n] + 1)
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
    return max(1, max(depth.values()))


def creates_cycle(g: Genome, src: int, dst: int) -> bool:
    """Would src->dst close a cycle, i.e. can dst already reach src? Disabled edges count too, so re-enabling any
    edge later can never create a cycle."""
    if src == dst:
        return True
    succ: dict[int, list[int]] = {}
    for a, b, _w, _en in g.conns.values():
        succ.setdefault(a, []).append(b)
    stack, seen = [dst], set()
    while stack:
        n = stack.pop()
        if n == src:
            return True
        if n not in seen:
            seen.add(n)
            stack.extend(succ.get(n, []))
    return False


def distance(a: Genome, b: Genome, cfg: NEATConfig) -> float:
    ka, kb = a.conns.keys(), b.conns.keys()
    match = ka & kb
    non_matching = len(ka) + len(kb) - 2 * len(match)
    n = max(len(ka), len(kb), 1)
    wdiff = sum(abs(a.conns[k][2] - b.conns[k][2]) for k in match) / max(len(match), 1)
    return cfg.c_disjoint * non_matching / n + cfg.c_weight * wdiff


def compile_population(genomes: list[Genome], D: int, H: int, device) -> tuple[torch.Tensor, ...]:
    """Dense matrices for batched evaluation. Node layout per genome: inputs 0..D-1, hidden slots D..D+H-1,
    outputs D+H, D+H+1. Returns Win [P, D, H+2], Wh [P, H+2, H+2], b [P, H+2], depth (int)."""
    P = len(genomes)
    Win = torch.zeros(P, D, H + 2)
    Wh = torch.zeros(P, H + 2, H + 2)
    b = torch.zeros(P, H + 2)
    depth = 1
    for p, g in enumerate(genomes):
        local = {n: i for i, n in enumerate(g.hidden)} | {D: H, D + 1: H + 1}
        for n, v in g.bias.items():
            b[p, local[n]] = v
        for src, dst, w, en in g.conns.values():
            if not en:
                continue
            if src < D:
                Win[p, src, local[dst]] += w
            else:
                Wh[p, local[src], local[dst]] += w
        depth = max(depth, depth_of(g, D))
    return Win.to(device), Wh.to(device), b.to(device), depth


def propagate(x: torch.Tensor, Win: torch.Tensor, Wh: torch.Tensor, b: torch.Tensor, depth: int) -> torch.Tensor:
    """x [P, B, D] -> actions [P, B, 2]. Hidden nodes tanh, outputs linear then clamped."""
    H = Wh.shape[1] - 2
    zin = torch.bmm(x, Win) + b.unsqueeze(1)
    h = torch.zeros_like(zin)
    for _ in range(depth):
        z = zin + torch.bmm(h, Wh)
        h = torch.cat([torch.tanh(z[..., :H]), z[..., H:]], dim=-1)
    return torch.clamp(h[..., H:], -1.0, 1.0)


class NEATPolicy:
    """A compiled NEAT genome as policy(obs) -> actions, for evaluation and export."""

    def __init__(self, Win, Wh, b, depth):
        self.Win, self.Wh, self.b, self.depth = Win, Wh, b, depth

    @classmethod
    def from_checkpoint(cls, ck: dict, device="cpu") -> NEATPolicy:
        return cls(ck["Win"].to(device), ck["Wh"].to(device), ck["b"].to(device), ck["depth"])

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        shape = obs.shape[:-1]
        x = obs.reshape(1, -1, obs.shape[-1])
        return propagate(x, self.Win, self.Wh, self.b, self.depth).reshape(*shape, 2)


# ---------------------------------------------------------------------------------------- trainer
class NEATTrainer(GATrainer):
    def __init__(self, world_cfg: WorldConfig, cfg: NEATConfig, out_dir: str | Path):
        if cfg.assign != "mixed":
            raise ValueError("NEAT supports only mixed assignment")
        super().__init__(world_cfg, cfg, out_dir)
        self.D = D = world_cfg.obs_dim
        self.rng = random.Random(cfg.seed)
        self.innov = Innovations(first_node=D + 2)
        self.genomes: dict[int, list[Genome]] = {}
        self.species: dict[int, list[dict]] = {}
        self.threshold = {sp: cfg.threshold for sp in cfg.learn}
        self.next_species = 0
        for sp in cfg.learn:
            gs = []
            for _ in range(cfg.population):
                g = Genome(self._new_id())
                for out in (D, D + 1):
                    g.bias[out] = 0.0
                    for i in range(D):
                        g.conns[self.innov.conn(i, out)] = [i, out, self.rng.gauss(0, cfg.init_weight_std), True]
                gs.append(g)
            self.genomes[sp] = gs
            self.species[sp] = []
            self._compile(sp)

    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def _compile(self, sp: int) -> None:
        self.pop[sp] = compile_population(self.genomes[sp], self.D, self.cfg.max_hidden, self.device)

    def _actions(self, sp: int, obs: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        Win, Wh, b, depth = self.pop[sp]
        P = self.cfg.population
        W, N, D = obs.shape
        x = obs.reshape(W, N // P, P, D).permute(2, 0, 1, 3).reshape(P, W * (N // P), D)
        a = propagate(x, Win[perm], Wh[perm], b[perm], depth)
        return a.reshape(P, W, N // P, 2).permute(1, 2, 0, 3).reshape(W, N, 2)

    # ------------------------------------------------------------------ mutation / crossover
    def _mutate(self, g: Genome) -> None:
        cfg, r, D = self.cfg, self.rng, self.D
        if r.random() < cfg.p_mutate_weights:
            for c in g.conns.values():
                if r.random() < cfg.weight_replace_rate:
                    c[2] = r.gauss(0, cfg.init_weight_std)
                elif r.random() < cfg.weight_mutate_rate:
                    c[2] += r.gauss(0, cfg.weight_sigma)
            for n in g.bias:
                if r.random() < cfg.weight_mutate_rate:
                    g.bias[n] += r.gauss(0, cfg.bias_sigma)
        if r.random() < cfg.p_add_node and len(g.hidden) < cfg.max_hidden:
            enabled = [k for k, c in g.conns.items() if c[3]]
            if enabled:
                src, dst, w, _ = g.conns[r.choice(enabled)]
                g.conns[self.innov.conn(src, dst)][3] = False
                n = self.innov.node(src, dst, g)
                g.hidden.append(n)
                g.bias[n] = 0.0
                g.conns[self.innov.conn(src, n)] = [src, n, 1.0, True]
                g.conns[self.innov.conn(n, dst)] = [n, dst, w, True]
        if r.random() < cfg.p_add_conn:
            sources = list(range(D)) + g.hidden
            targets = g.hidden + [D, D + 1]
            existing = {(c[0], c[1]) for c in g.conns.values()}
            for _ in range(20):
                src, dst = r.choice(sources), r.choice(targets)
                if (src, dst) not in existing and not creates_cycle(g, src, dst):
                    g.conns[self.innov.conn(src, dst)] = [src, dst, r.gauss(0, cfg.init_weight_std), True]
                    break
        if r.random() < cfg.p_toggle and g.conns:
            # safe either way: cycle checks count disabled edges too, so the full edge set is always acyclic
            c = g.conns[r.choice(list(g.conns))]
            c[3] = not c[3]

    def _crossover(self, fit: Genome, other: Genome) -> Genome:
        """Structure from the fitter parent (so the child stays acyclic); matching genes from either."""
        r = self.rng
        child = fit.copy(self._new_id())
        for k, c in child.conns.items():
            o = other.conns.get(k)
            if o is not None:
                if r.random() < 0.5:
                    c[2] = o[2]
                if not (c[3] and o[3]):
                    c[3] = r.random() >= self.cfg.disabled_inherit
        for n in child.bias:
            if n in other.bias and r.random() < 0.5:
                child.bias[n] = other.bias[n]
        return child

    # ------------------------------------------------------------------ speciation + reproduction
    def _speciate(self, sp: int, genomes: list[Genome]) -> list[dict]:
        cfg = self.cfg
        old = self.species[sp]
        new = [{"id": s["id"], "rep": s["rep"], "members": [], "best": s["best"], "since": s["since"]} for s in old]
        for g in genomes:
            for s in new:
                if distance(g, s["rep"], cfg) < self.threshold[sp]:
                    s["members"].append(g)
                    break
            else:
                self.next_species += 1
                new.append({"id": self.next_species, "rep": g, "members": [g], "best": -math.inf, "since": 0})
        new = [s for s in new if s["members"]]
        for s in new:
            s["rep"] = self.rng.choice(s["members"])
        n = len(new)  # steer the threshold toward target_species
        if n < cfg.target_species:
            self.threshold[sp] *= 0.9
        elif n > cfg.target_species:
            self.threshold[sp] *= 1.1
        return new

    def next_population(self, sp: int, fitness: torch.Tensor, samples: torch.Tensor) -> dict:
        cfg, r = self.cfg, self.rng
        P = cfg.population
        gs = self.genomes[sp]
        f = fitness.cpu().tolist()
        ok = [(s >= cfg.min_samples) and not math.isnan(x) for s, x in zip(samples.tolist(), f, strict=True)]
        valid = [x for x, o in zip(f, ok, strict=True) if o]
        floor = min(valid) if valid else 0.0
        score = {g.gid: (f[j] if ok[j] else floor) for j, g in enumerate(gs)}
        for j, g in enumerate(gs):
            if ok[j]:
                self.fit_hist[sp].setdefault(g.gid, []).append(f[j])

        species = self._speciate(sp, gs)
        for s in species:
            s["members"].sort(key=lambda g: score[g.gid], reverse=True)
            top = score[s["members"][0].gid]
            if top > s["best"]:
                s["best"], s["since"] = top, 0
            else:
                s["since"] += 1
        species.sort(key=lambda s: score[s["members"][0].gid], reverse=True)
        species = [s for i, s in enumerate(species) if i < 2 or s["since"] < cfg.stagnation]

        # offspring quotas from shared fitness: species mean of (score - global min)
        lo = min(score.values())
        adj = [sum(score[g.gid] - lo for g in s["members"]) / len(s["members"]) + 1e-9 for s in species]
        total = sum(adj)
        quota = [max(1, round(P * a / total)) for a in adj]
        while sum(quota) > P:
            quota[quota.index(max(quota))] -= 1
        while sum(quota) < P:
            quota[adj.index(max(adj))] += 1

        nxt: list[Genome] = []
        self.innov.new_generation()
        for s, q in zip(species, quota, strict=True):
            if q <= 0:
                continue
            members = s["members"]
            nxt.append(members[0])  # species champion survives unchanged (keeps its id and history)
            pool = members[: max(2, math.ceil(cfg.survival_frac * len(members)))]
            for _ in range(q - 1):
                a = r.choice(pool)
                if len(pool) > 1 and r.random() < cfg.p_crossover:
                    b = r.choice(pool)
                    if score[b.gid] > score[a.gid]:
                        a, b = b, a
                    child = self._crossover(a, b)
                else:
                    child = a.copy(self._new_id())
                self._mutate(child)
                nxt.append(child)
        self.genomes[sp] = nxt[:P]
        self.species[sp] = species
        self._compile(sp)

        hidden = [len(g.hidden) for g in nxt]
        return {
            "best": max(valid) if valid else float("nan"),
            "mean": sum(valid) / len(valid) if valid else float("nan"),
            "evaluated": len(valid),
            "species": len(species),
            "threshold": self.threshold[sp],
            "hidden_mean": sum(hidden) / len(hidden),
            "hidden_max": max(hidden),
            "conns_mean": sum(g.n_enabled() for g in nxt) / len(nxt),
            "depth": self.pop[sp][3],
        }

    def champion_genome(self, sp: int) -> Genome:
        """Best mean fitness over all evaluations, among genomes in the current population with >= 2 evaluations
        (species champions carry over, so they collect several); falls back to the best single evaluation."""
        gs = self.genomes[sp]
        hist = self.fit_hist[sp]

        def key(g):
            h = hist.get(g.gid, [])
            return (len(h) >= 2, sum(h) / len(h) if h else -math.inf)

        return max(gs, key=key)

    def champion(self, sp: int):  # not a flat vector for NEAT
        return self.champion_genome(sp)

    def save(self) -> None:
        for sp in self.cfg.learn:
            g = self.champion_genome(sp)
            Win, Wh, b, depth = compile_population([g], self.D, self.cfg.max_hidden, "cpu")
            name = SPECIES_NAME[sp]
            torch.save({"method": "neat", "obs_dim": self.D, "species": name, "Win": Win[0:1], "Wh": Wh[0:1],
                        "b": b[0:1], "depth": depth, "neat_config": asdict(self.cfg)}, self.out / f"{name}.pt")
            (self.out / f"{name}_genome.json").write_text(json.dumps({
                "hidden": g.hidden, "bias": {str(k): v for k, v in g.bias.items()},
                "conns": [[k, *c] for k, c in sorted(g.conns.items())],
                "n_hidden": len(g.hidden), "n_enabled": g.n_enabled(), "depth": depth}))
        (self.out / "history.json").write_text(json.dumps(self.history))
