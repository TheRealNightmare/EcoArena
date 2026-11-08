import math

import pytest
import torch

from ecoarena.env import PREDATOR, PREY, EcoWorld, load_config, rng
from ecoarena.policies import make_policy


def small_world(**overrides):
    overrides = {"refuge_count": 0, **overrides}
    cfg = load_config("S1", **overrides)
    return EcoWorld(cfg, n_worlds=3, seed=7)


def zeros(world):
    return [torch.zeros(world.W, sc.capacity, 2) for sc in world.species_cfg]


# ---------------------------------------------------------------- rng
def test_hash_is_deterministic_and_32bit():
    x = torch.arange(0, 1000, dtype=torch.int64)
    h1, h2 = rng.hash32(x), rng.hash32(x)
    assert torch.equal(h1, h2)
    assert int(h1.min()) >= 0 and int(h1.max()) < 2**32
    assert len(set(h1.tolist())) == 1000  # no collisions on a small range


def test_hash_known_values():
    # pinned so the TypeScript port can check it reproduces the same numbers
    got = rng.hash32(torch.tensor([0, 1, 2, 12345, 2**32 - 1], dtype=torch.int64)).tolist()
    assert got == PINNED_HASHES


# cross-checked against a plain-Python uint32 version and a JavaScript Math.imul version
PINNED_HASHES = [0, 1753845952, 3507691905, 2435775735, 1734902346]


def test_uniform_range_and_mean():
    w, s = rng.grid(4, 5000, "cpu")
    u = rng.uniform(1, 2, 3, w, s)
    assert u.min() >= 0 and u.max() < 1
    assert abs(u.mean().item() - 0.5) < 0.01


# ---------------------------------------------------------------- determinism
def test_same_seed_same_trajectory():
    runs = []
    for _ in range(2):
        w = small_world()
        pp, qp = make_policy("heuristic", w, PREY), make_policy("heuristic", w, PREDATOR)
        for _ in range(60):
            o1, o2 = w.observe()
            w.step(pp(o1), qp(o2))
        runs.append(w)
    for a, b in zip(runs[0].species, runs[1].species):
        assert torch.equal(a.alive, b.alive)
        assert torch.allclose(a.pos, b.pos)
        assert torch.allclose(a.energy, b.energy)
    assert torch.allclose(runs[0].grass, runs[1].grass)


def test_worlds_differ():
    w = small_world()
    assert not torch.allclose(w.species[PREY].pos[0], w.species[PREY].pos[1])


# ---------------------------------------------------------------- mechanics
def test_initial_counts():
    w = small_world()
    pops = w.populations()
    assert (pops[:, 0] == w.cfg.prey.init_count).all()
    assert (pops[:, 1] == w.cfg.predator.init_count).all()


def test_positions_stay_in_world():
    w = small_world()
    a = [torch.ones(w.W, sc.capacity, 2) for sc in w.species_cfg]  # full thrust, full turn
    for _ in range(30):
        w.step(*a)
    for st in w.species:
        assert (st.pos >= 0).all() and (st.pos < w.cfg.world_size).all()


def test_speed_respects_limits():
    w = small_world()
    a = [torch.ones(w.W, sc.capacity, 2) for sc in w.species_cfg]
    for _ in range(30):
        w.step(*a)
    for st, sc in zip(w.species, w.species_cfg):
        assert (st.speed <= sc.max_speed + 1e-6).all()


def test_wrap_min_image():
    w = small_world()
    d = torch.tensor([99.0, -99.0, 51.0, 10.0])
    assert torch.allclose(w.min_image(d), torch.tensor([-1.0, 1.0, -49.0, 10.0]))


def test_starvation_kills():
    w = small_world(**{"prey.base_drain": 200.0, "predator.base_drain": 200.0})
    w.step(*zeros(w))
    assert (w.populations() == 0).all()


def test_old_age_kills():
    w = small_world(**{"prey.max_age": 1, "predator.max_age": 1})
    w.step(*zeros(w))
    assert (w.populations() == 0).all()


def test_capture_one_prey_per_predator_and_once_per_prey():
    # breeding off so the energy check isn't muddied by a same-step birth
    cfg = load_config("S1", **{"prey.init_count": 3, "predator.init_count": 2, "predator.repro_threshold": 1000.0, "refuge_count": 0,
                               "predator.catch_prob": 1.0})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    prey, pred = w.species
    # three prey bunched next to two predators
    prey.pos[0, :3] = torch.tensor([[50.0, 50.0], [50.2, 50.0], [50.4, 50.0]])
    pred.pos[0, :2] = torch.tensor([[50.1, 50.5], [50.3, 50.5]])
    e0 = pred.energy[0, :2].clone()
    info = w.step(*zeros(w))
    assert int(info["captures"][0]) == 2
    assert int(prey.alive[0].sum()) == 1
    assert torch.allclose(pred.energy[0, :2], torch.clamp(e0 + cfg.predator.energy_per_catch, max=100) - cfg.predator.base_drain)


def test_digesting_predator_cannot_catch():
    cfg = load_config("S1", **{"prey.init_count": 2, "predator.init_count": 1, "refuge_count": 0,
                               "predator.digest_steps": 5, "predator.repro_threshold": 1000.0,
                               "predator.catch_prob": 1.0, "predator.miss_stun_steps": 0})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    prey, pred = w.species
    prey.pos[0, :2] = torch.tensor([[50.0, 50.0], [80.0, 80.0]])
    pred.pos[0, 0] = torch.tensor([50.5, 50.0])
    w.step(*zeros(w))
    assert int(prey.alive[0].sum()) == 1
    # move the second prey next to the predator: still digesting, so it survives 5 steps
    for i in range(6):
        prey.pos[0, 1] = pred.pos[0, 0] + torch.tensor([0.3, 0.0])
        prey.speed[0, 1] = 0.0
        pred.speed[0, 0] = 0.0
        w.step(*zeros(w))
        expect_alive = 1 if i < 5 else 0
        assert int(prey.alive[0].sum()) == expect_alive, f"step {i}"


def test_capture_across_wrap_edge():
    cfg = load_config("S1", **{"prey.init_count": 1, "predator.init_count": 1, "refuge_count": 0,
                               "predator.catch_prob": 1.0})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    w.species[PREY].pos[0, 0] = torch.tensor([0.2, 50.0])
    w.species[PREDATOR].pos[0, 0] = torch.tensor([99.6, 50.0])
    w.step(*zeros(w))
    assert int(w.species[PREY].alive[0].sum()) == 0


def test_births_fill_free_slots_and_split_energy():
    cfg = load_config("S1", **{"prey.init_count": 4, "predator.init_count": 1})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    prey = w.species[PREY]
    prey.energy[0, :4] = torch.tensor([90.0, 10.0, 95.0, 30.0])
    prey.pos[0, :4] = torch.tensor([[10.0, 10], [30, 30], [60, 60], [80, 80]])
    w.species[PREDATOR].pos[0, 0] = torch.tensor([5.0, 90.0])
    info = w.step(*zeros(w))
    assert int(info["births"][0][0]) == 2
    assert int(prey.alive[0].sum()) == 6
    _wi, child, par = info["birth_pairs"][PREY]
    assert sorted(par.tolist()) == [0, 2]
    assert sorted(child.tolist()) == [4, 5]
    assert (prey.energy[0, child] == cfg.prey.child_energy).all()
    assert (prey.cooldown[0, par] == cfg.prey.repro_cooldown).all()
    # children appear next to their parent (one body diameter away, with wrap)
    d = w.min_image(prey.pos[0, child] - prey.pos[0, par]).norm(dim=-1)
    assert torch.allclose(d, torch.full_like(d, 2 * cfg.prey.radius), atol=1e-4)


def test_no_birth_without_free_slot():
    cfg = load_config("S1", **{"prey.init_count": 5, "prey.capacity": 5, "predator.init_count": 1})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    w.species[PREY].energy[:] = 95.0
    w.species[PREDATOR].pos[0, 0] = torch.tensor([1.0, 1.0])
    w.species[PREY].pos[0] = torch.tensor([[50.0, 50], [60, 60], [70, 70], [80, 80], [40, 40]])
    info = w.step(*zeros(w))
    assert int(info["births"][PREY][0]) == 0


def test_grass_eaten_and_regrows():
    cfg = load_config("S1", **{"prey.init_count": 1, "predator.init_count": 1, "refuge_count": 0})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    w.species[PREY].pos[0, 0] = torch.tensor([51.0, 51.0])
    w.species[PREDATOR].pos[0, 0] = torch.tensor([5.0, 5.0])
    c = int(w.cell_index(w.species[PREY].pos[0, 0]))
    w.grass[0, c] = 0.5
    e0 = float(w.species[PREY].energy[0, 0])
    w.step(*zeros(w))
    g = 0.5 - cfg.grass_eat_per_step
    g = g + cfg.grass_regrow_rate * g * (1 - g) + cfg.grass_regrow_floor
    assert math.isclose(float(w.grass[0, c]), g, rel_tol=1e-5)
    gained = cfg.grass_eat_per_step * cfg.grass_energy_per_unit - cfg.prey.base_drain
    assert math.isclose(float(w.species[PREY].energy[0, 0]) - e0, gained, rel_tol=1e-5)


# ---------------------------------------------------------------- observations
def test_obs_shape_and_dead_slots_zero():
    w = small_world()
    o1, o2 = w.observe()
    assert o1.shape == (3, w.cfg.prey.capacity, w.cfg.obs_dim)
    assert o2.shape == (3, w.cfg.predator.capacity, w.cfg.obs_dim)
    dead = ~w.species[PREY].alive
    assert (o1[dead] == 0).all()


def test_obs_one_hot_is_valid():
    w = small_world(refuge_count=8)  # refuges on, so all five sector types can appear
    o1, _ = w.observe()
    alive = w.species[PREY].alive
    rays = o1[alive][:, : w.cfg.n_rays * 6].reshape(-1, w.cfg.n_rays, 6)
    assert torch.allclose(rays[..., 1:].sum(-1), torch.ones(rays.shape[:2]))
    assert (rays[..., 0] >= 0).all() and (rays[..., 0] <= 1).all()


def test_ray_sees_predator_straight_ahead():
    cfg = load_config("S1", **{"prey.init_count": 1, "predator.init_count": 1, "grass_init_min": 0.0, "grass_init_max": 0.0, "refuge_count": 0})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    prey, pred = w.species
    prey.pos[0, 0] = torch.tensor([50.0, 50.0])
    prey.heading[0, 0] = 0.0
    pred.pos[0, 0] = torch.tensor([60.0, 50.0])
    o1, _ = w.observe()
    rays = o1[0, 0, : cfg.n_rays * 6].reshape(cfg.n_rays, 6)
    # straight ahead is the boundary between the two middle sectors; the body surface is 10 - radius away
    mid = rays[cfg.n_rays // 2 - 1 : cfg.n_rays // 2 + 1]
    hit = mid[mid[:, 3] > 0.5]
    assert len(hit) >= 1
    expected = (10.0 - cfg.predator.radius) / cfg.prey.vision_range
    assert abs(float(hit[:, 0].min()) - expected) < 1e-4
    # and nothing is reported in the other sectors
    others = torch.cat([rays[: cfg.n_rays // 2 - 1], rays[cfg.n_rays // 2 + 1 :]])
    assert (others[:, 5] == 1).all()


def test_target_between_old_ray_angles_is_still_seen():
    # 8 degrees off-axis, far away: this fell between two thin rays in the first design
    cfg = load_config("S1", **{"prey.init_count": 1, "predator.init_count": 1, "grass_init_min": 0.0, "grass_init_max": 0.0, "refuge_count": 0})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    prey, pred = w.species
    prey.pos[0, 0] = torch.tensor([50.0, 50.0])
    prey.heading[0, 0] = 0.0
    a = math.radians(8)
    pred.pos[0, 0] = torch.tensor([50 + 15 * math.cos(a), 50 + 15 * math.sin(a)])
    o1, _ = w.observe()
    rays = o1[0, 0, : cfg.n_rays * 6].reshape(cfg.n_rays, 6)
    assert (rays[:, 3] > 0.5).sum() == 1


def test_target_behind_prey_outside_cone_is_unseen():
    cfg = load_config("S1", **{"prey.init_count": 1, "predator.init_count": 1, "grass_init_min": 0.0, "grass_init_max": 0.0, "refuge_count": 0})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    prey, pred = w.species
    prey.pos[0, 0] = torch.tensor([50.0, 50.0])
    prey.heading[0, 0] = 0.0
    pred.pos[0, 0] = torch.tensor([45.0, 50.0])  # directly behind; prey cone is 240 deg
    o1, _ = w.observe()
    rays = o1[0, 0, : cfg.n_rays * 6].reshape(cfg.n_rays, 6)
    assert (rays[:, 3] == 0).all()


def test_agents_do_not_see_themselves():
    cfg = load_config("S1", **{"prey.init_count": 1, "predator.init_count": 1, "grass_init_min": 0.0, "grass_init_max": 0.0, "refuge_count": 0})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    w.species[PREDATOR].pos[0, 0] = torch.tensor([5.0, 5.0])
    w.species[PREY].pos[0, 0] = torch.tensor([60.0, 60.0])
    o1, _ = w.observe()
    rays = o1[0, 0, : cfg.n_rays * 6].reshape(cfg.n_rays, 6)
    assert (rays[:, 5] == 1).all()  # all "none"


@pytest.mark.parametrize("scenario", ["S1", "S2", "S3", "S4", "S5"])
def test_scenarios_load(scenario):
    cfg = load_config(scenario)
    EcoWorld(cfg, n_worlds=1, seed=0).step(*zeros(EcoWorld(cfg, n_worlds=1, seed=0)))


# ---------------------------------------------------------------- refuges
def refuge_world():
    cfg = load_config("S1", **{"prey.init_count": 1, "predator.init_count": 1,
                               "predator.repro_threshold": 1000.0, "refuge_count": 4})
    return cfg, EcoWorld(cfg, n_worlds=2, seed=3)


def a_refuge_point(w):
    c = int(torch.nonzero(w.refuge[0])[0])
    G, cs = w.cfg.grass_grid, w.cfg.cell_size
    return torch.tensor([(c // G + 0.5) * cs, (c % G + 0.5) * cs])


def test_refuge_map_exists_and_is_shared_across_worlds():
    cfg, w = refuge_world()
    assert w.refuge.shape == (1, cfg.grass_grid ** 2)
    frac = w.refuge.float().mean().item()
    assert 0.02 < frac < 0.5


def test_prey_in_refuge_cannot_be_caught():
    _, w = refuge_world()
    p = a_refuge_point(w)
    for wi in range(w.W):
        w.species[PREY].pos[wi, 0] = p
        w.species[PREDATOR].pos[wi, 0] = p + torch.tensor([0.3, 0.0])
    assert w.in_refuge(w.species[PREY].pos)[:, 0].all()
    info = w.step(*zeros(w))
    assert int(info["captures"].sum()) == 0
    assert w.species[PREY].alive[:, 0].all()


def test_refuge_grass_is_capped():
    cfg, w = refuge_world()
    for _ in range(5):
        w.step(*zeros(w))
    assert (w.grass[w.refuge.expand(w.W, -1)] <= cfg.refuge_grass_cap + 1e-6).all()


def test_in_refuge_flag_in_observation():
    _, w = refuge_world()
    w.species[PREY].pos[:, 0] = a_refuge_point(w)
    o1, _ = w.observe()
    assert (o1[:, 0, -1] == 1).all()


# ---------------------------------------------------------------- territories and immigration
def test_crowded_predators_cannot_breed():
    cfg = load_config("S1", **{"prey.init_count": 1, "predator.init_count": 3, "refuge_count": 0,
                               "predator.territory_radius": 5.0, "predator.territory_max": 0})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    pred = w.species[PREDATOR]
    pred.energy[0, :3] = 95.0
    pred.pos[0, :3] = torch.tensor([[50.0, 50.0], [52.0, 50.0], [80.0, 80.0]])  # two crowded, one alone
    w.species[PREY].pos[0, 0] = torch.tensor([5.0, 5.0])
    info = w.step(*zeros(w))
    _wi, _child, par = info["birth_pairs"][PREDATOR]
    assert par.tolist() == [2]


def test_immigration_rescues_an_extinct_species():
    cfg = load_config("S1", **{"prey.init_count": 0, "refuge_count": 0,
                               "immigration_floor": 2, "immigration_interval": 10})
    w = EcoWorld(cfg, n_worlds=2, seed=0)
    arrivals = 0
    for _ in range(10):
        info = w.step(*zeros(w))
        arrivals += int(info["migrants"][PREY].sum())
    assert arrivals == 2  # one per world, on step 10
    assert (w.populations()[:, PREY] == 1).all()


def test_immigration_off_by_default():
    w = small_world(**{"prey.init_count": 0})
    for _ in range(100):
        w.step(*zeros(w))
    assert (w.populations()[:, PREY] == 0).all()


def test_scenarios_are_relative_to_the_base():
    small = load_config("S4")
    large = load_config("S4", base="s1_large.json")
    assert small.prey.init_count == 2 * load_config("S1").prey.init_count
    assert large.prey.init_count == 4 * small.prey.init_count
    assert load_config("S2").grass_regrow_rate == load_config("S1").grass_regrow_rate * 0.5


# ---------------------------------------------------------------- catch success
def lunge_world(prob, stun):
    cfg = load_config("S1", **{"prey.init_count": 150, "predator.init_count": 150, "refuge_count": 0,
                               "predator.capacity": 150, "predator.repro_threshold": 1000.0,
                               "prey.repro_threshold": 1000.0,
                               "predator.catch_prob": prob, "predator.miss_stun_steps": stun})
    w = EcoWorld(cfg, n_worlds=4, seed=1)
    prey, pred = w.species
    # pair every predator with its own prey, 0.5 apart, far from other pairs
    for i in range(150):
        x, y = 4.0 + 6.5 * (i % 15), 4.0 + 9.5 * (i // 15)
        prey.pos[:, i] = torch.tensor([x, y])
        pred.pos[:, i] = torch.tensor([x + 0.5, y])
    return w


def test_catch_prob_controls_success_rate():
    w = lunge_world(0.3, 50)
    info = w.step(*zeros(w))
    rate = info["captures"].sum().item() / (4 * 150)
    assert 0.24 < rate < 0.36


def test_missed_predator_is_stunned():
    w = lunge_world(0.3, 50)
    w.step(*zeros(w))
    pred = w.species[PREDATOR]
    alive_prey = w.species[PREY].alive
    # predators whose prey survived missed: stunned for 50 steps
    missed = alive_prey[:, :150]
    assert (pred.digest[missed] == 50).all()
    info = w.step(*zeros(w))
    assert int(info["captures"].sum()) == 0


def test_catch_prob_one_always_succeeds():
    w = lunge_world(1.0, 0)
    info = w.step(*zeros(w))
    assert int(info["captures"].sum()) == 4 * 150


def test_busy_flag_set_after_catch():
    cfg = load_config("S1", **{"prey.init_count": 1, "predator.init_count": 1, "refuge_count": 0,
                               "predator.repro_threshold": 1000.0, "predator.catch_prob": 1.0})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    w.species[PREY].pos[0, 0] = torch.tensor([50.0, 50.0])
    w.species[PREDATOR].pos[0, 0] = torch.tensor([50.5, 50.0])
    _, o2 = w.observe()
    assert o2[0, 0, -3] == 0
    w.step(*zeros(w))
    _, o2 = w.observe()
    qc = cfg.predator
    expected = qc.digest_steps / (max(qc.digest_steps, qc.miss_stun_steps) + 1)
    assert abs(float(o2[0, 0, -3]) - expected) < 1e-6


def test_heuristic_predator_coasts_while_digesting():
    w = small_world()
    pred = w.species[PREDATOR]
    pred.digest[:] = 50
    _, o2 = w.observe()
    a = make_policy("heuristic", w, PREDATOR)(o2)
    assert (a[..., 1][pred.alive] == 0).all()


# ---------------------------------------------------------------- drag world
def test_drag_decays_speed_without_thrust():
    cfg = load_config("S1", **{"prey.init_count": 1, "predator.init_count": 1, "refuge_count": 0,
                               "prey.drag": 0.1, "predator.drag": 0.1})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    w.species[PREY].speed[0, 0] = 1.0
    w.species[PREDATOR].speed[0, 0] = 1.0
    w.step(*zeros(w))
    assert abs(float(w.species[PREY].speed[0, 0]) - 0.9) < 1e-6
    assert abs(float(w.species[PREDATOR].speed[0, 0]) - 0.9) < 1e-6


def test_no_drag_means_free_coasting():
    cfg = load_config("S1", **{"prey.init_count": 1, "predator.init_count": 1, "refuge_count": 0})
    w = EcoWorld(cfg, n_worlds=1, seed=0)
    w.species[PREY].speed[0, 0] = 0.7
    w.step(*zeros(w))
    assert abs(float(w.species[PREY].speed[0, 0]) - 0.7) < 1e-6


def test_drag_world_config_loads():
    cfg = load_config("S1", base="s1_drag.json")
    assert cfg.prey.drag > 0 and cfg.predator.drag > 0
