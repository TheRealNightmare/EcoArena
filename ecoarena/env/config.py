"""World configuration: loaded from configs/*.json, shared with the TypeScript port."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, fields
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass(frozen=True)
class SpeciesConfig:
    init_count: int
    capacity: int
    radius: float
    max_speed: float
    max_accel: float
    max_turn: float
    energy_cap: float
    energy_init: float
    base_drain: float
    move_cost: float
    repro_threshold: float
    repro_cooldown: int
    child_energy: float
    max_age: int
    vision_cone_deg: float
    vision_range: float
    catch_radius: float = 0.0
    energy_per_catch: float = 0.0
    digest_steps: int = 0  # handling time after a catch (Holling type II); predators only
    catch_prob: float = 1.0  # chance a lunge at a prey in range succeeds; predators only
    miss_stun_steps: int = 0  # steps a predator can't catch after a failed lunge
    territory_radius: float = 0.0  # 0 = off. Breeding blocked when crowded by own species
    territory_max: int = 0  # max other same-species agents within territory_radius to still breed


@dataclass(frozen=True)
class WorldConfig:
    world_size: float
    grass_grid: int
    grass_regrow_rate: float
    grass_regrow_floor: float
    grass_init_min: float
    grass_init_max: float
    grass_eat_per_step: float
    grass_energy_per_unit: float
    grass_visible_threshold: float
    refuge_count: int
    refuge_radius: float
    refuge_grass_cap: float
    immigration_floor: int  # 0 = off. Below this many alive, one migrant arrives ...
    immigration_interval: int  # ... every this many steps
    n_rays: int
    ray_neighbors: int
    ray_grass_samples: int
    prey: SpeciesConfig
    predator: SpeciesConfig

    @property
    def cell_size(self) -> float:
        return self.world_size / self.grass_grid

    @property
    def obs_dim(self) -> int:
        # per sector: distance + one-hot(grass, prey, predator, refuge, none);
        # then energy, speed, age, busy (digesting/stunned), grass-under-self, in-refuge
        return self.n_rays * 6 + 6


def _set_dotted(d: dict, key: str, value) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d[p]
    if parts[-1] not in d:
        raise KeyError(f"unknown config key: {key}")
    if isinstance(value, str) and value.startswith("x"):  # relative: "x2" doubles the base value
        old = d[parts[-1]]
        new = old * float(value[1:])
        value = round(new) if isinstance(old, int) else new
    d[parts[-1]] = value


def _from_dict(raw: dict) -> WorldConfig:
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    names = {f.name for f in fields(WorldConfig)}
    unknown = set(raw) - names
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    raw["prey"] = SpeciesConfig(**raw["prey"])
    raw["predator"] = SpeciesConfig(**raw["predator"])
    return WorldConfig(**raw)


def load_config(scenario: str = "S1", base: str | Path = "s1_balanced.json", **overrides) -> WorldConfig:
    """Load the base config, apply a named scenario (S1–S5), then any dotted-key overrides.

    Example: load_config("S2", **{"predator.init_count": 40})
    """
    base_path = Path(base) if Path(base).is_absolute() else CONFIG_DIR / base
    raw = json.loads(base_path.read_text())
    scenarios = json.loads((CONFIG_DIR / "scenarios.json").read_text())
    if scenario not in scenarios:
        raise KeyError(f"unknown scenario {scenario}; have {[k for k in scenarios if not k.startswith('_')]}")
    raw = copy.deepcopy(raw)
    for k, v in {**scenarios[scenario], **overrides}.items():
        _set_dotted(raw, k, v)
    return _from_dict(raw)
