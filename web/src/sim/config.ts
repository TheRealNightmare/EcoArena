// World configuration: the same configs/*.json the Python env reads (one source of truth).
import scenariosRaw from "../../../configs/scenarios.json";
import balancedRaw from "../../../configs/s1_balanced.json";
import largeRaw from "../../../configs/s1_large.json";

export interface SpeciesConfig {
  init_count: number;
  capacity: number;
  radius: number;
  max_speed: number;
  max_accel: number;
  max_turn: number;
  energy_cap: number;
  energy_init: number;
  base_drain: number;
  move_cost: number;
  repro_threshold: number;
  repro_cooldown: number;
  child_energy: number;
  max_age: number;
  vision_cone_deg: number;
  vision_range: number;
  catch_radius: number;
  energy_per_catch: number;
  digest_steps: number;
  catch_prob: number;
  miss_stun_steps: number;
  territory_radius: number;
  territory_max: number;
  drag: number;
}

export interface WorldConfig {
  world_size: number;
  grass_grid: number;
  grass_regrow_rate: number;
  grass_regrow_floor: number;
  grass_init_min: number;
  grass_init_max: number;
  grass_eat_per_step: number;
  grass_energy_per_unit: number;
  grass_visible_threshold: number;
  refuge_count: number;
  refuge_radius: number;
  refuge_grass_cap: number;
  immigration_floor: number;
  immigration_interval: number;
  n_rays: number;
  ray_neighbors: number;
  ray_grass_samples: number;
  prey: SpeciesConfig;
  predator: SpeciesConfig;
}

const SPECIES_DEFAULTS = {
  catch_radius: 0, energy_per_catch: 0, digest_steps: 0, catch_prob: 1, miss_stun_steps: 0,
  territory_radius: 0, territory_max: 0, drag: 0,
};

// JSON cannot tell 200.0 from 200, so integer fields are listed (they match the Python dataclass types).
// A relative "xK" override on an integer field is rounded, as in Python.
const INT_KEYS = new Set([
  "grass_grid", "refuge_count", "immigration_floor", "immigration_interval", "n_rays", "ray_neighbors",
  "ray_grass_samples", "init_count", "capacity", "repro_cooldown", "max_age", "digest_steps",
  "miss_stun_steps", "territory_max",
]);

export const BASES: Record<string, unknown> = { "s1_large.json": largeRaw, "s1_balanced.json": balancedRaw };
export const SCENARIOS = Object.keys(scenariosRaw).filter((k) => !k.startsWith("_"));

function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function setDotted(d: Record<string, any>, key: string, value: unknown): void {
  const parts = key.split(".");
  for (const p of parts.slice(0, -1)) d = d[p];
  const last = parts[parts.length - 1];
  if (!(last in d)) throw new Error(`unknown config key: ${key}`);
  if (typeof value === "string" && value.startsWith("x")) {
    const v = d[last] * parseFloat(value.slice(1));
    value = INT_KEYS.has(last) ? roundHalfEven(v) : v;
  }
  d[last] = value;
}

export function loadConfig(scenario = "S1", base = "s1_large.json", overrides: Record<string, unknown> = {}): WorldConfig {
  const raw = structuredClone(BASES[base]) as Record<string, any>;
  if (!raw) throw new Error(`unknown base ${base}`);
  const sc = (scenariosRaw as unknown as Record<string, Record<string, unknown>>)[scenario];
  if (!sc) throw new Error(`unknown scenario ${scenario}`);
  raw.prey = { ...SPECIES_DEFAULTS, ...raw.prey }; // defaults first, so optional fields can be overridden
  raw.predator = { ...SPECIES_DEFAULTS, ...raw.predator };
  for (const [k, v] of Object.entries({ ...sc, ...overrides })) setDotted(raw, k, v);
  for (const k of Object.keys(raw)) if (k.startsWith("_")) delete raw[k];
  return raw as WorldConfig;
}

export const obsDim = (cfg: WorldConfig) => cfg.n_rays * 6 + 6;
export const cellSize = (cfg: WorldConfig) => cfg.world_size / cfg.grass_grid;
