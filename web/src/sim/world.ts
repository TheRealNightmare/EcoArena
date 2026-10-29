// EcoArena world, ported from ecoarena/env/world.py for one world (world index 0).
//
// The step order and every formula follow the Python file line by line; see its docstring. State is kept in
// float32 typed arrays and the main intermediate results are rounded to float32 (Math.fround), so a run tracks
// the PyTorch float32 run closely (the parity test checks this). Neighbour searches use a uniform grid instead
// of PyTorch's all-pairs tensors; ties are broken by lowest slot index, as torch.min does.
import { cellSize, obsDim, type SpeciesConfig, type WorldConfig } from "./config";
import * as rng from "./rng";

export const PREY = 0;
export const PREDATOR = 1;

const f = Math.fround;
const PI32 = f(Math.PI);
const TWO_PI32 = f(2 * Math.PI);

export function fmodPos(x: number, b: number): number {
  let r = x % b;
  if (r !== 0 && r < 0 !== b < 0) r += b;
  return f(r);
}

export function wrapAngle(a: number): number {
  return f(fmodPos(f(a + PI32), TWO_PI32) - PI32);
}

function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export interface SpeciesState {
  n: number;
  x: Float32Array;
  y: Float32Array;
  heading: Float32Array;
  speed: Float32Array;
  energy: Float32Array;
  age: Int32Array;
  cooldown: Int32Array;
  digest: Int32Array;
  alive: Uint8Array;
}

export interface StepInfo {
  t: number;
  captures: number;
  deaths: [number, number];
  births: [number, number];
  eaten: Uint8Array; // prey slots caught this step
  died: [Uint8Array, Uint8Array];
  newborn: [number[], number[]]; // child slots born this step, per species
  catches: [number, number][]; // (predator slot, prey slot) for each successful catch this step
}

/** Uniform bucket grid over the torus for neighbour queries. */
class Grid {
  dims: number;
  cs: number;
  start: Int32Array;
  items: Int32Array;
  count = 0;
  constructor(public L: number, cellTarget: number, cap: number) {
    this.dims = Math.max(1, Math.floor(L / cellTarget));
    this.cs = L / this.dims;
    this.start = new Int32Array(this.dims * this.dims + 1);
    this.items = new Int32Array(cap);
  }
  cellOf(x: number, y: number): number {
    const d = this.dims;
    const i = Math.min(d - 1, Math.max(0, Math.floor(x / this.cs)));
    const j = Math.min(d - 1, Math.max(0, Math.floor(y / this.cs)));
    return i * d + j;
  }
  build(xs: ArrayLike<number>[], ys: ArrayLike<number>[], alives: Uint8Array[], filters?: (Uint8Array | null)[]): void {
    const nc = this.dims * this.dims;
    const counts = new Int32Array(nc + 1);
    const cells: number[] = [];
    let k = 0;
    let base = 0;
    for (let g = 0; g < xs.length; g++) {
      const n = alives[g].length;
      for (let i = 0; i < n; i++) {
        if (!alives[g][i] || (filters && filters[g] && !filters[g]![i])) continue;
        const c = this.cellOf(xs[g][i], ys[g][i]);
        counts[c + 1]++;
        cells.push(c, base + i);
        k++;
      }
      base += n;
    }
    for (let c = 0; c < nc; c++) counts[c + 1] += counts[c];
    this.start.set(counts);
    const fill = counts.slice(0, nc);
    if (this.items.length < k) this.items = new Int32Array(k);
    for (let m = 0; m < cells.length; m += 2) this.items[fill[cells[m]]++] = cells[m + 1];
    this.count = k;
  }
  /** Writes the cell ids within `radius` of (x, y) into out (torus, each cell once); returns how many, or -1
   * meaning "every cell" (window as wide as the grid). */
  cellsNear(x: number, y: number, radius: number, out: Int32Array): number {
    const d = this.dims;
    const span = Math.ceil(radius / this.cs);
    if (2 * span + 1 >= d) return -1;
    const ci = Math.min(d - 1, Math.max(0, Math.floor(x / this.cs)));
    const cj = Math.min(d - 1, Math.max(0, Math.floor(y / this.cs)));
    let n = 0;
    for (let di = -span; di <= span; di++) {
      const i = (ci + di + d) % d;
      for (let dj = -span; dj <= span; dj++) out[n++] = i * d + ((cj + dj + d) % d);
    }
    return n;
  }

  /** Calls fn(id) for every item in cells within `radius` (torus), each cell once. */
  query(x: number, y: number, radius: number, fn: (id: number) => void): void {
    const d = this.dims;
    const span = Math.ceil(radius / this.cs);
    const ci = Math.min(d - 1, Math.max(0, Math.floor(x / this.cs)));
    const cj = Math.min(d - 1, Math.max(0, Math.floor(y / this.cs)));
    if (2 * span + 1 >= d) {
      for (let m = 0; m < this.count; m++) fn(this.items[m]);
      return;
    }
    for (let di = -span; di <= span; di++) {
      const i = (ci + di + d) % d;
      for (let dj = -span; dj <= span; dj++) {
        const c = i * d + ((cj + dj + d) % d);
        for (let m = this.start[c]; m < this.start[c + 1]; m++) fn(this.items[m]);
      }
    }
  }
}

export class World {
  cfg: WorldConfig;
  seed: number;
  t = 0;
  sc: [SpeciesConfig, SpeciesConfig];
  species!: [SpeciesState, SpeciesState];
  grass!: Float32Array;
  refuge!: Uint8Array;
  refugeCentres: [number, number][] = [];
  rayOffsets: [Float32Array, Float32Array];
  readonly L: number;
  readonly G: number;
  readonly cs: number;
  readonly D: number;

  constructor(cfg: WorldConfig, seed = 0) {
    this.cfg = cfg;
    this.seed = seed >>> 0;
    this.sc = [cfg.prey, cfg.predator];
    this.L = cfg.world_size;
    this.G = cfg.grass_grid;
    this.cs = cellSize(cfg);
    this.D = obsDim(cfg);
    this.rayOffsets = [this.makeRayOffsets(cfg.prey.vision_cone_deg), this.makeRayOffsets(cfg.predator.vision_cone_deg)];
    this.reset();
  }

  private makeRayOffsets(coneDeg: number): Float32Array {
    const R = this.cfg.n_rays;
    const rad = (coneDeg * Math.PI) / 180;
    const width = f(rad / R);
    const out = new Float32Array(R);
    for (let r = 0; r < R; r++) out[r] = f(f((r + 0.5) * width) - f(rad / 2));
    return out;
  }

  private stream(base: number, sp: number): number {
    return base + sp * rng.SPECIES_STREAM_OFFSET;
  }

  private initSpecies(sp: number): SpeciesState {
    const sc = this.sc[sp];
    const N = sc.capacity;
    const L = this.L;
    const s: SpeciesState = {
      n: N, x: new Float32Array(N), y: new Float32Array(N), heading: new Float32Array(N), speed: new Float32Array(N),
      energy: new Float32Array(N), age: new Int32Array(N), cooldown: new Int32Array(N), digest: new Int32Array(N),
      alive: new Uint8Array(N),
    };
    const u = (stream: number, i: number) => rng.uniform(this.seed, 0, this.stream(stream, sp), 0, i);
    for (let i = 0; i < N; i++) {
      s.x[i] = f(u(rng.S_INIT_X, i) * L);
      s.y[i] = f(u(rng.S_INIT_Y, i) * L);
      s.heading[i] = f(f(u(rng.S_INIT_HEADING, i) * TWO_PI32) - PI32);
      const alive = i < sc.init_count;
      s.alive[i] = alive ? 1 : 0;
      if (alive) {
        s.age[i] = Math.trunc(f(u(rng.S_INIT_AGE, i) * f(sc.max_age * 0.5)));
        s.energy[i] = Math.min(f(sc.energy_init * f(0.5 + u(rng.S_INIT_ENERGY, i))), sc.energy_cap);
      }
    }
    return s;
  }

  minImage(d: number): number {
    const L = this.L;
    if (d <= L / 2 && d >= -L / 2) return d; // round(d / L) == 0 (ties go to even, i.e. 0)
    return f(d - f(L * roundHalfEven(f(d / L))));
  }

  cellIndex(x: number, y: number): number {
    const G = this.G;
    const i = Math.min(G - 1, Math.max(0, Math.trunc(f(x / this.cs))));
    const j = Math.min(G - 1, Math.max(0, Math.trunc(f(y / this.cs))));
    return i * G + j;
  }

  reset(): void {
    const cfg = this.cfg;
    this.t = 0;
    this.species = [this.initSpecies(PREY), this.initSpecies(PREDATOR)];
    const G = this.G;
    this.grass = new Float32Array(G * G);
    for (let c = 0; c < G * G; c++) {
      const g = rng.uniform(this.seed, 0, rng.S_GRASS_INIT, 0, c);
      this.grass[c] = f(cfg.grass_init_min + f(g * f(cfg.grass_init_max - cfg.grass_init_min)));
    }
    this.refuge = new Uint8Array(G * G);
    this.refugeCentres = [];
    for (let k = 0; k < cfg.refuge_count; k++) {
      this.refugeCentres.push([
        f(rng.uniform(this.seed, 0, rng.S_REFUGE_X, 0, k) * this.L),
        f(rng.uniform(this.seed, 0, rng.S_REFUGE_Y, 0, k) * this.L),
      ]);
    }
    const cs = f(this.cs);
    for (let i = 0; i < G; i++) {
      const cx = f(f(i + 0.5) * cs);
      for (let j = 0; j < G; j++) {
        const cy = f(f(j + 0.5) * cs);
        for (const [rx, ry] of this.refugeCentres) {
          const dx = this.minImage(f(cx - rx));
          const dy = this.minImage(f(cy - ry));
          if (f(Math.sqrt(f(f(dx * dx) + f(dy * dy)))) <= cfg.refuge_radius) {
            this.refuge[i * G + j] = 1;
            break;
          }
        }
      }
    }
    this.capRefugeGrass();
  }

  private capRefugeGrass(): void {
    const cap = f(this.cfg.refuge_grass_cap);
    for (let c = 0; c < this.grass.length; c++) if (this.refuge[c] && this.grass[c] > cap) this.grass[c] = cap;
  }

  populations(): [number, number] {
    const count = (a: Uint8Array) => {
      let n = 0;
      for (let i = 0; i < a.length; i++) n += a[i];
      return n;
    };
    return [count(this.species[0].alive), count(this.species[1].alive)];
  }

  inRefuge(x: number, y: number): boolean {
    return this.refuge[this.cellIndex(x, y)] === 1;
  }

  // ------------------------------------------------------------------ step
  step(preyAct: Float32Array, predAct: Float32Array): StepInfo {
    const cfg = this.cfg;
    const L = this.L;
    const acts = [preyAct, predAct];
    const thrusts: Float32Array[] = [];

    // 1-2. actions and movement
    for (let sp = 0; sp < 2; sp++) {
      const st = this.species[sp];
      const sc = this.sc[sp];
      const a = acts[sp];
      const th = new Float32Array(st.n);
      const maxTurn = f(sc.max_turn);
      const maxAccel = f(sc.max_accel);
      for (let i = 0; i < st.n; i++) {
        const turn = f(Math.min(1, Math.max(-1, a[2 * i])));
        const thrust = f(Math.min(1, Math.max(-1, a[2 * i + 1])));
        const al = st.alive[i];
        st.heading[i] = wrapAngle(f(st.heading[i] + f(f(turn * maxTurn) * al)));
        st.speed[i] = f(Math.min(sc.max_speed, Math.max(0, f(st.speed[i] + f(thrust * maxAccel)))) * al);
        const vx = f(f(Math.cos(st.heading[i])) * st.speed[i]);
        const vy = f(f(Math.sin(st.heading[i])) * st.speed[i]);
        st.x[i] = fmodPos(f(st.x[i] + vx), L);
        st.y[i] = fmodPos(f(st.y[i] + vy), L);
        th[i] = f(thrust * al);
      }
      thrusts.push(th);
    }

    // 3. prey eat grass (cell shared equally by all prey standing on it)
    const prey = this.species[PREY];
    const pc = cfg.prey;
    const G2 = this.G * this.G;
    const counts = new Float32Array(G2);
    const ci = new Int32Array(prey.n);
    for (let i = 0; i < prey.n; i++) {
      ci[i] = this.cellIndex(prey.x[i], prey.y[i]);
      if (prey.alive[i]) counts[ci[i]] += 1;
    }
    const consumed = new Float32Array(G2);
    const eat = f(cfg.grass_eat_per_step);
    for (let c = 0; c < G2; c++) if (counts[c] > 0) consumed[c] = Math.min(this.grass[c], f(counts[c] * eat));
    const perUnit = f(cfg.grass_energy_per_unit);
    for (let i = 0; i < prey.n; i++) {
      if (!prey.alive[i]) continue;
      const per = f(consumed[ci[i]] / Math.max(counts[ci[i]], 1));
      prey.energy[i] = Math.min(f(prey.energy[i] + f(per * perUnit)), pc.energy_cap);
    }
    for (let c = 0; c < G2; c++) if (consumed[c] > 0) this.grass[c] = f(this.grass[c] - consumed[c]);

    // 4. captures
    const { eaten, captures, catches } = this.captures();

    // 5. drain, ageing, cooldowns
    for (let sp = 0; sp < 2; sp++) {
      const st = this.species[sp];
      const sc = this.sc[sp];
      const th = thrusts[sp];
      const base = f(sc.base_drain);
      const mc = f(sc.move_cost);
      for (let i = 0; i < st.n; i++) {
        if (st.alive[i]) {
          st.energy[i] = f(st.energy[i] - f(base + f(f(mc * th[i]) * th[i])));
          st.age[i] += 1;
        }
        if (st.cooldown[i] > 0) st.cooldown[i] -= 1;
        if (st.digest[i] > 0) st.digest[i] -= 1;
      }
    }

    // 6. deaths
    const deaths: [number, number] = [0, 0];
    const died: [Uint8Array, Uint8Array] = [new Uint8Array(prey.n), new Uint8Array(this.species[1].n)];
    for (let sp = 0; sp < 2; sp++) {
      const st = this.species[sp];
      const sc = this.sc[sp];
      for (let i = 0; i < st.n; i++) {
        if (!st.alive[i]) continue;
        const dead = st.energy[i] <= 0 || st.age[i] >= sc.max_age || (sp === PREY && eaten[i] === 1);
        if (dead) {
          st.alive[i] = 0;
          st.speed[i] = 0;
          deaths[sp]++;
          died[sp][i] = 1;
        }
      }
    }

    // 7. births, 8. immigration
    const newborn: [number[], number[]] = [[], []];
    const births: [number, number] = [this.births(PREY, newborn[0]), this.births(PREDATOR, newborn[1])];
    this.immigrate(PREY);
    this.immigrate(PREDATOR);

    // 9. grass regrowth (logistic + floor), refuge cap
    const r = f(cfg.grass_regrow_rate);
    const fl = f(cfg.grass_regrow_floor);
    for (let c = 0; c < G2; c++) {
      const g = this.grass[c];
      this.grass[c] = Math.min(1, Math.max(0, f(f(g + f(f(r * g) * f(1 - g))) + fl)));
    }
    this.capRefugeGrass();

    this.t += 1;
    return { t: this.t, captures, deaths, births, eaten, died, newborn, catches };
  }

  private predGrid?: Grid;

  private captures(): { eaten: Uint8Array; captures: number; catches: [number, number][] } {
    const [prey, pred] = this.species;
    const qc = this.cfg.predator;
    const r2 = f(qc.catch_radius * qc.catch_radius);
    const hunting = new Uint8Array(pred.n);
    for (let q = 0; q < pred.n; q++) hunting[q] = pred.alive[q] && pred.digest[q] <= 0 ? 1 : 0;
    this.predGrid ??= new Grid(this.L, Math.max(qc.catch_radius, 4), pred.n);
    const grid = this.predGrid;
    grid.build([pred.x], [pred.y], [hunting]);
    // each exposed prey picks its nearest hunting predator in range (ties: lowest predator slot)
    const bestD = new Float64Array(pred.n).fill(Infinity);
    const bestP = new Int32Array(pred.n).fill(-1);
    for (let p = 0; p < prey.n; p++) {
      if (!prey.alive[p] || this.inRefuge(prey.x[p], prey.y[p])) continue;
      let dmin = Infinity;
      let qarg = -1;
      grid.query(prey.x[p], prey.y[p], qc.catch_radius, (q) => {
        const dx = this.minImage(f(prey.x[p] - pred.x[q]));
        const dy = this.minImage(f(prey.y[p] - pred.y[q]));
        const d2 = f(f(dx * dx) + f(dy * dy));
        if (d2 <= r2 && (d2 < dmin || (d2 === dmin && q < qarg))) {
          dmin = d2;
          qarg = q;
        }
      });
      // each predator keeps its nearest candidate prey (ties: lowest prey slot, since p ascends)
      if (qarg >= 0 && dmin < bestD[qarg]) {
        bestD[qarg] = dmin;
        bestP[qarg] = p;
      }
    }
    const eaten = new Uint8Array(prey.n);
    let captures = 0;
    const catches: [number, number][] = [];
    const stream = this.stream(rng.S_CATCH, PREDATOR);
    for (let q = 0; q < pred.n; q++) {
      if (bestP[q] < 0) continue;
      const success = qc.catch_prob >= 1 || rng.uniform(this.seed, this.t, stream, 0, q) < qc.catch_prob;
      if (success) {
        eaten[bestP[q]] = 1;
        catches.push([q, bestP[q]]);
        pred.energy[q] = Math.min(f(pred.energy[q] + qc.energy_per_catch), qc.energy_cap);
        pred.digest[q] = qc.digest_steps + 1; // +1: step 5 decrements in this same step
        captures++;
      } else {
        pred.digest[q] = qc.miss_stun_steps + 1;
      }
    }
    return { eaten, captures, catches };
  }

  private births(sp: number, newborn: number[] = []): number {
    const st = this.species[sp];
    const sc = this.sc[sp];
    const N = st.n;
    const parents: number[] = [];
    let grid: Grid | null = null;
    const tr2 = f(sc.territory_radius * sc.territory_radius);
    if (sc.territory_radius > 0) {
      grid = new Grid(this.L, Math.max(sc.territory_radius, 4), N);
      grid.build([st.x], [st.y], [st.alive]);
    }
    for (let i = 0; i < N; i++) {
      if (!st.alive[i] || st.energy[i] < sc.repro_threshold || st.cooldown[i] > 0) continue;
      if (grid) {
        let crowd = -1; // exclude self
        grid.query(st.x[i], st.y[i], sc.territory_radius, (j) => {
          const dx = this.minImage(f(st.x[j] - st.x[i]));
          const dy = this.minImage(f(st.y[j] - st.y[i]));
          if (f(f(dx * dx) + f(dy * dy)) <= tr2) crowd++;
        });
        if (crowd > sc.territory_max) continue;
      }
      parents.push(i);
    }
    const free: number[] = [];
    for (let i = 0; i < N && free.length < parents.length; i++) if (!st.alive[i]) free.push(i);
    const k = Math.min(parents.length, free.length);
    const off = f(2 * sc.radius);
    for (let m = 0; m < k; m++) {
      const child = free[m];
      const par = parents[m];
      const u = (stream: number) => rng.uniform(this.seed, this.t, this.stream(stream, sp), 0, child);
      const ang = f(u(rng.S_BIRTH_ANGLE) * TWO_PI32);
      st.x[child] = fmodPos(f(st.x[par] + f(f(Math.cos(ang)) * off)), this.L);
      st.y[child] = fmodPos(f(st.y[par] + f(f(Math.sin(ang)) * off)), this.L);
      st.heading[child] = f(f(u(rng.S_BIRTH_HEADING) * TWO_PI32) - PI32);
      st.speed[child] = 0;
      st.energy[child] = sc.child_energy;
      st.energy[par] = f(st.energy[par] - sc.child_energy);
      st.age[child] = 0;
      st.cooldown[child] = sc.repro_cooldown;
      st.cooldown[par] = sc.repro_cooldown;
      st.digest[child] = 0;
      st.alive[child] = 1;
      newborn.push(child);
    }
    return k;
  }

  private immigrate(sp: number): void {
    const cfg = this.cfg;
    if (cfg.immigration_floor <= 0 || (this.t + 1) % cfg.immigration_interval !== 0) return;
    const st = this.species[sp];
    const sc = this.sc[sp];
    let n = 0;
    let slot = -1;
    for (let i = 0; i < st.n; i++) {
      n += st.alive[i];
      if (!st.alive[i] && slot < 0) slot = i;
    }
    if (n >= cfg.immigration_floor || slot < 0) return;
    const u = (stream: number) => rng.uniform(this.seed, this.t, this.stream(stream, sp), 0, slot);
    st.x[slot] = f(u(rng.S_MIGRANT_X) * this.L);
    st.y[slot] = f(u(rng.S_MIGRANT_Y) * this.L);
    st.heading[slot] = f(f(u(rng.S_MIGRANT_HEADING) * TWO_PI32) - PI32);
    st.speed[slot] = 0;
    st.energy[slot] = sc.energy_init;
    st.age[slot] = 0;
    st.cooldown[slot] = 0;
    st.digest[slot] = 0;
    st.alive[slot] = 1;
  }

  // ------------------------------------------------------------------ observations
  private obsGrid?: Grid;

  /** Observations for (prey, predator): Float32Array [N * D] each, zeros for free slots. */
  observe(): [Float32Array, Float32Array] {
    const [prey, pred] = this.species;
    const reachMax = Math.max(this.cfg.prey.vision_range, this.cfg.predator.vision_range) + 1;
    this.obsGrid ??= new Grid(this.L, Math.max(reachMax / 2, 5), prey.n + pred.n);
    this.obsGrid.build([prey.x, pred.x], [prey.y, pred.y], [prey.alive, pred.alive]);
    return [this.observeSpecies(PREY), this.observeSpecies(PREDATOR)];
  }

  private observeSpecies(sp: number): Float32Array {
    const cfg = this.cfg;
    const st = this.species[sp];
    const sc = this.sc[sp];
    const [prey, pred] = this.species;
    const Np = prey.n;
    const Nall = prey.n + pred.n;
    const R = cfg.n_rays;
    const K = Math.min(cfg.ray_neighbors, Nall - 1);
    const range = f(sc.vision_range);
    const radMax = f(Math.max(cfg.prey.radius, cfg.predator.radius));
    const reach2 = f((range + radMax) * (range + radMax));
    const reach = Math.sqrt(reach2);
    const radius = [f(cfg.prey.radius), f(cfg.predator.radius)];
    const halfD = (sc.vision_cone_deg * Math.PI) / 180 / 2;
    const half = f(halfD);
    const width = f((2 * halfD) / R);
    const S = cfg.ray_grass_samples;
    const stepLen = f(range / S);
    const steps = new Float32Array(S);
    for (let s = 0; s < S; s++) steps[s] = f(f(s + 0.5) * stepLen);
    const thr = f(cfg.grass_visible_threshold);
    const busySpan = Math.max(sc.digest_steps, sc.miss_stun_steps) + 1;
    const D = this.D;
    const out = new Float32Array(st.n * D);
    const selfBase = sp === PREY ? 0 : Np;

    const candD = new Float64Array(K + 1);
    const candId = new Int32Array(K + 1);
    const candX = new Float32Array(K + 1);
    const candY = new Float32Array(K + 1);
    const cellBuf = new Int32Array(1024);
    const dPrey = new Float64Array(R);
    const dPred = new Float64Array(R);

    for (let i = 0; i < st.n; i++) {
      if (!st.alive[i]) continue;
      const px = st.x[i];
      const py = st.y[i];
      const hd = st.heading[i];
      // K nearest alive agents within reach (excluding self)
      let nc = 0;
      const grid = this.obsGrid!;
      const ncell = grid.cellsNear(px, py, reach, cellBuf);
      const lo = ncell < 0 ? 0 : -1;
      for (let cc = 0; cc < (ncell < 0 ? 1 : ncell); cc++) {
        const a0 = ncell < 0 ? lo : grid.start[cellBuf[cc]];
        const a1 = ncell < 0 ? grid.count : grid.start[cellBuf[cc] + 1];
        for (let m = a0; m < a1; m++) {
          const id = grid.items[m];
          if (id === selfBase + i) continue;
          const ox = id < Np ? prey.x[id] : pred.x[id - Np];
          const oy = id < Np ? prey.y[id] : pred.y[id - Np];
          const dx = this.minImage(f(ox - px));
          const dy = this.minImage(f(oy - py));
          const d2 = f(f(dx * dx) + f(dy * dy));
          if (d2 > reach2) continue;
          // keep the K nearest (ties: lower id), insertion into a sorted list of at most K
          let pos = nc < K ? nc : K;
          while (pos > 0 && (candD[pos - 1] > d2 || (candD[pos - 1] === d2 && candId[pos - 1] > id))) {
            if (pos < K) {
              candD[pos] = candD[pos - 1];
              candId[pos] = candId[pos - 1];
              candX[pos] = candX[pos - 1];
              candY[pos] = candY[pos - 1];
            }
            pos--;
          }
          if (pos < K) {
            candD[pos] = d2;
            candId[pos] = id;
            candX[pos] = dx;
            candY[pos] = dy;
            if (nc < K) nc++;
          }
        }
      }
      dPrey.fill(Infinity);
      dPred.fill(Infinity);
      for (let c = 0; c < nc; c++) {
        const id = candId[c];
        const kind = id < Np ? PREY : PREDATOR;
        const bearing = wrapAngle(f(f(Math.atan2(candY[c], candX[c])) - hd));
        const sector = Math.min(R - 1, Math.max(0, Math.floor(f(f(bearing + half) / width))));
        const surf = Math.max(f(f(Math.sqrt(candD[c])) - radius[kind]), 0);
        if (Math.abs(bearing) <= half && surf <= range) {
          const arr = kind === PREY ? dPrey : dPred;
          if (surf < arr[sector]) arr[sector] = surf;
        }
      }
      const o = i * D;
      const offs = this.rayOffsets[sp];
      for (let r = 0; r < R; r++) {
        const agentD = Math.min(dPrey[r], dPred[r]);
        const agentKind = dPred[r] < dPrey[r] ? PREDATOR : PREY;
        // grass and refuges along the sector's centre line
        const ang = f(hd + offs[r]);
        const ux = f(Math.cos(ang));
        const uy = f(Math.sin(ang));
        let grassD = Infinity;
        let refugeD = Infinity;
        for (let s = 0; s < S; s++) {
          const qx = fmodPos(f(px + f(ux * steps[s])), this.L);
          const qy = fmodPos(f(py + f(uy * steps[s])), this.L);
          const c = this.cellIndex(qx, qy);
          const ref = this.refuge[c] === 1;
          if (ref && refugeD === Infinity) refugeD = steps[s];
          if (!ref && this.grass[c] > thr && grassD === Infinity) grassD = steps[s];
          if (refugeD !== Infinity && grassD !== Infinity) break;
        }
        const isAgent = Number.isFinite(agentD) && agentD <= refugeD && agentD <= grassD;
        const isRefuge = Number.isFinite(refugeD) && !isAgent && refugeD <= grassD;
        const isGrass = Number.isFinite(grassD) && !isAgent && !isRefuge;
        const none = !(isAgent || isRefuge || isGrass);
        const dist = Math.min(agentD, refugeD, grassD);
        const b = o + r * 6;
        out[b] = none ? 1 : f(dist / range);
        out[b + 1] = isGrass ? 1 : 0;
        out[b + 2] = isAgent && agentKind === PREY ? 1 : 0;
        out[b + 3] = isAgent && agentKind === PREDATOR ? 1 : 0;
        out[b + 4] = isRefuge ? 1 : 0;
        out[b + 5] = none ? 1 : 0;
      }
      const b = o + R * 6;
      out[b] = f(st.energy[i] / sc.energy_cap);
      out[b + 1] = f(st.speed[i] / sc.max_speed);
      out[b + 2] = f(st.age[i] / sc.max_age);
      out[b + 3] = Math.min(1, Math.max(0, f(st.digest[i] / busySpan)));
      const c = this.cellIndex(px, py);
      out[b + 4] = this.grass[c];
      out[b + 5] = this.refuge[c];
    }
    return out;
  }
}
