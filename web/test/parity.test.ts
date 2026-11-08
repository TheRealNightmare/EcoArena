// Python -> TypeScript parity: the same config, seed and policies must give the same world.
// Fixtures come from `python scripts/export_web.py parity` (PyTorch float32 run, one world).
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { loadConfig } from "../src/sim/config";
import { brainPolicy, heuristicPolicy, parseBrain, randomPolicy, type Policy } from "../src/sim/policies";
import { hash32, uniform } from "../src/sim/rng";
import { World } from "../src/sim/world";

const fx = (name: string) => JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8"));

describe("rng", () => {
  it("matches the pinned Python hashes", () => {
    expect([0, 1, 2, 12345, 2 ** 32 - 1].map(hash32)).toEqual([0, 1753845952, 3507691905, 2435775735, 1734902346]);
    expect(uniform(7, 0, 1, 0, 0)).toBeGreaterThanOrEqual(0);
  });
});

function expectClose(a: ArrayLike<number>, b: ArrayLike<number>, tol: number, what: string) {
  expect(a.length, `${what} length`).toBe(b.length);
  let worst = 0;
  let at = -1;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > worst) {
      worst = d;
      at = i;
    }
  }
  expect(worst, `${what}: max |diff| ${worst} at ${at} (${a[at]} vs ${b[at]})`).toBeLessThanOrEqual(tol);
}

function compareState(w: World, snap: any, tol: number) {
  expect(w.t).toBe(snap.t);
  ["prey", "predator"].forEach((name, sp) => {
    const st = w.species[sp];
    const s = snap[name];
    const alive: number[] = [];
    for (let i = 0; i < st.n; i++) if (st.alive[i]) alive.push(i);
    expect(alive, `${name} alive slots at t=${snap.t}`).toEqual(s.alive);
    const pick = (arr: ArrayLike<number>) => alive.map((i) => arr[i]);
    expect(pick(st.age), `${name} age`).toEqual(s.age);
    expect(pick(st.cooldown), `${name} cooldown`).toEqual(s.cooldown);
    expect(pick(st.digest), `${name} digest`).toEqual(s.digest);
    expectClose(pick(st.x), s.x, tol, `${name} x t=${snap.t}`);
    expectClose(pick(st.y), s.y, tol, `${name} y t=${snap.t}`);
    expectClose(pick(st.energy), s.energy, tol * 10, `${name} energy t=${snap.t}`);
    expectClose(pick(st.speed), s.speed, tol, `${name} speed t=${snap.t}`);
  });
  expectClose(w.grass.slice(0, 50), snap.grass_head, 1e-5, `grass t=${snap.t}`);
}

function runTrace(name: string, mk: (sp: number) => Policy) {
  const tr = fx(name);
  const w = new World(loadConfig(tr.scenario, tr.base, tr.overrides ?? {}), tr.seed);
  const pols = [mk(0), mk(1)];
  const acts = [new Float32Array(w.species[0].n * 2), new Float32Array(w.species[1].n * 2)];
  compareState(w, tr.snapshots[0], 1e-5);
  let k = 1;
  for (let t = 0; t < tr.steps; t++) {
    const obs = w.observe();
    if (tr.obs[String(t)]) {
      ["prey", "predator"].forEach((nm, sp) => {
        const ref = tr.obs[String(t)][nm];
        const D = w.D;
        const mine: number[] = [];
        for (const i of ref.alive) for (let d = 0; d < D; d++) mine.push(obs[sp][i * D + d]);
        expectClose(mine, ref.obs.flat(), 1e-4, `${nm} obs t=${t}`);
      });
    }
    pols[0](w, obs[0], acts[0]);
    pols[1](w, obs[1], acts[1]);
    const info = w.step(acts[0], acts[1]);
    const ev = tr.events[t];
    expect([info.captures, ...info.births, ...info.deaths], `events t=${t}`).toEqual([ev.captures, ...ev.births, ...ev.deaths]);
    if ((t + 1) % 10 === 0) compareState(w, tr.snapshots[k++], 2e-3);
  }
}

describe("world parity with PyTorch", () => {
  it("random agents, 150 steps (pure dynamics)", () => runTrace("trace_random.json", randomPolicy));
  it("heuristic agents, 150 steps (observations + policy + dynamics)", () => runTrace("trace_heuristic.json", heuristicPolicy));
  it("drag world, random agents, 150 steps", () => runTrace("trace_drag.json", randomPolicy));
});

describe("brains", () => {
  it("exported networks give the PyTorch actions", () => {
    const cases = fx("brains.json");
    for (const c of cases) {
      const brain = parseBrain(JSON.parse(readFileSync(new URL(`../public/brains/${c.file}`, import.meta.url), "utf8")));
      const sp = c.species === "prey" ? 0 : 1;
      const n = c.obs.length;
      const D = c.obs[0].length;
      const fake = { D, species: [{ n, alive: new Uint8Array(n).fill(1) }, { n, alive: new Uint8Array(n).fill(1) }] } as any;
      const out = new Float32Array(n * 2);
      brainPolicy(sp, brain)(fake, Float32Array.from(c.obs.flat()), out);
      expectClose(out, c.actions.flat(), 1e-4, c.file);
    }
  });
});
