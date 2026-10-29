// Simulation worker: steps the world off the UI thread and posts compact frames.
/// <reference lib="webworker" />
import { loadBrain } from "./brains";
import type { BrainName, Frame, Herd, Selected, SimSettings, ToWorker } from "./protocol";
import { loadConfig } from "./sim/config";
import { brainPolicy, heuristicPolicy, randomPolicy, type Policy } from "./sim/policies";
import { PREDATOR, PREY, World } from "./sim/world";

const ctx = self as unknown as DedicatedWorkerGlobalScope;

let world: World | null = null;
let settings: SimSettings | null = null;
let policies: [Policy, Policy] = [randomPolicy(0), randomPolicy(1)];
let acts: [Float32Array, Float32Array] = [new Float32Array(0), new Float32Array(0)];
let running = false;
let targetSps = 30;
let timer: ReturnType<typeof setTimeout> | null = null;
let popBuf: number[] = [];
let bornBuf: number[] = [];
let catchBuf: number[] = [];
let deathBuf: number[] = [];
let isReset = true;
let eating = new Uint8Array(0); // predator slots digesting a catch (as opposed to stunned after a miss)
let captures = 0;
let births: [number, number] = [0, 0];
let sendStatic = true;
let frameNo = 0;
let extinct: Frame["extinct"] = null;
let selected: { sp: number; slot: number; diedAt: number | null } | null = null;
let spsMeasured = 0;

function makePolicy(name: BrainName, sp: number): Policy {
  if (name === "random") return randomPolicy(sp);
  if (name === "heuristic") return heuristicPolicy(sp);
  const b = loadBrain(name, sp === PREY ? "prey" : "predator");
  return b ? brainPolicy(sp, b) : heuristicPolicy(sp);
}

function init(s: SimSettings) {
  settings = s;
  world = new World(loadConfig(s.scenario, s.base), s.seed);
  policies = [makePolicy(s.prey, PREY), makePolicy(s.predator, PREDATOR)];
  acts = [new Float32Array(world.species[0].n * 2), new Float32Array(world.species[1].n * 2)];
  popBuf = [...world.populations()];
  bornBuf = [];
  catchBuf = [];
  deathBuf = [];
  isReset = true;
  captures = 0;
  births = [0, 0];
  extinct = null;
  selected = null;
  sendStatic = true;
  frameNo = 0;
}

function stepOnce() {
  const w = world!;
  // positions before the step, to place death effects where the animal was
  const obs = w.observe();
  policies[0](w, obs[0], acts[0]);
  policies[1](w, obs[1], acts[1]);
  const preyX = w.species[0].x.slice(), preyY = w.species[0].y.slice();
  const predX = w.species[1].x.slice(), predY = w.species[1].y.slice();
  const info = w.step(acts[0], acts[1]);
  for (let q = 0; q < eating.length; q++) if (w.species[1].digest[q] === 0 || !w.species[1].alive[q]) eating[q] = 0;
  for (const [q, p] of info.catches) {
    eating[q] = 1;
    if (catchBuf.length < 4 * 400) catchBuf.push(preyX[p], preyY[p], q, p);
  }
  for (let sp = 0; sp < 2; sp++) {
    for (const c of info.newborn[sp]) if (bornBuf.length < 2 * 600) bornBuf.push(sp, c);
    const died = info.died[sp];
    const xs = sp === 0 ? preyX : predX, ys = sp === 0 ? preyY : predY;
    for (let i = 0; i < died.length; i++) {
      if (!died[i] || (sp === 0 && info.eaten[i]) || deathBuf.length >= 4 * 300) continue;
      deathBuf.push(sp, xs[i], ys[i], i);
    }
  }
  captures += info.captures;
  births[0] += info.births[0];
  births[1] += info.births[1];
  const p = w.populations();
  popBuf.push(p[0], p[1]);
  if (!extinct && (p[0] === 0 || p[1] === 0)) extinct = { species: p[0] === 0 ? "prey" : "predator", t: w.t };
  if (selected && selected.diedAt === null && !w.species[selected.sp].alive[selected.slot]) selected.diedAt = w.t;
}

function herd(sp: number): Herd {
  const w = world!;
  const st = w.species[sp];
  const sc = w.sc[sp];
  let n = 0;
  for (let i = 0; i < st.n; i++) n += st.alive[i];
  const h: Herd = {
    slot: new Int32Array(n), x: new Float32Array(n), y: new Float32Array(n), h: new Float32Array(n),
    spd: new Float32Array(n), flags: new Uint8Array(n), energy: new Float32Array(n),
  };
  let k = 0;
  for (let i = 0; i < st.n; i++) {
    if (!st.alive[i]) continue;
    h.slot[k] = i;
    h.x[k] = st.x[i];
    h.y[k] = st.y[i];
    h.h[k] = st.heading[i];
    h.spd[k] = st.speed[i] / sc.max_speed;
    h.flags[k] = (st.digest[i] > 0 ? 1 : 0) | (w.inRefuge(st.x[i], st.y[i]) ? 2 : 0) | (sp === PREDATOR && eating[i] ? 4 : 0);
    h.energy[k] = st.energy[i] / sc.energy_cap;
    k++;
  }
  return h;
}

function selectedInfo(): Selected | null {
  if (!selected || !world) return null;
  const w = world;
  const { sp, slot } = selected;
  const st = w.species[sp];
  const sc = w.sc[sp];
  const alive = st.alive[slot] === 1 && selected.diedAt === null;
  // what the brain sees right now (observed fresh, so it matches the drawn position)
  const obs = alive ? Array.from(currentObsRow(sp, slot)) : [];
  return {
    species: sp === PREY ? "prey" : "predator",
    slot,
    alive,
    diedAt: selected.diedAt,
    x: st.x[slot],
    y: st.y[slot],
    heading: st.heading[slot],
    energy: st.energy[slot],
    energyCap: sc.energy_cap,
    age: st.age[slot],
    maxAge: sc.max_age,
    speed: st.speed[slot],
    maxSpeed: sc.max_speed,
    digest: st.digest[slot],
    inRefuge: w.inRefuge(st.x[slot], st.y[slot]),
    obs,
    coneDeg: sc.vision_cone_deg,
    range: sc.vision_range,
    action: [acts[sp][2 * slot], acts[sp][2 * slot + 1]],
  };
}

function currentObsRow(sp: number, slot: number): Float32Array {
  const o = world!.observe(); // once per frame, only while an animal is selected
  return o[sp].subarray(slot * world!.D, (slot + 1) * world!.D);
}

function post() {
  const w = world!;
  const prey = herd(PREY);
  const pred = herd(PREDATOR);
  const sendGrass = sendStatic || frameNo % 3 === 0;
  const frame: Frame = {
    type: "frame",
    t: w.t,
    world: w.L,
    grid: w.G,
    capacity: [w.species[0].n, w.species[1].n],
    prey,
    pred,
    grass: sendGrass ? w.grass.slice() : null,
    refuge: sendStatic ? w.refuge.slice() : null,
    refugeCentres: sendStatic ? w.refugeCentres : null,
    refugeRadius: w.cfg.refuge_radius,
    reset: isReset,
    popHistory: Int32Array.from(popBuf),
    captures,
    births,
    bornSlots: Int32Array.from(bornBuf),
    catchEvents: Float32Array.from(catchBuf),
    deathEvents: Float32Array.from(deathBuf),
    stepsPerSec: spsMeasured,
    running,
    extinct,
    selected: selectedInfo(),
  };
  popBuf = [];
  bornBuf = [];
  catchBuf = [];
  deathBuf = [];
  captures = 0;
  births = [0, 0];
  sendStatic = false;
  isReset = false;
  frameNo++;
  const transfer: Transferable[] = [frame.popHistory.buffer, frame.bornSlots.buffer, frame.catchEvents.buffer,
    frame.deathEvents.buffer];
  for (const hd of [prey, pred]) for (const a of [hd.slot, hd.x, hd.y, hd.h, hd.spd, hd.flags, hd.energy]) transfer.push(a.buffer);
  if (frame.grass) transfer.push(frame.grass.buffer);
  if (frame.refuge) transfer.push(frame.refuge.buffer);
  ctx.postMessage(frame, transfer);
}

// Run loop: aim for targetSps steps per second, posting at most ~30 frames per second.
let lastTick = performance.now();
let owed = 0;
let spsWindow: number[] = [];
function loop() {
  timer = null;
  if (!running || !world) return;
  const now = performance.now();
  const dt = Math.min(0.25, (now - lastTick) / 1000);
  lastTick = now;
  const budgetMs = 28;
  let steps = 0;
  if (targetSps <= 0) {
    while (performance.now() - now < budgetMs) {
      stepOnce();
      steps++;
    }
  } else {
    owed = Math.min(owed + dt * targetSps, targetSps * 0.25 + 1);
    while (owed >= 1 && performance.now() - now < budgetMs) {
      stepOnce();
      owed -= 1;
      steps++;
    }
  }
  spsWindow.push(now, steps);
  while (spsWindow.length > 2 && now - spsWindow[0] > 1000) spsWindow.splice(0, 2);
  let total = 0;
  for (let i = 1; i < spsWindow.length; i += 2) total += spsWindow[i];
  const span = Math.max(0.25, (now - spsWindow[0]) / 1000);
  spsMeasured = total / span;
  post();
  timer = setTimeout(loop, Math.max(0, 33 - (performance.now() - now)));
}

function start() {
  if (timer === null) {
    lastTick = performance.now();
    owed = 0;
    timer = setTimeout(loop, 0);
  }
}

ctx.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  switch (m.type) {
    case "init":
      init(m.settings);
      running = m.running;
      targetSps = m.speed;
      post();
      if (running) start();
      break;
    case "brain":
      if (!world || !settings) break;
      settings = { ...settings, [m.species]: m.brain };
      policies[m.species === "prey" ? 0 : 1] = makePolicy(m.brain, m.species === "prey" ? PREY : PREDATOR);
      break;
    case "run":
      running = m.running;
      if (running) start();
      else post();
      break;
    case "speed":
      targetSps = m.stepsPerSec;
      break;
    case "step":
      if (world) {
        stepOnce();
        post();
      }
      break;
    case "select": {
      if (!world) break;
      let best = 36; // within 6 world units
      let pick: { sp: number; slot: number } | null = null;
      for (const sp of [PREDATOR, PREY]) {
        const st = world.species[sp];
        for (let i = 0; i < st.n; i++) {
          if (!st.alive[i]) continue;
          const dx = world.minImage(st.x[i] - m.x);
          const dy = world.minImage(st.y[i] - m.y);
          const d2 = dx * dx + dy * dy;
          if (d2 < best) {
            best = d2;
            pick = { sp, slot: i };
          }
        }
      }
      selected = pick ? { ...pick, diedAt: null } : null;
      post();
      break;
    }
    case "selectSlot":
      if (world && world.species[m.species]?.alive[m.slot]) {
        selected = { sp: m.species, slot: m.slot, diedAt: null };
        post();
      }
      break;
    case "deselect":
      selected = null;
      post();
      break;
  }
};
