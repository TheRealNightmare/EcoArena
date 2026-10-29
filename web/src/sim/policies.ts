// Policies: random (B0), hand-coded heuristic (B1), and trained brains (MLP for PPO/GA/ES/hybrid, NEAT graphs).
// All of them act only on the observation vector, like the Python versions in ecoarena/policies/baselines.py.
import * as rng from "./rng";
import { fmodPos, PREY, World } from "./world";

const f = Math.fround;
const PER_SECTOR = 6;
const D_ = 0, GRASS = 1, PREY_HIT = 2, PRED_HIT = 3, REFUGE = 4;

export type Policy = (world: World, obs: Float32Array, out: Float32Array) => void;

export function randomPolicy(sp: number): Policy {
  return (w, _obs, out) => {
    const N = w.species[sp].n;
    const st = rng.S_POLICY + sp * rng.SPECIES_STREAM_OFFSET;
    for (let s = 0; s < N; s++) {
      out[2 * s] = f(rng.uniform(w.seed, w.t, st, 0, s) * 2 - 1);
      out[2 * s + 1] = f(rng.uniform(w.seed, w.t, st + 1, 0, s) * 2 - 1);
    }
  };
}

export function heuristicPolicy(sp: number): Policy {
  return (w, obs, out) => {
    const R = w.cfg.n_rays;
    const D = w.D;
    const offs = w.rayOffsets[sp];
    const maxTurn = f(w.sc[sp].max_turn);
    const N = w.species[sp].n;
    const alive = w.species[sp].alive;
    const wanderStream = rng.S_POLICY + 2 + sp * rng.SPECIES_STREAM_OFFSET;
    const toward = (a: number) => Math.min(1, Math.max(-1, f(a / maxTurn)));
    const nearest = (o: number, ch: number): number => {
      let best = 9;
      let j = -1;
      for (let r = 0; r < R; r++) {
        const b = o + r * PER_SECTOR;
        const d = obs[b + ch] > 0.5 ? obs[b + D_] : 9;
        if (d < best) {
          best = d;
          j = r;
        }
      }
      return j; // -1: none seen
    };
    const TWO_PI = f(2 * Math.PI);
    const PI = f(Math.PI);
    for (let s = 0; s < N; s++) {
      if (!alive[s]) {
        out[2 * s] = 0;
        out[2 * s + 1] = 0;
        continue;
      }
      const o = s * D;
      let turn = f(f(rng.uniform(w.seed, w.t, wanderStream, 0, s) * 2 - 1) * f(0.3));
      let thrust = 0.5;
      if (sp === PREY) {
        const jg = nearest(o, GRASS);
        if (jg >= 0) turn = toward(offs[jg]);
        if (obs[o + D - 2] > 0.25) {
          thrust = -1;
          turn = 0;
        }
        const jp = nearest(o, PRED_HIT);
        if (jp >= 0) {
          const inside = obs[o + D - 1] > 0.5;
          const jr = nearest(o, REFUGE);
          const away = f(fmodPos(f(offs[jp] + TWO_PI), TWO_PI) - PI);
          turn = inside ? 0 : jr >= 0 ? toward(offs[jr]) : toward(away);
          thrust = inside ? -1 : 1;
        }
      } else {
        const jp = nearest(o, PREY_HIT);
        if (jp >= 0) turn = toward(offs[jp]);
        thrust = jp >= 0 ? 1 : 0.6;
        if (obs[o + D - 3] > 0) thrust = 0;
      }
      out[2 * s] = turn;
      out[2 * s + 1] = thrust;
    }
  };
}

// ---------------------------------------------------------------------------------------- trained brains
export interface MLPBrain {
  kind: "mlp";
  layers: { in: number; out: number; w: Float32Array; b: Float32Array }[]; // w is [out][in], like nn.Linear
}

export interface NEATBrain {
  kind: "neat";
  D: number;
  H: number; // hidden slots; layout per genome: hidden 0..H-1, outputs H, H+1
  depth: number;
  Win: Float32Array; // [D][H+2]
  Wh: Float32Array; // [H+2][H+2]
  b: Float32Array; // [H+2]
  nHidden: number;
}

export type Brain = MLPBrain | NEATBrain;

export function brainPolicy(sp: number, brain: Brain): Policy {
  if (brain.kind === "mlp") {
    // Layer 1 is applied sparsely: most observation entries are 0 (one-hot sectors), so W1 is stored
    // transposed ([in][out]) and only nonzero inputs are accumulated.
    const [L1, ...rest] = brain.layers;
    const W1T = new Float32Array(L1.in * L1.out);
    for (let o = 0; o < L1.out; o++) for (let i = 0; i < L1.in; i++) W1T[i * L1.out + o] = L1.w[o * L1.in + i];
    const maxW = Math.max(...brain.layers.map((l) => l.out));
    const ha = new Float32Array(maxW);
    const hb = new Float32Array(maxW);
    return (w, obs, out) => {
      const st = w.species[sp];
      const D = w.D;
      const n1 = L1.out;
      for (let s = 0; s < st.n; s++) {
        if (!st.alive[s]) {
          out[2 * s] = 0;
          out[2 * s + 1] = 0;
          continue;
        }
        const base = s * D;
        for (let o = 0; o < n1; o++) ha[o] = L1.b[o];
        for (let i = 0; i < L1.in; i++) {
          const x = obs[base + i];
          if (x === 0) continue;
          const row = i * n1;
          for (let o = 0; o < n1; o++) ha[o] += x * W1T[row + o];
        }
        for (let o = 0; o < n1; o++) ha[o] = Math.tanh(ha[o]);
        let src = ha;
        let dst = hb;
        for (let li = 0; li < rest.length; li++) {
          const L = rest[li];
          const last = li === rest.length - 1;
          const nin = L.in;
          for (let o = 0; o < L.out; o++) {
            let z = L.b[o];
            const row = o * nin;
            for (let i = 0; i < nin; i++) z += L.w[row + i] * src[i];
            dst[o] = last ? z : Math.tanh(z);
          }
          const tmp = src;
          src = dst;
          dst = tmp;
        }
        out[2 * s] = Math.min(1, Math.max(-1, src[0]));
        out[2 * s + 1] = Math.min(1, Math.max(-1, src[1]));
      }
    };
  }
  const { D, H, depth, Win, Wh, b } = brain;
  const H2 = H + 2;
  // only the hidden slots a genome uses matter; the rest have zero weights
  const used: number[] = [];
  for (let j = 0; j < H; j++) {
    let any = b[j] !== 0;
    for (let i = 0; i < D && !any; i++) any = Win[i * H2 + j] !== 0;
    for (let i = 0; i < H2 && !any; i++) any = Wh[i * H2 + j] !== 0 || Wh[j * H2 + i] !== 0;
    if (any) used.push(j);
  }
  used.push(H, H + 1);
  const zin = new Float32Array(H2);
  const h = new Float32Array(H2);
  const z = new Float32Array(H2);
  return (w, obs, out) => {
    const st = w.species[sp];
    for (let s = 0; s < st.n; s++) {
      if (!st.alive[s]) {
        out[2 * s] = 0;
        out[2 * s + 1] = 0;
        continue;
      }
      const o = s * w.D;
      for (const j of used) {
        let acc = b[j];
        for (let i = 0; i < D; i++) acc += obs[o + i] * Win[i * H2 + j];
        zin[j] = acc;
        h[j] = 0;
      }
      for (let k = 0; k < depth; k++) {
        for (const j of used) {
          let acc = zin[j];
          for (const i of used) acc += h[i] * Wh[i * H2 + j];
          z[j] = acc;
        }
        for (const j of used) h[j] = j < H ? Math.tanh(z[j]) : z[j];
      }
      out[2 * s] = Math.min(1, Math.max(-1, h[H]));
      out[2 * s + 1] = Math.min(1, Math.max(-1, h[H + 1]));
    }
  };
}

function b64ToF32(s: string): Float32Array {
  const bin = atob(s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/** Parse a brain exported by scripts/export_web.py. */
export function parseBrain(j: any): Brain {
  if (j.kind === "mlp") {
    return {
      kind: "mlp",
      layers: j.layers.map((l: any) => ({ in: l.in, out: l.out, w: b64ToF32(l.w), b: b64ToF32(l.b) })),
    };
  }
  return { kind: "neat", D: j.D, H: j.H, depth: j.depth, Win: b64ToF32(j.Win), Wh: b64ToF32(j.Wh), b: b64ToF32(j.b), nHidden: j.n_hidden };
}
