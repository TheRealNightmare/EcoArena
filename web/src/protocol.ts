// Messages between the UI thread and the simulation worker.

export type BrainName = "random" | "heuristic" | "ppo" | "ga" | "es" | "neat" | "hybrid";

export interface SimSettings {
  base: "s1_large.json" | "s1_balanced.json";
  scenario: string;
  seed: number;
  prey: BrainName;
  predator: BrainName;
}

export type ToWorker =
  | { type: "init"; settings: SimSettings; running: boolean; speed: number }
  | { type: "brain"; species: "prey" | "predator"; brain: BrainName }
  | { type: "run"; running: boolean }
  | { type: "speed"; stepsPerSec: number } // 0 = as fast as possible
  | { type: "step" }
  | { type: "select"; x: number; y: number } // world coordinates; picks the nearest animal
  | { type: "selectSlot"; species: number; slot: number } // a specific animal (the camera tour)
  | { type: "deselect" };

export interface Selected {
  species: "prey" | "predator";
  slot: number;
  alive: boolean;
  diedAt: number | null;
  x: number;
  y: number;
  heading: number;
  energy: number;
  energyCap: number;
  age: number;
  maxAge: number;
  speed: number;
  maxSpeed: number;
  digest: number;
  inRefuge: boolean;
  obs: number[]; // 16 sectors x [dist, grass, prey, predator, refuge, none]
  coneDeg: number;
  range: number;
  action: [number, number];
}

/** Animals of one species, alive ones only, keyed by slot so the renderer can track each animal over time. */
export interface Herd {
  slot: Int32Array;
  x: Float32Array;
  y: Float32Array;
  h: Float32Array; // heading, radians
  spd: Float32Array; // speed / max speed, 0..1
  flags: Uint8Array; // bit 0: can't catch (digesting or stunned), bit 1: in a refuge, bit 2: digesting
  energy: Float32Array; // energy / cap, 0..1
}

export interface Frame {
  type: "frame";
  t: number;
  world: number;
  grid: number;
  capacity: [number, number];
  prey: Herd;
  pred: Herd;
  grass: Float32Array | null; // sent every few frames
  refuge: Uint8Array | null; // sent on reset
  refugeCentres: [number, number][] | null;
  refugeRadius: number;
  reset: boolean; // a new world started with this frame
  popHistory: Int32Array; // [prey, pred] pairs for every step since the last frame
  captures: number; // since last frame
  births: [number, number];
  // events since the last frame, for effects
  bornSlots: Int32Array; // [species, slot] pairs
  catchEvents: Float32Array; // [x, y, predatorSlot, preySlot] per catch
  deathEvents: Float32Array; // [species, x, y, slot] per death that wasn't a catch
  stepsPerSec: number;
  running: boolean;
  extinct: { species: "prey" | "predator"; t: number } | null;
  selected: Selected | null;
}
