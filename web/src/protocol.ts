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

export interface Frame {
  type: "frame";
  t: number;
  world: number;
  grid: number;
  prey: { x: Float32Array; y: Float32Array; h: Float32Array };
  pred: { x: Float32Array; y: Float32Array; h: Float32Array; busy: Uint8Array };
  grass: Float32Array | null; // sent every few frames
  refuge: Uint8Array | null; // sent on reset
  refugeCentres: [number, number][] | null;
  refugeRadius: number;
  popHistory: Int32Array; // [prey, pred] pairs for every step since the last frame
  captures: number; // since last frame
  births: [number, number];
  stepsPerSec: number;
  running: boolean;
  extinct: { species: "prey" | "predator"; t: number } | null;
  selected: Selected | null;
}
