// Counter-based random numbers, bit-identical to ecoarena/env/rng.py:
// every draw is lowbias32 hash of (seed, step, stream, world, slot).

export const S_INIT_X = 1;
export const S_INIT_Y = 2;
export const S_INIT_HEADING = 3;
export const S_INIT_AGE = 4;
export const S_GRASS_INIT = 5;
export const S_REFUGE_X = 6;
export const S_REFUGE_Y = 7;
export const S_INIT_ENERGY = 8;
export const S_BIRTH_ANGLE = 10;
export const S_BIRTH_HEADING = 11;
export const S_MIGRANT_X = 12;
export const S_MIGRANT_Y = 13;
export const S_MIGRANT_HEADING = 14;
export const S_CATCH = 15;
export const S_POLICY = 20;
export const SPECIES_STREAM_OFFSET = 100;

export function hash32(x: number): number {
  x >>>= 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

/** Uniform float32 in [0, 1). The browser always runs world 0. */
export function uniform(seed: number, step: number, stream: number, world: number, slot: number): number {
  let h = hash32(seed >>> 0);
  h = hash32((h ^ (step >>> 0)) >>> 0);
  h = hash32((h ^ (stream >>> 0)) >>> 0);
  h = hash32((h ^ (world >>> 0)) >>> 0);
  h = hash32((h ^ (slot >>> 0)) >>> 0);
  return Math.fround(h / 4294967296);
}
