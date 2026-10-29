// Trained brain weights, bundled into the simulation worker (exported by scripts/export_web.py brains).
import { parseBrain, type Brain } from "./sim/policies";

const files = import.meta.glob("../public/brains/*-*.json", { eager: true, import: "default" }) as Record<string, unknown>;

export function loadBrain(method: string, species: string): Brain | null {
  const raw = files[`../public/brains/${method}-${species}.json`];
  return raw ? parseBrain(raw) : null;
}
