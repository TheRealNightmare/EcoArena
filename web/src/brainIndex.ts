// Which trained brains exist (small; the weights themselves are bundled only into the worker).
import index from "../public/brains/index.json";

export interface BrainInfo {
  method: string;
  species: "prey" | "predator";
  label: string;
  kind: string;
  file: string;
  note?: string;
}

export const BRAIN_INDEX = index as BrainInfo[];
