# EcoArena web demo

The EcoArena world running live in the browser: a TypeScript port of `ecoarena/env/world.py`, the hand-coded and
trained brains, and a three.js top-down view.

```bash
pnpm install
pnpm dev          # http://localhost:5173
pnpm test         # Python -> TypeScript parity test
pnpm build        # dist/ (static site) + dist/ecoarena.html (single file)
```

- `src/sim/` — the world, RNG, config loader and policies. Reads the same `configs/*.json` as Python.
- `src/worker.ts` — runs the simulation off the UI thread and posts compact frames.
- `src/view/` — three.js map (grass texture, refuges, instanced animals, vision fan) and the population chart.
- `public/brains/` — exported networks. Pick checkpoints in `brains.manifest.json`, then
  `python scripts/export_web.py brains`.

## Parity

`python scripts/export_web.py parity` records PyTorch runs (seed 7, 100 × 100 world, one world) with random and
hand-coded agents. `test/parity.test.ts` replays them: every alive slot, age, cooldown, rest timer, capture, birth
and death must match exactly for 150 steps, positions and energy to float32 rounding, and the observation
vectors to 1e-4. Exported brains must reproduce the PyTorch actions. Long runs drift apart slowly (sin/cos and
float rounding differ between the two runtimes), which is expected: the test proves the rules are the same.

## Deploy

`dist/` is a static site (Vercel, Netlify or GitHub Pages). `dist/ecoarena.html` is the same app in one file.
