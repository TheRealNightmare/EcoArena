# EcoArena

A non-episodic 2D predator–prey ecosystem for comparing **reinforcement learning**, **neuroevolution** and **hybrids** on the same world, judged by ecosystem-level outcomes (persistence, extinction, stability, Lotka–Volterra fit) as well as agent-level scores.

Status: week 1 — environment core, baselines and tests. Training code comes next.

## Quick start

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch                        # CUDA build; Blackwell cards (RTX 50xx) need CUDA 12.8+
uv pip install -e ".[dev]"
pytest -q                                   # unit tests (CPU, ~10 s)
python scripts/run_baseline.py --policy heuristic --worlds 5 --steps 10000
python scripts/timing.py --device cuda      # throughput for 1, 16, 64 parallel worlds
```

`run_baseline.py` writes a population plot, the raw curves (`.npy`) and a JSON summary to `results/baselines/`.

## How the world works

- 100 × 100 continuous world with wrap-around edges; a 50 × 50 grass grid with logistic regrowth.
- **Prey** eat grass; **predators** eat prey. Both burn energy every step (more when thrusting), breed above an energy threshold (the child takes energy from the parent), and die of starvation, old age or being eaten.
- The world never resets. Births reuse free slots in fixed-size tensors, so many worlds step in parallel on the GPU.
- **Observation** (84 numbers): 16 vision sectors across the field of view (240° prey, 120° predators, range 20), each giving distance + what is nearest (grass / prey / predator / nothing), plus own energy, speed, age and grass underfoot.
- **Action**: `[turn, thrust]` in [-1, 1].
- All randomness is a counter-based hash of (seed, step, stream, world, slot), reproducible bit-for-bit in TypeScript for the browser demo.

All rules live in [`configs/s1_balanced.json`](configs/s1_balanced.json); scenarios S2–S5 are overrides in [`configs/scenarios.json`](configs/scenarios.json). Step order and details are documented at the top of [`ecoarena/env/world.py`](ecoarena/env/world.py).

## Layout

```
configs/            world rules (shared with the web demo)
ecoarena/env/       batched PyTorch environment, RNG, config loader
ecoarena/policies/  B0 random and B1 heuristic baselines
ecoarena/trainers/  PPO, GA, ES, NEAT, hybrid (coming)
ecoarena/eval/      benchmark runner and metrics (coming)
scripts/            run_baseline.py, sweep.py, timing.py
tests/              environment unit tests
web/                Next.js + three.js demo (coming)
paper/              LaTeX source (IEEE template)
results/            generated CSVs and figures
```
