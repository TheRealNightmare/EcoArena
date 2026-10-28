# EcoArena

A non-episodic 2D predator–prey ecosystem for comparing **reinforcement learning**, **neuroevolution** and **hybrids** on the same world, judged by ecosystem-level outcomes (persistence, extinction, stability, Lotka–Volterra fit) as well as agent-level scores.

Status: week 1 done — environment, baselines, tests, and a world that passes the survival gate. Training code (PPO first) comes next.

Scenarios S2–S5 in `configs/scenarios.json` are relative (`"x2"` doubles the base value), so they apply to either world.

## Quick start

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch                        # CUDA build; Blackwell cards (RTX 50xx) need CUDA 12.8+
uv pip install -e ".[dev]"
pytest -q                                   # unit tests (CPU, ~10 s)
python scripts/run_baseline.py --policy heuristic --worlds 5 --steps 10000 --base s1_large.json
python scripts/timing.py --device cuda      # throughput for 1, 16, 64 parallel worlds
python scripts/gate_sweep.py                # survival gate: candidate rules x {heuristic, random}
```

`run_baseline.py` writes a population plot, the raw curves (`.npy`) and a JSON summary to `results/baselines/`.

## How the world works

- Continuous world with wrap-around edges and a grass grid with logistic regrowth. The benchmark world is
  200 × 200 (`configs/s1_large.json`); the 100 × 100 world (`s1_balanced.json`) has the same rules and is
  used for tests and quick debugging. On 100 × 100, predator numbers fall to 1–3 at the bottom of a cycle
  and usually die out by chance. **Survival gate passed** on 200 × 200: both species stayed alive in
  10/10 worlds for 10,000 steps with heuristic agents and with random agents (`results/gate_sweep_r3/`).
- Refuges: patches where prey can't be caught and grass grows sparsely. Predators take time to digest
  after a catch and can't breed when crowded by other predators. A lunge succeeds half the time
  (`catch_prob`), and a miss stuns the predator for 40 steps (`miss_stun_steps`).
- **Prey** eat grass; **predators** eat prey. Both burn energy every step (more when thrusting), breed above an energy threshold (the child takes energy from the parent), and die of starvation, old age or being eaten.
- The world never resets. Births reuse free slots in fixed-size tensors, so many worlds step in parallel on the GPU.
- **Observation** (102 numbers): 16 vision sectors across the field of view (prey: 240°, range 20; predators: 120°, range 12), each giving distance + what is nearest (grass / prey / predator / refuge / nothing), plus own energy, speed, age, whether it is busy (a predator digesting or stunned can't catch), grass underfoot and whether it is in a refuge.
- **Action**: `[turn, thrust]` in [-1, 1].
- All randomness is a counter-based hash of (seed, step, stream, world, slot), reproducible bit-for-bit in TypeScript for the browser demo.

All rules live in [`configs/s1_balanced.json`](configs/s1_balanced.json); scenarios S2–S5 are overrides in [`configs/scenarios.json`](configs/scenarios.json). Step order and details are documented at the top of [`ecoarena/env/world.py`](ecoarena/env/world.py).

## Layout

```
configs/            world rules (shared with the web demo)
ecoarena/env/       batched PyTorch environment, RNG, config loader
ecoarena/policies/  B0 random and B1 heuristic baselines
ecoarena/trainers/  PPO, GA, ES, NEAT; hybrid = scripts/train_hybrid.py (GA/ES then PPO)
ecoarena/eval/      benchmark runner and metrics (coming)
scripts/            run_baseline.py, sweep.py, timing.py
tests/              environment unit tests
web/                Next.js + three.js demo (coming)
paper/              LaTeX source (IEEE template)
results/            generated CSVs and figures
```
