# EcoArena week-1 setup log (GPU machine)

Local log, not committed. Date: 2026-10-26.

## Step 1: machine

- `nvidia-smi`: NVIDIA GeForce RTX 5060 Ti, 16311 MiB, driver 595.91.07, CUDA 13.2 (driver max)
- System python: 3.14.8 (not used)
- uv 0.12.3 (already installed)

## Step 2: environment

- `uv venv --python 3.12` → CPython 3.12.13 in `.venv`
- `uv pip install torch --index-url https://download.pytorch.org/whl/cu128` → torch 2.11.0+cu128, triton 3.6.0
- `uv pip install -e ".[dev]"` → ok (pytest 9.1.1, ruff 0.17.2); torch left at cu128
- Verify: `2.11.0+cu128 12.8 True NVIDIA GeForce RTX 5060 Ti (12, 0)`. cu128 worked first try, no sm_120 warnings.
- GPU matmul smoke test ran fine.

## Step 3: tests and lint

- `pytest -q` → 36 passed in 3.27s
- `ruff check ecoarena tests scripts` → All checks passed!

## Step 4: timing

```
device=cuda torch=2.11.0+cu128 base=s1_balanced.json
NVIDIA GeForce RTX 5060 Ti
worlds=  1  agent-steps/s=      46,627  world-steps/s=      188  20M agent-steps ≈  0.12 h (env only, no learning)
worlds= 16  agent-steps/s=     434,923  world-steps/s=    1,740  20M agent-steps ≈  0.01 h (env only, no learning)
worlds= 64  agent-steps/s=     571,409  world-steps/s=    2,283  20M agent-steps ≈  0.01 h (env only, no learning)
```

```
device=cuda torch=2.11.0+cu128 base=s1_large.json
NVIDIA GeForce RTX 5060 Ti
worlds=  1  agent-steps/s=     128,849  world-steps/s=      129  20M agent-steps ≈  0.04 h (env only, no learning)
worlds=  4  agent-steps/s=     186,171  world-steps/s=      186  20M agent-steps ≈  0.03 h (env only, no learning)
worlds= 16  agent-steps/s=     196,175  world-steps/s=      196  20M agent-steps ≈  0.03 h (env only, no learning)
```

No OOM at 16 worlds. Notes:
- world-steps/s = steps × worlds / time, so agent-steps/s ÷ world-steps/s = agents per world
  (~250 on 100×100, ~1000 on 200×200).
- Large world saturates early: 4 → 16 worlds gives only +5% agent-steps/s.

## Step 5: survival gate (heuristic, s1_large, 5 × 10,000, cuda) — FAILED (2/5, needs 4/5)

Summary: worlds_survived 2/5, extinct_at_step [null, null, 2598, 3230, 2325],
mean_pop_second_half [661.3, 28.0], cv_second_half [0.226, 0.093], wall 261.9 s,
190.9 world-steps/s, 127,373 agent-steps/s, captures_per_world [8739, 9855, 2367, 2222, 2168], no migrants.

Per world (from pops.npy):
- seed 0: survives. Prey 600 → boom ~980 (step ~200) → trough ~370 (~1000–1300) → flat ~630 from ~2500. Predators peak ~204 (~800), then slow monotone decline to 46 at end (min 44 @ 9874). One damped prey oscillation, no real cycle; predators drifting down.
- seed 1: same shape as seed 0. Prey trough ~285, flat ~640. Predators 206 → 42 (min 41 @ 9965), still declining.
- seed 2: prey crash to 14 at ~1900; predators (peak 220) starve, last predator dies ~2400 (extinct 2598). Prey then cycle alone 300 ↔ 1000 (period ~1700) on grass.
- seed 3: same; prey min 26, predators extinct 3230 (min 1 @ 2877).
- seed 4: same; prey min 15, predators extinct 2325.

Note: commit 2ada846 says tuning was checked on 2 worlds × 3000 steps with "prey trough ~230, predators never below ~120".
Here 3/5 worlds have prey troughs of 14–26 before step 2000, and the 2 survivors' predators fall well below 120.

## Step 6: random baseline (s1_large, 5 × 3000, cuda) — unexpected: 5/5 survive

worlds_survived 5/5, extinct_at_step all null, mean_pop_second_half [712.3, 78.7], cv [0.661, 0.606], wall 97.4 s, 124,772 agent-steps/s.
Plot shows a clean LV cycle in every world: prey boom ~1150 (~350), predators peak ~300 (~1000), prey trough 93–179 (~1600–1700),
predators trough 16–78 (~2250–2530), prey second boom 1300–1530 (~2550). Random does NOT collapse in 3000 steps;
it outlasts the heuristic over the same horizon (heuristic predators over-hunt prey in the first ~2000 steps).

## Extra: CPU vs GPU check (heuristic, s1_large, 5 × 3000, cpu; output in scratchpad, not results/)

Reason: tuning commit claimed prey trough ~230 / predators ≥120 (likely on CPU); GPU showed 3/5 crashes. Test whether device matters.

- CPU: worlds_survived 4/5, extinct_at_step [null, null, null, null, 2846], wall 1486.9 s (24.8 min), 5,914 agent-steps/s (~21× slower than GPU).
- CPU and GPU curves are identical until step 100, then diverge (float differences, expected).
- Same seeds fail on both devices: seeds 2–4 crash (CPU prey min 49 / 37 / 25; GPU 14 / 26 / 15), seeds 0–1 healthy.
- CPU's "4/5" is misleading: seeds 2 and 3 end at step 3000 with only 4 predators each.
- Conclusion: not a device bug. The tuned S1 rules are not robust across seeds; the 2-world tuning check got lucky with seeds 0–1.

CPU timing (contended, during the run): `worlds=5 agent-steps/s=3,559 world-steps/s=4`.
