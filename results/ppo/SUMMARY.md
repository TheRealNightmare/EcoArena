# PPO (M1) results — 200×200 world, 20M agent-steps per learning species, 1 training seed

Evaluated on 4 fresh worlds × 3000 steps. **Attack rate** = captures per (prey × predator × step) × 10⁶,
the Lotka–Volterra interaction coefficient. It does not depend on how many animals are alive, so it compares
skill across ecosystem states: lower = better prey, higher = better predators.

| Run | Learner | Attack rate: PPO | Hand-coded | Random | Worlds keeping both species (PPO) |
|---|---|---|---|---|---|
| `prey-v2` | prey vs hand-coded predators | **11.4** | 14.2 | 14.0 | 4/4 |
| `predator-v2` | predators vs hand-coded prey | **13.2** | 14.2 | 5.5 | 4/4 |
| `both-v2` | both together | 12.9 | 14.2 | 8.0 | 4/4 |

- PPO prey are 20% harder to catch than hand-coded prey and sustain more prey (567 vs 493 alive).
- PPO predators catch 2.4× as well as random predators and come within 7% of the hand-coded hunters.
- `both-v2` is not a fair run: prey trained for 113M steps vs 20M for predators (budget bug, fixed in the
  next commit: each species now freezes at its own budget). Rerun as `both-v3`.
- One training seed only; the benchmark will use 5 seeds before any claim goes in the paper.

Earlier runs (`*-vs-heuristic-s1_large-seed0`, v1) had no entropy bonus; prey entropy collapsed to 0.
