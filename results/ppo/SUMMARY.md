# PPO (M1) results — 200×200 world, 1 training seed

Evaluated on 4 fresh worlds × 3000 steps. **Attack rate** = captures per (prey × predator × step) × 10⁶,
the Lotka–Volterra interaction coefficient. It does not depend on how many animals are alive, so it compares
skill across ecosystem states: lower = better prey, higher = better predators.

| Run | Learner | World-steps (≈) | Attack rate: PPO | Hand-coded | Random | Worlds keeping both species (PPO) |
|---|---|---|---|---|---|---|
| `prey-v2` | prey vs hand-coded predators | 43k | **11.4** | 14.2 | 14.0 | 4/4 |
| `predator-v2` | predators vs hand-coded prey | 139k | **13.2** | 14.2 | 5.5 | 4/4 |
| `both-v2` | both, learning simultaneously throughout | 193k | 12.9 | 14.2 | 8.0 | **4/4** |
| `both-v3` | both, each frozen at 20M agent-steps | 157k | 13.1 | 14.2 | 8.0 | **1/4** |

- PPO prey are 20% harder to catch than hand-coded prey and sustain more prey (567 vs 493 alive).
- PPO predators catch 2.4× as well as random predators and come within 7% of the hand-coded hunters.
- **Budget rule.** `both-v3` froze each species at 20M agent-steps. Prey (≈5× more numerous) hit that at
  iteration 123, while predators had 2.1M steps and were near extinction. Predators then trained 490 more
  iterations against prey that could no longer adapt, over-exploited them (prey 128 vs predators 145 at the
  end of training), and only 1/4 evaluation worlds kept both species. Agent-step budgets also depend on
  population size, which is an outcome of the method. **The benchmark therefore budgets world-steps**
  (`--world-steps`, default 150k): every method gets the same simulation, and all learners train throughout.
  Under that rule `both-v2` is the stage-3 result.
- This "frozen opponent → over-exploitation → collapse" effect is worth a sentence in the paper's discussion.
- One training seed only; the benchmark will use 5 seeds before any claim goes in the paper.

Earlier runs (`*-vs-heuristic-s1_large-seed0`, v1) had no entropy bonus; prey entropy collapsed to 0.
