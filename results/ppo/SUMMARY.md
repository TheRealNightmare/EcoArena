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

## Reward control: PPO without the birth bonus (2026-10-29)

Same as prey-v2 / predator-v2 but `--birth-bonus 0`, i.e. the exact fitness GA/ES/NEAT use.

| run | attack rate | catches / 1k pred-steps | pred lifespan | prey lifespan | worlds |
|---|---|---|---|---|---|
| prey-v2 (bonus 1) | 11.4 | 6.41 | 997 | 504 | 4/4 |
| **prey-nobonus** | **8.2** | 5.77 | 742 | **958** | 4/4 |
| predator-v2 (bonus 1) | 13.2 | 5.30 | 997 | 589 | 4/4 |
| **predator-nobonus** | 12.1 | **7.45** | **1618** | 446 | 4/4 |
| heuristic | 14.2 | 5.92 | 962 | 505 | 4/4 |

The +1 birth bonus was a confound. Without it PPO prey are the best prey of any method so far, and PPO
predators reach the evolutionary methods' level (GA 8.18, ES 8.03, NEAT 7.59). The earlier "evolution makes
better predators" pattern was mostly the reward difference. From now on every method uses birth bonus 0
(the default in `common.step_rewards`, `PPOConfig` and `train_ppo.py`). The benchmark seed-0 PPO and hybrid
runs, trained with bonus 1, moved to `results/ppo/bench-bonus1/` and will be retrained.
