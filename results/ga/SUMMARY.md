# GA (M2) — prey stage, findings so far

All runs: s1_large, 4 worlds, 150k world-steps, population 50, heuristic predators.

## Champion evaluation (scripts/train_ga.py, 4 fresh worlds x 3000 steps)

| run | birth bonus | attack rate (lower = better prey) | worlds survived |
|---|---|---|---|
| g128-s02 | 1 | 16.0 | 4/4 |
| g128-s05 | 1 | 16.1 | 4/4 |
| g64-s02 | 1 | 15.5 | 4/4 |
| nb-g128-s02 | 0 | 15.1 | 0/4 |
| nb-g128-s05 | 0 | 16.5 | 2/4 |
| nb-g64-s02 | 0 | 16.3 | 4/4 |
| heuristic | – | 14.2 | 4/4 |
| random | – | 14.0 | 0/4 |
| PPO prey-v2 | 1 | 11.4 | 4/4 |

By attack rate alone GA looks worse than random. That is misleading.

## Shared-arena check (scripts/ga_signal.py, CPU, 4 worlds)

Policies share the same worlds, slot s -> "genome" s % 50 -> policy. Fitness = reward per agent-step
without birth bonus (the GA fitness). 128-step windows after a 200-step warm-up.

| policy | run 1 (6 windows) | run 2 (4 windows) |
|---|---|---|
| PPO prey-v2 | **-0.0023** | |
| GA nb-g128-s02 champion | -0.0043 | **-0.0016** |
| GA g128-s02 champion (bonus 1) | -0.0040 | |
| heuristic | -0.0037 | -0.0020 |
| random actions | -0.0084 | |
| untrained GA networks (3) | | -0.0078, -0.0023, -0.0059 |

(Absolute values differ between runs because the mix of policies changes the ecosystem; compare within a column.)

- **GA does learn.** Its champions roughly halve the loss of random actions and of untrained networks and
  land at heuristic level. PPO is clearly better again.
- **The fitness signal is usable:** in a single 128-step window, 70% of the top-10% genomes were PPO
  copies (chance = 20%). Noise is not what stops GA; the search is (sigma-sized steps in ~11k weights, ~300
  generations under the 150k budget).
- **Attack rate and reward disagree for random prey**: they are caught at a low rate per encounter but
  starve. The evaluator now reports `prey_reward_per_1k_agent_steps` next to the attack rate.

## All three stages (2026-10-28, birth bonus 0, g128, sigma 0.02)

| stage | method | attack rate | catches / 1k predator-steps | predator lifespan | prey reward / 1k | predator reward / 1k | worlds |
|---|---|---|---|---|---|---|---|
| prey learn | GA (`--elite-score mean`) | 15.2 | 5.53 | 807 | -1.11 | -1.31 | 4/4 |
| prey learn | PPO prey-v2 | 11.4 | 6.41 | 997 | – | – | 4/4 |
| predator learn | **GA** | 12.4 | **8.18** | **1238** | -1.11 | **-0.53** | 4/4 |
| predator learn | PPO predator-v2 | 13.2 | 5.30 | 997 | – | – | 4/4 |
| both learn | GA | 11.0 | 6.54 | 784 | -0.76 | -1.39 | 4/4 |
| both learn | PPO both-v2 | 12.9 | 5.03 | 1014 | – | – | 4/4 |
| (reference) | heuristic | 14.2 | 5.92 | 962 | -1.05 | -0.93 | 4/4 |

(PPO runs predate the reward columns.)

- **GA predators are the best hunters so far**: 38% more catches per predator-step than the heuristic, 54% more than
  PPO, and they live longest. Their attack rate is *lower* than the heuristic's because prey are more numerous
  (660 vs 493) and a predator can only eat so fast (digest 60 steps + miss stun), so captures per
  prey x predator fall as prey rise. For predators, report catches per predator-step; attack rate is
  confounded by handling time.
- **Co-evolution**: prey come out ahead (attack rate 11.0, prey reward -0.76 vs heuristic -1.05).
- Separate-world reward is confounded by ecosystem state (random prey score -1.00 alone vs -1.05 heuristic,
  yet lose clearly in the shared arena). The shared-arena check stays the fair per-policy comparison.
