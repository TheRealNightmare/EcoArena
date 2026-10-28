# Hybrid (M5) — GA (or ES) for 75k world-steps, then PPO from the champion for 75k (2026-10-29)

PPO phase: fresh critic, 20 value-only warm-up iterations, action std 0.37 at hand-over. Same 150k total.

| stage | run | attack rate | catches / 1k pred-steps | pred lifespan | prey lifespan | worlds |
|---|---|---|---|---|---|---|
| prey learn | hybrid (GA->PPO) | 15.0 | 6.52 | 1132 | 347 | 4/4 |
| prey learn | GA phase only (75k) | 15.3 | 5.81 | 896 | 567 | 3/4 |
| prey learn | PPO alone (150k) | **11.4** | 6.41 | 997 | 504 | 4/4 |
| predator learn | hybrid (GA->PPO) | 14.5 | 7.64 | **1498** | 337 | 4/4 |
| predator learn | GA phase only (75k) | 12.1 | **7.79** | 1341 | 525 | 4/4 |
| predator learn | GA alone (150k) | 12.4 | 8.18 | 1238 | – | 4/4 |
| predator learn | hybrid (ES->PPO) | 14.1 | 5.40 | 938 | 596 | 4/4 |
| predator learn | ES phase only (75k) | 10.6 | 6.23 | 1458 | 407 | 4/4 |
| both learn | hybrid (GA->PPO) | 12.3 | 6.03 | 1303 | 366 | 4/4 |
| both learn | GA phase only (75k) | 13.1 | 6.71 | 1061 | 488 | 4/4 |
| reference | heuristic | 14.2 | 5.92 | 962 | 505 | 4/4 |

- **No gain from the hand-over as configured.** For prey, PPO from a GA start ends at 15.0, far from PPO from
  scratch (11.4) with the same total budget; prey lifespan fell in the PPO phase. For predators, PPO kept
  GA's hunting rate (7.64 vs 7.79) and lengthened predator life, but undid most of ES's advantage (6.23 -> 5.40).
- **GA predators are good after half the budget** (7.79 at 75k vs 8.18 at 150k): evolution finds hunting fast.
- Candidate explanation to check: PPO optimises a different reward (birth bonus +1) from the evolutionary
  methods (bonus 0), so the PPO phase pulls evolved policies toward PPO's own solution. Control run queued:
  `train_ppo.py --birth-bonus 0`.
