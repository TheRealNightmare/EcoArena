# ES (M3) — all three stages (2026-10-28)

OpenAI-ES, population 50 (25 antithetic pairs), sigma 0.02, Adam lr 0.01 unless noted, g128, 150k world-steps,
s1_large, 4 worlds. Same evaluation, fitness and budget as GA.

| stage | run | attack rate | catches / 1k pred-steps | pred lifespan | prey reward / 1k | pred reward / 1k | worlds |
|---|---|---|---|---|---|---|---|
| prey learn | es-prey | 16.0 | 5.45 | 775 | -1.05 | -1.40 | 1/4 |
| prey learn | es-prey-lr003 | 14.8 | 5.59 | 807 | -0.88 | -1.32 | 2/4 |
| predator learn | **es-predator** | 13.5 | **8.03** | **1648** | -1.33 | **-0.11** | 4/4 |
| both learn | es-both | 11.6 | 6.11 | 1215 | -1.22 | -0.61 | 4/4 |
| reference | heuristic | 14.2 | 5.92 | 962 | -1.05 | -0.93 | 4/4 |

Training fitness (population mean, by quarter of the run):
- es-predator: -0.00135 -> -0.00053 -> -0.00042 -> -0.00031 (steady climb: ES clearly learns predators)
- es-prey: -0.00112 -> -0.00094 -> -0.00097 -> -0.00094 (flat after the first quarter)
- es-both: prey flat, predators flat around -0.0007

## Pattern across methods so far
- **Predators**: both evolutionary methods beat PPO and the heuristic on catches per predator-step (GA 8.18,
  ES 8.03, PPO 5.30, heuristic 5.92); ES predators live longest and have the best predator reward.
- **Prey**: only PPO clearly learns (attack rate 11.4). GA and ES prey stay at or below heuristic level.
- **Both**: under GA co-evolution prey come out ahead; under ES predators do.
- Working hypothesis for the paper: prey fitness is dominated by rare, discrete deaths (high variance per
  genome-window), predator fitness by frequent +35-energy catches (low variance), so population-level
  black-box methods get a usable signal for predators but not prey, while PPO's per-step credit assignment
  (value function + GAE) handles the rare-event case. Testable: compare per-genome fitness variance of the two
  species with ga_signal.py.

## Shared-arena signal check, GPU, 20 windows (results/ga/signal.json)
Fitness without birth bonus: PPO -0.0018, heuristic -0.0025, GA-old champion -0.0027, GA-nb champion -0.0033,
random -0.0059. Same-policy spread in one window 0.0030; top-10% genomes from PPO 42% (chance 20%). Noisier than
the 6-window CPU check suggested: for prey, one 128-step window ranks policies only weakly.
