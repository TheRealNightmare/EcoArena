# Benchmark results

Setup: every method controls both species, trained together for 150k world-steps (no birth bonus), 5 training
seeds per method. Each seed is evaluated on 5 fresh worlds × 10,000 steps in each scenario (S1 baseline,
S2 slow grass, S3 fast prey, S4 doubled start, S5 doubled predators). Full numbers with 95% CIs and p-values:
`STATS.md`; figures: `figures/`; per-world rows: `per_world.csv`.

## 1. Learned ecosystems collapse, and always the same way

Worlds with both species alive after 10,000 steps (all scenarios pooled, 125 worlds per learned method, 25 per
baseline):

| Random | ES | Hybrid | Hand-coded | PPO | GA | NEAT |
|---|---|---|---|---|---|---|
| 92% | 77% | 64% | 64% | 22% | 10% | 0% |

- Every one of the collapses is **predator extinction**. Prey never went extinct in any run.
- Collapses are slow (median predator extinction 2–9k steps). That is why the 1,000-step checks during training
  all survived.
- Log-rank tests against PPO: GA and NEAT are worse on S1 (p < 0.001). ES and Hybrid are better on S2–S5
  (p < 0.001).
- Random agents form the steadiest ecosystem (prey CV ≈ 0.03). Their predators are inefficient, but so are their
  prey, so the populations stay balanced.

Interpretation: each method optimises each animal's own energy over a short horizon (128-step generations / PPO
rollouts). Nothing in the objective rewards a sustainable predator population, and learned prey get good enough
at evading that predators slowly starve.

## 2. Hunting efficiency is not survival

Results for S1, mean of 5 seeds:

| | PPO | GA | ES | NEAT | Hybrid | Hand-coded | Random |
|---|---|---|---|---|---|---|---|
| Catches / 1k predator-steps | 5.44 | 6.44 | 5.56 | 6.41 | 5.74 | 6.14 | 6.46 |
| Predator lifespan (steps) | 1196 | 782 | 1059 | 633 | 1055 | 1056 | 1072 |
| Prey lifespan (steps) | 501 | 1190 | 734 | 1260 | 728 | 440 | 683 |

- GA and NEAT predators catch the most prey per step, yet they live the shortest lives (p = 0.008 vs PPO). They
  spend more energy chasing than they recover.
- GA and NEAT prey live longest: about 2.4× PPO prey.
- PPO predators live longest but catch the least.

## 3. Seed sensitivity is large

Worlds surviving per training seed (out of 25):

| Method | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| ES | 25 | 22 | 23 | 1 | 25 |
| Hybrid | 6 | 14 | 15 | 20 | 25 |
| PPO | 5 | 6 | 1 | 9 | 6 |
| GA | 5 | 5 | 0 | 2 | 0 |
| NEAT | 0 | 0 | 0 | 0 | 0 |

A single training run per method, as in the earlier per-species experiments, would have given an unreliable
ranking.

## Notes for the paper

- With 5 seeds the smallest possible permutation p-value is 0.008. Many comparisons hit that floor.
- The baselines (Random, Hand-coded) have no training seeds, so each of their evaluation worlds counts as one
  sample.
- The web demo now uses these benchmark brains (most stable seed per method; ES is the default).
