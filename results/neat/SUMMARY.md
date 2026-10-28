# NEAT (M4) — all three stages (2026-10-29)

Population 50, max 48 hidden nodes, g128, 150k world-steps, s1_large, 4 worlds. Same evaluation, fitness and
budget as GA/ES. Genomes start linear (102 -> 2) and grow.

| stage | attack rate | catches / 1k pred-steps | pred lifespan | prey lifespan | prey reward / 1k | pred reward / 1k | worlds |
|---|---|---|---|---|---|---|---|
| prey learn | 16.0 | 5.62 | 812 | 556 | -1.16 | -1.29 | 4/4 |
| predator learn | 10.6 | **7.59** | 873 | 825 | -0.74 | -1.14 | 4/4 |
| both learn | 7.9 | 5.51 | 622 | 1054 | -0.69 | -2.05 | 3/4 |
| heuristic | 14.2 | 5.92 | 962 | 505 | -1.05 | -0.93 | 4/4 |

Structure at the end (population mean / champion): ~16–18 hidden nodes, ~255–265 enabled connections,
depth 3–5. Hidden-node counts are almost identical across the population (mean 17.7, max 18): one structure
took over, and the compatibility threshold fell to ~0.07 to keep 5 species, so species differ mainly in weights.

- **Prey**: no learning (training fitness flat; attack rate 16.0, worse than random prey).
- **Predators**: more catches per predator-step than the heuristic (+28%), fewer than GA (8.18) and ES (8.03);
  shorter-lived than heuristic predators. Training mean fell after the first quarter while worlds reset 9 times
  — fitness tracked the ecosystem crashing, not policy quality.
- **Both**: prey win outright; predators collapse to ~37 per world and one world loses them.
- NEAT is the weakest of the three evolutionary methods here. Likely reasons: it starts from a linear policy
  and spends early budget growing structure, and a 102-input minimal network gives each add-node mutation
  little leverage.
