# E7: the optimised recipe

Family: **PPO**. Survival = worlds with both species alive (S1, S2, S4). Skill: 1.0 = an average benchmark method (see `scripts/report_opt.py`).

| brain | S1 | S2 | S4 | survival (95% CI) | predator skill | prey skill | skill |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO | 80% | 0% | 4% | 28% (19–39), n=75 | 1.34 | 1.63 | 1.48 |
| GA | 4% | 0% | 8% | 4% (0–9), n=75 | 0.77 | 0.83 | 0.80 |
| ES | 80% | 60% | 80% | 73% (63–83), n=75 | 0.95 | 1.00 | 0.98 |
| NEAT | 0% | 0% | 0% | 0% (0–0), n=75 | 0.76 | 0.78 | 0.77 |
| Hybrid | 88% | 40% | 76% | 68% (57–77), n=75 | 1.18 | 1.16 | 1.17 |
| ppo-opt | 92% | 60% | 64% | 72% (65–79), n=150 | 1.30 | 1.29 | 1.30 |
