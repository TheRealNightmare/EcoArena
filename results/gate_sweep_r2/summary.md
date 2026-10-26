# Gate sweep · s1_large.json · 5 worlds × 6000 steps · device cuda

Pass = every policy keeps both species alive in ≥ 4/5 worlds.

| Candidate | heuristic survived | random survived | heuristic lowest predators (2nd half) | random lowest predators (2nd half) | Pass |
|---|---|---|---|---|---|
| E0_current | 0/5 | 5/5 | 201 | 30 | no |
| E1_pred_drain0.2 | 0/5 | 5/5 | 0 | 89 | no |
| E2_catch28 | 0/5 | 5/5 | 0 | 14 | no |
| E3_pred_init_half | 0/5 | 5/5 | 116 | 21 | no |
| E4_E3_catch28 | 0/5 | 5/5 | 0 | 11 | no |
| E5_p0.5_stun20 | 5/5 | 5/5 | 217 | 3 | yes |
| E6_p0.5_stun40_digest60 | 5/5 | 5/5 | 12 | 114 | yes |

Total wall time: 1714 s
