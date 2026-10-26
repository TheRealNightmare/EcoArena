"""Survival-gate sweep on the GPU: candidate rule sets × {heuristic, random} × N worlds.

A candidate passes the gate when BOTH policies keep both species alive in >= --need of --worlds.
Testing two very different hunting skills guards against tuning the world to one policy.

    python scripts/gate_sweep.py                      # screening: 5 worlds x 6000 steps
    python scripts/gate_sweep.py --only C5 --steps 10000   # confirm one candidate

Writes results/gate_sweep/summary.md, one JSON per run and one plot per run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_baseline import plot, run, summarize

# Overrides on top of configs/s1_large.json.
#
# Round 1 (2026-10-26, 5 x 6000): no candidate passed. Lower catch_prob starved predators even with
# prey plentiful, so predators were not too efficient: they barely broke even. Cause: predators
# could not observe that they were digesting, and the heuristic chased at full thrust for the 83%
# of its time it could not eat (0.27 vs 0.15 energy/step). Round 2 adds a "busy" observation and
# the heuristic coasts while busy. Predators now keep more energy, so candidates bracket both ways.
CANDIDATES: dict[str, dict] = {
    "E0_current": {},
    "E1_pred_drain0.2": {"predator.base_drain": 0.2},
    "E2_catch28": {"predator.energy_per_catch": 28.0},
    "E3_pred_init_half": {"predator.init_count": "x0.5"},
    "E4_E3_catch28": {"predator.init_count": "x0.5", "predator.energy_per_catch": 28.0},
    "E5_p0.5_stun20": {"predator.catch_prob": 0.5, "predator.miss_stun_steps": 20},
    "E6_p0.5_stun40_digest60": {"predator.catch_prob": 0.5, "predator.miss_stun_steps": 40,
                                "predator.digest_steps": 60},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--worlds", type=int, default=5)
    p.add_argument("--need", type=int, default=4)
    p.add_argument("--base", default="s1_large.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--only", nargs="*")
    p.add_argument("--policies", nargs="*", default=["heuristic", "random"])
    p.add_argument("--out", default="results/gate_sweep")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    t_all = time.time()
    for name, ov in CANDIDATES.items():
        if a.only and name not in a.only:
            continue
        res = {}
        for pol in a.policies:
            _, pops, capt, secs = run(pol, "S1", a.worlds, a.steps, 0, a.device, ov, a.base)
            s = summarize(pops)
            tail = pops[len(pops) // 2 :]
            s.update({
                "candidate": name, "policy": pol, "overrides": ov, "steps": a.steps, "base": a.base,
                "wall_seconds": round(secs, 1),
                "min_pop": pops.min(axis=0).tolist(),  # per world: [prey, predators]
                "min_pred_second_half": tail[..., 1].min(axis=0).tolist(),
                "captures_per_world": capt.tolist(),
            })
            stem = f"{name}-{pol}-{a.steps}"
            (out / f"{stem}.json").write_text(json.dumps(s, indent=2))
            plot(pops, out / f"{stem}.png", f"{name} · {pol} · {s['worlds_survived']}/{a.worlds} keep both species")
            res[pol] = s
            print(json.dumps({"cand": name, "policy": pol, "survived": s["worlds_survived"],
                              "extinct_at": s["extinct_at_step"],
                              "min_pred_2nd_half": s["min_pred_second_half"],
                              "mean_pop_2nd_half": s["mean_pop_second_half"], "secs": round(secs)}), flush=True)
        passed = all(r["worlds_survived"] >= a.need for r in res.values())
        rows.append((name, res, passed))

    lines = [
        f"# Gate sweep · {a.base} · {a.worlds} worlds × {a.steps} steps · device {a.device}",
        "",
        f"Pass = every policy keeps both species alive in ≥ {a.need}/{a.worlds} worlds.",
        "",
        "| Candidate | " + " | ".join(f"{p} survived" for p in a.policies)
        + " | " + " | ".join(f"{p} lowest predators (2nd half)" for p in a.policies) + " | Pass |",
        "|---" * (1 + 2 * len(a.policies) + 1) + "|",
    ]
    for name, res, passed in rows:
        surv = " | ".join(f"{res[p]['worlds_survived']}/{a.worlds}" for p in a.policies)
        low = " | ".join(str(min(res[p]["min_pred_second_half"])) for p in a.policies)
        lines.append(f"| {name} | {surv} | {low} | {'yes' if passed else 'no'} |")
    lines += ["", f"Total wall time: {round(time.time() - t_all)} s", ""]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
