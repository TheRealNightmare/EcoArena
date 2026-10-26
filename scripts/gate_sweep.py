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
# Diagnosis (logs.md, 2026-10-26): heuristic predators catch at the maximum rate digestion allows,
# so they either strip prey (3/5 worlds) or sit at break-even and dwindle (2/5). Random predators
# cycle cleanly. Candidates make a lunge uncertain and stun the predator after a miss.
CANDIDATES: dict[str, dict] = {
    "C0_current": {},
    "C1_p0.5_stun20": {"predator.catch_prob": 0.5, "predator.miss_stun_steps": 20},
    "C2_p0.3_stun20": {"predator.catch_prob": 0.3, "predator.miss_stun_steps": 20},
    "C3_p0.3_stun40": {"predator.catch_prob": 0.3, "predator.miss_stun_steps": 40},
    "C4_p0.3_stun20_catch45": {"predator.catch_prob": 0.3, "predator.miss_stun_steps": 20,
                               "predator.energy_per_catch": 45.0},
    "C5_p0.3_stun20_digest60": {"predator.catch_prob": 0.3, "predator.miss_stun_steps": 20,
                                "predator.digest_steps": 60},
    "C6_p0.5_stun40_digest60": {"predator.catch_prob": 0.5, "predator.miss_stun_steps": 40,
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
