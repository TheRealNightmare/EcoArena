"""Quick parameter sweep for the survivability gate (heuristic agents, several seeds each).

    python scripts/sweep.py --steps 3000 --worlds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_baseline import plot, run, summarize

BASE: dict = {}  # configs/s1_balanced.json already holds the current best guess

# Goal: shrink the opening prey boom (150 -> ~380 in 130 steps) that feeds a predator boom.
SPARSE = {"grass_init_min": 0.1, "grass_init_max": 0.4}
SLOW_PREY = {"prey.repro_cooldown": 150}

# U3 removed the boom but predators could not break even (~0.3 energy/step vs. <= 1 catch/100 steps).
U3 = {**SPARSE, **SLOW_PREY, "predator.digest_steps": 100}
CHEAP_PRED = {"predator.base_drain": 0.1, "predator.move_cost": 0.1}

# Refuges with grass capped at 0.15 can't hold residents, so they don't act as a reservoir.
V4 = {**U3, "predator.base_drain": 0.15, "predator.move_cost": 0.15}

CANDIDATES = {
    "W1_V4_refgrass0.5": {**V4, "refuge_grass_cap": 0.5},
    "W2_V4_refgrass0.35": {**V4, "refuge_grass_cap": 0.35},
    "W3_V1_refgrass0.5": {**U3, **CHEAP_PRED, "refuge_grass_cap": 0.5},
    "W4_U3_refgrass0.5": {**U3, "refuge_grass_cap": 0.5},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--worlds", type=int, default=5)
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    for name, ov in CANDIDATES.items():
        if a.only and name not in a.only:
            continue
        _, pops, _capt, secs = run("heuristic", "S1", a.worlds, a.steps, 0, "cpu", {**BASE, **ov})
        s = summarize(pops)
        out = Path("results/tune/sweep")
        out.mkdir(parents=True, exist_ok=True)
        plot(pops, out / f"{name}.png", f"{name} · {s['worlds_survived']}/{a.worlds} survive")
        (out / f"{name}.json").write_text(json.dumps({"overrides": {**BASE, **ov}, **s}, indent=2))
        print(json.dumps({"cand": name, "survived": s["worlds_survived"], "extinct_at": s["extinct_at_step"],
                          "mean_pop": s["mean_pop_second_half"], "cv": s["cv_second_half"], "secs": round(secs)}), flush=True)


if __name__ == "__main__":
    main()
