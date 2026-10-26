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

FAST = {"predator.max_speed": 1.3, "predator.move_cost": 0.15, "predator.max_turn": 0.15}

CANDIDATES = {
    "R1_refuge8": {},
    "R2_refuge12": {"refuge_count": 12},
    "R3_refuge8_fast": FAST,
    "R4_refuge12_fast": {**FAST, "refuge_count": 12},
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
