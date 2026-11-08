"""E7 report: the optimised recipe against the benchmark, judged on survival AND skill.

    python scripts/report_opt.py --family es        # results/study/OPT.md + results/study/figures/opt_pareto.png

Survival: share of test worlds (S1, S2, S4 x 5 worlds x 10k steps, every seed) where both species are alive.
Skill (from `study.py skill`): a brain's predators hunt the prey of the five benchmark methods, and its prey are
hunted by their predators. For each opponent we divide by the average benchmark method against that opponent, so
1.0 = an average benchmark method:
    predator skill = mean over opponents of  attack rate achieved / benchmark mean attack rate on that prey
    prey skill     = mean over opponents of  benchmark mean attack rate by that predator / attack rate suffered
    skill          = geometric mean of the two
A recipe that keeps the ecosystem alive by making useless animals scores low on skill.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LEARNED = ["ppo", "ga", "es", "neat", "hybrid"]
NAMES = {"ppo": "PPO", "ga": "GA", "es": "ES", "neat": "NEAT", "hybrid": "Hybrid"}
COLORS = {"ppo": "#3b6fb6", "ga": "#c8553d", "es": "#2e8b57", "neat": "#8e5ea2", "hybrid": "#d29a2a"}
SCEN = ["S1", "S2", "S4"]


def survival(study: Path, bench: Path, cond: str) -> tuple[dict, list[int]]:
    """Per-scenario survival share and the flat list of 0/1 outcomes (for the bootstrap)."""
    per, flat = {}, []
    for sc in SCEN:
        d = bench / "eval" / sc if cond in LEARNED else study / "eval" / sc
        ok = []
        for f in sorted(d.glob(f"{cond}-seed*.json")):
            if not f.stem.removeprefix(f"{cond}-seed").isdigit():
                continue
            ok += [int(w["survived"]) for w in json.loads(f.read_text())["per_world"]]
        per[sc] = (float(np.mean(ok)) if ok else None, len(ok))
        flat += ok
    return per, flat


def boot_ci(x: list[int], n: int = 4000) -> tuple[float, float]:
    if not x:
        return (np.nan, np.nan)
    r = np.random.default_rng(0).choice(x, (n, len(x))).mean(1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def skill_table(study: Path) -> dict:
    """attack[(cond, role, opponent)] -> mean attack rate over seeds."""
    acc = defaultdict(list)
    for f in (study / "skill").glob("*.json"):
        cond_seed, vs = f.stem.split("-pred_vs_") if "-pred_vs_" in f.stem else f.stem.split("-prey_vs_")
        role = "pred" if "-pred_vs_" in f.stem else "prey"
        cond = cond_seed.rsplit("-seed", 1)[0]
        acc[(cond, role, vs)].append(json.loads(f.read_text())["summary"]["attack_rate_e6"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def skill(att: dict, cond: str) -> tuple[float, float, float] | None:
    preds, preys = [], []
    for m in LEARNED:
        ref_pred = [att[(b, "pred", m)] for b in LEARNED if (b, "pred", m) in att]
        ref_prey = [att[(b, "prey", m)] for b in LEARNED if (b, "prey", m) in att]
        if (cond, "pred", m) in att and ref_pred:
            preds.append(att[(cond, "pred", m)] / np.mean(ref_pred))
        if (cond, "prey", m) in att and ref_prey:
            preys.append(np.mean(ref_prey) / max(att[(cond, "prey", m)], 1e-9))
    if not preds or not preys:
        return None
    p, q = float(np.mean(preds)), float(np.mean(preys))
    return p, q, float(np.sqrt(p * q))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["ga", "es", "ppo"])
    ap.add_argument("--study", default=str(ROOT / "results" / "study"))
    ap.add_argument("--bench", default=str(ROOT / "results" / "bench"))
    a = ap.parse_args()
    study, bench = Path(a.study), Path(a.bench)
    sys.path.insert(0, str(ROOT / "scripts"))
    from study import CONDITIONS

    fam = a.family
    conds = LEARNED + [c for c, (e, m, _) in CONDITIONS.items() if e == "E7" and m == fam]
    att = skill_table(study)
    rows = []
    for c in conds:
        per, flat = survival(study, bench, c)
        if not flat:
            continue
        sk = skill(att, c)
        rows.append({"cond": c, "per": per, "surv": float(np.mean(flat)), "ci": boot_ci(flat), "n": len(flat),
                     "skill": sk})

    L = ["# E7: the optimised recipe", "", f"Family: **{fam.upper()}**. Survival = worlds with both species alive "
         "(S1, S2, S4). Skill: 1.0 = an average benchmark method (see `scripts/report_opt.py`).", "",
         "| brain | S1 | S2 | S4 | survival (95% CI) | predator skill | prey skill | skill |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    pct = lambda v: "–" if v is None else f"{100 * v:.0f}%"
    for r in rows:
        sk = r["skill"] or (np.nan, np.nan, np.nan)
        L.append(f"| {NAMES.get(r['cond'], r['cond'])} | " + " | ".join(pct(r["per"][s][0]) for s in SCEN)
                 + f" | {100 * r['surv']:.0f}% ({100 * r['ci'][0]:.0f}–{100 * r['ci'][1]:.0f}), n={r['n']} | "
                 f"{sk[0]:.2f} | {sk[1]:.2f} | {sk[2]:.2f} |")
    (study / "OPT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for r in rows:
        if r["skill"] is None:
            continue
        c = r["cond"]
        base = c.split("-")[0]
        is_opt = c == f"{fam}-opt"
        ax.errorbar(r["skill"][2], 100 * r["surv"],
                    yerr=[[100 * (r["surv"] - r["ci"][0])], [100 * (r["ci"][1] - r["surv"])]],
                    fmt="*" if is_opt else ("o" if c in LEARNED else "s"), ms=13 if is_opt else 6,
                    color=COLORS[base], mfc=COLORS[base] if (c in LEARNED or is_opt) else "white", lw=0.8, capsize=2)
        ax.annotate(NAMES.get(c, c.replace(f"{fam}-", "")), (r["skill"][2], 100 * r["surv"]), fontsize=7,
                    xytext=(5, 3), textcoords="offset points")
    ax.axvline(1, color="#bbb", lw=0.8)
    ax.set_xlabel("skill (1 = average benchmark method)")
    ax.set_ylabel("worlds with both species alive (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Survival and skill: benchmark vs optimised recipe", fontsize=9)
    fig.tight_layout()
    (study / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(study / "figures" / "opt_pareto.png", dpi=220)


if __name__ == "__main__":
    main()
