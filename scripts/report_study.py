"""Study 2 report: cross-play matrices, predator energy budgets and tactics.

    python scripts/report_study.py energy    # thrust and energy budget of every benchmark brain (CPU ok, ~20 min)
    python scripts/report_study.py report    # results/study/REPORT.md + results/study/figures/*.png

Cross-play: prey of method A against predators of method B on S1 (results/study/crossplay/, made by
`study.py crossplay`); the diagonal is the benchmark itself (results/bench/eval/S1/).
Energy: movement costs move_cost × thrust² per step and speed carries over between steps, so a predator can
cruise almost for free. Income = catches per predator-step (benchmark, S1) × energy per catch.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecoarena.env import PREDATOR, PREY, EcoWorld, load_config
from ecoarena.policies import make_policy
from ecoarena.trainers.ppo import load_policy

OUT = ROOT / "results" / "study"
BENCH = ROOT / "results" / "bench"
M = ["random", "heuristic", "ppo", "ga", "es", "neat", "hybrid"]
NAMES = {"random": "Random", "heuristic": "Hand-coded", "ppo": "PPO", "ga": "GA", "es": "ES", "neat": "NEAT",
         "hybrid": "Hybrid"}
LEARNED = ["ppo", "ga", "es", "neat", "hybrid"]
BASE = "s1_large.json"


# ------------------------------------------------------------------------------------------------ energy
@torch.no_grad()
def thrust_stats(m: str, s: int | None, steps: int, warm: int, device: str, study: bool = False) -> dict:
    cfg = load_config("S1", base=BASE)
    w = EcoWorld(cfg, n_worlds=1, seed=20_000, device=device)
    if s is None:
        pol = {sp: make_policy(m, w, sp) for sp in (PREY, PREDATOR)}
    else:
        d = (OUT if study else BENCH) / "train" / m / f"{m}-seed{s}"
        pol = {PREY: load_policy(d / "prey.pt", device), PREDATOR: load_policy(d / "predator.pt", device)}
    acc = {k: [] for k in ("pred_th2", "pred_flip", "pred_mean", "prey_th2", "prey_flip")}
    prev = {PREY: None, PREDATOR: None}
    for t in range(warm + steps):
        o = w.observe()
        a = {sp: pol[sp](o[sp]) for sp in (PREY, PREDATOR)}
        for sp, tag in ((PREY, "prey"), (PREDATOR, "pred")):
            thr = a[sp][0, :, 1].clamp(-1, 1)
            al = w.species[sp].alive[0]
            if t >= warm and bool(al.any()):
                acc[f"{tag}_th2"].append(float((thr[al] ** 2).mean()))
                if tag == "pred":
                    acc["pred_mean"].append(float(thr[al].mean()))
                if prev[sp] is not None:
                    acc[f"{tag}_flip"].append(float(((thr * prev[sp]) < 0)[al].float().mean()))
            prev[sp] = thr.clone()
        w.step(a[PREY], a[PREDATOR])
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
    qc = cfg.predator
    out["pred_cost"] = qc.base_drain + qc.move_cost * out["pred_th2"]
    f = (OUT if study else BENCH) / "eval" / "S1" / (f"{m}.json" if s is None else f"{m}-seed{s}.json")
    pw = json.loads(f.read_text())["per_world"]
    out["catches_per_1k"] = float(np.mean([x["pred_catches_per_1k_agent_steps"] for x in pw]))
    out["pred_income"] = out["catches_per_1k"] / 1000 * qc.energy_per_catch
    out["pred_net"] = out["pred_income"] - out["pred_cost"]
    out["pred_extinct_share"] = float(np.mean([x["extinct_at"]["predator"] is not None for x in pw]))
    return out


def cmd_energy(a) -> None:
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    f = OUT / (a.file or "energy.json")
    res = json.loads(f.read_text()) if f.exists() else {}
    jobs = [] if a.only_conds else [(m, None, False) for m in ("random", "heuristic")] + [(m, s, False) for m in LEARNED for s in range(5)]
    jobs += [(c, s, True) for c in a.conds for s in range(10)]  # study conditions (results/study)
    for m, s, study in jobs:
        key = m if s is None else f"{m}-seed{s}"
        if key in res:
            continue
        if study and not (OUT / "train" / m / key / "predator.pt").exists():
            continue
        res[key] = {"method": m, "seed": s, "study": study, **thrust_stats(m, s, a.steps, a.warm, device, study)}
        r = res[key]
        print(f"{key:14s} thrust² {r['pred_th2']:.2f}  flips {r['pred_flip']:.2f}  cost {r['pred_cost']:.3f}  "
              f"income {r['pred_income']:.3f}  net {r['pred_net']:+.3f}", flush=True)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(res, indent=1))


# ------------------------------------------------------------------------------------------------ report
def load_crossplay() -> dict:
    R: dict = collections.defaultdict(list)
    for f in glob.glob(str(OUT / "crossplay" / "*.json")):
        d = json.loads(Path(f).read_text())
        mp, mq = d["method"].split("|")
        R[(mp, mq)] += d["per_world"]
    for f in glob.glob(str(BENCH / "eval" / "S1" / "*.json")):
        d = json.loads(Path(f).read_text())
        R[(d["method"], d["method"])] += d["per_world"]
    return R


def matrix(R, fn) -> np.ndarray:
    return np.array([[fn(R[(p, q)]) if R.get((p, q)) else np.nan for q in M] for p in M])


def md_matrix(title: str, A: np.ndarray, fmt: str) -> list[str]:
    out = [f"**{title}** (rows: prey brain, columns: predator brain; diagonal = benchmark)", "",
           "| prey \\ predators | " + " | ".join(NAMES[q] for q in M) + " |", "|---" * (len(M) + 1) + "|"]
    for i, p in enumerate(M):
        out.append(f"| {NAMES[p]} | " + " | ".join("–" if np.isnan(v) else fmt.format(v) for v in A[i]) + " |")
    return [*out, ""]


def cmd_report(a) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = load_crossplay()
    surv = matrix(R, lambda ws: 100 * np.mean([w["survived"] for w in ws]))
    pext = matrix(R, lambda ws: 100 * np.mean([w["extinct_at"]["predator"] is not None for w in ws]))
    yext = matrix(R, lambda ws: 100 * np.mean([w["extinct_at"]["prey"] is not None for w in ws]))
    life = matrix(R, lambda ws: np.mean([w["pred_mean_lifespan"] for w in ws]))

    figs = OUT / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, A, title, cmap in ((axes[0], surv, "Worlds with both species alive (%)", "RdYlGn"),
                               (axes[1], pext, "Worlds where the predators died out (%)", "RdYlGn_r")):
        ax.imshow(A, cmap=cmap, vmin=0, vmax=100)
        for i in range(len(M)):
            for j in range(len(M)):
                if not np.isnan(A[i, j]):
                    ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=8.5,
                            fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(M)), [NAMES[m] for m in M], rotation=35, ha="right")
        ax.set_yticks(range(len(M)), [NAMES[m] for m in M])
        ax.set_xlabel("predator brain")
        ax.set_ylabel("prey brain")
        ax.set_title(title)
    fig.suptitle("Cross-play on S1: every prey brain against every predator brain (25 worlds per cell; diagonal = benchmark)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(figs / "crossplay.png", dpi=200)
    plt.close(fig)

    lines = ["# Study 2: cross-play, energy and tactics", "",
             "Generated by `scripts/report_study.py report`. S1, 10,000 steps. Learned × learned cells pair the same training seed (5 seeds × 5 worlds = 25 worlds); cells with a fixed baseline also use 5 seeds; Random × Hand-coded use 5 worlds.", ""]
    # robustness summaries
    off = ~np.eye(len(M), dtype=bool)
    lines += ["## Who is strong", "",
              "| brain | predators die out, mean over the other methods' prey | as prey: other methods' predators die out |",
              "|---|---|---|"]
    for i, m in enumerate(M):
        as_pred = np.nanmean(pext[:, i][off[:, i]])
        as_prey = np.nanmean(pext[i, :][off[i, :]])
        lines.append(f"| {NAMES[m]} | {as_pred:.0f}% | {as_prey:.0f}% |")
    lines += ["", "Low in the first column = robust predators; high in the second = prey that starve any predator.", ""]
    lines += md_matrix("Both species alive (%)", surv, "{:.0f}")
    lines += md_matrix("Predators died out (%)", pext, "{:.0f}")
    lines += md_matrix("Prey died out (%)", yext, "{:.0f}")
    lines += md_matrix("Predator lifespan (steps)", life, "{:.0f}")

    ef = OUT / "energy.json"
    if ef.exists():
        E = json.loads(ef.read_text())
        by = collections.defaultdict(list)
        E = {k: r for k, r in E.items() if not r.get("study")}
        for r in E.values():
            by[r["method"]].append(r)
        lines += ["## Predator energy budget", "",
                  "Movement costs 0.15 × thrust² per step on top of a 0.15 base drain; speed carries over, so cruising is nearly free. Income = catches per predator-step (benchmark, S1) × 35. Thrust measured over 300 steps after a 100-step warm-up, one world, S1. Mean over seeds (range).", "",
                  "| predators | thrust² | thrust sign flips / step | cost / step | income / step | net / step | worlds where predators died out |",
                  "|---|---|---|---|---|---|---|"]

        def cell(rs, k, fmt):
            v = np.array([r[k] for r in rs])
            return fmt.format(v.mean()) + (f" ({fmt.format(v.min())}–{fmt.format(v.max())})" if len(v) > 1 else "")

        for m in M:
            rs = by.get(m)
            if not rs:
                continue
            lines.append(f"| {NAMES[m]} | {cell(rs, 'pred_th2', '{:.2f}')} | {cell(rs, 'pred_flip', '{:.2f}')} | "
                         f"{cell(rs, 'pred_cost', '{:.3f}')} | {cell(rs, 'pred_income', '{:.3f}')} | "
                         f"{cell(rs, 'pred_net', '{:+.3f}')} | {100 * np.mean([r['pred_extinct_share'] for r in rs]):.0f}% |")
        xs = np.array([r["pred_net"] for r in E.values()])
        ys = np.array([r["pred_extinct_share"] for r in E.values()])
        rho = np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1]
        lines += ["", f"Across all {len(xs)} brains, the rank correlation between net energy and the share of worlds where predators died out is {rho:+.2f}.", ""]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        col = {"random": "#9aa0a6", "heuristic": "#5f6368", "ppo": "#2e6b3b", "ga": "#c98a1e", "es": "#d65f2b",
               "neat": "#7b5ea7", "hybrid": "#2b7bb9"}
        for m in M:
            rs = by.get(m, [])
            ax.scatter([r["pred_net"] for r in rs], [100 * r["pred_extinct_share"] for r in rs], s=34, color=col[m],
                       label=NAMES[m], edgecolor="black", linewidth=0.4, zorder=3)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("predator net energy per step (income − cost)")
        ax.set_ylabel("worlds where predators died out (%)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(figs / "energy.png", dpi=200)
        plt.close(fig)

    beh = collections.defaultdict(list)
    for f in glob.glob(str(OUT / "behaviour" / "*.json")):
        d = json.loads(Path(f).read_text())
        beh[d["method"]].append(d)
    if beh:
        keys = [("prey_in_refuge", "prey in refuges", "{:.2f}"), ("flee_alignment", "prey flee alignment", "{:+.2f}"),
                ("prey_clustering", "prey clustering", "{:.2f}"), ("pred_speed", "predator speed (share of max)", "{:.2f}"),
                ("pursuit_share", "predators pursuing", "{:.2f}"), ("pred_sit_and_wait", "predators sitting and waiting", "{:.2f}"),
                ("ambush_index", "ambush index (1 = chance)", "{:.2f}"), ("edge_catch_share", "catches within 3 of a refuge", "{:.2f}")]
        lines += ["## Tactics (S1, 3 worlds × 3,000 steps, mean over seeds)", "",
                  "| measure | " + " | ".join(NAMES[m] for m in M if m in beh) + " |", "|---" * (len(beh) + 1) + "|"]
        for k, lab, fmt in keys:
            lines.append(f"| {lab} | " + " | ".join(fmt.format(np.nanmean([r[k] for r in beh[m]])) for m in M if m in beh) + " |")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["energy", "report"])
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warm", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--conds", nargs="*", default=[], help="study conditions to add to the energy table")
    p.add_argument("--file", default=None, help="energy table file name (default energy.json)")
    p.add_argument("--only-conds", action="store_true", help="measure only --conds, not the benchmark brains")
    p.add_argument("--base", default="s1_large.json", help="world; drag world: s1_drag.json with --bench/--out")
    p.add_argument("--bench", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    global BASE, BENCH, OUT
    BASE = a.base
    if a.bench:
        BENCH = Path(a.bench)
    if a.out:
        OUT = Path(a.out)
    {"energy": cmd_energy, "report": cmd_report}[a.command](a)


if __name__ == "__main__":
    main()
