"""Paper figures and statistics from the benchmark evaluation (results/bench/eval/).

    python scripts/analyze.py            # writes results/bench/figures/*.png and results/bench/STATS.md

Unit of analysis: a training seed. Each seed's 5 evaluation worlds are averaged first, so a method has n = 5
independent values per scenario (worlds within one seed share the same trained brains, so they are not
independent of each other). The fixed baselines (Random, Hand-coded) have no training seeds, so
each of their evaluation worlds counts as one sample. Survival curves use every world (time to first extinction of either species,
censored at the end of the run).

Statistics (numpy only, no scipy needed):
  - mean with a 95% bootstrap confidence interval over training seeds
  - two-sided permutation test of each method against PPO and against the hand-coded baseline (exact over all
    C(10, 5) = 252 relabellings when both groups have 5 seeds)
  - log-rank test of survival curves against PPO
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["random", "heuristic", "ppo", "ga", "es", "neat", "hybrid"]
NAMES = {
    "random": "Random",
    "heuristic": "Hand-coded",
    "ppo": "PPO",
    "ga": "GA",
    "es": "ES",
    "neat": "NEAT",
    "hybrid": "Hybrid",
}
COLORS = {
    "random": "#9aa0a6",
    "heuristic": "#5f6368",
    "ppo": "#2e6b3b",
    "ga": "#c98a1e",
    "es": "#d65f2b",
    "neat": "#7b5ea7",
    "hybrid": "#2b7bb9",
}
SCEN = {
    "S1": "S1 baseline",
    "S2": "S2 slow grass",
    "S3": "S3 fast prey",
    "S4": "S4 crowded start",
    "S5": "S5 many predators",
}
METRICS = [
    ("survived", "Worlds with both species alive", "higher"),
    ("attack_rate_e6", "Attack rate (catches per prey·predator·step ×10⁶)", "lower for prey"),
    ("pred_catches_per_1k_agent_steps", "Catches per 1,000 predator-steps", "higher for predators"),
    ("prey_mean_lifespan", "Prey mean lifespan (steps)", "higher"),
    ("pred_mean_lifespan", "Predator mean lifespan (steps)", "higher"),
    ("cv_prey", "Prey population CV, 2nd half", "lower = steadier"),
    ("cv_pred", "Predator population CV, 2nd half", "lower = steadier"),
]


def load(eval_dir: Path):
    """Returns recs[scenario][method] = list of per-seed dicts (metric -> mean over that seed's worlds) and
    surv[scenario][method] = list of (time, event) per world."""
    recs: dict = defaultdict(lambda: defaultdict(list))
    surv: dict = defaultdict(lambda: defaultdict(list))
    steps_of = {}
    for f in sorted(eval_dir.glob("S*/*.json")):
        d = json.loads(f.read_text())
        sc, m = d["scenario"], d["method"]
        steps_of[sc] = d["steps"]
        pw = d["per_world"]
        # learned methods: one sample per training seed (mean of its worlds); the fixed baselines have no
        # training seeds, so each of their worlds is one sample
        groups = [[w] for w in pw] if m in ("random", "heuristic") else [pw]
        for g in groups:
            row = {"survived": np.mean([w["survived"] for w in g]), "policy": d["policy"]}
            for k, *_ in METRICS[1:]:
                if k.startswith("cv_"):
                    row[k] = np.mean([w["cv_2nd_half"][0 if k == "cv_prey" else 1] for w in g])
                else:
                    row[k] = np.mean([w[k] for w in g])
            recs[sc][m].append(row)
        for w in pw:
            ts = [t for t in (w["extinct_at"]["prey"], w["extinct_at"]["predator"]) if t]
            surv[sc][m].append((min(ts), 1) if ts else (d["steps"], 0))
    return recs, surv, steps_of


def boot_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 2:
        return float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def perm_test(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided permutation test on the difference of means (exact when small)."""
    allv = np.concatenate([a, b])
    obs = abs(a.mean() - b.mean())
    n = len(a)
    combos = list(itertools.combinations(range(len(allv)), n))
    if len(combos) > 20000:
        rng = np.random.default_rng(0)
        combos = [tuple(rng.choice(len(allv), n, replace=False)) for _ in range(20000)]
    hits = 0
    for c in combos:
        mask = np.zeros(len(allv), bool)
        mask[list(c)] = True
        if abs(allv[mask].mean() - allv[~mask].mean()) >= obs - 1e-12:
            hits += 1
    return hits / len(combos)


def kaplan_meier(data: list[tuple[int, int]]):
    t = np.array([d[0] for d in data], float)
    e = np.array([d[1] for d in data], int)
    times = np.unique(t[e == 1])
    s, xs, ys = 1.0, [0.0], [1.0]
    for u in times:
        at_risk = (t >= u).sum()
        died = ((t == u) & (e == 1)).sum()
        s *= 1 - died / at_risk
        xs += [u, u]
        ys += [ys[-1], s]
    return np.array(xs), np.array(ys)


def logrank(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    """Two-group log-rank test p-value (chi-square, 1 dof)."""
    from math import erfc, sqrt

    t = np.array([x[0] for x in a + b], float)
    e = np.array([x[1] for x in a + b], int)
    g = np.array([0] * len(a) + [1] * len(b))
    O_E, V = 0.0, 0.0
    for u in np.unique(t[e == 1]):
        r = t >= u
        n, n1 = r.sum(), (r & (g == 0)).sum()
        d = ((t == u) & (e == 1)).sum()
        d1 = ((t == u) & (e == 1) & (g == 0)).sum()
        O_E += d1 - d * n1 / n
        if n > 1:
            V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    if V <= 0:
        return 1.0
    z = abs(O_E) / sqrt(V)
    return erfc(z / sqrt(2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(ROOT / "results" / "bench"))
    a = ap.parse_args()
    bench = Path(a.bench)
    recs, surv, steps_of = load(bench / "eval")
    if not recs:
        raise SystemExit("no evaluation files; run `python scripts/benchmark.py eval` first")
    figs = bench / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    scen = [s for s in SCEN if s in recs]
    methods = [m for m in ORDER if any(m in recs[s] for s in scen)]

    # ---------------------------------------------------------------- survival curves, one panel per scenario
    fig, axes = plt.subplots(1, len(scen), figsize=(3.0 * len(scen), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sc in zip(axes, scen, strict=True):
        for m in methods:
            if m not in surv[sc]:
                continue
            xs, ys = kaplan_meier(surv[sc][m])
            xs = np.append(xs, steps_of[sc])
            ys = np.append(ys, ys[-1])
            ax.plot(xs / 1000, ys, color=COLORS[m], lw=1.6, label=NAMES[m])
        ax.set_title(SCEN[sc])
        ax.set_xlabel("step (thousands)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("share of worlds with both species")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(figs / "survival.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- metric bars on S1 (mean ± 95% CI over seeds)
    s1 = "S1" if "S1" in recs else scen[0]
    keys = [m for m in METRICS if m[0] != "survived"]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    for ax, (k, label, better) in zip(axes.ravel(), keys, strict=False):
        ms = [m for m in methods if m in recs[s1]]
        vals = [np.array([r[k] for r in recs[s1][m]]) for m in ms]
        means = [v.mean() for v in vals]
        cis = [boot_ci(v) for v in vals]
        err = np.array(
            [
                [mu - lo for mu, (lo, _) in zip(means, cis, strict=True)],
                [hi - mu for mu, (_, hi) in zip(means, cis, strict=True)],
            ]
        )
        ax.bar(range(len(ms)), means, yerr=err, color=[COLORS[m] for m in ms], capsize=3, width=0.7)
        for i, v in enumerate(vals):
            ax.scatter(
                np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)),
                v,
                s=8,
                color="black",
                alpha=0.5,
                zorder=3,
            )
        ax.set_xticks(range(len(ms)), [NAMES[m] for m in ms], rotation=35, ha="right")
        ax.set_title(f"{label}\n({better})", fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[len(keys) :]:
        ax.axis("off")
    fig.suptitle(
        f"{SCEN[s1]}: mean with 95% bootstrap CI; dots = training seeds (Random and Hand-coded: evaluation worlds)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(figs / "metrics_S1.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- robustness: survival share across scenarios
    fig, ax = plt.subplots(figsize=(7, 2.8))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        ys = [
            np.mean([r["survived"] for r in recs[sc].get(m, [])]) if m in recs[sc] else np.nan
            for sc in scen
        ]
        ax.bar(
            np.arange(len(scen)) + (j - len(methods) / 2 + 0.5) * w,
            ys,
            width=w,
            color=COLORS[m],
            label=NAMES[m],
        )
    ax.set_xticks(range(len(scen)), [SCEN[s] for s in scen])
    ax.set_ylabel(f"worlds surviving {max(steps_of.values()) // 1000 or max(steps_of.values())}{'k' if max(steps_of.values()) >= 1000 else ''} steps")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=4, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figs / "robustness.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- example population traces on S1
    fig, axes = plt.subplots(1, len(methods), figsize=(2.2 * len(methods), 2.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, methods, strict=True):
        f = (
            bench
            / "eval"
            / s1
            / (f"{m}.pops.npy" if m in ("random", "heuristic") else f"{m}-seed0.pops.npy")
        )
        if not f.exists():
            ax.axis("off")
            continue
        pops = np.load(f)  # [T/10, W, 2]
        x = np.arange(pops.shape[0]) * 10 / 1000
        ax.plot(x, pops[:, 0, 0], color="#a27a1a", lw=1, label="prey")
        ax2 = ax.twinx()
        ax2.plot(x, pops[:, 0, 1], color="#c4401f", lw=1, label="predators")
        ax.set_title(NAMES[m])
        ax.set_xlabel("step (k)")
        ax2.spines["top"].set_visible(False)
    axes[0].set_ylabel("prey", color="#a27a1a")
    ax2.set_ylabel("predators", color="#c4401f")
    fig.suptitle(f"{SCEN[s1]}: populations in one evaluation world (seed 0)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "populations_S1.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- statistics
    lines = [
        "# Benchmark statistics",
        "",
        (
            "Generated by `scripts/analyze.py`. n = training seeds (each the mean of its evaluation worlds); "
            "for Random and Hand-coded, n = evaluation worlds. "
            "p-values: two-sided exact permutation tests on seed means; log-rank on per-world time to first "
            "extinction. With 5 seeds per method the smallest possible permutation p-value is 2/252 ≈ 0.008."
        ),
        "",
    ]
    for sc in scen:
        lines += [
            f"## {SCEN[sc]}",
            "",
            "| metric | " + " | ".join(NAMES[m] for m in methods) + " |",
            "|---" * (len(methods) + 1) + "|",
        ]
        for k, label, _ in METRICS:
            cells = []
            for m in methods:
                if m not in recs[sc]:
                    cells.append("–")
                    continue
                v = np.array([r[k] for r in recs[sc][m]])
                lo, hi = boot_ci(v)
                cells.append(
                    f"{v.mean():.2f} [{lo:.2f}, {hi:.2f}]" if len(v) > 1 else f"{v.mean():.2f}"
                )
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "p-values against PPO / against Hand-coded (learned methods only):",
            "",
            "| metric | "
            + " | ".join(NAMES[m] for m in methods if m not in ("random", "heuristic"))
            + " |",
            "|---" * (len([m for m in methods if m not in ("random", "heuristic")]) + 1) + "|",
        ]
        for k, label, _ in METRICS:
            cells = []
            for m in methods:
                if m in ("random", "heuristic"):
                    continue
                if m not in recs[sc]:
                    cells.append("–")
                    continue
                v = np.array([r[k] for r in recs[sc][m]])
                p_ppo = (
                    perm_test(v, np.array([r[k] for r in recs[sc]["ppo"]]))
                    if m != "ppo" and "ppo" in recs[sc]
                    else None
                )
                # the baseline has no training seeds: compare against its world-level spread replicated per seed
                p_h = None
                if "heuristic" in recs[sc]:
                    h = np.array([r[k] for r in recs[sc]["heuristic"]])
                    if len(h) > 1:
                        p_h = perm_test(v, h)
                cells.append(
                    f"{'—' if p_ppo is None else f'{p_ppo:.3f}'} / {'—' if p_h is None else f'{p_h:.3f}'}"
                )
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        if "ppo" in surv[sc]:
            lr = [
                f"{NAMES[m]} {logrank(surv[sc][m], surv[sc]['ppo']):.3f}"
                for m in methods
                if m != "ppo" and m in surv[sc]
            ]
            lines += ["", "Log-rank p against PPO (survival): " + ", ".join(lr), ""]
    (bench / "STATS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nfigures in {figs}")


if __name__ == "__main__":
    main()
