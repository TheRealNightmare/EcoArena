"""Population dynamics of the benchmark: cycles, lags, Lotka-Volterra fits and how fast predators decline.

    python scripts/dynamics.py [--eval results/bench/eval] [--scenario S1]

Writes results/bench/DYNAMICS.md and results/bench/figures/phase_<S>.png, dynamics_<S>.png. CPU only.

Per world, the series (sampled every 10 steps) is cut at the first extinction and the first 1,000 steps are
dropped as a transient. Then:
  period      first autocorrelation peak of the detrended prey series (steps); nan if no clear peak
  lag         predator peak lag behind prey, from the cross-correlation (steps)
  pred_trend  slope of log(predators) per 1,000 steps (negative = predators dwindling)
  LV fit      d ln(prey)/dt = a - b * predators;  d ln(predators)/dt = -c + d * prey  (least squares, R^2)
              x* = c/d and y* = a/b are the fitted equilibria
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["random", "heuristic", "ppo", "ga", "es", "neat", "hybrid"]
NAMES = {"random": "Random", "heuristic": "Hand-coded", "ppo": "PPO", "ga": "GA", "es": "ES", "neat": "NEAT",
         "hybrid": "Hybrid"}
COLORS = {"random": "#9aa0a6", "heuristic": "#5f6368", "ppo": "#2e6b3b", "ga": "#c98a1e", "es": "#d65f2b",
          "neat": "#7b5ea7", "hybrid": "#2b7bb9"}
DT = 10  # steps between samples
SKIP = 1000


def detrend(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x))
    return x - np.polyval(np.polyfit(t, x, 1), t)


def acf_period(x: np.ndarray) -> float:
    x = detrend(x.astype(float))
    if x.std() < 1e-9 or len(x) < 40:
        return np.nan
    x = (x - x.mean()) / x.std()
    ac = np.correlate(x, x, "full")[len(x) - 1 :] / len(x)
    neg = np.nonzero(ac < 0)[0]
    if not len(neg):
        return np.nan
    start = neg[0]
    seg = ac[start : len(ac) // 2]
    if len(seg) < 3:
        return np.nan
    k = start + int(np.argmax(seg))
    return k * DT if ac[k] > 0.1 else np.nan


def xcorr_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> float:
    x, y = detrend(x.astype(float)), detrend(y.astype(float))
    if x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan
    x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    lags = range(min(max_lag, len(x) // 3))
    c = [np.mean(x[: len(x) - k] * y[k:]) for k in lags]
    return int(np.argmax(c)) * DT


def lv_fit(x: np.ndarray, y: np.ndarray, stride: int = 10) -> dict:
    """Fit on 100-step increments (stride samples): 10-step changes are dominated by birth/death noise."""
    x, y = x[::stride], y[::stride]
    dt = DT * stride
    ok = (x[:-1] > 0) & (x[1:] > 0) & (y[:-1] > 0) & (y[1:] > 0)
    if ok.sum() < 12:
        return {}
    dlx = (np.log(x[1:]) - np.log(x[:-1]))[ok] / dt
    dly = (np.log(y[1:]) - np.log(y[:-1]))[ok] / dt
    X1 = np.c_[np.ones(ok.sum()), -y[:-1][ok]]
    X2 = np.c_[-np.ones(ok.sum()), x[:-1][ok]]
    (a, b), *_ = np.linalg.lstsq(X1, dlx, rcond=None)
    (c, d), *_ = np.linalg.lstsq(X2, dly, rcond=None)
    r2 = lambda X, p, z: 1 - np.sum((z - X @ p) ** 2) / max(np.sum((z - z.mean()) ** 2), 1e-12)
    return {"a": a, "b": b, "c": c, "d": d, "r2_prey": r2(X1, np.array([a, b]), dlx),
            "r2_pred": r2(X2, np.array([c, d]), dly), "x_star": c / d if d else np.nan, "y_star": a / b if b else np.nan}


def analyse(pops: np.ndarray, steps: int) -> list[dict]:
    out = []
    for w in range(pops.shape[1]):
        p = pops[:, w].astype(float)
        dead = np.nonzero((p == 0).any(1))[0]
        end = dead[0] if len(dead) else len(p)
        seg = p[SKIP // DT : end]
        rec = {"survived": not len(dead), "extinct_step": int(end * DT) if len(dead) else None, "n": len(seg)}
        if len(seg) >= 60:
            x, y = seg[:, 0], seg[:, 1]
            per = acf_period(x)
            rec.update(period=per, lag=xcorr_lag(x, y, int((3000 if np.isnan(per) else per) / DT)),
                       cv_prey=x.std() / x.mean(), cv_pred=y.std() / max(y.mean(), 1e-9),
                       pred_trend=np.polyfit(np.arange(len(y)) * DT / 1000, np.log(np.maximum(y, 0.5)), 1)[0],
                       mean_prey=x.mean(), mean_pred=y.mean(), **lv_fit(x, y))
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(ROOT / "results" / "bench" / "eval"))
    ap.add_argument("--scenario", default="S1")
    a = ap.parse_args()
    ev = Path(a.eval)
    bench = ev.parent
    rows: dict[str, list[dict]] = defaultdict(list)
    series: dict[str, np.ndarray] = {}
    for f in sorted((ev / a.scenario).glob("*.json")):
        d = json.loads(f.read_text())
        pops = np.load(f.with_suffix(".pops.npy"))
        rows[d["method"]] += analyse(pops, d["steps"])
        if d["policy"] in ("random", "heuristic") or d["policy"].endswith("seed0"):
            series[d["method"]] = pops
    methods = [m for m in ORDER if m in rows]

    def agg(m: str, k: str) -> str:
        v = np.array([r[k] for r in rows[m] if k in r and r[k] is not None], float)
        v = v[np.isfinite(v)]
        return f"{np.median(v):.3g} ({len(v)})" if len(v) else "–"

    keys = [("period", "cycle period, steps"), ("lag", "predator lag, steps"), ("cv_prey", "prey CV"),
            ("cv_pred", "predator CV"), ("pred_trend", "predator trend, log/1k steps"), ("mean_prey", "mean prey"),
            ("mean_pred", "mean predators"), ("r2_prey", "LV fit R² prey"), ("r2_pred", "LV fit R² predators"),
            ("x_star", "LV prey equilibrium x*"), ("y_star", "LV predator equilibrium y*")]
    md = [f"# Population dynamics ({a.scenario})", "",
          ("Generated by `scripts/dynamics.py`. Median over evaluation worlds (count in brackets), each cut at its "
           "first extinction, first 1,000 steps dropped."), "",
          "| measure | " + " | ".join(NAMES[m] for m in methods) + " |", "|---" * (len(methods) + 1) + "|"]
    md += [f"| {lab} | " + " | ".join(agg(m, k) for m in methods) + " |" for k, lab in keys]
    (bench / f"DYNAMICS_{a.scenario}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # phase portraits: prey vs predators, world 0 of seed 0, coloured by time
    fig, axes = plt.subplots(1, len(methods), figsize=(2.3 * len(methods), 2.5))
    for ax, m in zip(np.atleast_1d(axes), methods, strict=True):
        p = series[m][:, 0].astype(float)
        t = np.arange(len(p)) * DT / 1000
        ax.scatter(p[:, 0], p[:, 1], c=t, cmap="viridis", s=1.5)
        ax.plot(p[0, 0], p[0, 1], "o", color="black", ms=3)
        ax.set_title(NAMES[m])
        ax.set_xlabel("prey")
    np.atleast_1d(axes)[0].set_ylabel("predators")
    fig.suptitle(f"{a.scenario}: phase portraits (one world, seed 0; colour = time, dot = start)", fontsize=9)
    fig.tight_layout()
    fig.savefig(bench / "figures" / f"phase_{a.scenario}.png", dpi=200)
    plt.close(fig)

    # predator decline: median log predator population over time across all worlds of each method
    fig, ax = plt.subplots(figsize=(5.2, 3))
    allp: dict[str, list[np.ndarray]] = defaultdict(list)
    for f in sorted((ev / a.scenario).glob("*.json")):
        d = json.loads(f.read_text())
        allp[d["method"]].append(np.load(f.with_suffix(".pops.npy"))[:, :, 1])
    for m in methods:
        y = np.concatenate(allp[m], axis=1).astype(float)  # [T, worlds]
        t = np.arange(y.shape[0]) * DT / 1000
        ax.plot(t, np.median(y, 1), color=COLORS[m], lw=1.6, label=NAMES[m])
        ax.fill_between(t, np.percentile(y, 25, 1), np.percentile(y, 75, 1), color=COLORS[m], alpha=0.12)
    ax.set_xlabel("step (thousands)")
    ax.set_ylabel("predators (median, IQR)")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(bench / "figures" / f"predators_{a.scenario}.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
