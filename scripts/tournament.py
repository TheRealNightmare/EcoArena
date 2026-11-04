"""Ecological tournament: cross-play treated as a two-population game between prey brains and predator brains.

    python scripts/tournament.py      # results/study/TOURNAMENT.md + results/study/figures/tournament.png

Strategies: the 7 brains (Random, Hand-coded, PPO, GA, ES, NEAT, Hybrid) for each species. Payoffs from the S1
cross-play (results/study/crossplay/) plus the benchmark diagonal, mean over seeds and worlds. Two games:

  objective game  each species' payoff is its own reward per 1,000 animal-steps: what every method was trained on
  survival game   each species' payoff is the share of worlds in which it is still alive after 10,000 steps

For each game: pure-strategy Nash equilibria, all Nash equilibria (support enumeration), alpha-Rank (Omidshafiei et
al. 2019: a Markov chain of strategy swaps under selection, ranked by its stationary mass) and best-response cycles
(intransitivity: no stable answer, each side keeps switching).
"""

from __future__ import annotations

import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
M = ["random", "heuristic", "ppo", "ga", "es", "neat", "hybrid"]
NAMES = {"random": "Random", "heuristic": "Hand-coded", "ppo": "PPO", "ga": "GA", "es": "ES", "neat": "NEAT",
         "hybrid": "Hybrid"}
OUT = ROOT / "results" / "study"


def load() -> dict:
    R: dict = collections.defaultdict(list)
    for f in glob.glob(str(OUT / "crossplay" / "*.json")):
        d = json.loads(Path(f).read_text())
        p, q = d["method"].split("|")
        R[(p, q)] += d["per_world"]
    for f in glob.glob(str(ROOT / "results" / "bench" / "eval" / "S1" / "*.json")):
        d = json.loads(Path(f).read_text())
        R[(d["method"], d["method"])] += d["per_world"]
    return R


def payoff(R, fn) -> np.ndarray:
    return np.array([[np.mean([fn(w) for w in R[(p, q)]]) for q in M] for p in M])


def alpharank(A: np.ndarray, B: np.ndarray, alpha: float = 50.0, m: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Two-population alpha-Rank. A[i, j] = row (prey) payoff, B[i, j] = column (predator) payoff, each scaled to
    [0, 1]. Returns the stationary mass of each prey strategy and each predator strategy."""
    n1, n2 = A.shape
    A = (A - A.min()) / max(A.max() - A.min(), 1e-12)
    B = (B - B.min()) / max(B.max() - B.min(), 1e-12)
    eta = 1.0 / ((n1 - 1) + (n2 - 1))

    def rho(f: float) -> float:
        if abs(f) < 1e-12:
            return 1.0 / m
        x = np.clip(-alpha * f, -700, 700)
        return float(np.expm1(x) / np.expm1(np.clip(m * x, -700, 700)))

    S = n1 * n2
    T = np.zeros((S, S))
    for i in range(n1):
        for j in range(n2):
            s = i * n2 + j
            for k in range(n1):
                if k != i:
                    T[s, k * n2 + j] = eta * rho(A[k, j] - A[i, j])
            for k in range(n2):
                if k != j:
                    T[s, i * n2 + k] = eta * rho(B[i, k] - B[i, j])
            T[s, s] = 1.0 - T[s].sum()
    w, v = np.linalg.eig(T.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    pi = np.abs(pi) / np.abs(pi).sum()
    P = pi.reshape(n1, n2)
    return P.sum(1), P.sum(0)


def br_cycles(A: np.ndarray, B: np.ndarray) -> tuple[list, list]:
    """Pure Nash equilibria and best-response cycles (alternating best responses that never settle)."""
    n1, n2 = A.shape
    pure = [(i, j) for i in range(n1) for j in range(n2) if A[i, j] >= A[:, j].max() - 1e-12 and B[i, j] >= B[i, :].max() - 1e-12]
    cycles = set()
    for i0 in range(n1):
        for j0 in range(n2):
            seen, path, (i, j), turn = {}, [], (i0, j0), 0
            while (i, j, turn) not in seen:
                seen[(i, j, turn)] = len(path)
                path.append((i, j))
                if turn == 0:
                    i = int(np.argmax(A[:, j]))
                else:
                    j = int(np.argmax(B[i, :]))
                turn ^= 1
            cyc = path[seen[(i, j, turn)]:]
            profiles = tuple(sorted(set(cyc)))
            if len(profiles) > 1:
                cycles.add(profiles)
    return pure, sorted(cycles)


def nash_all(A: np.ndarray, B: np.ndarray) -> list:
    import nashpy

    out = []
    for x, y in nashpy.Game(A, B).support_enumeration():
        out.append((np.round(x, 3), np.round(y, 3)))
    return out


def main() -> None:
    R = load()
    games = {
        "objective": (payoff(R, lambda w: w["prey_reward_per_1k_agent_steps"]),
                      payoff(R, lambda w: w["pred_reward_per_1k_agent_steps"]),
                      "reward per 1,000 animal-steps (the training objective)"),
        "survival": (payoff(R, lambda w: float(w["extinct_at"]["prey"] is None)),
                     payoff(R, lambda w: float(w["extinct_at"]["predator"] is None)),
                     "share of worlds in which the species survives 10,000 steps"),
    }
    lines = ["# Ecological tournament", "",
             "Generated by `scripts/tournament.py`. Prey brains (rows) against predator brains (columns) on S1; payoffs are means over the cross-play worlds (diagonal: benchmark). See the script for the method.", ""]
    ranks = {}
    for g, (A, B, what) in games.items():
        pa, pb = alpharank(A, B)
        ranks[g] = (pa, pb)
        pure, cyc = br_cycles(A, B)
        lines += [f"## {g.capitalize()} game: payoff = {what}", "",
                  "| brain | alpha-Rank mass as prey | alpha-Rank mass as predators |", "|---|---|---|"]
        for k in np.argsort(-(pa + pb)):
            lines.append(f"| {NAMES[M[k]]} | {pa[k]:.3f} | {pb[k]:.3f} |")
        lines += ["", "Pure Nash equilibria (prey, predators): " + (", ".join(f"({NAMES[M[i]]}, {NAMES[M[j]]})" for i, j in pure) or "none") + "."]
        if cyc:
            lines.append("Best-response cycles (intransitive, no stable answer): " + "; ".join(
                " → ".join(f"({NAMES[M[i]]}, {NAMES[M[j]]})" for i, j in c) for c in cyc) + ".")
        else:
            lines.append("No best-response cycles: best responses settle.")
        try:
            eqs = nash_all(A, B)
            lines.append(f"All Nash equilibria ({len(eqs)}):")
            for x, y in eqs:
                fx = ", ".join(f"{NAMES[M[i]]} {x[i]:.2f}" for i in range(len(M)) if x[i] > 0)
                fy = ", ".join(f"{NAMES[M[j]]} {y[j]:.2f}" for j in range(len(M)) if y[j] > 0)
                lines.append(f"- prey: {fx} | predators: {fy}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"(Nash enumeration failed: {e})")
        lines.append("")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"random": "#9aa0a6", "heuristic": "#5f6368", "ppo": "#2e6b3b", "ga": "#c98a1e", "es": "#d65f2b",
           "neat": "#7b5ea7", "hybrid": "#2b7bb9"}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0), sharey=True)
    for ax, (g, side, k) in zip(axes, [("objective", "prey", 0), ("objective", "predators", 1),
                                        ("survival", "prey", 0), ("survival", "predators", 1)], strict=True):
        v = ranks[g][k]
        order = np.argsort(-v)
        ax.barh(range(len(M)), v[order], color=[col[M[i]] for i in order])
        ax.set_yticks(range(len(M)), [NAMES[M[i]] for i in order])
        ax.invert_yaxis()
        ax.set_title(f"{g} game: {side}", fontsize=9)
        ax.set_xlabel("alpha-Rank mass")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "tournament.png", dpi=200)
    (OUT / "TOURNAMENT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
