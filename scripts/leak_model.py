"""Analytic model of the energy leak: when can a predator lineage replace itself, and can selection see the leak?

    python scripts/leak_model.py            # reads results/study/energy.json and energy_signal.json

Two parts, both built only from the world rules (configs/s1_large.json) and measured behaviour, with no fitted
parameters:

1. Lineage replacement. A predator lives at most max_age steps and pays base_drain + move_cost * thrust^2 per step.
   Its catches over a lifetime are Poisson with the brain's measured catch rate; each catch gives energy_per_catch.
   A newborn starts with child_energy and breeds at repro_threshold, handing child_energy to the child, so each child
   costs child_energy of banked surplus (at most max_age / repro_cooldown children). The expected number of children
   per predator,
       R0 = E_k[ min(floor((energy_per_catch * k - cost * L) / child_energy), L / cooldown)^+ ],  k ~ Poisson(rate*L),
   must reach 1 for the predators to persist. We compare R0 with the measured predator extinction of every brain.

2. Selection resolution. GA, ES and NEAT rank genomes by mean reward over a T-step window. The energy a jittering
   predator wastes is a fixed per-step difference D; the luck of catches and deaths gives noise sd(T). We fit
   sd(T) ~ T^-b to the measured spreads (independent steps would give b = 1/2), predict the share of elites a
   Gaussian ranking would take from the better half, and the share of selection pressure that goes to the energy
   trait, d^2 / (1 + d^2) with d = D / sd(T) (the trait's share of fitness variance).

Writes results/study/leak_model.json, results/study/LEAK_MODEL.md and paper/figures/leak_model.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, poisson, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecoarena.env import load_config

STUDY = ROOT / "results" / "study"
COLORS = {"ppo": "#3b6fb6", "ga": "#c8553d", "es": "#2e8b57", "neat": "#8e5ea2", "hybrid": "#d29a2a",
          "random": "#888888", "heuristic": "#888888"}


def r0(rate: float, th2: float, q, L: int | None = None) -> float:
    """Expected children per predator lifetime (catch rate per predator-step, mean thrust^2)."""
    L = L or q.max_age
    cost = q.base_drain + q.move_cost * th2
    k = np.arange(0, int(rate * L * 4 + 60))
    kids = np.clip(np.floor((q.energy_per_catch * k - cost * L) / q.child_energy), 0, L // q.repro_cooldown)
    return float((poisson.pmf(k, rate * L) * kids).sum())


def r0_contour(q, level: float, th2s: np.ndarray) -> np.ndarray:
    """Catch rate (per 1k predator-steps) at which R0 = level, for each thrust^2."""
    out = []
    for th2 in th2s:
        lo, hi = 1e-4, 0.03
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if r0(mid, th2, q) < level else (lo, mid)
        out.append(1000 * (lo + hi) / 2)
    return np.array(out)


def elite_share(d: float, P: int = 50, half: int = 25, elites: int = 5, n: int = 100_000) -> float:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, P))
    x[:, :half] += d
    return float((np.argsort(-x, 1)[:, :elites] < half).mean())


def holdout(energy: Path, base: str, cut: float) -> dict:
    """Out-of-sample test: R0 for brains the model never saw (another world), judged with the main world's cut."""
    q = load_config("S1", base=base).predator
    rows = []
    for name, v in json.loads(energy.read_text()).items():
        rows.append({"brain": name, "th2": v["pred_th2"], "catches_per_1k": v["catches_per_1k"], "net": v["pred_net"],
                     "r0": r0(v["catches_per_1k"] / 1000, v["pred_th2"], q), "extinct_share": v["pred_extinct_share"]})
    R = np.array([r["r0"] for r in rows])
    ext = np.array([r["extinct_share"] for r in rows]) >= 0.5
    pos, neg = R[ext], R[~ext]
    auc = float(np.mean([(a < b) + 0.5 * (a == b) for a in pos for b in neg])) if len(pos) and len(neg) else None
    return {"base": base, "n": len(rows), "n_extinct": int(ext.sum()), "accuracy_r0_1": float(((R < 1) == ext).mean()),
            "main_world_cut": cut, "accuracy_main_cut": float(((R < cut) == ext).mean()), "auc": auc, "brains": rows}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=None, help="energy.json of another world, e.g. results/study_drag/energy.json")
    ap.add_argument("--holdout-base", default="s1_drag.json")
    args = ap.parse_args()
    if args.holdout:
        cut = json.loads((STUDY / "leak_model.json").read_text())["lineage"]["best_cut"]
        h = holdout(Path(args.holdout), args.holdout_base, cut)
        out = Path(args.holdout).with_name("leak_model_holdout.json")
        out.write_text(json.dumps(h, indent=1))
        print(f"{h['n']} unseen brains ({h['n_extinct']} with predators extinct): R0<1 rule {100 * h['accuracy_r0_1']:.0f}%, "
              f"main-world cut R0<{cut:.2f} {100 * h['accuracy_main_cut']:.0f}%, AUC {h['auc']}")
        for r in h["brains"]:
            print(f"  {r['brain']:18s} th2 {r['th2']:.2f} catch {r['catches_per_1k']:.2f} R0 {r['r0']:.2f} ext {r['extinct_share']:.1f}")
        return
    q = load_config("S1", base="s1_large.json").predator
    brains = json.loads((STUDY / "energy.json").read_text())
    sig = json.loads((STUDY / "energy_signal.json").read_text())

    # ---------------------------------------------------------------- 1. lineage replacement
    rows = []
    for name, v in brains.items():
        R = r0(v["catches_per_1k"] / 1000, v["pred_th2"], q)
        rows.append({"brain": name, "method": v["method"], "th2": v["pred_th2"], "catches_per_1k": v["catches_per_1k"],
                     "net": v["pred_net"], "r0": R, "extinct_share": v["pred_extinct_share"]})
    R = np.array([r["r0"] for r in rows])
    ext = np.array([r["extinct_share"] for r in rows]) >= 0.5
    rho, p = spearmanr(R, [r["extinct_share"] for r in rows])
    rule1 = float(((R < 1) == ext).mean())  # zero-parameter rule: extinct iff R0 < 1
    # best single cut (fitted, reported as such) and AUC
    cuts = np.sort(R)
    accs = [((R < c) == ext).mean() for c in cuts]
    best = float(cuts[int(np.argmax(accs))])
    pos, neg = R[ext], R[~ext]
    auc = float(np.mean([(a < b) + 0.5 * (a == b) for a in pos for b in neg]))
    lineage = {"rule": "extinct iff R0 < 1", "accuracy_r0_1": rule1, "n": len(rows), "spearman_rho": float(rho),
               "spearman_p": float(p), "auc": auc, "best_cut": best, "accuracy_best_cut": float(max(accs)),
               "breakeven_catches_per_1k_at_th2_0": float(r0_contour(q, 1.0, np.array([0.0]))[0]),
               "breakeven_catches_per_1k_at_th2_0.6": float(r0_contour(q, 1.0, np.array([0.6]))[0])}

    # ---------------------------------------------------------------- 2. selection resolution
    D = sig["saving_per_agent_step"]
    T = np.array(sorted(int(t) for t in sig["windows"]))
    sd = np.array([sig["windows"][str(t)]["noise_sd"] for t in T])
    measured_share = np.array([sig["windows"][str(t)]["elites_from_credited_half"] for t in T])
    gens = np.array([sig["windows"][str(t)]["generations"] for t in T])
    slope, icpt = np.polyfit(np.log(T), np.log(sd), 1)
    snr = D / sd
    t_star_fit = float(np.exp((np.log(D) - icpt) / slope))  # SNR = 1
    t_star_iid = float(T[0] * (1 / snr[0]) ** 2)  # SNR ~ sqrt(T) from the 128-step point
    sel = {"saving_per_step": D, "windows": T.tolist(), "noise_sd": sd.tolist(), "snr": snr.tolist(),
           "noise_exponent": float(-slope), "t_star_fit": t_star_fit, "t_star_iid": t_star_iid,
           "elites_measured": measured_share.tolist(),
           "elites_measured_se": np.sqrt(measured_share * (1 - measured_share) / (5 * gens)).tolist(),
           "elites_gaussian": [elite_share(d) for d in snr],
           "selection_share": (snr ** 2 / (1 + snr ** 2)).tolist()}

    out = {"lineage": lineage, "selection": sel, "brains": rows}
    (STUDY / "leak_model.json").write_text(json.dumps(out, indent=1))

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    th = np.linspace(0, 1.0, 201)
    a = ax[0]
    a.plot(th, r0_contour(q, 1.0, th), "k-", lw=1.4, label=r"$R_0=1$ (model)")
    a.fill_between(th, 0, r0_contour(q, 1.0, th), color="#c8553d", alpha=0.07)
    for r in rows:
        c = COLORS.get(r["method"].split("-")[0], "#555")
        mk = "D" if "cost5" in r["brain"] else "o"  # energy fix (movement cost x5 in training)
        a.scatter(r["th2"], r["catches_per_1k"], s=22 if mk == "D" else 26, marker=mk,
                  color=c if r["extinct_share"] < 0.5 else "white", edgecolor=c, lw=1.2, zorder=3)
    for m in ("ppo", "ga", "es", "neat", "hybrid"):
        a.scatter([], [], color=COLORS[m], s=26, label=m.upper() if m != "hybrid" else "Hybrid")
    a.scatter([], [], color="#888", s=26, label="Random / Hand-coded")
    a.scatter([], [], color="#555", marker="D", s=22, label="trained with cost x5")
    a.scatter([], [], color="white", edgecolor="#555", s=26, label="open: predators extinct")
    a.set_xlabel(r"predator thrust$^2$ (energy leak)")
    a.set_ylabel("catches per 1k predator-steps")
    a.set_title(f"A. Lineage replacement: {100 * rule1:.0f}% of {len(rows)} brains, AUC {auc:.2f}", fontsize=9)
    a.legend(fontsize=6.3, loc="lower right", frameon=True, framealpha=0.9, edgecolor="none", ncol=2)
    a.set_xlim(0, 1.02)
    a.set_ylim(1.2, 7.8)

    b = ax[1]
    tt = np.geomspace(64, 8192, 60)
    b.plot(tt, D / np.exp(icpt + slope * np.log(tt)), "k-", lw=1.4, label=f"fit: SNR $\\propto T^{{{-slope:.2f}}}$")
    b.plot(tt, snr[0] * np.sqrt(tt / T[0]), "k--", lw=1, label=r"independent steps: $\propto\sqrt{T}$")
    b.scatter(T, snr, color="#c8553d", zorder=3, label="measured (GA champion)")
    b.axhline(1, color="#999", lw=0.8)
    b.axvline(128, color="#c8553d", lw=0.8, ls=":")
    b.text(140, 0.62, "GA/ES/NEAT\nwindow", fontsize=7, color="#c8553d")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("fitness window T (steps)")
    b.set_ylabel("signal / noise of the leak")
    b.set_title(f"B. Selection sees the leak only for T > {t_star_fit:.0f}", fontsize=9)
    b.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    for f in (ROOT / "paper" / "figures" / "leak_model.png", STUDY / "figures" / "leak_model.png"):
        f.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f, dpi=220)

    # ---------------------------------------------------------------- report
    L = [
        "# Analytic model of the energy leak",
        "",
        "Generated by `scripts/leak_model.py` from the world rules and `energy.json` / `energy_signal.json`.",
        "",
        "## 1. Lineage replacement",
        "",
        f"R0 = expected children per predator lifetime (Poisson catches over max_age = {q.max_age} steps, "
        f"{q.energy_per_catch:g} energy per catch, cost {q.base_drain:g} + {q.move_cost:g} x thrust^2 per step, "
        f"{q.child_energy:g} energy per child, at most {q.max_age // q.repro_cooldown} children).",
        "",
        f"- Zero-parameter rule *extinct iff R0 < 1*: **{100 * rule1:.0f}%** of {len(rows)} brains classified correctly.",
        f"- Rank agreement R0 vs extinction share: Spearman rho = {rho:.2f} (p = {p:.1e}); AUC = {auc:.2f}.",
        f"- Best single cut (fitted): R0 < {best:.2f} gives {100 * max(accs):.0f}%. The model is slightly pessimistic "
        "(mean-rate catches ignore that good hunters breed more).",
        f"- Break-even catch rate: {lineage['breakeven_catches_per_1k_at_th2_0']:.2f} per 1k steps when coasting, "
        f"{lineage['breakeven_catches_per_1k_at_th2_0.6']:.2f} at GA's thrust^2 of 0.6.",
        "",
        "| brain | thrust^2 | catches/1k | net/step | R0 | extinct |",
        "|---|---:|---:|---:|---:|---:|",
        *[f"| {r['brain']} | {r['th2']:.2f} | {r['catches_per_1k']:.2f} | {r['net']:+.4f} | {r['r0']:.2f} | "
          f"{100 * r['extinct_share']:.0f}% |" for r in rows],
        "",
        "## 2. Selection resolution",
        "",
        f"Saving of coasting over jittering: D = {D:.5f} reward per predator-step.",
        "",
        "| window T | noise sd | SNR | elites from credited half: measured | Gaussian model | share of selection on the leak |",
        "|---:|---:|---:|---:|---:|---:|",
        *[f"| {t} | {s:.5f} | {d:.2f} | {100 * m:.0f}% (±{100 * se:.0f}) | {100 * g:.0f}% | {100 * h:.0f}% |"
          for t, s, d, m, se, g, h in zip(T, sd, snr, measured_share, sel["elites_measured_se"],
                                          sel["elites_gaussian"], sel["selection_share"])],
        "",
        f"- Noise falls as T^-{-slope:.2f} (independent steps: T^-0.50), so SNR = 1 at T ≈ {t_star_fit:.0f} steps "
        f"({t_star_iid:.0f} if steps were independent). GA, ES and NEAT use T = 128.",
        "- The Gaussian model under-predicts the measured elite shares, plausibly because the noise is skewed: the -1 death "
        "penalty, which spreads the bottom of the ranking more than the top where elites are chosen.",
        "",
    ]
    (STUDY / "LEAK_MODEL.md").write_text("\n".join(L))
    print("\n".join(L[:14] + L[-12:]))


if __name__ == "__main__":
    main()
