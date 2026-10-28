"""The main benchmark: every method controls BOTH species, 5 training seeds, evaluated on scenarios S1-S5.

    python scripts/benchmark.py train                      # 5 methods x 5 seeds on S1 (resumable, skips done runs)
    python scripts/benchmark.py train --methods ppo ga --seeds 0 1   # a chunk
    python scripts/benchmark.py eval                       # every checkpoint + baselines on S1-S5
    python scripts/benchmark.py table                      # tables and CSV from the eval files

Everything lands in results/bench/. Each step skips work whose output already exists, so it can be stopped
and restarted at any time (e.g. when the GPU has to go off).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecoarena.env import PREDATOR, PREY, load_config
from ecoarena.eval.benchmark import evaluate_worlds
from ecoarena.trainers.ppo import load_policy

METHODS = ["ppo", "ga", "es", "neat", "hybrid"]
BASELINES = ["random", "heuristic"]
SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]
OUT = ROOT / "results" / "bench"
SCRIPT = {"ppo": ["train_ppo.py"], "ga": ["train_ga.py", "--method", "ga"], "es": ["train_ga.py", "--method", "es"],
          "neat": ["train_ga.py", "--method", "neat"], "hybrid": ["train_hybrid.py"]}


def set_out(path: str | None) -> None:
    global OUT
    if path:
        OUT = Path(path)


def run_dir(method: str, seed: int) -> Path:
    return OUT / "train" / method / f"{method}-seed{seed}"


def cmd_train(a) -> None:
    jobs = [(m, s) for s in a.seeds for m in a.methods]
    todo = [(m, s) for m, s in jobs if not (run_dir(m, s) / "eval.json").exists()]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} runs already done; {len(todo)} to go", flush=True)
    for i, (m, s) in enumerate(todo, 1):
        d = run_dir(m, s)
        cmd = [sys.executable, str(ROOT / "scripts" / SCRIPT[m][0]), *SCRIPT[m][1:],
               "--learn", "prey", "predator", "--seed", str(s), "--name", d.name, "--out", str(d.parent),
               "--world-steps", str(a.world_steps), "--base", a.base,
               "--eval-worlds", "2", "--eval-steps", str(a.train_eval_steps)]
        if a.device:
            cmd += ["--device", a.device]
        print(f"\n[{i}/{len(todo)}] {m} seed {s}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, check=False)
        if r.returncode:
            sys.exit(f"{m} seed {s} failed (exit {r.returncode}); fix and rerun, finished runs are kept")
        print(f"[{i}/{len(todo)}] {m} seed {s} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def policy_sets(methods, seeds, device):
    for b in BASELINES:
        yield b, b, {PREY: b, PREDATOR: b}
    for m in methods:
        for s in seeds:
            d = run_dir(m, s)
            if (d / "prey.pt").exists() and (d / "predator.pt").exists():
                yield f"{m}-seed{s}", m, {PREY: load_policy(d / "prey.pt", device),
                                          PREDATOR: load_policy(d / "predator.pt", device)}


def cmd_eval(a) -> None:
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for sc in a.scenarios:
        cfg = load_config(sc, base=a.base)
        for name, method, pols in policy_sets(a.methods, a.seeds, device):
            f = OUT / "eval" / sc / f"{name}.json"
            if f.exists():
                continue
            t0 = time.time()
            res = evaluate_worlds(cfg, pols, a.worlds, a.steps, 20_000, device)
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f.with_suffix(".pops.npy"), res.pop("pops"))
            f.write_text(json.dumps({"scenario": sc, "policy": name, "method": method, "worlds": a.worlds,
                                     "steps": a.steps, **res}, indent=1))
            s = res["summary"]
            print(f"{sc} {name:16s} survived {s['worlds_survived']}/{a.worlds}  attack {s['attack_rate_e6']:6.2f}  "
                  f"catch/1k {s['pred_catches_per_1k_agent_steps']:5.2f}  ({time.time() - t0:.0f}s)", flush=True)


METRICS = ["worlds_survived", "attack_rate_e6", "pred_catches_per_1k_agent_steps", "prey_mean_lifespan",
           "pred_mean_lifespan", "prey_reward_per_1k_agent_steps", "pred_reward_per_1k_agent_steps"]


def cmd_table(a) -> None:
    rows = []
    for f in sorted((OUT / "eval").glob("S*/*.json")):
        d = json.loads(f.read_text())
        for i, w in enumerate(d["per_world"]):
            rows.append({"scenario": d["scenario"], "method": d["method"], "policy": d["policy"], "world": i,
                         "survived": int(w["survived"]),
                         "prey_extinct_at": w["extinct_at"]["prey"], "pred_extinct_at": w["extinct_at"]["predator"],
                         "mean_prey": w["mean_pop_2nd_half"][0], "mean_pred": w["mean_pop_2nd_half"][1],
                         "cv_prey": w["cv_2nd_half"][0], "cv_pred": w["cv_2nd_half"][1],
                         **{k: w[k] for k in METRICS[1:]}})
    if not rows:
        sys.exit("no eval files yet; run `benchmark.py eval` first")
    with open(OUT / "per_world.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    order = BASELINES + METHODS
    cols = ["survived", "attack_rate_e6", "pred_catches_per_1k_agent_steps", "prey_mean_lifespan",
            "pred_mean_lifespan", "cv_prey", "cv_pred"]
    head = ["survived", "attack rate", "catches/1k pred-steps", "prey life", "pred life", "CV prey", "CV pred"]
    md = ["# Benchmark results (mean ± std over all evaluation worlds of all training seeds)\n",
          "Generated by `scripts/benchmark.py table`. `survived` = share of worlds with both species alive at the end.\n"]
    for sc in SCENARIOS:
        sub = [r for r in rows if r["scenario"] == sc]
        if not sub:
            continue
        md += [f"\n## {sc}\n", "| method | n | " + " | ".join(head) + " |", "|---" * (len(head) + 2) + "|"]
        for m in order:
            mr = [r for r in sub if r["method"] == m]
            if not mr:
                continue
            cells = []
            for c in cols:
                v = np.array([r[c] for r in mr], dtype=float)
                cells.append(f"{v.mean():.2f}" if c == "survived" else f"{v.mean():.2f} ± {v.std():.2f}")
            md.append(f"| {m} | {len(mr)} | " + " | ".join(cells) + " |")
    (OUT / "TABLES.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["train", "eval", "table"])
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    p.add_argument("--world-steps", type=int, default=150_000)
    p.add_argument("--worlds", type=int, default=5, help="evaluation worlds per scenario")
    p.add_argument("--steps", type=int, default=10_000, help="evaluation length")
    p.add_argument("--device", default=None)
    p.add_argument("--base", default="s1_large.json", help="world; s1_balanced.json only for quick tests")
    p.add_argument("--train-eval-steps", type=int, default=1000, help="quick check each training script runs")
    p.add_argument("--out", default=None, help="default results/bench")
    a = p.parse_args()
    set_out(a.out)
    {"train": cmd_train, "eval": cmd_eval, "table": cmd_table}[a.command](a)


if __name__ == "__main__":
    main()
