"""Study 2: why do learned ecosystems collapse, and what prevents it?

The benchmark showed that when both species learn, the predators usually starve out (prey never die out), and
how often depends on the method. Each experiment here tests one explanation:

  E1 horizon     - learners only look ~128 steps ahead; does a longer window / discount stop the collapse?
  E2 selection   - individual vs group (one genome per world) selection, crossed with a shared ecosystem-health
                   reward ("care"): is it what learners optimise or the level at which selection acts?
  E3 asymmetry   - the life-dinner principle: prey run for their lives, predators for a meal. Slow down or
                   freeze one species' learning: is the collapse an arms race the prey win?
  X  cross-play  - prey of method A against predators of method B (benchmark brains), no training.

Commands (all resumable; finished runs are skipped):

    python scripts/study.py probe                          # GPU speed check + time estimate for every run
    python scripts/study.py train --exp E2 --seeds 0       # train (filter by experiment, condition, seeds)
    python scripts/study.py eval                           # evaluate every trained run (S1, S2, S4 by default)
    python scripts/study.py crossplay                      # prey x predator matrix of benchmark brains on S1
    python scripts/study.py behaviour                      # tactic measurements (refuges, ambush, pursuit...)
    python scripts/study.py list                           # conditions and progress

Runs go to results/study/train/<condition>/<condition>-seed<s>; evaluations to results/study/eval/<S>/.
The benchmark's ppo / ga / es runs are the baseline condition of every experiment (same settings, reused).
"""

from __future__ import annotations

import argparse
import json
import shutil
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

OUT = ROOT / "results" / "study"
BENCH = ROOT / "results" / "bench"
BASE = "s1_large.json"  # world B: --base s1_drag.json --bench results/bench_drag --out results/study_drag
SEEDS = [0, 1, 2, 3, 4]
EVAL_SEED = 20_000  # same as the benchmark, so baseline evaluations are reused as they are

GA = ["train_ga.py", "--method", "ga"]
ES = ["train_ga.py", "--method", "es"]
PPO = ["train_ppo.py"]
NEAT = ["train_ga.py", "--method", "neat"]
# E2 runs in episodic worlds: 24 fresh worlds per generation, one per genome under group selection
EPI = ["--episodic", "--worlds", "24", "--population", "24", "--gen-steps", "1000", "--world-steps", "1200000"]
GA_EPI = ["--elite-frac", "0.125"]  # 3 elites of 24

# name -> (experiment, method, script + args). Baselines (exp "base") are the benchmark runs, not retrained.
CONDITIONS: dict[str, tuple[str, str, list[str]]] = {
    "ppo": ("base", "ppo", []),
    "ga": ("base", "ga", []),
    "es": ("base", "es", []),
    "neat": ("base", "neat", []),
    # E1: horizon. GA/ES keep the benchmark's 293 generations, so a longer window costs more world-steps; the
    # budget control gives ES the h512 compute at the old window. PPO: longer discount at the same budget.
    "es-h512": ("E1x", "es", [*ES, "--gen-steps", "512", "--world-steps", "600000"]),
    "es-h1024": ("E1x", "es", [*ES, "--gen-steps", "1024", "--world-steps", "1200000"]),
    "es-budget4x": ("E1x", "es", [*ES, "--world-steps", "600000"]),
    "ga-h512": ("E1", "ga", [*GA, "--gen-steps", "512", "--world-steps", "600000"]),
    # control for ga-h512: the same 4x compute at the old 128-step window (is it just more training?)
    "ga-budget4x": ("E1", "ga", [*GA, "--world-steps", "600000"]),
    "ga-h1024": ("E1x", "ga", [*GA, "--gen-steps", "1024", "--world-steps", "1200000"]),
    "ppo-g995": ("E1", "ppo", [*PPO, "--gamma", "0.995"]),
    "ppo-g999": ("E1", "ppo", [*PPO, "--gamma", "0.999"]),
    # E2: level of selection x objective (2 x 2), GA and ES, identical episodic setting
    "es-mix": ("E2x", "es", [*ES, *EPI]),
    "es-grp": ("E2x", "es", [*ES, *EPI, "--assign", "group"]),
    "es-mix-eco": ("E2x", "es", [*ES, *EPI, "--eco-lambda", "5"]),
    "es-grp-eco": ("E2x", "es", [*ES, *EPI, "--assign", "group", "--eco-lambda", "5"]),
    "ga-mix": ("E2x", "ga", [*GA, *EPI, *GA_EPI]),
    "ga-grp": ("E2x", "ga", [*GA, *EPI, *GA_EPI, "--assign", "group"]),
    "ga-mix-eco": ("E2x", "ga", [*GA, *EPI, *GA_EPI, "--eco-lambda", "5"]),
    "ga-grp-eco": ("E2x", "ga", [*GA, *EPI, *GA_EPI, "--assign", "group", "--eco-lambda", "5"]),
    "ppo-eco1": ("E2", "ppo", [*PPO, "--eco-lambda", "1"]),
    "ppo-eco5": ("E2", "ppo", [*PPO, "--eco-lambda", "5"]),
    # E3: learning-speed asymmetry (benchmark settings otherwise)
    "es-prey4": ("E3", "es", [*ES, "--prey-every", "4"]),
    "es-preyfrozen": ("E3", "es", [*ES, "--prey-every", "0"]),
    "es-pred4": ("E3", "es", [*ES, "--pred-every", "4"]),
    "ppo-prey4": ("E3", "ppo", [*PPO, "--prey-every", "4"]),
    "ppo-preyfrozen": ("E3", "ppo", [*PPO, "--prey-every", "0"]),
    "ppo-pred4": ("E3", "ppo", [*PPO, "--pred-every", "4"]),
    # E4: the energy leak. GA/NEAT predators jitter (thrust sign flips ~40% of steps) and burn more energy than they
    # catch; the movement cost is tiny next to the noise of catches in a 128-step fitness window. Fix: count the
    # movement cost 5x in the training reward (the world itself is unchanged). E1's longer windows are the other fix.
    "ga-cost5": ("E4", "ga", [*GA, "--cost-weight", "5"]),
    "neat-cost5": ("E4", "neat", [*NEAT, "--cost-weight", "5"]),
    "es-cost5": ("E4", "es", [*ES, "--cost-weight", "5"]),
    # E5: Homeostatic Co-Training (our method): a species learns only while the other species holds at least half
    # its starting population; for PPO also the extinction penalty instead of a silent reset.
    "ppo-hct": ("E5", "ppo", [*PPO, "--homeostat", "0.5"]),
    "ppo-hct-ext": ("E5", "ppo", [*PPO, "--homeostat", "0.5", "--extinction-penalty", "5"]),
    "ppo-ext": ("E5", "ppo", [*PPO, "--extinction-penalty", "5"]),
    "es-hct": ("E5", "es", [*ES, "--homeostat", "0.5"]),
    "ga-hct": ("E5", "ga", [*GA, "--homeostat", "0.5"]),
    "neat-hct": ("E5", "neat", [*NEAT, "--homeostat", "0.5"]),
    # E6: hyperparameter check. Each method's main step size at 0.5x and 2x the benchmark value (the 1x runs are the
    # benchmark's own seeds 0-2). Is the collapse ranking an artefact of untuned settings?
    "ppo-lr0.5x": ("E6", "ppo", [*PPO, "--lr", "1.5e-4"]),
    "ppo-lr2x": ("E6", "ppo", [*PPO, "--lr", "6e-4"]),
    "ga-sig0.5x": ("E6", "ga", [*GA, "--sigma", "0.01"]),
    "ga-sig2x": ("E6", "ga", [*GA, "--sigma", "0.04"]),
    "es-lr0.5x": ("E6", "es", [*ES, "--lr", "0.005"]),
    "es-lr2x": ("E6", "es", [*ES, "--lr", "0.02"]),
    "neat-sig0.5x": ("E6", "neat", [*NEAT, "--neat-weight-sigma", "0.025"]),
    "neat-sig2x": ("E6", "neat", [*NEAT, "--neat-weight-sigma", "0.1"]),
}
# E7: the optimised recipe. Stack the fixes that each worked alone, then remove one at a time (ablation).
#   evolution (GA, ES): movement cost x5 (E4) + prey learn 4x slower (E3) + Homeostatic Co-Training (E5)
#   PPO:                gamma 0.999 (E1)     + prey learn 4x slower (E3) + Homeostatic Co-Training (E5)
# Only the family chosen after the E5 results is run. Names: <method>-opt, <method>-opt-no<ingredient>.
_FIX = {"cost": ["--cost-weight", "5"], "g999": ["--gamma", "0.999"], "prey4": ["--prey-every", "4"],
        "hct": ["--homeostat", "0.5"]}
_RECIPE = {"ga": (GA, ["cost", "prey4", "hct"]), "es": (ES, ["cost", "prey4", "hct"]),
           "ppo": (PPO, ["g999", "prey4", "hct"])}
for _m, (_base, _ing) in _RECIPE.items():
    CONDITIONS[f"{_m}-opt"] = ("E7", _m, [*_base, *[x for i in _ing for x in _FIX[i]]])
    for _drop in _ing:
        CONDITIONS[f"{_m}-opt-no{_drop}"] = ("E7", _m, [*_base, *[x for i in _ing if i != _drop for x in _FIX[i]]])
# rough cost in benchmark-run units (one benchmark run = 150k world-steps on 4 worlds, ~15 min on the RTX 5060 Ti)
COST = {"es-h512": 4, "es-h1024": 8, "es-budget4x": 4, "ga-h512": 4, "ga-h1024": 8, "ga-budget4x": 4}
for _n, (_e, _m, _a) in CONDITIONS.items():
    if _e == "E2x":
        COST[_n] = 8  # 1.2M world-steps, but on 24 worlds at once (the probe measures the real speed-up)


def run_dir(cond: str, seed: int) -> Path:
    if CONDITIONS[cond][0] == "base":
        return BENCH / "train" / cond / f"{cond}-seed{seed}"
    return OUT / "train" / cond / f"{cond}-seed{seed}"


def done(cond: str, seed: int) -> bool:
    d = run_dir(cond, seed)
    return (d / "eval.json").exists() and (d / "prey.pt").exists() and (d / "predator.pt").exists()


def select(a) -> list[str]:
    names = [n for n, (e, _, _) in CONDITIONS.items() if e != "base"]
    if a.exp:
        names = [n for n in names if CONDITIONS[n][0] in a.exp]
    if a.cond:
        names = [n for n in names if n in a.cond]
    return names


# ------------------------------------------------------------------------------------------------ train
def cmd_train(a) -> None:
    jobs = [(c, s) for s in a.seeds for c in select(a)]  # seed-major: every condition gets seed 0 first
    todo = [(c, s) for c, s in jobs if not done(c, s)]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} runs already done; {len(todo)} to go", flush=True)
    for i, (c, s) in enumerate(todo, 1):
        d = run_dir(c, s)
        _, _, args = CONDITIONS[c]
        cmd = [sys.executable, str(ROOT / "scripts" / args[0]), *args[1:], "--learn", "prey", "predator",
               "--seed", str(s), "--name", d.name, "--out", str(d.parent), "--base", BASE,
               "--eval-worlds", "0"]
        if a.device:
            cmd += ["--device", a.device]
        print(f"\n[{i}/{len(todo)}] {c} seed {s}: {' '.join(cmd[2:])}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT, check=False)
        if r.returncode:
            sys.exit(f"{c} seed {s} failed (exit {r.returncode}); fix and rerun, finished runs are kept")
        print(f"[{i}/{len(todo)}] {c} seed {s} done in {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------------------------------------ eval
def _eval_one(f: Path, cfg, pols, method: str, policy: str, sc: str, a, device) -> None:
    t0 = time.time()
    res = evaluate_worlds(cfg, pols, a.worlds, a.steps, EVAL_SEED, device)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f.with_suffix(".pops.npy"), res.pop("pops"))
    f.write_text(json.dumps({"scenario": sc, "policy": policy, "method": method, "worlds": a.worlds,
                             "steps": a.steps, **res}, indent=1))
    s = res["summary"]
    print(f"{sc} {policy:22s} survived {s['worlds_survived']}/{a.worlds}  attack {s['attack_rate_e6']:6.2f}  "
          f"catch/1k {s['pred_catches_per_1k_agent_steps']:5.2f}  ({time.time() - t0:.0f}s)", flush=True)


def cmd_eval(a) -> None:
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    conds = select(a) + (["ppo", "ga", "es", "neat"] if not a.cond else [c for c in a.cond if CONDITIONS[c][0] == "base"])
    for sc in a.scenarios:
        cfg = load_config(sc, base=BASE)
        for c in conds:
            for s in a.seeds:
                f = OUT / "eval" / sc / f"{c}-seed{s}.json"
                if f.exists() or not done(c, s):
                    continue
                bench_f = BENCH / "eval" / sc / f"{c}-seed{s}.json"
                if CONDITIONS[c][0] == "base" and bench_f.exists() and a.worlds == 5 and a.steps == 10_000:
                    f.parent.mkdir(parents=True, exist_ok=True)  # identical setting: reuse the benchmark's
                    shutil.copy(bench_f, f)
                    shutil.copy(bench_f.with_suffix(".pops.npy"), f.with_suffix(".pops.npy"))
                    continue
                d = run_dir(c, s)
                pols = {PREY: load_policy(d / "prey.pt", device), PREDATOR: load_policy(d / "predator.pt", device)}
                _eval_one(f, cfg, pols, c, f"{c}-seed{s}", sc, a, device)


# ------------------------------------------------------------------------------------------------ cross-play
def cmd_crossplay(a) -> None:
    """Prey of one method against predators of another (S1). Learned x learned pairs use the same seed index."""
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("S1", base=BASE)
    learned = ["ppo", "ga", "es", "neat", "hybrid"]
    fixed = ["heuristic", "random"]

    def brain(m: str, sp: str, s: int):
        if m in fixed:
            return m
        return load_policy(BENCH / "train" / m / f"{m}-seed{s}" / f"{sp}.pt", device)

    for s in a.seeds:
        for mp in learned + fixed:
            for mq in learned + fixed:
                if mp == mq:
                    continue  # the diagonal is the benchmark itself
                if mp in fixed and mq in fixed and s > 0:
                    continue
                name = f"{mp}-prey_vs_{mq}-pred-seed{s}"
                f = OUT / "crossplay" / f"{name}.json"
                if f.exists():
                    continue
                pols = {PREY: brain(mp, "prey", s), PREDATOR: brain(mq, "predator", s)}
                _eval_one(f, cfg, pols, f"{mp}|{mq}", name, "S1", a, device)


# ------------------------------------------------------------------------------------------------ skill
def cmd_skill(a) -> None:
    """Agent skill, so a 'safe' recipe can't win by being useless: each brain's predators hunt the prey of the five
    benchmark methods, and its prey are hunted by their predators (S1, opponents of seed s mod 5). Short games,
    since the attack rate (catches per prey-predator encounter) is a rate. Benchmark methods get the same test."""
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("S1", base=BASE)
    learned = ["ppo", "ga", "es", "neat", "hybrid"]
    bench = lambda m, sp, s: load_policy(BENCH / "train" / m / f"{m}-seed{s % 5}" / f"{sp}.pt", device)
    own = lambda c, sp, s: load_policy(run_dir(c, s) / f"{sp}.pt", device)
    # default: the five benchmark methods (the reference); --exp/--cond: study conditions (base names allowed)
    conds = list(a.cond) if a.cond else (select(a) if a.exp else learned)
    sub = argparse.Namespace(worlds=a.skill_worlds, steps=a.skill_steps)
    for c in conds:
        for s in a.seeds:
            if c in learned:
                if not (BENCH / "train" / c / f"{c}-seed{s}" / "predator.pt").exists():
                    continue
                mine = lambda sp, c=c, s=s: load_policy(BENCH / "train" / c / f"{c}-seed{s}" / f"{sp}.pt", device)
            elif done(c, s):
                mine = lambda sp, c=c, s=s: own(c, sp, s)
            else:
                continue
            for m in learned:
                for role, pols in (("pred", {PREY: bench(m, "prey", s), PREDATOR: mine("predator")}),
                                   ("prey", {PREY: mine("prey"), PREDATOR: bench(m, "predator", s)})):
                    f = OUT / "skill" / f"{c}-seed{s}-{role}_vs_{m}.json"
                    if not f.exists():
                        _eval_one(f, cfg, pols, c, f"{c}-seed{s} {role} vs {m}", "S1", sub, device)


# ------------------------------------------------------------------------------------------------ behaviour
def cmd_behaviour(a) -> None:
    """Tactic measurements (ecoarena.eval.behaviour) for every benchmark brain and every study condition on S1."""
    from ecoarena.eval.behaviour import measure_tactics

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config("S1", base=BASE)
    jobs = [(b, None) for b in ("heuristic", "random")]
    jobs += [(m, s) for m in ("ppo", "ga", "es", "neat", "hybrid") for s in a.seeds]
    jobs += [(c, s) for c in select(a) for s in a.seeds if done(c, s)]
    for m, s in jobs:
        name = m if s is None else f"{m}-seed{s}"
        f = OUT / "behaviour" / f"{name}.json"
        if f.exists():
            continue
        if s is None:
            pols = {PREY: m, PREDATOR: m}
        else:
            d = BENCH / "train" / m / name if m in ("ppo", "ga", "es", "neat", "hybrid") else run_dir(m, s)
            pols = {PREY: load_policy(d / "prey.pt", device), PREDATOR: load_policy(d / "predator.pt", device)}
        t0 = time.time()
        res = measure_tactics(cfg, pols, a.tactic_worlds, a.tactic_steps, EVAL_SEED, device)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"policy": name, "method": m, **res}, indent=1))
        print(f"{name:22s} refuge {res['prey_in_refuge']:.2f}  ambush {res['ambush_index']:.2f}  "
              f"edge-catch {res['edge_catch_share']:.2f}  pursuit {res['pursuit_share']:.2f}  "
              f"flee {res['flee_alignment']:+.2f}  ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------------ probe / list
def cmd_probe(a) -> None:
    """Measure world-steps per second for the two settings the study uses, then estimate every run's time."""
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from ecoarena.env import EcoWorld
    from ecoarena.trainers.ga import init_population, world_actions

    cfg = load_config("S1", base=BASE)
    speed = {}
    for W in (4, 24):
        w = EcoWorld(cfg, n_worlds=W, seed=0, device=device)
        g = torch.Generator().manual_seed(0)
        pop = {sp: init_population(W, cfg.obs_dim, 0.5, g, torch.device(device)) for sp in (PREY, PREDATOR)}
        wperm = torch.arange(W, device=device)
        for t in range(a.probe_steps + 20):
            if t == 20:
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
            obs = w.observe()
            w.step(world_actions(pop[PREY], obs[PREY], wperm), world_actions(pop[PREDATOR], obs[PREDATOR], wperm))
        if device == "cuda":
            torch.cuda.synchronize()
        speed[W] = W * a.probe_steps / (time.time() - t0)
        print(f"{W:2d} worlds: {speed[W]:,.0f} world-steps/s (forward passes only, no learning)", flush=True)
    ratio = speed[24] / speed[4]
    bench_min = []
    for m in ("ppo", "ga", "es"):
        for s in SEEDS:
            h = BENCH / "train" / m / f"{m}-seed{s}" / "history.json"
            if h.exists():
                bench_min.append(json.loads(h.read_text())[-1]["secs"] / 60)
    unit = float(np.mean(bench_min)) if bench_min else 15.0
    print(f"\none benchmark run took {unit:.1f} min on average; 24 worlds run {ratio:.1f}x faster per world-step\n")
    total = 0.0
    for n, (e, m, _) in CONDITIONS.items():
        if e == "base":
            continue
        c = COST.get(n, 1)
        if e == "E2x":
            c = c / max(ratio, 1.0)
        mins = c * unit * len(SEEDS)
        total += mins
        print(f"{e:3s} {n:16s} ~{c * unit:5.0f} min/run  x{len(SEEDS)} seeds = {mins / 60:5.1f} h")
    print(f"\ntraining total ~{total / 60:.0f} h; evaluation adds ~1 min per run per scenario")


def cmd_list(a) -> None:
    for n, (e, m, args) in CONDITIONS.items():
        got = sum(done(n, s) for s in SEEDS)
        print(f"{e:4s} {n:16s} {m:4s} {got}/{len(SEEDS)} trained   {' '.join(args[1:]) if args else '(benchmark)'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["train", "eval", "crossplay", "behaviour", "skill", "probe", "list"])
    # E1x / E2x: optional extras (ES/GA long windows, group selection), too costly for the main schedule because
    # parallel worlds give no speed-up on our GPU (probe, 7 Nov: 195 vs 206 world-steps/s)
    p.add_argument("--exp", nargs="+", choices=["E1", "E1x", "E2", "E2x", "E3", "E4", "E5", "E6", "E7"])
    p.add_argument("--cond", nargs="+", choices=list(CONDITIONS))
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--scenarios", nargs="+", default=["S1", "S2", "S4"])
    p.add_argument("--worlds", type=int, default=5)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--skill-worlds", type=int, default=3)
    p.add_argument("--skill-steps", type=int, default=3000)
    p.add_argument("--probe-steps", type=int, default=200)
    p.add_argument("--tactic-worlds", type=int, default=3)
    p.add_argument("--tactic-steps", type=int, default=3000)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="default results/study")
    p.add_argument("--base", default="s1_large.json", help="world config; s1_drag.json = world B (drag)")
    p.add_argument("--bench", default=None, help="benchmark dir whose runs are the baselines (default results/bench)")
    a = p.parse_args()
    global OUT, BENCH, BASE
    if a.out:
        OUT = Path(a.out)
    if a.bench:
        BENCH = Path(a.bench)
    BASE = a.base
    if BASE != "s1_large.json" and (not a.out or not a.bench):
        sys.exit("another world needs its own --out and --bench, so its runs never mix with the main world's")
    {"train": cmd_train, "eval": cmd_eval, "crossplay": cmd_crossplay, "behaviour": cmd_behaviour, "skill": cmd_skill,
     "probe": cmd_probe, "list": cmd_list}[
        a.command](a)


if __name__ == "__main__":
    main()
