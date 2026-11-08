"""Export trained brains for the browser demo (web/public/brains/) and write the parity fixtures.

    python scripts/export_web.py brains          # checkpoints listed in web/brains.manifest.json -> JSON
    python scripts/export_web.py parity          # Python reference traces for web/test/parity.test.ts

Brain format: weights as base64 little-endian float32.
  MLP (PPO, GA, ES, hybrid):  {"kind": "mlp", "layers": [{"in", "out", "w": [out][in], "b"}, ...]}
  NEAT:                       {"kind": "neat", "D", "H", "depth", "Win": [D][H+2], "Wh": [H+2][H+2], "b": [H+2]}
"""

from __future__ import annotations

import argparse
import base64
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

WEB = ROOT / "web"


def b64(t: torch.Tensor) -> str:
    return base64.b64encode(t.detach().cpu().contiguous().numpy().astype("<f4").tobytes()).decode()


def export_brain(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("method") == "neat":
        Win, Wh, b = ck["Win"][0], ck["Wh"][0], ck["b"][0]
        H = Wh.shape[0] - 2
        used = int(((Win.abs().sum(0)[:H] > 0) | (Wh.abs().sum(0)[:H] > 0) | (Wh.abs().sum(1)[:H] > 0)).sum())
        return {"kind": "neat", "D": int(Win.shape[0]), "H": H, "depth": int(ck["depth"]), "n_hidden": used,
                "Win": b64(Win), "Wh": b64(Wh), "b": b64(b)}
    sd = ck["state_dict"]
    layers = []
    for i in (0, 2, 4):
        w, bias = sd[f"actor.{i}.weight"], sd[f"actor.{i}.bias"]
        layers.append({"in": int(w.shape[1]), "out": int(w.shape[0]), "w": b64(w), "b": b64(bias)})
    return {"kind": "mlp", "layers": layers}


def cmd_brains() -> None:
    manifest = json.loads((WEB / "brains.manifest.json").read_text())
    out = WEB / "public" / "brains"
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for entry in manifest["brains"]:
        src = ROOT / entry["source"]
        if not src.exists():
            print(f"skip (missing): {src}")
            continue
        data = export_brain(src)
        data.update({k: entry[k] for k in ("method", "species", "label", "note") if k in entry})
        data["source"] = entry["source"]
        fname = f"{entry['method']}-{entry['species']}.json"
        (out / fname).write_text(json.dumps(data))
        index.append({k: data[k] for k in ("method", "species", "label", "kind") if k in data} | {"file": fname,
                      "note": entry.get("note", "")})
        print(f"wrote {fname} ({data['kind']}) from {entry['source']}")
    (out / "index.json").write_text(json.dumps(index, indent=1))


# ---------------------------------------------------------------------------------------- parity
def snapshot(w: EcoWorld) -> dict:
    out = {"t": w.t, "grass_sum": float(w.grass.sum()), "grass_head": w.grass[0, :50].tolist()}
    for name, st in (("prey", w.species[PREY]), ("predator", w.species[PREDATOR])):
        al = st.alive[0]
        idx = torch.nonzero(al).squeeze(1)
        out[name] = {
            "alive": idx.tolist(),
            "x": st.pos[0, idx, 0].tolist(), "y": st.pos[0, idx, 1].tolist(),
            "heading": st.heading[0, idx].tolist(), "speed": st.speed[0, idx].tolist(),
            "energy": st.energy[0, idx].tolist(), "age": st.age[0, idx].tolist(),
            "cooldown": st.cooldown[0, idx].tolist(), "digest": st.digest[0, idx].tolist(),
        }
    return out


def obs_snapshot(w: EcoWorld) -> dict:
    o = w.observe()
    out = {}
    for name, sp in (("prey", PREY), ("predator", PREDATOR)):
        idx = torch.nonzero(w.species[sp].alive[0]).squeeze(1)
        out[name] = {"alive": idx.tolist(), "obs": o[sp][0, idx].tolist()}
    return out


def trace(policy: str, steps: int, every: int, obs_at: tuple[int, ...], **overrides) -> dict:
    cfg = load_config("S1", base="s1_balanced.json", **overrides)
    w = EcoWorld(cfg, n_worlds=1, seed=7)
    pol = [make_policy(policy, w, PREY), make_policy(policy, w, PREDATOR)]
    snaps, obs, events = [snapshot(w)], {}, []
    for t in range(steps):
        o = w.observe()
        if t in obs_at:
            obs[str(t)] = obs_snapshot(w)
        info = w.step(pol[0](o[PREY]), pol[1](o[PREDATOR]))
        events.append({"captures": int(info["captures"][0]), "births": [int(b[0]) for b in info["births"]],
                       "deaths": [int(d[0]) for d in info["deaths"]],
                       "pop": info["populations"][0].tolist()})
        if (t + 1) % every == 0:
            snaps.append(snapshot(w))
    return {"policy": policy, "base": "s1_balanced.json", "scenario": "S1", "seed": 7, "steps": steps,
            "overrides": overrides, "snapshots": snaps, "obs": obs, "events": events}


def brain_cases() -> list[dict]:
    """Actions of exported brains on fixed observations (taken from the heuristic world at t=0)."""
    cfg = load_config("S1", base="s1_balanced.json")
    w = EcoWorld(cfg, n_worlds=1, seed=7)
    o = w.observe()
    cases = []
    index = json.loads((WEB / "public" / "brains" / "index.json").read_text())
    for e in index:
        sp = PREY if e["species"] == "prey" else PREDATOR
        manifest = {b["method"] + "-" + b["species"]: b for b in
                    json.loads((WEB / "brains.manifest.json").read_text())["brains"]}
        src = ROOT / manifest[f"{e['method']}-{e['species']}"]["source"]
        pol = load_policy(src)
        idx = torch.nonzero(w.species[sp].alive[0]).squeeze(1)[:20]
        x = o[sp][0, idx]
        cases.append({"file": e["file"], "species": e["species"], "obs": x.tolist(),
                      "actions": pol(x.unsqueeze(0))[0].tolist()})
    return cases


def cmd_parity() -> None:
    out = WEB / "test" / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    (out / "trace_random.json").write_text(json.dumps(trace("random", 150, 10, (0,))))
    (out / "trace_heuristic.json").write_text(json.dumps(trace("heuristic", 150, 10, (0, 20, 60))))
    (out / "trace_drag.json").write_text(json.dumps(trace("random", 150, 10, (0,), **{"prey.drag": 0.05, "predator.drag": 0.05})))
    (out / "brains.json").write_text(json.dumps(brain_cases()))
    for p in sorted(out.glob("*.json")):
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["brains", "parity"])
    a = ap.parse_args()
    np.set_printoptions(precision=4)
    {"brains": cmd_brains, "parity": cmd_parity}[a.command]()
