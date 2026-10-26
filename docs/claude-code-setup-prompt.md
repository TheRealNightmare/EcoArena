# Claude Code prompt: set up EcoArena on the GPU machine and run the week-1 checks

**How to use:** clone the repo, open a terminal in it, start `claude`, and paste everything below the line.

```bash
git clone https://github.com/TheRealNightmare/EcoArena.git
cd EcoArena
claude
```

---

You are setting up the EcoArena research repo on my GPU machine (Ubuntu, NVIDIA RTX 5060 Ti 16 GB) and running the week-1 checks. EcoArena is a batched PyTorch predator–prey ecosystem for an ML course paper comparing reinforcement learning and neuroevolution. Read `README.md` first, then do the steps below in order. Stop and tell me if a step fails in a way the notes don't cover.

## Rules

- Do not change anything under `ecoarena/`, `configs/` or `tests/`. This is a measurement run, not a coding task. If something looks like a bug, report it with the exact error; don't fix it.
- Do not commit or push. Generated files under `results/` stay local.
- Use a virtual environment in the repo (`.venv`), never the system Python.
- Long commands: run them in the foreground with a generous timeout (up to 60 minutes) and don't interrupt them because they look quiet. Training-free runs print nothing until the end.

## Step 1: check the machine

Run and report:

```bash
nvidia-smi
python3 --version
uv --version || echo "uv missing"
```

If `uv` is missing, install it: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then open a new shell or `source ~/.local/bin/env`.

## Step 2: create the environment

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[dev]"
```

The RTX 5060 Ti is a Blackwell card (compute capability 12.0), so PyTorch must be a CUDA 12.8 or newer build. Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Expected: `cuda` is 12.8 or higher, `is_available()` is `True`, the name contains `5060 Ti`, capability `(12, 0)`.

If `is_available()` is `False`, or you see "no kernel image is available for execution on the device", or a warning that sm_120 is not supported: reinstall torch from the newest CUDA index PyTorch offers (try `cu129`, then `cu130`) and verify again. Report which index worked.

## Step 3: tests and lint

```bash
pytest -q
ruff check ecoarena tests scripts
```

Expected: all tests pass (36 at the time of writing) and ruff prints no errors. Tests run on CPU. If anything fails, stop and report the full failure output.

## Step 4: GPU throughput

```bash
python scripts/timing.py --device cuda
python scripts/timing.py --device cuda --base s1_large.json --worlds 1 4 16
```

The first command times the 100×100 debug world; the second times the 200×200 benchmark world. Keep the full output of both.

If the large world runs out of GPU memory at 16 worlds, rerun it with `--worlds 1 4 8` and note that 16 failed.

## Step 5: the survival gate

This is the week-1 gate: with hand-coded (heuristic) agents, both species must stay alive for 10,000 steps in at least 4 of 5 worlds on the benchmark world.

```bash
python scripts/run_baseline.py --policy heuristic --base s1_large.json --worlds 5 --steps 10000 --device cuda
```

This can take a while. It prints a JSON summary at the end and writes these files to `results/baselines/`:

- `heuristic-s1_large-S1-seed0.png`: population plot, one panel per world
- `heuristic-s1_large-S1-seed0-summary.json`: the summary
- `heuristic-s1_large-S1-seed0-pops.npy`: raw population curves

Open the PNG and describe what you see in each panel. Points to cover:

- whether both lines stay above zero to the end
- the lowest predator count, and roughly when it happens
- whether prey and predators cycle (prey rise, predators follow, prey fall), settle flat, or drift

## Step 6: random baseline (sanity check)

```bash
python scripts/run_baseline.py --policy random --base s1_large.json --worlds 5 --steps 3000 --device cuda
```

Random agents are expected to collapse quickly; this just confirms the baseline runs.

## Step 7: report back

Finish with one short report I can paste into my chat with Claude, in this format:

```
SETUP
- GPU / driver / CUDA (from nvidia-smi):
- torch version, torch.version.cuda, capability:
- CUDA index that worked:
- tests: N passed / failed; ruff: clean / errors

TIMING (paste both timing.py outputs verbatim)

GATE (s1_large, heuristic, 5 x 10,000)
- worlds_survived: X/5
- extinct_at_step: [...]
- mean_pop_second_half: [prey, predators]
- wall_seconds / agent_steps_per_sec:
- what the plot shows, per world (1 line each):

RANDOM BASELINE
- worlds_survived, extinct_at_step:

PROBLEMS
- anything that failed, warned, or looked odd (exact messages)
```

Also give me the full paths of the two PNG files so I can attach them to the chat.
