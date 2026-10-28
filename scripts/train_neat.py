"""Train NEAT (M4). Same options as train_ga.py; writes results/neat/<name>/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ga import main

if __name__ == "__main__":
    main("neat")
