"""Audit a temporal visual policy on disjoint seeds and levels."""

from __future__ import annotations

import argparse
from pathlib import Path

from env import RICH_LEVEL_VARIANTS
from train_recurrent_cnn import configure_process
from train_visual_temporal import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, action="append")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    configure_process(2)
    for level in RICH_LEVEL_VARIANTS[:-1]:
        for seed_start in args.seed_start or [5000, 6000, 7000, 8000]:
            result = evaluate(args.model, args.episodes, level, seed_start)
            print(
                f"level={level} start={seed_start} episodes={args.episodes} "
                f"mean_steps={result['steps']:.1f} collision_rate={result['collision']:.2f}"
            )


if __name__ == "__main__":
    main()
