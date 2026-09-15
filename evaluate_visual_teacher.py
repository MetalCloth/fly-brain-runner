"""Audit a bootstrapped visual policy on disjoint seeds and levels."""

from __future__ import annotations

import argparse
from pathlib import Path

from env import RICH_EVAL_LEVEL_VARIANTS, RICH_LEVEL_VARIANTS
from train_recurrent_cnn import configure_process
from train_visual_teacher import evaluate, identity_view, near_field_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, action="append")
    parser.add_argument(
        "--level",
        choices=RICH_EVAL_LEVEL_VARIANTS,
        action="append",
        help="evaluate only the requested level(s); defaults to standard/dense/fast",
    )
    parser.add_argument(
        "--motion-hold",
        type=int,
        default=0,
        help="hold an accepted jump/roll against short opposite-motion flips",
    )
    parser.add_argument(
        "--motion-flip-delay",
        type=int,
        default=0,
        help="delay an opposite motion only after three repeated prior motions",
    )
    parser.add_argument(
        "--near-field",
        action="store_true",
        help="evaluate a six-channel checkpoint with the appended near-field view",
    )
    args = parser.parse_args()
    if (
        args.episodes < 1
        or args.motion_hold < 0
        or args.motion_flip_delay < 0
        or (args.motion_hold and args.motion_flip_delay)
    ):
        parser.error(
            "episodes must be positive, timing values cannot be negative, "
            "and only one timing rule may be enabled"
        )

    configure_process(2)
    seed_starts = args.seed_start or [0, 1000, 2000]
    levels = args.level or RICH_LEVEL_VARIANTS[:-1]
    for level in levels:
        for seed_start in seed_starts:
            result = evaluate(
                args.model,
                args.episodes,
                level,
                seed_start,
                args.motion_hold,
                args.motion_flip_delay,
                6 if args.near_field else 3,
                near_field_view if args.near_field else identity_view,
            )
            print(
                f"level={level} start={seed_start} episodes={args.episodes} "
                f"mean_steps={result['steps']:.1f} "
                f"collision_rate={result['collision']:.2f}"
            )


if __name__ == "__main__":
    main()
