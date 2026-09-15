"""Train a visual policy with an appended near-field pixel view."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from env import RICH_LEVEL_VARIANTS
from train_recurrent_cnn import SAFE_MAX_RSS_MB, SAFE_MIN_AVAILABLE_MB, configure_process
from train_visual_teacher import (
    VisualPolicy,
    collect,
    collect_student,
    evaluate,
    human_visual_teacher_action,
    fit,
    initialize_near_field,
    near_field_view,
    visual_teacher_action,
)


def train(
    teacher_checkpoint: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    train_samples: int,
    val_samples: int,
    dagger_rounds: int,
    dagger_samples: int,
    dagger_level: str,
    max_rss_mb: int,
    min_available_mb: int,
    teacher_policy: str = "safe",
    seed: int = 7,
) -> Path:
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(teacher_checkpoint)
    configure_process(2)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if teacher_policy not in ("safe", "human"):
        raise ValueError("teacher_policy must be 'safe' or 'human'")
    label_policy = (
        visual_teacher_action
        if teacher_policy == "safe"
        else human_visual_teacher_action
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base_frames, base_labels = collect(
        0, 120, "mixed", train_samples // 2, label_policy, near_field_view
    )
    maneuver_frames, maneuver_labels = collect(
        200,
        120,
        "mixed",
        train_samples - train_samples // 2,
        label_policy,
        near_field_view,
    )
    train_frames = np.concatenate((base_frames, maneuver_frames))
    train_labels = np.concatenate((base_labels, maneuver_labels))
    val_frames, val_labels = collect(
        1000, 50, "mixed", val_samples, label_policy, near_field_view
    )
    print(
        f"dataset train={len(train_labels)} val={len(val_labels)} "
        f"shape={train_frames.shape[1:]} "
        f"train_actions={np.bincount(train_labels, minlength=6).tolist()}"
    )

    model = VisualPolicy(input_channels=6)
    initialize_near_field(model, teacher_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_path = output_dir / "nearfield_visual_teacher_best.pt"
    best_accuracy = fit(
        model,
        optimizer,
        train_frames,
        train_labels,
        val_frames,
        val_labels,
        epochs,
        batch_size,
        -1.0,
        best_path,
        max_rss_mb,
        min_available_mb,
    )
    for dagger_round in range(dagger_rounds):
        model.load_state_dict(
            torch.load(best_path, map_location="cpu", weights_only=True)
        )
        extra_frames, extra_labels = collect_student(
            model,
            2000 + dagger_round * 100,
            100,
            dagger_level,
            dagger_samples,
            near_field_view,
            policy=label_policy,
        )
        train_frames = np.concatenate((train_frames, extra_frames))
        train_labels = np.concatenate((train_labels, extra_labels))
        print(f"dagger_round={dagger_round + 1} samples={len(extra_labels)}")
        best_accuracy = fit(
            model,
            optimizer,
            train_frames,
            train_labels,
            val_frames,
            val_labels,
            max(3, epochs // 3),
            batch_size,
            best_accuracy,
            best_path,
            max_rss_mb,
            min_available_mb,
        )
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("results/visual_dagger_dense_v2/visual_teacher_best.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/visual_nearfield_v1")
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=12_000)
    parser.add_argument("--val-samples", type=int, default=3_000)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-samples", type=int, default=4_000)
    parser.add_argument("--dagger-level", choices=RICH_LEVEL_VARIANTS, default="dense")
    parser.add_argument(
        "--teacher-policy",
        choices=("safe", "human"),
        default="safe",
        help="label policy; human adds soft pickup-versus-risk trade-offs",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-rss-mb", type=int, default=SAFE_MAX_RSS_MB)
    parser.add_argument("--min-available-mb", type=int, default=SAFE_MIN_AVAILABLE_MB)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.train_samples, args.val_samples) < 1:
        parser.error("training counts must be positive")
    if args.dagger_rounds < 0 or args.dagger_samples < 1:
        parser.error("dagger rounds must be nonnegative and samples positive")
    path = train(
        args.teacher_checkpoint,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.train_samples,
        args.val_samples,
        args.dagger_rounds,
        args.dagger_samples,
        args.dagger_level,
        args.max_rss_mb,
        args.min_available_mb,
        args.teacher_policy,
        args.seed,
    )
    print(f"saved model: {path}")
    for level in RICH_LEVEL_VARIANTS[:-1]:
        result = evaluate(
            path,
            50,
            level,
            0,
            input_channels=6,
            view=near_field_view,
        )
        print(
            f"level={level} episodes=50 mean_steps={result['steps']:.1f} "
            f"collision_rate={result['collision']:.2f}"
        )


if __name__ == "__main__":
    main()
