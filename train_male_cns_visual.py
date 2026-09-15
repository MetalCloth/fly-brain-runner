"""Train a small action head on top of the fixed MaleCNS visual graph."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from env import RICH_ACTIONS, RICH_LEVEL_VARIANTS
from fly_graph import RETINA_SIZE, pool_perspective_retina, pool_retina
from gym_env import RichPixelGymRunnerEnv
from male_cns_neuron_graph import MaleCNSNeuronGraphExtractor
from rich_benchmark import teacher_action
from torch import nn
from train_recurrent_cnn import (
    SAFE_MAX_RSS_MB,
    SAFE_MIN_AVAILABLE_MB,
    configure_process,
)
from train_visual_teacher import (
    collect,
    collect_student,
    fit,
)


class MaleCNSVisualPolicy(nn.Module):
    """Fixed-topology MaleCNS graph with plastic edge strengths and a head."""

    def __init__(
        self,
        observation_space,
        retina_mode: str = "perspective",
        trainable_edges: bool = True,
        retina_skip: bool = True,
    ) -> None:
        super().__init__()
        self.graph = MaleCNSNeuronGraphExtractor(
            observation_space,
            trainable_edges=trainable_edges,
            retina_mode=retina_mode,
        )
        # ponytail: one compact head is enough after graph pooling; add memory
        # only if a held-out partial-observation gate shows it is necessary.
        self.retina_mode = retina_mode
        self.retina_skip = retina_skip
        input_features = self.graph.output_features + (
            RETINA_SIZE if retina_skip else 0
        )
        self.head = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.Tanh(),
            nn.Linear(64, len(RICH_ACTIONS)),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        graph_features = self.graph(frames)
        if not self.retina_skip:
            return self.head(graph_features)
        retina = (
            pool_perspective_retina(frames)
            if self.retina_mode == "perspective"
            else pool_retina(frames)
        )
        return self.head(torch.cat((graph_features, retina), dim=1))


def train(
    output_dir: Path,
    epochs: int,
    batch_size: int,
    train_samples: int,
    val_samples: int,
    max_rss_mb: int,
    min_available_mb: int,
    dagger_rounds: int,
    dagger_samples: int,
    dagger_level: str,
    retina_mode: str,
    trainable_edges: bool,
    retina_skip: bool,
) -> Path:
    configure_process(2)
    if dagger_rounds < 0 or dagger_samples < 1:
        raise ValueError("dagger rounds must be nonnegative and samples positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_frames, base_labels = collect(
        0, 120, "mixed", train_samples // 2, teacher_action
    )
    maneuver_frames, maneuver_labels = collect(
        200, 120, "mixed", train_samples - train_samples // 2
    )
    train_frames = np.concatenate((base_frames, maneuver_frames))
    train_labels = np.concatenate((base_labels, maneuver_labels))
    val_frames, val_labels = collect(1000, 50, "mixed", val_samples)
    print(
        f"dataset train={len(train_labels)} val={len(val_labels)} "
        f"train_actions={np.bincount(train_labels, minlength=len(RICH_ACTIONS)).tolist()}"
    )

    env = RichPixelGymRunnerEnv(
        max_steps=300,
        spawn_interval=8,
        rich_layer=2,
        level_variant="mixed",
    )
    try:
        model = MaleCNSVisualPolicy(
            env.observation_space, retina_mode, trainable_edges, retina_skip
        )
    finally:
        env.close()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_path = output_dir / "male_cns_visual_best.pt"
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

    print(f"best_validation_accuracy={best_accuracy:.3f}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/male_cns_visual_dagger_v1")
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=12_000)
    parser.add_argument("--val-samples", type=int, default=4_000)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-samples", type=int, default=4_000)
    parser.add_argument("--dagger-level", choices=RICH_LEVEL_VARIANTS, default="dense")
    parser.add_argument("--retina-mode", choices=("fixed", "perspective"), default="perspective")
    parser.add_argument(
        "--fixed-edges",
        action="store_true",
        help="keep MaleCNS edge strengths fixed for an ablation",
    )
    parser.add_argument(
        "--no-retina-skip",
        action="store_true",
        help="use only the graph output for a pure-graph ablation",
    )
    parser.add_argument("--max-rss-mb", type=int, default=SAFE_MAX_RSS_MB)
    parser.add_argument("--min-available-mb", type=int, default=SAFE_MIN_AVAILABLE_MB)
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.train_samples < 1
        or args.val_samples < 1
        or args.dagger_rounds < 0
        or args.dagger_samples < 1
    ):
        parser.error("training counts must be positive; dagger rounds may be zero")
    model_path = train(
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.train_samples,
        args.val_samples,
        args.max_rss_mb,
        args.min_available_mb,
        args.dagger_rounds,
        args.dagger_samples,
        args.dagger_level,
        args.retina_mode,
        not args.fixed_edges,
        not args.no_retina_skip,
    )
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
