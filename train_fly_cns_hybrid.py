"""Train a residual MaleCNS visual branch beside the proven pixel policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from env import RICH_ACTIONS, RICH_LEVEL_VARIANTS
from fly_graph import RETINA_SIZE
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
    VisualPolicy,
    collect,
    collect_student,
    fit,
    near_field_view,
)


class FlyCNSHybridPolicy(nn.Module):
    """Nearfield policy plus a learned-retina MaleCNS residual action path."""

    def __init__(
        self,
        observation_space,
        init_checkpoint: Path | None = None,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.residual_scale = residual_scale
        self.base = VisualPolicy(input_channels=6)
        if init_checkpoint is not None and init_checkpoint.exists():
            self.base.load_state_dict(
                torch.load(init_checkpoint, map_location="cpu", weights_only=True)
            )
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.graph = MaleCNSNeuronGraphExtractor(observation_space)
        self.retina_projection = nn.Linear(128, RETINA_SIZE)
        self.residual = nn.Linear(
            RETINA_SIZE + self.graph.output_features, len(RICH_ACTIONS)
        )
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.base.features(frames / 255.0)
        latent = self.base.head[:2](features)
        base_logits = self.base.head[2](latent)
        retina = torch.sigmoid(self.retina_projection(latent))
        graph_features = self.graph.forward_retina(retina)
        # Keep the learned fly correction bounded so the verified base policy
        # remains the fallback when the branch has weak evidence.
        return base_logits + self.residual_scale * torch.tanh(
            self.residual(torch.cat((retina, graph_features), dim=1))
        )


def train(
    output_dir: Path,
    init_checkpoint: Path | None,
    epochs: int,
    batch_size: int,
    train_samples: int,
    val_samples: int,
    max_rss_mb: int,
    min_available_mb: int,
    dagger_rounds: int,
    dagger_samples: int,
    dagger_level: str,
) -> Path:
    configure_process(2)
    if dagger_rounds < 0 or dagger_samples < 1:
        raise ValueError("dagger rounds must be nonnegative and samples positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_frames, base_labels = collect(
        0,
        120,
        "mixed",
        train_samples // 2,
        teacher_action,
        near_field_view,
    )
    maneuver_frames, maneuver_labels = collect(
        200,
        120,
        "mixed",
        train_samples - train_samples // 2,
        view=near_field_view,
    )
    train_frames = np.concatenate((base_frames, maneuver_frames))
    train_labels = np.concatenate((base_labels, maneuver_labels))
    val_frames, val_labels = collect(
        1000, 50, "mixed", val_samples, view=near_field_view
    )
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
        model = FlyCNSHybridPolicy(env.observation_space, init_checkpoint)
    finally:
        env.close()

    optimizer = torch.optim.AdamW(
        [*model.retina_projection.parameters(), *model.residual.parameters()],
        lr=5e-4,
        weight_decay=1e-4,
    )
    best_path = output_dir / "fly_cns_hybrid_best.pt"
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
        "--output-dir", type=Path, default=Path("results/fly_cns_hybrid_v1")
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path("results/visual_nearfield_v1/nearfield_visual_teacher_best.pt"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=9000)
    parser.add_argument("--val-samples", type=int, default=3000)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-samples", type=int, default=3000)
    parser.add_argument("--dagger-level", choices=RICH_LEVEL_VARIANTS, default="dense")
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
        args.init_checkpoint,
        args.epochs,
        args.batch_size,
        args.train_samples,
        args.val_samples,
        args.max_rss_mb,
        args.min_available_mb,
        args.dagger_rounds,
        args.dagger_samples,
        args.dagger_level,
    )
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
