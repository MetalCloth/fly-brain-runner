"""Train a MaleCNS action policy from a learned visual retina."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from env import RICH_ACTIONS, RICH_LEVEL_VARIANTS
from fly_graph import RETINA_SIZE
from gym_env import RichPixelGymRunnerEnv
from male_cns_neuron_graph import MaleCNSNeuronGraphExtractor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from train_recurrent_cnn import (
    SAFE_MAX_RSS_MB,
    SAFE_MIN_AVAILABLE_MB,
    configure_process,
)
from train_visual_teacher import (
    VisualPolicy,
    collect,
    collect_student,
    human_visual_teacher_action,
    fit,
    near_field_view,
    resource_ok,
    validation_accuracy,
    visual_teacher_action,
)


class FlyCNSRetinaPolicy(nn.Module):
    """Frozen visual adapter, learned retina, MaleCNS graph, action head."""

    def __init__(
        self,
        observation_space,
        init_checkpoint: Path | None = None,
        disable_graph: bool = False,
        init_policy: Path | None = None,
    ) -> None:
        super().__init__()
        self.visual = VisualPolicy(input_channels=6)
        if init_checkpoint is not None and init_checkpoint.exists():
            self.visual.load_state_dict(
                torch.load(init_checkpoint, map_location="cpu", weights_only=True)
            )
        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        self.graph = MaleCNSNeuronGraphExtractor(observation_space)
        self.disable_graph = disable_graph
        self.retina_projection = nn.Linear(128, RETINA_SIZE)
        self.action_head = nn.Sequential(
            nn.Linear(self.graph.output_features, 96),
            nn.ReLU(),
            nn.Linear(96, len(RICH_ACTIONS)),
        )
        if init_policy is not None:
            if not init_policy.is_file():
                raise FileNotFoundError(init_policy)
            self.load_state_dict(
                torch.load(init_policy, map_location="cpu", weights_only=True)
            )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.visual.features(frames / 255.0)
        latent = self.visual.head[:2](features)
        retina = torch.sigmoid(self.retina_projection(latent))
        graph_features = self.graph.forward_retina(retina)
        if self.disable_graph:
            graph_features = torch.zeros_like(graph_features)
        return self.action_head(graph_features)


def fit_preserving(
    model: nn.Module,
    anchor: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_frames: np.ndarray,
    train_labels: np.ndarray,
    val_frames: np.ndarray,
    val_labels: np.ndarray,
    epochs: int,
    batch_size: int,
    best_accuracy: float,
    best_path: Path,
    max_rss_mb: int,
    min_available_mb: int,
) -> float:
    """Fine-tune labels while distilling the accepted policy's behavior."""

    train_data = TensorDataset(
        torch.from_numpy(train_frames).permute(0, 3, 1, 2),
        torch.from_numpy(train_labels),
    )
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    counts = np.bincount(train_labels, minlength=len(RICH_ACTIONS)).astype(np.float32)
    weights = torch.from_numpy(
        np.power(len(train_labels) / np.maximum(counts, 1), 0.35)
    )
    weights /= weights.mean()
    temperature = 2.0
    stale_epochs = 0

    anchor.eval()
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch_index, (images, labels) in enumerate(loader):
            if batch_index % 32 == 0 and not resource_ok(max_rss_mb, min_available_mb):
                raise RuntimeError("visual training stopped by resource guard")
            brightness = 0.95 + 0.10 * torch.rand((images.shape[0], 1, 1, 1))
            inputs = images.float() * brightness
            logits = model(inputs)
            with torch.no_grad():
                anchor_logits = anchor(inputs)
            anchor_actions = anchor_logits.argmax(1)
            # The only allowed override is a lane-change pickup preference
            # while the accepted policy itself says noop. Hazard actions stay
            # anchored even when the human-labelled dataset is noisy.
            coin_override = (anchor_actions == 0) & ((labels == 1) | (labels == 2))
            protected_labels = torch.where(coin_override, labels, anchor_actions)
            imitation = F.cross_entropy(logits, protected_labels, weight=weights)
            preserve = F.kl_div(
                F.log_softmax(logits / temperature, dim=1),
                F.softmax(anchor_logits / temperature, dim=1),
                reduction="batchmean",
            ) * temperature**2
            loss = 0.35 * imitation + preserve
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        accuracy = validation_accuracy(model, val_frames, val_labels)
        print(f"loss={mean(losses):.4f} val_accuracy={accuracy:.3f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_path)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 3:
                break
    return best_accuracy


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
    teacher_policy: str = "safe",
    seed: int = 7,
    init_policy: Path | None = None,
    preserve_base: bool = False,
) -> Path:
    configure_process(2)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dagger_rounds < 0 or dagger_samples < 1:
        raise ValueError("dagger rounds must be nonnegative and samples positive")
    if teacher_policy not in ("safe", "human"):
        raise ValueError("teacher_policy must be 'safe' or 'human'")
    if preserve_base and init_policy is None:
        raise ValueError("preserve_base requires init_policy")
    label_policy = (
        visual_teacher_action
        if teacher_policy == "safe"
        else human_visual_teacher_action
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base_frames, base_labels = collect(
        0,
        120,
        "mixed",
        train_samples // 2,
        label_policy,
        near_field_view,
    )
    maneuver_frames, maneuver_labels = collect(
        200,
        120,
        "mixed",
        train_samples - train_samples // 2,
        policy=label_policy,
        view=near_field_view,
    )
    train_frames = np.concatenate((base_frames, maneuver_frames))
    train_labels = np.concatenate((base_labels, maneuver_labels))
    val_frames, val_labels = collect(
        1000, 50, "mixed", val_samples, label_policy, view=near_field_view
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
        model = FlyCNSRetinaPolicy(
            env.observation_space, init_checkpoint, init_policy=init_policy
        )
        anchor = (
            FlyCNSRetinaPolicy(
                env.observation_space, init_checkpoint, init_policy=init_policy
            )
            if preserve_base
            else None
        )
    finally:
        env.close()
    if anchor is not None:
        anchor.eval()
        for parameter in anchor.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [*model.retina_projection.parameters(), *model.action_head.parameters()],
        lr=1e-4 if init_policy is not None else 1e-3,
        weight_decay=1e-4,
    )
    best_path = output_dir / "fly_cns_retina_best.pt"
    if anchor is None:
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
    else:
        best_accuracy = fit_preserving(
            model,
            anchor,
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
        if anchor is not None:
            best_accuracy = fit_preserving(
                model,
                anchor,
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
        else:
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
        "--output-dir", type=Path, default=Path("results/fly_cns_retina_v1")
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path("results/visual_nearfield_v1/nearfield_visual_teacher_best.pt"),
    )
    parser.add_argument(
        "--init-policy",
        type=Path,
        help="optional fly checkpoint to fine-tune instead of starting the graph head fresh",
    )
    parser.add_argument(
        "--preserve-base",
        action="store_true",
        help="distill the init policy while learning human-like pickup labels",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--val-samples", type=int, default=4000)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-samples", type=int, default=4000)
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
        args.teacher_policy,
        args.seed,
        args.init_policy,
        args.preserve_base,
    )
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
