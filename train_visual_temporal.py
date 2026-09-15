"""Train a four-frame visual policy with the verified rich-environment teacher."""

from __future__ import annotations

import argparse
from pathlib import Path
from random import Random
from statistics import mean

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from env import RICH_ACTIONS, RICH_LEVEL_VARIANTS, RunnerEnv
from rich_benchmark import teacher_action
from train_recurrent_cnn import SAFE_MAX_RSS_MB, SAFE_MIN_AVAILABLE_MB, configure_process
from train_visual_teacher import (
    frame,
    pack_buckets,
    resource_ok,
    visual_teacher_action,
)


FRAME_COUNT = 4
FRAME_CHANNELS = 3
FEATURE_DIM = 128


class TemporalVisualPolicy(nn.Module):
    """The static visual policy with four consecutive RGB frames as input."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(FRAME_COUNT * FRAME_CHANNELS, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, FEATURE_DIM),
            nn.ReLU(),
            nn.Linear(FEATURE_DIM, len(RICH_ACTIONS)),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 5:
            frames = frames.reshape(frames.shape[0], -1, 84, 84)
        return self.head(self.features(frames.float() / 255.0))


def stack_history(history: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([item.transpose(2, 0, 1) for item in history], axis=0)


def collect_temporal(
    seed_start: int,
    episodes: int,
    level_variant: str,
    max_samples: int,
    policy,
) -> tuple[np.ndarray, np.ndarray]:
    buckets: list[list[np.ndarray]] = [[] for _ in RICH_ACTIONS]
    per_action = max(1, max_samples // len(RICH_ACTIONS))
    for episode in range(episodes):
        seed = seed_start + episode
        env = RunnerEnv(
            seed=seed,
            max_steps=300,
            spawn_interval=8,
            rich_mechanics=True,
            rich_layer=2,
            level_variant=level_variant,
        )
        observation, _ = env.reset(seed=seed)
        initial = frame(env)
        history = [initial.copy() for _ in range(FRAME_COUNT)]
        rng = Random(seed + 100_000)
        while True:
            action = policy(observation, rng)
            if len(buckets[action]) < per_action:
                buckets[action].append(stack_history(history))
            if sum(len(bucket) for bucket in buckets) >= max_samples:
                break
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            history = [*history[1:], frame(env)]
        if sum(len(bucket) for bucket in buckets) >= max_samples:
            break
    return pack_buckets(buckets, seed_start)


def collect_student_temporal(
    model: TemporalVisualPolicy,
    seed_start: int,
    episodes: int,
    level_variant: str,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    buckets: list[list[np.ndarray]] = [[] for _ in RICH_ACTIONS]
    per_action = max(1, max_samples // len(RICH_ACTIONS))
    model.eval()
    for episode in range(episodes):
        seed = seed_start + episode
        env = RunnerEnv(
            seed=seed,
            max_steps=300,
            spawn_interval=8,
            rich_mechanics=True,
            rich_layer=2,
            level_variant=level_variant,
        )
        observation, _ = env.reset(seed=seed)
        initial = frame(env)
        history = [initial.copy() for _ in range(FRAME_COUNT)]
        rng = Random(seed + 100_000)
        while True:
            current = stack_history(history)
            label = visual_teacher_action(observation, rng)
            if len(buckets[label]) < per_action:
                buckets[label].append(current)
            with torch.no_grad():
                logits = model(torch.from_numpy(current).unsqueeze(0))
            student_action = int(logits.argmax(1).item())
            observation, _, terminated, truncated, _ = env.step(student_action)
            if terminated or truncated:
                break
            history = [*history[1:], frame(env)]
            if sum(len(bucket) for bucket in buckets) >= max_samples:
                break
        if sum(len(bucket) for bucket in buckets) >= max_samples:
            break
    return pack_buckets(buckets, seed_start)


def initialize_from_static(model: TemporalVisualPolicy, checkpoint: Path) -> None:
    from train_visual_teacher import VisualPolicy

    teacher = VisualPolicy()
    teacher.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.no_grad():
        model.features[0].weight.zero_()
        model.features[0].weight[:, -FRAME_CHANNELS:].copy_(teacher.features[0].weight)
        model.features[0].bias.copy_(teacher.features[0].bias)
        model.features[2].load_state_dict(teacher.features[2].state_dict())
        model.features[4].load_state_dict(teacher.features[4].state_dict())
        model.head[0].load_state_dict(teacher.head[0].state_dict())
        model.head[2].load_state_dict(teacher.head[2].state_dict())


def validation_accuracy(
    model: TemporalVisualPolicy,
    frames: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
) -> float:
    model.eval()
    dataset = TensorDataset(torch.from_numpy(frames), torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=batch_size)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, batch_labels in loader:
            correct += int((model(images).argmax(1) == batch_labels).sum())
            total += len(batch_labels)
    return correct / max(1, total)


def fit(
    model: TemporalVisualPolicy,
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
    dataset = TensorDataset(torch.from_numpy(train_frames), torch.from_numpy(train_labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    counts = np.bincount(train_labels, minlength=len(RICH_ACTIONS)).astype(np.float32)
    weights = torch.from_numpy(np.power(len(train_labels) / np.maximum(counts, 1), 0.35))
    weights /= weights.mean()
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch_index, (images, labels) in enumerate(loader):
            if batch_index % 32 == 0 and not resource_ok(max_rss_mb, min_available_mb):
                raise RuntimeError("temporal visual training stopped by resource guard")
            brightness = 0.95 + 0.10 * torch.rand((images.shape[0], 1, 1, 1))
            loss = F.cross_entropy(model(images * brightness), labels, weight=weights)
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
) -> Path:
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(teacher_checkpoint)
    configure_process(2)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_frames, base_labels = collect_temporal(0, 120, "mixed", train_samples // 2, teacher_action)
    maneuver_frames, maneuver_labels = collect_temporal(200, 120, "mixed", train_samples - train_samples // 2, visual_teacher_action)
    train_frames = np.concatenate((base_frames, maneuver_frames))
    train_labels = np.concatenate((base_labels, maneuver_labels))
    val_frames, val_labels = collect_temporal(1000, 50, "mixed", val_samples, visual_teacher_action)
    print(
        f"dataset train={len(train_labels)} val={len(val_labels)} "
        f"train_actions={np.bincount(train_labels, minlength=len(RICH_ACTIONS)).tolist()}"
    )

    model = TemporalVisualPolicy()
    initialize_from_static(model, teacher_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_path = output_dir / "temporal_visual_teacher_best.pt"
    best_accuracy = fit(
        model, optimizer, train_frames, train_labels, val_frames, val_labels,
        epochs, batch_size, -1.0, best_path, max_rss_mb, min_available_mb
    )
    for dagger_round in range(dagger_rounds):
        model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
        extra_frames, extra_labels = collect_student_temporal(
            model, 2000 + dagger_round * 100, 100, dagger_level, dagger_samples
        )
        train_frames = np.concatenate((train_frames, extra_frames))
        train_labels = np.concatenate((train_labels, extra_labels))
        print(f"dagger_round={dagger_round + 1} samples={len(extra_labels)}")
        best_accuracy = fit(
            model, optimizer, train_frames, train_labels, val_frames, val_labels,
            max(3, epochs // 3), batch_size, best_accuracy, best_path,
            max_rss_mb, min_available_mb
        )
    return best_path


def evaluate(model_path: Path, episodes: int, level_variant: str, seed_start: int) -> dict[str, float]:
    model = TemporalVisualPolicy()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    results: list[dict[str, float]] = []
    for seed in range(seed_start, seed_start + episodes):
        env = RunnerEnv(
            seed=seed, max_steps=300, spawn_interval=8, rich_mechanics=True,
            rich_layer=2, level_variant=level_variant,
        )
        observation, _ = env.reset(seed=seed)
        initial = frame(env)
        history = [initial.copy() for _ in range(FRAME_COUNT)]
        steps = 0
        while True:
            with torch.no_grad():
                action = int(model(torch.from_numpy(stack_history(history)).unsqueeze(0)).argmax(1).item())
            observation, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                results.append({"steps": float(steps), "collision": float(info["collision"])})
                break
            history = [*history[1:], frame(env)]
    return {
        "steps": mean(item["steps"] for item in results),
        "collision": mean(item["collision"] for item in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, default=Path("results/visual_dagger_dense_v2/visual_teacher_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/visual_temporal_dagger_v1"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=8_000)
    parser.add_argument("--val-samples", type=int, default=2_000)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--dagger-samples", type=int, default=2_000)
    parser.add_argument("--dagger-level", choices=RICH_LEVEL_VARIANTS, default="dense")
    parser.add_argument("--max-rss-mb", type=int, default=SAFE_MAX_RSS_MB)
    parser.add_argument("--min-available-mb", type=int, default=SAFE_MIN_AVAILABLE_MB)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.train_samples, args.val_samples, args.dagger_samples) < 1:
        parser.error("training counts must be positive")
    path = train(
        args.teacher_checkpoint, args.output_dir, args.epochs, args.batch_size,
        args.train_samples, args.val_samples, args.dagger_rounds, args.dagger_samples,
        args.dagger_level, args.max_rss_mb, args.min_available_mb,
    )
    print(f"saved model: {path}")
    for level in RICH_LEVEL_VARIANTS[:-1]:
        result = evaluate(path, 50, level, 0)
        print(f"level={level} episodes=50 mean_steps={result['steps']:.1f} collision_rate={result['collision']:.2f}")


if __name__ == "__main__":
    main()
