"""Bootstrap a pixel policy from the verified rich-environment teacher."""

from __future__ import annotations

import argparse
from pathlib import Path
from random import Random
import resource
from statistics import mean
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from env import RICH_ACTIONS, RICH_LEVEL_VARIANTS, RunnerEnv
from gym_env import RichPixelGymRunnerEnv
from rich_benchmark import human_teacher_action, teacher_action
from train_recurrent_cnn import (
    SAFE_MAX_RSS_MB,
    SAFE_MIN_AVAILABLE_MB,
    configure_process,
)


MOTION_ACTIONS = (3, 4)


class VisualPolicy(nn.Module):
    """Small 84x84 RGB classifier used only for the visual bootstrap."""

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, len(RICH_ACTIONS)),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(frames / 255.0))


def resource_ok(max_rss_mb: int, min_available_mb: int) -> bool:
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    available_mb = None
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    available_mb = float(line.split()[1]) / 1024.0
                    break
    except OSError:
        pass
    if rss_mb > max_rss_mb:
        print(f"resource guard: stopping at {rss_mb:.0f} MiB process RSS")
        return False
    if available_mb is not None and available_mb < min_available_mb:
        print(f"resource guard: stopping at {available_mb:.0f} MiB available RAM")
        return False
    return True


def frame(env: RunnerEnv) -> np.ndarray:
    return np.frombuffer(env.render_rgb(), dtype=np.uint8).reshape((84, 84, 3)).copy()


def near_field_view(pixels: np.ndarray) -> np.ndarray:
    """Append a vertical nearest-neighbor zoom of the lower track."""

    rows = np.linspace(32, 83, 84).round().astype(np.int64)
    zoom = pixels[rows]
    return np.concatenate((pixels, zoom), axis=2)


def identity_view(pixels: np.ndarray) -> np.ndarray:
    return pixels


def visual_teacher_action(observation: dict, rng: Random) -> int:
    """Prefer the required jump/roll before using a lane change escape."""

    current = [
        item
        for item in observation["obstacles"]
        if item["lane"] == observation["lane"] and item["distance"] <= 7.0
    ]
    if current:
        nearest = min(current, key=lambda item: item["distance"])
        if nearest["kind"] in ("jump", "gap", "either"):
            return 3
        if nearest["kind"] == "roll":
            return 4
    return teacher_action(observation, rng)


def human_visual_teacher_action(observation: dict, rng: Random) -> int:
    """Use soft pickup trade-offs, with a last-second recovery guard."""

    imminent = any(
        item["lane"] == observation["lane"] and item["distance"] <= 2.5
        for item in observation.get("obstacles", ())
    )
    if imminent:
        return visual_teacher_action(observation, rng)
    return human_teacher_action(observation, rng)


def timed_action(
    proposed: int,
    locked_motion: int | None,
    lock_remaining: int,
    hold_steps: int,
) -> tuple[int, int | None, int]:
    """Suppress a short opposite-motion flip without changing lane actions."""

    if hold_steps <= 0:
        return proposed, None, 0
    if locked_motion in MOTION_ACTIONS and lock_remaining > 0:
        if proposed == locked_motion:
            return proposed, locked_motion, hold_steps
        if proposed in MOTION_ACTIONS:
            return locked_motion, locked_motion, lock_remaining - 1
        return proposed, locked_motion, lock_remaining - 1
    if proposed in MOTION_ACTIONS:
        return proposed, proposed, hold_steps
    return proposed, None, 0


def guarded_motion_action(
    proposed: int,
    last_action: int | None,
    action_streak: int,
    pending_motion: int | None,
    pending_steps: int,
    delay_steps: int,
) -> tuple[int, int | None, int, int | None, int]:
    """Delay only an opposite motion after a sustained prior motion."""

    if (
        delay_steps > 0
        and proposed in MOTION_ACTIONS
        and last_action in MOTION_ACTIONS
        and proposed != last_action
        and action_streak >= 3
    ):
        pending_steps = pending_steps + 1 if pending_motion == proposed else 1
        pending_motion = proposed
        if pending_steps <= delay_steps:
            return (
                last_action,
                last_action,
                action_streak + 1,
                pending_motion,
                pending_steps,
            )
        return proposed, proposed, 1, None, 0

    pending_motion = None
    pending_steps = 0
    if proposed in MOTION_ACTIONS:
        streak = action_streak + 1 if proposed == last_action else 1
        return proposed, proposed, streak, pending_motion, pending_steps
    return proposed, None, 0, pending_motion, pending_steps


def initialize_near_field(model: VisualPolicy, checkpoint: Path) -> None:
    """Copy v2 into the original channels and leave the zoom branch neutral."""

    teacher = VisualPolicy()
    teacher.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    if model.features[0].in_channels != 6:
        raise ValueError("near-field model must have six input channels")
    with torch.no_grad():
        model.features[0].weight.zero_()
        model.features[0].weight[:, :3].copy_(teacher.features[0].weight)
        model.features[0].bias.copy_(teacher.features[0].bias)
        model.features[2].load_state_dict(teacher.features[2].state_dict())
        model.features[4].load_state_dict(teacher.features[4].state_dict())
        model.head[0].load_state_dict(teacher.head[0].state_dict())
        model.head[2].load_state_dict(teacher.head[2].state_dict())


TeacherPolicy = Callable[[dict, Random], int]


def pack_buckets(
    buckets: list[list[np.ndarray]], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    frames = np.concatenate([np.stack(bucket) for bucket in buckets if bucket])
    labels = np.concatenate(
        [
            np.full(len(bucket), action, dtype=np.int64)
            for action, bucket in enumerate(buckets)
            if bucket
        ]
    )
    order = np.random.default_rng(seed).permutation(len(labels))
    return frames[order], labels[order]


def collect(
    seed_start: int,
    episodes: int,
    level_variant: str,
    max_samples: int,
    policy: TeacherPolicy = visual_teacher_action,
    view=identity_view,
) -> tuple[np.ndarray, np.ndarray]:
    buckets: list[list[np.ndarray]] = [[] for _ in RICH_ACTIONS]
    per_action = max(1, max_samples // len(RICH_ACTIONS))
    for episode in range(episodes):
        env = RunnerEnv(
            seed=seed_start + episode,
            max_steps=300,
            spawn_interval=8,
            rich_mechanics=True,
            rich_layer=2,
            level_variant=level_variant,
        )
        observation, _ = env.reset(seed=seed_start + episode)
        rng = Random(seed_start + episode + 100_000)
        while True:
            action = policy(observation, rng)
            if len(buckets[action]) < per_action:
                buckets[action].append(view(frame(env)))
            if sum(len(bucket) for bucket in buckets) >= max_samples:
                break
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        if sum(len(bucket) for bucket in buckets) >= max_samples:
            break

    return pack_buckets(buckets, seed_start)


def collect_student(
    model: VisualPolicy,
    seed_start: int,
    episodes: int,
    level_variant: str,
    max_samples: int,
    view=identity_view,
    policy: TeacherPolicy = visual_teacher_action,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect teacher labels from states visited by the current student."""

    # Give imminent hazards their own budget; otherwise common safe frames can
    # fill an action bucket before the rare last-second recovery examples.
    critical_budget = max_samples // 2
    general_budget = max_samples - critical_budget
    critical_buckets: list[list[np.ndarray]] = [[] for _ in RICH_ACTIONS]
    general_buckets: list[list[np.ndarray]] = [[] for _ in RICH_ACTIONS]
    critical_per_action = max(1, critical_budget // len(RICH_ACTIONS))
    general_per_action = max(1, general_budget // len(RICH_ACTIONS))

    def sample_count() -> int:
        return sum(
            len(bucket)
            for group in (critical_buckets, general_buckets)
            for bucket in group
        )

    model.eval()
    for episode in range(episodes):
        env = RunnerEnv(
            seed=seed_start + episode,
            max_steps=300,
            spawn_interval=8,
            rich_mechanics=True,
            rich_layer=2,
            level_variant=level_variant,
        )
        observation, _ = env.reset(seed=seed_start + episode)
        rng = Random(seed_start + episode + 100_000)
        while True:
            current_frame = view(frame(env))
            label = policy(observation, rng)
            imminent = any(
                item["lane"] == observation["lane"] and item["distance"] <= 2.5
                for item in observation["obstacles"]
            )
            buckets = critical_buckets if imminent else general_buckets
            per_action = critical_per_action if imminent else general_per_action
            if len(buckets[label]) < per_action:
                buckets[label].append(current_frame)
            with torch.no_grad():
                logits = model(
                    torch.from_numpy(current_frame)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                )
            student_action = int(logits.argmax(1).item())
            observation, _, terminated, truncated, _ = env.step(student_action)
            if terminated or truncated:
                break
            if sample_count() >= max_samples:
                break
        if sample_count() >= max_samples:
            break
    buckets = [
        critical_buckets[action] + general_buckets[action]
        for action in range(len(RICH_ACTIONS))
    ]
    return pack_buckets(buckets, seed_start)


def validation_accuracy(model: VisualPolicy, frames: np.ndarray, labels: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(frames).permute(0, 3, 1, 2).float())
        return float((logits.argmax(1) == torch.from_numpy(labels)).float().mean())


def fit(
    model: VisualPolicy,
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
    stale_epochs = 0

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for batch_index, (images, labels) in enumerate(loader):
            if batch_index % 32 == 0 and not resource_ok(max_rss_mb, min_available_mb):
                raise RuntimeError("visual training stopped by resource guard")
            brightness = 0.95 + 0.10 * torch.rand((images.shape[0], 1, 1, 1))
            logits = model(images.float() * brightness)
            loss = F.cross_entropy(logits, labels, weight=weights)
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
) -> Path:
    configure_process(2)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if dagger_rounds < 0 or dagger_samples < 1:
        raise ValueError("dagger rounds must be nonnegative and samples positive")
    if teacher_policy not in ("safe", "human"):
        raise ValueError("teacher_policy must be 'safe' or 'human'")
    label_policy = (
        visual_teacher_action
        if teacher_policy == "safe"
        else human_visual_teacher_action
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base_frames, base_labels = collect(
        0, 120, "mixed", train_samples // 2, label_policy
    )
    maneuver_frames, maneuver_labels = collect(
        200, 120, "mixed", train_samples - train_samples // 2, label_policy
    )
    train_frames = np.concatenate((base_frames, maneuver_frames))
    train_labels = np.concatenate((base_labels, maneuver_labels))
    val_frames, val_labels = collect(1000, 50, "mixed", val_samples, label_policy)
    print(
        f"dataset train={len(train_labels)} val={len(val_labels)} "
        f"train_actions={np.bincount(train_labels, minlength=len(RICH_ACTIONS)).tolist()}"
    )

    model = VisualPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_path = output_dir / "visual_teacher_best.pt"
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


def evaluate(
    model_path: Path,
    episodes: int,
    level_variant: str,
    seed_start: int,
    motion_hold: int = 0,
    motion_flip_delay: int = 0,
    input_channels: int = 3,
    view=identity_view,
) -> dict[str, float]:
    model = VisualPolicy(input_channels=input_channels)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    env = RichPixelGymRunnerEnv(
        max_steps=300, spawn_interval=8, rich_layer=2, level_variant=level_variant
    )
    results: list[dict[str, float]] = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            observation, _ = env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            locked_motion = None
            lock_remaining = 0
            last_action = None
            action_streak = 0
            pending_motion = None
            pending_steps = 0
            while True:
                with torch.no_grad():
                    logits = model(
                        torch.from_numpy(view(observation))
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                    )
                proposed = int(logits.argmax(1).item())
                if motion_flip_delay:
                    (
                        action,
                        last_action,
                        action_streak,
                        pending_motion,
                        pending_steps,
                    ) = guarded_motion_action(
                        proposed,
                        last_action,
                        action_streak,
                        pending_motion,
                        pending_steps,
                        motion_flip_delay,
                    )
                else:
                    action, locked_motion, lock_remaining = timed_action(
                        proposed, locked_motion, lock_remaining, motion_hold
                    )
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    results.append(
                        {
                            "steps": float(steps),
                            "reward": total_reward,
                            "collision": float(info["collision"]),
                        }
                    )
                    break
    finally:
        env.close()
    return {
        "steps": mean(item["steps"] for item in results),
        "collision": mean(item["collision"] for item in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/visual_teacher"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=12_000)
    parser.add_argument("--val-samples", type=int, default=4_000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--dagger-samples", type=int, default=3_000)
    parser.add_argument("--dagger-level", choices=RICH_LEVEL_VARIANTS, default="mixed")
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
    )
    print(f"saved model: {model_path}")
    for level in RICH_LEVEL_VARIANTS[:-1]:
        for seed_start in (0, 1000):
            result = evaluate(model_path, args.episodes, level, seed_start)
            print(
                f"level={level} start={seed_start} episodes={args.episodes} "
                f"mean_steps={result['steps']:.1f} collision_rate={result['collision']:.2f}"
            )


if __name__ == "__main__":
    main()
