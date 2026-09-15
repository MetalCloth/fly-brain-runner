"""Evaluate numeric PPO checkpoints on disjoint seed bands."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median

from stable_baselines3 import PPO

from env import RICH_LEVEL_VARIANTS
from gym_env import RichGymRunnerEnv
from train_recurrent_cnn import configure_process


def evaluate(
    model_path: Path,
    episodes: int,
    seed_start: int,
    rich_layer: int,
    level_variant: str,
) -> dict[str, float]:
    model = PPO.load(model_path, device="cpu")
    env = RichGymRunnerEnv(
        max_steps=300,
        spawn_interval=8,
        rich_layer=rich_layer,
        level_variant=level_variant,
    )
    results: list[dict[str, float]] = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            observation, _ = env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(
                    int(action)
                )
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    results.append(
                        {
                            "steps": float(steps),
                            "reward": total_reward,
                            "score": float(info["score"]),
                            "collision": float(info["collision"]),
                        }
                    )
                    break
    finally:
        env.close()

    return {
        "mean_steps": mean(item["steps"] for item in results),
        "median_steps": median(item["steps"] for item in results),
        "mean_score": mean(item["score"] for item in results),
        "collision_rate": mean(item["collision"] for item in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--rich-layer", type=int, choices=(1, 2), default=2)
    parser.add_argument("--level-variant", choices=RICH_LEVEL_VARIANTS, default="standard")
    parser.add_argument(
        "--seed-start",
        dest="seed_starts",
        action="append",
        type=int,
        help="evaluate another disjoint seed band (repeatable)",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    configure_process(2)
    seed_starts = args.seed_starts or [0, 1000]
    for model_path in args.models:
        summaries = [
            (
                start,
                evaluate(
                    model_path,
                    args.episodes,
                    start,
                    args.rich_layer,
                    args.level_variant,
                ),
            )
            for start in seed_starts
        ]
        fields = [f"model={model_path}"]
        for start, summary in summaries:
            fields.extend(
                (
                    f"start_{start}_steps={summary['mean_steps']:.1f}",
                    f"start_{start}_collision={summary['collision_rate']:.2f}",
                )
            )
        if len(summaries) == 2:
            fields.append(
                f"steps_gap={summaries[0][1]['mean_steps'] - summaries[1][1]['mean_steps']:.1f}"
            )
        print(f"level={args.level_variant} " + " ".join(fields))


if __name__ == "__main__":
    main()
