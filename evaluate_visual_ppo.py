"""Audit a PPO-fine-tuned visual policy on disjoint seeds and levels."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from stable_baselines3 import PPO

from env import RICH_LEVEL_VARIANTS
from gym_env import RichPixelGymRunnerEnv
from train_recurrent_cnn import configure_process


def evaluate(model_path: Path, episodes: int, level_variant: str, seed_start: int) -> dict[str, float]:
    model = PPO.load(model_path, device="cpu")
    env = RichPixelGymRunnerEnv(
        max_steps=300, spawn_interval=8, rich_layer=2, level_variant=level_variant
    )
    results: list[dict[str, float]] = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            observation, _ = env.reset(seed=seed)
            steps = 0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(int(action))
                steps += 1
                if terminated or truncated:
                    results.append(
                        {"steps": float(steps), "collision": float(info["collision"])}
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
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, action="append")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    configure_process(2)
    seed_starts = args.seed_start or [5000, 6000, 7000, 8000]
    for level in RICH_LEVEL_VARIANTS[:-1]:
        for seed_start in seed_starts:
            result = evaluate(args.model, args.episodes, level, seed_start)
            print(
                f"level={level} start={seed_start} episodes={args.episodes} "
                f"mean_steps={result['steps']:.1f} collision_rate={result['collision']:.2f}"
            )


if __name__ == "__main__":
    main()
