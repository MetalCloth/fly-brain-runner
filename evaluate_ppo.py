"""Evaluate a saved PPO model on fixed seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median

from stable_baselines3 import PPO

from gym_env import GymRunnerEnv, PartialPixelGymRunnerEnv, PixelGymRunnerEnv


def evaluate(
    model_path: Path,
    episodes: int,
    max_steps: int,
    pixels: bool = False,
    partial_pixels: bool = False,
) -> list[dict[str, float]]:
    if pixels and partial_pixels:
        raise ValueError("pixels and partial_pixels are mutually exclusive")
    env_class = (
        PartialPixelGymRunnerEnv
        if partial_pixels
        else PixelGymRunnerEnv
        if pixels
        else GymRunnerEnv
    )
    env = env_class(max_steps=max_steps, spawn_interval=8)
    model = PPO.load(model_path, env=env, device="cpu")
    results: list[dict[str, float]] = []
    try:
        for seed in range(episodes):
            observation, _ = env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(int(action))
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
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument("--pixels", action="store_true")
    view_group.add_argument("--partial-pixels", action="store_true")
    args = parser.parse_args()

    results = evaluate(
        args.model,
        args.episodes,
        args.max_steps,
        pixels=args.pixels,
        partial_pixels=args.partial_pixels,
    )
    print(
        f"episodes={len(results)}  "
        f"mean_steps={mean(item['steps'] for item in results):.1f}  "
        f"median_steps={median(item['steps'] for item in results):.1f}  "
        f"mean_score={mean(item['score'] for item in results):.2f}  "
        f"collision_rate={mean(item['collision'] for item in results):.2f}"
    )


if __name__ == "__main__":
    main()
