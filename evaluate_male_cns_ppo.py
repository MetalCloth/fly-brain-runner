"""Evaluate the fixed MaleCNS graph controller on fixed seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_env import PartialPixelGymRunnerEnv
from male_cns_graph import MaleCNSGraphExtractor  # noqa: F401 - model loader needs it


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    env = DummyVecEnv(
        [lambda: PartialPixelGymRunnerEnv(max_steps=args.max_steps, spawn_interval=8)]
    )
    model = RecurrentPPO.load(args.model, env=env, device="cpu")
    results: list[dict[str, float]] = []
    try:
        for seed in range(args.episodes):
            env.seed(seed)
            observation = env.reset()
            lstm_state = None
            episode_start = np.ones((env.num_envs,), dtype=bool)
            total_reward = 0.0
            steps = 0
            while True:
                action, lstm_state = model.predict(
                    observation,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                observation, rewards, dones, infos = env.step(action)
                total_reward += float(rewards[0])
                steps += 1
                episode_start = dones
                if dones[0]:
                    info = infos[0]
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

    print(
        f"episodes={len(results)}  "
        f"mean_steps={mean(item['steps'] for item in results):.1f}  "
        f"median_steps={median(item['steps'] for item in results):.1f}  "
        f"mean_score={mean(item['score'] for item in results):.2f}  "
        f"collision_rate={mean(item['collision'] for item in results):.2f}"
    )


if __name__ == "__main__":
    main()
