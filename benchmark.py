"""Compare simple policies before adding PPO."""

from __future__ import annotations

from random import Random
from statistics import mean, median
from typing import Callable

from demo import choose_action
from env import ACTIONS, RunnerEnv


Policy = Callable[[dict, Random], int]


def random_policy(_: dict, rng: Random) -> int:
    return rng.randrange(len(ACTIONS))


def heuristic_policy(observation: dict, _: Random) -> int:
    return choose_action(observation)


def run_episode(policy: Policy, seed: int, max_steps: int = 300) -> dict[str, float]:
    env = RunnerEnv(seed=seed, max_steps=max_steps)
    observation, _ = env.reset(seed=seed)
    rng = Random(seed + 100_000)
    total_reward = 0.0
    steps = 0

    while True:
        action = policy(observation, rng)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            return {
                "steps": float(steps),
                "reward": total_reward,
                "score": float(info["score"]),
                "collision": float(info["collision"]),
            }


def report(name: str, policy: Policy, episodes: int = 20) -> None:
    results = [run_episode(policy, seed) for seed in range(episodes)]
    print(
        f"{name:9s}  "
        f"mean_steps={mean(item['steps'] for item in results):6.1f}  "
        f"median_steps={median(item['steps'] for item in results):6.1f}  "
        f"mean_score={mean(item['score'] for item in results):5.2f}  "
        f"collision_rate={mean(item['collision'] for item in results):.2f}"
    )


if __name__ == "__main__":
    report("random", random_policy)
    report("heuristic", heuristic_policy)
