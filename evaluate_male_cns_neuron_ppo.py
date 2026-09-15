"""Evaluate the fixed neuron-level MaleCNS controller on fixed seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env import RICH_EVAL_LEVEL_VARIANTS
from gym_env import (
    PartialPixelGymRunnerEnv,
    PixelGymRunnerEnv,
    RichPartialPixelGymRunnerEnv,
    RichPixelGymRunnerEnv,
)
from male_cns_neuron_graph import MaleCNSNeuronGraphExtractor  # noqa: F401


def evaluate(
    model_path: Path,
    episodes: int,
    max_steps: int,
    speed: float,
    spawn_interval: int,
    seed_start: int,
    full_view: bool,
    rich: bool,
    level_variant: str,
    rich_layer: int,
) -> list[dict[str, float]]:
    env_class = (
        RichPixelGymRunnerEnv
        if rich and full_view
        else RichPartialPixelGymRunnerEnv
        if rich
        else PixelGymRunnerEnv
        if full_view
        else PartialPixelGymRunnerEnv
    )
    runner_kwargs = {
        "max_steps": max_steps,
        "speed": speed,
        "spawn_interval": spawn_interval,
    }
    if rich:
        runner_kwargs.update(rich_layer=rich_layer, level_variant=level_variant)
    env = DummyVecEnv([lambda: env_class(**runner_kwargs)])
    model = RecurrentPPO.load(model_path, env=env, device="cpu")
    results: list[dict[str, float]] = []
    try:
        for seed in range(episodes):
            env.seed(seed_start + seed)
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
    return results


def print_summary(label: str, results: list[dict[str, float]]) -> None:
    print(
        f"{label} episodes={len(results)}  "
        f"mean_steps={mean(item['steps'] for item in results):.1f}  "
        f"median_steps={median(item['steps'] for item in results):.1f}  "
        f"mean_score={mean(item['score'] for item in results):.2f}  "
        f"collision_rate={mean(item['collision'] for item in results):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--spawn-interval", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--seed-start",
        type=int,
        action="append",
        help="evaluate another deterministic seed band (repeatable)",
    )
    parser.add_argument("--full-view", action="store_true")
    parser.add_argument("--rich", action="store_true")
    parser.add_argument(
        "--level-variant",
        choices=RICH_EVAL_LEVEL_VARIANTS,
        default="standard",
        help="rich evaluation profile; surprise is held out from training",
    )
    parser.add_argument("--rich-layer", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if not args.rich and args.level_variant != "standard":
        parser.error("--level-variant requires --rich")
    seed_starts = args.seed_start or [args.seed_offset]

    for seed_start in seed_starts:
        results = evaluate(
            args.model,
            args.episodes,
            args.max_steps,
            args.speed,
            args.spawn_interval,
            seed_start,
            args.full_view,
            args.rich,
            args.level_variant,
            args.rich_layer,
        )
        print_summary(f"level={args.level_variant} start={seed_start}", results)


if __name__ == "__main__":
    main()
