"""Train an MLP PPO baseline, or a CNN PPO baseline from pixels."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from gym_env import GymRunnerEnv, PartialPixelGymRunnerEnv, PixelGymRunnerEnv


def train(
    total_timesteps: int,
    seed: int,
    output_dir: Path,
    pixels: bool = False,
    partial_pixels: bool = False,
) -> Path:
    if pixels and partial_pixels:
        raise ValueError("pixels and partial_pixels are mutually exclusive")
    output_dir.mkdir(parents=True, exist_ok=True)
    visual = pixels or partial_pixels
    env_class = (
        PartialPixelGymRunnerEnv
        if partial_pixels
        else PixelGymRunnerEnv
        if pixels
        else GymRunnerEnv
    )
    env = Monitor(
        env_class(seed=seed, max_steps=300, spawn_interval=8),
        filename=str(output_dir / "monitor.csv"),
    )
    model = PPO(
        "CnnPolicy" if visual else "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [64, 64]},
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        seed=seed,
        device="cpu",
    )
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        model_name = (
            "cnn_partial_ppo" if partial_pixels else "cnn_ppo" if pixels else "mlp_ppo"
        )
        model_path = output_dir / model_name
        model.save(model_path)
        return model_path.with_suffix(".zip")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=7)
    view_group = parser.add_mutually_exclusive_group()
    view_group.add_argument("--pixels", action="store_true")
    view_group.add_argument("--partial-pixels", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path(
        "results/cnn_partial_ppo"
        if args.partial_pixels
        else "results/cnn_ppo"
        if args.pixels
        else "results/mlp_ppo"
    )
    model_path = train(
        args.timesteps,
        args.seed,
        output_dir,
        pixels=args.pixels,
        partial_pixels=args.partial_pixels,
    )
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
