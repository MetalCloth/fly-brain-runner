"""Train a sparse fly-inspired policy with ordinary PPO."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from fly_features import SparseFlyExtractor
from gym_env import GymRunnerEnv


def train(total_timesteps: int, seed: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(
        GymRunnerEnv(seed=seed, max_steps=300, spawn_interval=8),
        filename=str(output_dir / "monitor.csv"),
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": SparseFlyExtractor,
            "features_extractor_kwargs": {"features_dim": 24},
            "net_arch": [64, 64],
        },
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
        model_path = output_dir / "fly_sparse_ppo"
        model.save(model_path)
        return model_path.with_suffix(".zip")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/fly_sparse_ppo")
    )
    args = parser.parse_args()
    model_path = train(args.timesteps, args.seed, args.output_dir)
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
