"""Train a reduced fly-inspired graph, or its dense-readout control."""

from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor

from fly_graph import DenseFlyExtractor, FlyGraphExtractor
from gym_env import PartialPixelGymRunnerEnv


def train(
    total_timesteps: int, seed: int, output_dir: Path, dense: bool = False
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(
        PartialPixelGymRunnerEnv(seed=seed, max_steps=300, spawn_interval=8),
        filename=str(output_dir / "monitor.csv"),
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": DenseFlyExtractor if dense else FlyGraphExtractor,
            "features_extractor_kwargs": {"features_dim": 24},
            "lstm_hidden_size": 32,
            "n_lstm_layers": 1,
            "net_arch": {"pi": [], "vf": []},
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
        model_name = "dense_fly_recurrent_ppo" if dense else "fly_graph_recurrent_ppo"
        model_path = output_dir / model_name
        model.save(model_path)
        return model_path.with_suffix(".zip")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path
    )
    args = parser.parse_args()
    output_dir = args.output_dir or Path(
        "results/dense_fly_recurrent_ppo"
        if args.dense
        else "results/fly_graph_recurrent_ppo"
    )
    model_path = train(args.timesteps, args.seed, output_dir, dense=args.dense)
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
