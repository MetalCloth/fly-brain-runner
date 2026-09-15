"""Train the recurrent/action readout on the fixed MaleCNS graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor

from gym_env import PartialPixelGymRunnerEnv
from male_cns_graph import FEATURES_DIM, MaleCNSGraphExtractor


def train(total_timesteps: int, seed: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(
        PartialPixelGymRunnerEnv(seed=seed, max_steps=300, spawn_interval=8),
        filename=str(output_dir / "monitor.csv"),
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": MaleCNSGraphExtractor,
            "features_extractor_kwargs": {"features_dim": FEATURES_DIM},
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
        model_path = output_dir / "male_cns_recurrent_ppo"
        model.save(model_path)
        return model_path.with_suffix(".zip")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/male_cns_recurrent_ppo")
    )
    args = parser.parse_args()
    model_path = train(args.timesteps, args.seed, args.output_dir)
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
