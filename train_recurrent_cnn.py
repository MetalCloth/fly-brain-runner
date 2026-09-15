"""Train a recurrent CNN PPO policy on alternating partially observed frames."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from gym_env import PartialPixelGymRunnerEnv, RichPixelGymRunnerEnv


SAFE_DEFAULT_THREADS = 2
SAFE_MAX_RSS_MB = 3072
SAFE_MIN_AVAILABLE_MB = 2048


class ResourceGuardCallback(BaseCallback):
    """Stop cleanly before a training process can crowd out the desktop."""

    def __init__(
        self,
        max_rss_mb: int = SAFE_MAX_RSS_MB,
        min_available_mb: int = SAFE_MIN_AVAILABLE_MB,
        check_every: int = 256,
    ) -> None:
        if max_rss_mb < 1 or min_available_mb < 1 or check_every < 1:
            raise ValueError("resource guard limits must be positive")
        super().__init__(verbose=0)
        self.max_rss_mb = max_rss_mb
        self.min_available_mb = min_available_mb
        self.check_every = check_every

    @staticmethod
    def _available_memory_mb() -> float | None:
        try:
            with open("/proc/meminfo", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        return float(line.split()[1]) / 1024.0
        except OSError:
            return None
        return None

    def _on_step(self) -> bool:
        if self.n_calls % self.check_every:
            return True
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        available_mb = self._available_memory_mb()
        if rss_mb > self.max_rss_mb:
            print(f"resource guard: stopping at {rss_mb:.0f} MiB process RSS")
            return False
        if available_mb is not None and available_mb < self.min_available_mb:
            print(f"resource guard: stopping at {available_mb:.0f} MiB available RAM")
            return False
        return True


def configure_process(threads: int) -> None:
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A caller may already have initialized Torch's inter-op pool.
        pass
    try:
        os.nice(10)
    except OSError:
        pass


def train(
    total_timesteps: int,
    seed: int,
    output_dir: Path,
    rich: bool = False,
    curriculum: bool = False,
    threads: int = SAFE_DEFAULT_THREADS,
    max_rss_mb: int = SAFE_MAX_RSS_MB,
    min_available_mb: int = SAFE_MIN_AVAILABLE_MB,
) -> Path:
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive")
    if curriculum and not rich:
        raise ValueError("curriculum requires rich=True")
    configure_process(threads)

    output_dir.mkdir(parents=True, exist_ok=True)
    env_class = RichPixelGymRunnerEnv if rich else PartialPixelGymRunnerEnv

    def make_env(layer: int = 2):
        return Monitor(
            env_class(
                seed=seed,
                max_steps=300,
                spawn_interval=8,
                rich_layer=layer,
            ),
            filename=str(output_dir / f"monitor_layer{layer}.csv"),
        )

    env = make_env(1 if curriculum else 2)
    checkpoint_callback = CheckpointCallback(
        save_freq=2048,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="rich_recurrent_cnn" if rich else "recurrent_cnn",
    )
    resource_callback = ResourceGuardCallback(
        max_rss_mb=max_rss_mb,
        min_available_mb=min_available_mb,
    )
    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        policy_kwargs={"lstm_hidden_size": 32, "n_lstm_layers": 1},
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
        if curriculum:
            first_phase = max(1, total_timesteps // 2)
            model.learn(
                total_timesteps=first_phase,
                callback=[checkpoint_callback, resource_callback],
                progress_bar=False,
            )
            env.close()
            env = make_env(2)
            model.set_env(env)
            remaining = total_timesteps - first_phase
            if remaining:
                model.learn(
                    total_timesteps=remaining,
                    callback=[checkpoint_callback, resource_callback],
                    progress_bar=False,
                    reset_num_timesteps=False,
                )
        else:
            model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback, resource_callback],
                progress_bar=False,
            )
        model_path = output_dir / ("rich_recurrent_cnn_ppo" if rich else "recurrent_cnn_ppo")
        model.save(model_path)
        return model_path.with_suffix(".zip")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/recurrent_cnn_ppo")
    )
    parser.add_argument(
        "--rich",
        action="store_true",
        help="train on the rich Subway-style mechanics environment",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="train rich Layer 1 first, then continue on Layer 2",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=SAFE_DEFAULT_THREADS,
        help="PyTorch CPU threads (kept low by default for desktop safety)",
    )
    parser.add_argument(
        "--max-rss-mb",
        type=int,
        default=SAFE_MAX_RSS_MB,
        help="stop if this process exceeds the RSS ceiling",
    )
    parser.add_argument(
        "--min-available-mb",
        type=int,
        default=SAFE_MIN_AVAILABLE_MB,
        help="stop if system available RAM falls below this floor",
    )
    args = parser.parse_args()
    if args.curriculum and not args.rich:
        parser.error("--curriculum requires --rich")
    if args.rich and args.output_dir == Path("results/recurrent_cnn_ppo"):
        args.output_dir = Path(
            "results/rich_curriculum_recurrent_cnn"
            if args.curriculum
            else "results/rich_recurrent_cnn_ppo"
        )
    model_path = train(
        args.timesteps,
        args.seed,
        args.output_dir,
        rich=args.rich,
        curriculum=args.curriculum,
        threads=args.threads,
        max_rss_mb=args.max_rss_mb,
        min_available_mb=args.min_available_mb,
    )
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
