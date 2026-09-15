"""Fine-tune the visual DAgger policy with ordinary PPO in the simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
from random import Random

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import RICH_LEVEL_VARIANTS
from gym_env import RichPixelGymRunnerEnv
from train_recurrent_cnn import ResourceGuardCallback, configure_process
from train_visual_teacher import VisualPolicy, visual_teacher_action


class VisualBootstrapExtractor(BaseFeaturesExtractor):
    """Reuse the DAgger encoder while PPO learns value/action improvements."""

    def __init__(self, observation_space, features_dim: int = 128) -> None:
        if features_dim != 128:
            raise ValueError("the visual bootstrap has a fixed 128-feature head")
        super().__init__(observation_space, features_dim)
        encoder = VisualPolicy()
        encoder.head = nn.Sequential(*list(encoder.head.children())[:2])
        self.encoder = encoder

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)


class TeacherRewardWrapper(gym.Wrapper):
    """Add a small agreement signal so PPO cannot ignore the verified teacher."""

    def __init__(self, env: gym.Env, weight: float, seed: int) -> None:
        super().__init__(env)
        if weight < 0:
            raise ValueError("teacher reward weight must be nonnegative")
        self.weight = weight
        self.rng = Random(seed)

    def step(self, action):
        core = getattr(self.env, "core", None)
        teacher_action = (
            visual_teacher_action(core._observation(), self.rng)
            if core is not None
            else int(action)
        )
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self.weight:
            reward += self.weight if int(action) == teacher_action else -self.weight
        info = dict(info)
        info["teacher_action"] = teacher_action
        return observation, reward, terminated, truncated, info


def load_bootstrap(model: PPO, checkpoint: Path) -> None:
    teacher = VisualPolicy()
    teacher.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    extractor = model.policy.features_extractor
    extractor.encoder.features.load_state_dict(teacher.features.state_dict())
    extractor.encoder.head[0].load_state_dict(teacher.head[0].state_dict())
    if tuple(model.policy.action_net.weight.shape) != tuple(teacher.head[2].weight.shape):
        raise RuntimeError("PPO action head does not match the visual bootstrap")
    with torch.no_grad():
        model.policy.action_net.weight.copy_(teacher.head[2].weight)
        model.policy.action_net.bias.copy_(teacher.head[2].bias)
    for parameter in model.policy.features_extractor.parameters():
        parameter.requires_grad = False


def train(
    teacher_checkpoint: Path,
    total_timesteps: int,
    seed: int,
    output_dir: Path,
    n_envs: int,
    train_level: str,
    teacher_reward_weight: float,
) -> Path:
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(teacher_checkpoint)
    if total_timesteps < 1 or n_envs < 1:
        raise ValueError("timesteps and n_envs must be positive")
    configure_process(2)
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_env(index: int):
        return Monitor(
            TeacherRewardWrapper(
                RichPixelGymRunnerEnv(
                    seed=seed + index * 1009,
                    max_steps=300,
                    spawn_interval=8,
                    rich_layer=2,
                    level_variant=train_level,
                ),
                weight=teacher_reward_weight,
                seed=seed + index * 1009 + 50_000,
            ),
            filename=str(output_dir / f"monitor_{index}.csv"),
        )

    env = DummyVecEnv([lambda index=index: make_env(index) for index in range(n_envs)])
    # ponytail: single-frame PPO keeps this transfer experiment small; recurrent memory is deferred until it earns a held-out gain.
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": VisualBootstrapExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": {"pi": [], "vf": []},
            "normalize_images": False,
        },
        n_steps=512,
        batch_size=64,
        learning_rate=3e-6,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.0,
        n_epochs=1,
        target_kl=0.005,
        verbose=1,
        seed=seed,
        device="cpu",
    )
    load_bootstrap(model, teacher_checkpoint)
    try:
        callbacks = [
            ResourceGuardCallback(),
            CheckpointCallback(
                save_freq=max(1, 4096 // n_envs),
                save_path=str(output_dir / "checkpoints"),
                name_prefix="visual_ppo",
            ),
        ]
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
        model_path = output_dir / "visual_ppo"
        model.save(model_path)
        return model_path.with_suffix(".zip")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("results/visual_dagger_dense_v2/visual_teacher_best.pt"),
    )
    parser.add_argument("--timesteps", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--train-level", choices=RICH_LEVEL_VARIANTS, default="mixed")
    parser.add_argument("--teacher-reward-weight", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path("results/visual_ppo"))
    args = parser.parse_args()
    if args.teacher_reward_weight < 0:
        parser.error("--teacher-reward-weight must be nonnegative")
    print(
        train(
            args.teacher_checkpoint,
            args.timesteps,
            args.seed,
            args.output_dir,
            args.n_envs,
            args.train_level,
            args.teacher_reward_weight,
        )
    )


if __name__ == "__main__":
    main()
