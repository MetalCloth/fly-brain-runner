"""Probe rich RL learning with the readable state before relying on pixels."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env import RICH_LEVEL_VARIANTS
from gym_env import RichGymRunnerEnv
from train_recurrent_cnn import ResourceGuardCallback, configure_process


def run(
    timesteps: int,
    episodes: int,
    output_dir: Path,
    seed: int,
    rich_layer: int,
    policy_kind: str,
    n_envs: int,
    train_level: str,
    eval_level: str,
    curriculum: bool,
) -> Path:
    configure_process(2)
    if n_envs < 1:
        raise ValueError("n_envs must be positive")
    if curriculum and rich_layer != 2:
        raise ValueError("curriculum requires rich_layer=2")
    output_dir.mkdir(parents=True, exist_ok=True)

    def make_env(index: int, layer: int):
        return Monitor(
            RichGymRunnerEnv(
                seed=seed + index * 1009,
                max_steps=300,
                spawn_interval=8,
                rich_layer=layer,
                level_variant=train_level,
            ),
            filename=str(output_dir / f"monitor_layer{layer}_{index}.csv"),
        )

    def make_train_env(layer: int):
        if n_envs == 1:
            return make_env(0, layer)
        return DummyVecEnv(
            [lambda index=index: make_env(index, layer) for index in range(n_envs)]
        )

    train_env = make_train_env(1 if curriculum else rich_layer)
    model_class = RecurrentPPO if policy_kind == "recurrent" else PPO
    model = model_class(
        "MlpLstmPolicy" if policy_kind == "recurrent" else "MlpPolicy",
        train_env,
        policy_kwargs=(
            {"lstm_hidden_size": 32, "n_lstm_layers": 1}
            if policy_kind == "recurrent"
            else {"net_arch": [64, 64]}
        ),
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
        device="cpu",
    )
    try:
        checkpoint_callback = CheckpointCallback(
            save_freq=8192,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="rich_numeric_probe",
        )
        callbacks = [ResourceGuardCallback(), checkpoint_callback]
        if curriculum:
            first_phase = max(1, timesteps // 2)
            model.learn(
                total_timesteps=first_phase,
                callback=callbacks,
                progress_bar=False,
            )
            train_env.close()
            train_env = make_train_env(rich_layer)
            model.set_env(train_env)
            remaining = timesteps - first_phase
            if remaining:
                model.learn(
                    total_timesteps=remaining,
                    callback=callbacks,
                    progress_bar=False,
                    reset_num_timesteps=False,
                )
        else:
            model.learn(
                total_timesteps=timesteps,
                callback=callbacks,
                progress_bar=False,
            )
        model_path = output_dir / "rich_numeric_probe"
        model.save(model_path)
    finally:
        train_env.close()

    def evaluate(seed_start: int) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        eval_env = RichGymRunnerEnv(
            max_steps=300,
            spawn_interval=8,
            rich_layer=rich_layer,
            level_variant=eval_level,
        )
        try:
            for episode_seed in range(seed_start, seed_start + episodes):
                observation, _ = eval_env.reset(seed=episode_seed)
                state = None
                episode_start = np.ones((1,), dtype=bool)
                total_reward = 0.0
                steps = 0
                while True:
                    if policy_kind == "recurrent":
                        action, state = model.predict(
                            observation,
                            state=state,
                            episode_start=episode_start,
                            deterministic=True,
                        )
                    else:
                        action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = eval_env.step(
                        int(np.asarray(action).reshape(-1)[0])
                    )
                    total_reward += float(reward)
                    steps += 1
                    episode_start[:] = False
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
            eval_env.close()
        return results

    for label, seed_start in (("seed_band_a", 0), ("seed_band_b", 1000)):
        results = evaluate(seed_start)
        print(
            f"{label} level={eval_level} episodes={len(results)} "
            f"mean_steps={mean(item['steps'] for item in results):.1f} "
            f"median_steps={median(item['steps'] for item in results):.1f} "
            f"mean_score={mean(item['score'] for item in results):.2f} "
            f"collision_rate={mean(item['collision'] for item in results):.2f}"
        )
    return model_path.with_suffix(".zip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=16_384)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rich-layer", type=int, choices=(1, 2), default=2)
    parser.add_argument("--policy", choices=("mlp", "recurrent"), default="mlp")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--train-level", choices=RICH_LEVEL_VARIANTS, default="mixed")
    parser.add_argument("--eval-level", choices=RICH_LEVEL_VARIANTS, default="standard")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/rich_numeric_probe")
    )
    args = parser.parse_args()
    print(
        run(
            args.timesteps,
            args.episodes,
            args.output_dir,
            args.seed,
            args.rich_layer,
            args.policy,
            args.n_envs,
            args.train_level,
            args.eval_level,
            args.curriculum,
        )
    )


if __name__ == "__main__":
    main()
