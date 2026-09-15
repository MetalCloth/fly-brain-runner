"""Train the recurrent/action readout on the fixed neuron-level graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor

from env import RICH_LEVEL_VARIANTS
from gym_env import (
    PartialPixelGymRunnerEnv,
    PixelGymRunnerEnv,
    RichPartialPixelGymRunnerEnv,
    RichPixelGymRunnerEnv,
)
from male_cns_neuron_graph import MaleCNSNeuronGraphExtractor
from train_recurrent_cnn import (
    SAFE_DEFAULT_THREADS,
    SAFE_MAX_RSS_MB,
    SAFE_MIN_AVAILABLE_MB,
    ResourceGuardCallback,
    configure_process,
)


def train(
    total_timesteps: int,
    seed: int,
    output_dir: Path,
    randomized: bool = False,
    partial_view: bool = True,
    trainable_edges: bool = False,
    rich: bool = False,
    level_variant: str = "standard",
    rich_layer: int = 2,
    threads: int = SAFE_DEFAULT_THREADS,
    max_rss_mb: int = SAFE_MAX_RSS_MB,
    min_available_mb: int = SAFE_MIN_AVAILABLE_MB,
) -> Path:
    if rich and level_variant not in RICH_LEVEL_VARIANTS:
        raise ValueError(
            f"training level_variant must be one of {RICH_LEVEL_VARIANTS}"
        )
    if not rich and level_variant != "standard":
        raise ValueError("level_variant requires rich=True")
    if rich_layer not in (1, 2):
        raise ValueError("rich_layer must be 1 or 2")
    if not rich and rich_layer != 2:
        raise ValueError("rich_layer requires rich=True")
    configure_process(threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    env_class = (
        RichPartialPixelGymRunnerEnv
        if rich and partial_view
        else RichPixelGymRunnerEnv
        if rich
        else PartialPixelGymRunnerEnv
        if partial_view
        else PixelGymRunnerEnv
    )
    runner_kwargs = {"seed": seed, "max_steps": 300, "spawn_interval": 8}
    if rich:
        runner_kwargs.update(rich_layer=rich_layer, level_variant=level_variant)
    env = Monitor(
        env_class(**runner_kwargs),
        filename=str(output_dir / "monitor.csv"),
    )
    resource_callback = ResourceGuardCallback(
        max_rss_mb=max_rss_mb,
        min_available_mb=min_available_mb,
    )
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": MaleCNSNeuronGraphExtractor,
            "features_extractor_kwargs": {
                "randomized": randomized,
                "graph_seed": 13,
                "trainable_edges": trainable_edges,
            },
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
        model.learn(
            total_timesteps=total_timesteps,
            callback=resource_callback,
            progress_bar=False,
        )
        model_name = "male_cns_neuron_recurrent_ppo"
        if randomized:
            model_name = "male_cns_neuron_random_recurrent_ppo"
        if trainable_edges:
            model_name = model_name.replace("recurrent_ppo", "plastic_recurrent_ppo")
        if not partial_view:
            model_name = model_name.replace("recurrent_ppo", "full_recurrent_ppo")
        if rich:
            model_name = model_name.replace("recurrent_ppo", "rich_recurrent_ppo")
        model_path = output_dir / model_name
        model.save(model_path)
        return model_path.with_suffix(".zip")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--randomized", action="store_true")
    parser.add_argument("--full-view", action="store_true")
    parser.add_argument("--trainable-edges", action="store_true")
    parser.add_argument(
        "--rich",
        action="store_true",
        help="train on the rich Subway-style mechanics environment",
    )
    parser.add_argument(
        "--level-variant",
        choices=RICH_LEVEL_VARIANTS,
        default="standard",
        help="training profile; mixed samples standard, dense, and fast",
    )
    parser.add_argument("--rich-layer", type=int, choices=(1, 2), default=2)
    parser.add_argument("--threads", type=int, default=SAFE_DEFAULT_THREADS)
    parser.add_argument("--max-rss-mb", type=int, default=SAFE_MAX_RSS_MB)
    parser.add_argument(
        "--min-available-mb", type=int, default=SAFE_MIN_AVAILABLE_MB
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    args = parser.parse_args()
    output_name = "male_cns_neuron"
    if args.randomized:
        output_name += "_random"
    if args.trainable_edges:
        output_name += "_plastic"
    if args.full_view:
        output_name += "_full"
    if args.rich:
        output_name += "_rich"
    output_dir = args.output_dir or Path(f"results/{output_name}_recurrent_ppo")
    if not args.rich and args.level_variant != "standard":
        parser.error("--level-variant requires --rich")
    model_path = train(
        args.timesteps,
        args.seed,
        output_dir,
        randomized=args.randomized,
        partial_view=not args.full_view,
        trainable_edges=args.trainable_edges,
        rich=args.rich,
        level_variant=args.level_variant,
        rich_layer=args.rich_layer,
        threads=args.threads,
        max_rss_mb=args.max_rss_mb,
        min_available_mb=args.min_available_mb,
    )
    print(f"saved model: {model_path}")


if __name__ == "__main__":
    main()
