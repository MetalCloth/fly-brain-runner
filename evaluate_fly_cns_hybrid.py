"""Audit the hybrid visual policy on held-out seeds and levels."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import torch
from env import RICH_EVAL_LEVEL_VARIANTS
from gym_env import RichPixelGymRunnerEnv
from train_fly_cns_hybrid import FlyCNSHybridPolicy
from train_recurrent_cnn import configure_process
from train_visual_teacher import near_field_view


def evaluate(
    model_path: Path,
    episodes: int,
    level_variant: str,
    seed_start: int,
    disable_fly_residual: bool = False,
    residual_scale: float = 0.25,
) -> dict[str, float]:
    env = RichPixelGymRunnerEnv(
        max_steps=300,
        spawn_interval=8,
        rich_layer=2,
        level_variant=level_variant,
    )
    model = FlyCNSHybridPolicy(env.observation_space, residual_scale=residual_scale)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    if disable_fly_residual:
        model.residual.weight.data.zero_()
        model.residual.bias.data.zero_()
    model.eval()
    results: list[tuple[float, float]] = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            observation, _ = env.reset(seed=seed)
            steps = 0
            while True:
                with torch.no_grad():
                    logits = model(
                        torch.from_numpy(near_field_view(observation))
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                    )
                action = int(logits.argmax(1).item())
                observation, _, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated:
                    results.append((float(steps), float(info["collision"])))
                    break
    finally:
        env.close()
    return {
        "steps": mean(item[0] for item in results),
        "collision": mean(item[1] for item in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, action="append")
    parser.add_argument("--level", choices=RICH_EVAL_LEVEL_VARIANTS, action="append")
    parser.add_argument(
        "--disable-fly-residual",
        action="store_true",
        help="ablate the MaleCNS correction branch for a paired comparison",
    )
    parser.add_argument("--residual-scale", type=float, default=0.25)
    args = parser.parse_args()
    if args.episodes < 1 or args.residual_scale <= 0:
        parser.error("episodes and residual-scale must be positive")

    configure_process(2)
    seed_starts = args.seed_start or [5000, 6000, 7000]
    levels = args.level or tuple(RICH_EVAL_LEVEL_VARIANTS)
    for level in levels:
        for seed_start in seed_starts:
            result = evaluate(
                args.model,
                args.episodes,
                level,
                seed_start,
                args.disable_fly_residual,
                args.residual_scale,
            )
            print(
                f"level={level} start={seed_start} episodes={args.episodes} "
                f"mean_steps={result['steps']:.1f} collision_rate={result['collision']:.2f}"
            )


if __name__ == "__main__":
    main()
