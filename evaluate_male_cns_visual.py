"""Audit the fly-graph visual policy on held-out seeds and levels."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import torch
from env import RICH_EVAL_LEVEL_VARIANTS
from gym_env import RichPixelGymRunnerEnv
from train_male_cns_visual import MaleCNSVisualPolicy
from train_recurrent_cnn import configure_process


def evaluate(
    model_path: Path,
    episodes: int,
    level_variant: str,
    seed_start: int,
    retina_mode: str,
    trainable_edges: bool,
    retina_skip: bool,
) -> dict[str, float]:
    env = RichPixelGymRunnerEnv(
        max_steps=300,
        spawn_interval=8,
        rich_layer=2,
        level_variant=level_variant,
    )
    model = MaleCNSVisualPolicy(
        env.observation_space, retina_mode, trainable_edges, retina_skip
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    results: list[tuple[float, float]] = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            observation, _ = env.reset(seed=seed)
            steps = 0
            while True:
                with torch.no_grad():
                    logits = model(
                        torch.from_numpy(observation)
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
    parser.add_argument("--retina-mode", choices=("fixed", "perspective"), default="perspective")
    parser.add_argument("--fixed-edges", action="store_true")
    parser.add_argument("--no-retina-skip", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")

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
                args.retina_mode,
                not args.fixed_edges,
                not args.no_retina_skip,
            )
            print(
                f"level={level} start={seed_start} episodes={args.episodes} "
                f"mean_steps={result['steps']:.1f} collision_rate={result['collision']:.2f}"
            )


if __name__ == "__main__":
    main()
