"""Audit the learned-retina MaleCNS policy on held-out levels."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

import torch
from env import RICH_EVAL_LEVEL_VARIANTS
from gym_env import RichPixelGymRunnerEnv
from train_fly_cns_retina import FlyCNSRetinaPolicy
from train_recurrent_cnn import configure_process
from train_visual_teacher import near_field_view


def evaluate(
    model_path: Path,
    episodes: int,
    level_variant: str,
    seed_start: int,
    disable_graph: bool = False,
) -> dict[str, float]:
    env = RichPixelGymRunnerEnv(
        max_steps=300,
        spawn_interval=8,
        rich_layer=2,
        level_variant=level_variant,
    )
    model = FlyCNSRetinaPolicy(env.observation_space, disable_graph=disable_graph)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    results: list[dict[str, float]] = []
    try:
        for seed in range(seed_start, seed_start + episodes):
            observation, _ = env.reset(seed=seed)
            steps = 0
            total_reward = 0.0
            near_misses = 0
            coin_opportunities = 0
            while True:
                with torch.no_grad():
                    logits = model(
                        torch.from_numpy(near_field_view(observation))
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                    )
                action = int(logits.argmax(1).item())
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                near_misses += int(info.get("near_miss", False))
                coin_opportunities += sum(
                    kind == "coin" for kind in info.get("pickup_opportunities", ())
                )
                steps += 1
                if terminated or truncated:
                    results.append(
                        {
                            "steps": float(steps),
                            "reward": total_reward,
                            "score": float(info["score"]),
                            "coins": float(info.get("coins", 0)),
                            "coin_opportunities": float(coin_opportunities),
                            "near_misses": float(near_misses),
                            "collision": float(info["collision"]),
                        }
                    )
                    break
    finally:
        env.close()
    opportunities = sum(item["coin_opportunities"] for item in results)
    return {
        "steps": mean(item["steps"] for item in results),
        "reward": mean(item["reward"] for item in results),
        "score": mean(item["score"] for item in results),
        "coins": mean(item["coins"] for item in results),
        "coin_capture": sum(item["coins"] for item in results) / max(opportunities, 1.0),
        "near_misses": mean(item["near_misses"] for item in results),
        "collision": mean(item["collision"] for item in results),
    }


def gate(candidate: dict[str, float], baseline: dict[str, float]) -> tuple[bool, str]:
    """Keep safety stable while allowing a small utility improvement."""

    if candidate["collision"] > baseline["collision"] + 0.03:
        return False, "collision regression > 3 percentage points"
    if candidate["steps"] < baseline["steps"] - 8.0:
        return False, "mean survival dropped by more than 8 steps"
    if candidate["coins"] < baseline["coins"] - 0.25:
        return False, "coin collection regressed by more than 0.25 per episode"
    return True, "safety and utility guard passed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, action="append")
    parser.add_argument("--level", choices=RICH_EVAL_LEVEL_VARIANTS, action="append")
    parser.add_argument("--disable-graph", action="store_true")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="compare each band with a baseline checkpoint",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="exit nonzero if any baseline comparison regresses",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")

    configure_process(2)
    seed_starts = args.seed_start or [5000, 6000, 7000, 8000]
    levels = args.level or tuple(RICH_EVAL_LEVEL_VARIANTS)
    all_passed = True
    for level in levels:
        for seed_start in seed_starts:
            result = evaluate(
                args.model, args.episodes, level, seed_start, args.disable_graph
            )
            print(
                f"level={level} start={seed_start} episodes={args.episodes} "
                f"mean_steps={result['steps']:.1f} mean_score={result['score']:.2f} "
                f"mean_coins={result['coins']:.2f} coin_capture={result['coin_capture']:.2f} "
                f"near_misses={result['near_misses']:.2f} "
                f"collision_rate={result['collision']:.2f}"
            )
            if args.baseline is not None:
                baseline = evaluate(
                    args.baseline, args.episodes, level, seed_start, False
                )
                passed, reason = gate(result, baseline)
                all_passed = all_passed and passed
                print(
                    f"gate level={level} start={seed_start} "
                    f"baseline_steps={baseline['steps']:.1f} "
                    f"baseline_coins={baseline['coins']:.2f} "
                    f"status={'PASS' if passed else 'FAIL'} reason={reason}"
                )
    if args.gate and not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
