"""Find where a visual checkpoint disagrees with the rich teacher."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from random import Random

import torch

from env import RICH_ACTIONS, RunnerEnv
from train_visual_teacher import VisualPolicy, frame, visual_teacher_action


def bucket(observation: dict) -> tuple[str, str]:
    current = [
        item
        for item in observation["obstacles"]
        if item["lane"] == observation["lane"] and item["distance"] <= 7.0
    ]
    if not current:
        return "none", "none"
    nearest = min(current, key=lambda item: item["distance"])
    distance = nearest["distance"]
    distance_bucket = "0-2" if distance < 2 else "2-4" if distance < 4 else "4-7"
    return nearest["kind"], distance_bucket


def analyze(model_path: Path, episodes: int, level: str, seed_start: int) -> dict:
    model = VisualPolicy()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    totals = Counter()
    by_kind: dict[str, Counter] = defaultdict(Counter)
    by_distance: dict[str, Counter] = defaultdict(Counter)
    failures: list[dict] = []
    for seed in range(seed_start, seed_start + episodes):
        env = RunnerEnv(
            seed=seed,
            max_steps=300,
            spawn_interval=8,
            rich_mechanics=True,
            rich_layer=2,
            level_variant=level,
        )
        observation, _ = env.reset(seed=seed)
        rng = Random(seed + 100_000)
        history: deque[dict[str, object]] = deque(maxlen=6)
        while True:
            kind, distance = bucket(observation)
            pixels = frame(env)
            with torch.no_grad():
                logits = model(
                    torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float()
                )[0]
            action = int(logits.argmax())
            teacher = visual_teacher_action(observation, rng)
            confidence = float(torch.softmax(logits, dim=0)[action])
            history.append(
                {
                    "step": observation["step"],
                    "kind": kind,
                    "distance": distance,
                    "predicted_action": RICH_ACTIONS[action],
                    "teacher_action": RICH_ACTIONS[teacher],
                    "confidence": round(confidence, 4),
                }
            )
            totals["steps"] += 1
            by_kind[kind]["steps"] += 1
            by_distance[distance]["steps"] += 1
            if action != teacher:
                totals["disagreements"] += 1
                by_kind[kind]["disagreements"] += 1
                by_distance[distance]["disagreements"] += 1
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info["collision"]:
                    totals["collisions"] += 1
                    by_kind[kind]["collisions"] += 1
                    by_distance[distance]["collisions"] += 1
                    if len(failures) < 40:
                        failures.append(
                            {
                                "seed": seed,
                                "step": observation["step"],
                                "kind": kind,
                                "distance": distance,
                                "lane": observation["lane"],
                                "motion": observation["motion"],
                                "predicted_action": RICH_ACTIONS[action],
                                "teacher_action": RICH_ACTIONS[teacher],
                                "confidence": round(confidence, 4),
                                "decision_history": list(history),
                            }
                        )
                break

    def serialise(values: dict[str, Counter]) -> dict[str, dict[str, int]]:
        return {key: dict(sorted(value.items())) for key, value in sorted(values.items())}

    report = {
        "model": str(model_path),
        "level": level,
        "seed_start": seed_start,
        "episodes": episodes,
        "totals": dict(totals),
        "disagreement_rate": totals["disagreements"] / max(1, totals["steps"]),
        "collision_rate": totals["collisions"] / max(1, episodes),
        "by_kind": serialise(by_kind),
        "by_distance": serialise(by_distance),
        "failures": failures,
    }
    print(
        f"level={level} start={seed_start} episodes={episodes} "
        f"disagreement_rate={report['disagreement_rate']:.3f} "
        f"collision_rate={report['collision_rate']:.3f} "
        f"failures_logged={len(failures)}"
    )
    for kind, values in report["by_kind"].items():
        print(
            f"kind={kind} steps={values.get('steps', 0)} "
            f"disagreements={values.get('disagreements', 0)} "
            f"collisions={values.get('collisions', 0)}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--level", choices=("standard", "dense", "fast"), default="dense")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    report = analyze(args.model, args.episodes, args.level, args.seed_start)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"saved analysis: {args.output}")


if __name__ == "__main__":
    main()
