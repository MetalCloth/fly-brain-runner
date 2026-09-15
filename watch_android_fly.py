"""Watch Android screenshots with the fly checkpoint without sending input."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from android_bridge import AndroidBridge, parse_crop
from env import RICH_ACTIONS
from run_android_policy import decode_png
from train_fly_cns_retina import FlyCNSRetinaPolicy
from train_visual_teacher import near_field_view


FRAME_SHAPE = (84, 84, 3)
DEFAULT_MODEL = Path("results/fly_cns_retina_v1/fly_cns_retina_best.pt")


def load_model(model_path: Path) -> FlyCNSRetinaPolicy:
    """Load the custom fly checkpoint used by the simulator evaluator."""

    observation_space = spaces.Box(
        low=0,
        high=255,
        shape=FRAME_SHAPE,
        dtype=np.uint8,
    )
    model = FlyCNSRetinaPolicy(observation_space)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def predict(
    model: FlyCNSRetinaPolicy, frame: np.ndarray
) -> tuple[int, float, list[dict[str, float | str]]]:
    """Return the top action and probabilities for one RGB frame."""

    if frame.shape != FRAME_SHAPE:
        raise ValueError(f"expected frame shape {FRAME_SHAPE}, got {frame.shape}")
    six_channel_frame = near_field_view(frame)
    tensor = (
        torch.from_numpy(six_channel_frame)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    ranking = torch.argsort(probabilities, descending=True)[:3]
    top_three = [
        {
            "action": RICH_ACTIONS[int(index)],
            "probability": float(probabilities[index]),
        }
        for index in ranking
    ]
    action = int(ranking[0])
    return action, float(probabilities[action]), top_three


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--serial")
    parser.add_argument("--crop", type=parse_crop, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.15)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("results/android_observation/fly_watch_only.jsonl"),
    )
    args = parser.parse_args()
    if args.steps < 1 or args.interval <= 0:
        raise ValueError("steps must be positive and interval must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    bridge = AndroidBridge(serial=args.serial)
    width, height = bridge.screen_size()
    left, top, crop_width, crop_height = args.crop
    if left < 0 or top < 0 or left + crop_width > width or top + crop_height > height:
        raise ValueError("crop extends beyond the reported Android display")
    model = load_model(args.model)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"WATCH ONLY: {args.steps} screenshot(s) on {width}x{height}, "
        f"crop={args.crop}; no phone input will be sent"
    )
    with args.log.open("a", encoding="utf-8") as log:
        for step in range(args.steps):
            captured_at = time.time()
            frame = decode_png(bridge.screenshot_png(), args.crop)
            action, confidence, top_three = predict(model, frame)
            record = {
                "mode": "watch-only",
                "executed": False,
                "step": step,
                "action": action,
                "action_name": RICH_ACTIONS[action],
                "confidence": confidence,
                "top_three": top_three,
                "screen": {"width": width, "height": height},
                "crop": list(args.crop),
                "frame_shape": list(frame.shape),
                "captured_at": captured_at,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(
                f"step={step:04d} action={RICH_ACTIONS[action]} "
                f"confidence={confidence:.3f}"
            )
            if step + 1 < args.steps:
                time.sleep(args.interval)
    print(f"saved watch-only log: {args.log}")


if __name__ == "__main__":
    main()
