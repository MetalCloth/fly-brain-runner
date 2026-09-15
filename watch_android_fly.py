"""Run the fly checkpoint on Android screenshots, watching by default."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from android_bridge import ACTIONS, AndroidBridge, parse_crop
from env import RICH_ACTIONS
from run_android_policy import decode_png
from run_android_session import ScreenState, classify_screen, screen_signature
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
        "--menu-reference",
        type=Path,
        help="menu screenshot used to prevent input outside gameplay",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send supported model swipes after the menu gate passes",
    )
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
    if args.execute and args.menu_reference is None:
        raise ValueError("--execute requires --menu-reference")
    if args.menu_reference is not None and not args.menu_reference.is_file():
        raise FileNotFoundError(args.menu_reference)

    bridge = AndroidBridge(serial=args.serial)
    width, height = bridge.screen_size()
    left, top, crop_width, crop_height = args.crop
    if left < 0 or top < 0 or left + crop_width > width or top + crop_height > height:
        raise ValueError("crop extends beyond the reported Android display")
    model = load_model(args.model)
    menu_signature = None
    if args.menu_reference is not None:
        menu_signature = screen_signature(
            decode_png(args.menu_reference.read_bytes(), args.crop)
        )
    args.log.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"{'EXECUTE' if args.execute else 'WATCH ONLY'}: {args.steps} screenshot(s) "
        f"on {width}x{height}, crop={args.crop}"
    )
    with args.log.open("a", encoding="utf-8") as log:
        for step in range(args.steps):
            captured_at = time.time()
            frame = decode_png(bridge.screenshot_png(), args.crop)
            state = classify_screen(frame, menu_signature)
            if args.execute and state is not ScreenState.ACTIVE:
                record = {
                    "mode": "execute",
                    "executed": False,
                    "input_sent": False,
                    "state": state.value,
                    "step": step,
                    "screen": {"width": width, "height": height},
                    "crop": list(args.crop),
                    "frame_shape": list(frame.shape),
                    "captured_at": captured_at,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(f"step={step:04d} gated state={state.value}; no input sent")
                if step + 1 < args.steps:
                    time.sleep(args.interval)
                continue

            action, confidence, top_three = predict(model, frame)
            input_sent = False
            action_note = None
            if args.execute:
                if action in ACTIONS:
                    bridge.send_action(
                        action,
                        width=width,
                        height=height,
                    )
                    input_sent = action != 0
                else:
                    action_note = "unsupported_android_action"
            record = {
                "mode": "execute" if args.execute else "watch-only",
                "executed": args.execute,
                "input_sent": input_sent,
                "state": state.value,
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
            if action_note is not None:
                record["note"] = action_note
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(
                f"step={step:04d} action={RICH_ACTIONS[action]} "
                f"confidence={confidence:.3f} "
                f"{'sent' if input_sent else 'no-input'}"
            )
            if step + 1 < args.steps:
                time.sleep(args.interval)
    print(f"saved watch-only log: {args.log}")


if __name__ == "__main__":
    main()
