"""Capture a labelled Android frame sequence without sending gestures by default."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from android_bridge import ACTIONS, AndroidBridge


def parse_actions(value: str) -> tuple[int, ...]:
    """Parse a comma-separated local action script."""

    try:
        actions = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("actions must be comma-separated integers") from error
    if not actions or any(action not in ACTIONS for action in actions):
        raise ValueError(f"actions must contain only {sorted(ACTIONS)}")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/android_sequence"))
    parser.add_argument("--serial")
    parser.add_argument("--actions", type=parse_actions, default=(0,))
    parser.add_argument("--interval", type=float, default=0.15)
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")

    bridge = AndroidBridge(serial=args.serial)
    width, height = bridge.screen_size()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.jsonl"
    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"{mode}: capturing {len(args.actions)} frames from {width}x{height}")
    with metadata_path.open("w", encoding="utf-8") as metadata:
        for index, action in enumerate(args.actions):
            frame_path = args.output_dir / f"frame_{index:05d}.png"
            frame_path.write_bytes(bridge.screenshot_png())
            metadata.write(
                json.dumps(
                    {
                        "frame": frame_path.name,
                        "action": action,
                        "action_name": ACTIONS[action],
                        "screen_width": width,
                        "screen_height": height,
                        "captured_at": time.time(),
                    }
                )
                + "\n"
            )
            if args.execute:
                bridge.send_action(
                    action,
                    width=width,
                    height=height,
                    duration_ms=args.duration_ms,
                )
            time.sleep(args.interval)
    print(f"saved {len(args.actions)} frames and {metadata_path}")


if __name__ == "__main__":
    main()
