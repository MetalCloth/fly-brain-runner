"""Run a trained recurrent policy against Android screenshots.

The command is dry-run by default. Pass ``--execute`` only after the game
viewport and swipe geometry have been checked on the connected phone.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO

from android_bridge import AndroidBridge, parse_crop
from male_cns_neuron_graph import MaleCNSNeuronGraphExtractor  # noqa: F401


FRAME_SIZE = (84, 84, 3)


def decode_png(png: bytes, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    """Decode and resize a PNG with the system FFmpeg binary."""

    filters = []
    if crop is not None:
        left, top, width, height = crop
        filters.append(f"crop={width}:{height}:{left}:{top}")
    filters.append("scale=84:84")
    result = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-vf",
            ",".join(filters),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        input=png,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    expected = int(np.prod(FRAME_SIZE))
    if len(result.stdout) != expected:
        raise RuntimeError(f"expected {expected} decoded bytes, got {len(result.stdout)}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(FRAME_SIZE).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--crop", type=parse_crop, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--interval", type=float, default=0.15)
    parser.add_argument("--horizontal-fraction", type=float, default=0.22)
    parser.add_argument("--vertical-fraction", type=float, default=0.18)
    parser.add_argument("--swipe-y-fraction", type=float, default=0.72)
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.interval <= 0:
        raise ValueError("steps must be positive and interval must be positive")

    bridge = AndroidBridge(serial=args.serial)
    screen_width, screen_height = bridge.screen_size()
    left, top, crop_width, crop_height = args.crop
    if left + crop_width > screen_width or top + crop_height > screen_height:
        raise ValueError("crop extends beyond the reported Android display")

    model = RecurrentPPO.load(args.model, device="cpu")
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)
    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"{mode}: {args.steps} steps on {screen_width}x{screen_height}, crop={args.crop}")
    for step in range(args.steps):
        frame = decode_png(bridge.screenshot_png(), args.crop)
        action, lstm_state = model.predict(
            frame[None, ...],
            state=lstm_state,
            episode_start=episode_start,
            deterministic=True,
        )
        action_id = int(np.asarray(action).reshape(-1)[0])
        print(f"step={step:04d} action={action_id}")
        if args.execute:
            bridge.send_action(
                action_id,
                width=screen_width,
                height=screen_height,
                horizontal_fraction=args.horizontal_fraction,
                vertical_fraction=args.vertical_fraction,
                start_y_fraction=args.swipe_y_fraction,
                duration_ms=args.duration_ms,
            )
        episode_start[:] = False
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
