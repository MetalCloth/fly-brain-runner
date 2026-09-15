"""Capture one Android frame for manual viewport calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

from android_bridge import AndroidBridge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/android_frame.png"))
    parser.add_argument("--serial")
    args = parser.parse_args()

    bridge = AndroidBridge(serial=args.serial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bridge.screenshot_png())
    width, height = bridge.screen_size()
    print(f"saved {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
