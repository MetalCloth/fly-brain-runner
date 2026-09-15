"""Record labeled human play from a browser game using screen capture and keys."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from browser_pipeline import (
    ACTIONS,
    ScreenCapturer,
    capture_rgb,
    parse_region,
    resize_nearest,
)


KEY_ACTIONS = {
    "left": 1,
    "right": 2,
    "up": 3,
    "down": 4,
    "a": 1,
    "d": 2,
    "w": 3,
    "s": 4,
    "space": 3,
}


def load_capture_dependencies():
    try:
        from pynput import keyboard
    except ImportError as error:
        raise SystemExit(
            "Browser recording needs pynput. Run: "
            "./.venv/bin/python -m pip install -r requirements.txt"
        ) from error
    return keyboard


class KeyTracker:
    """Observe movement keys without sending any input to the game."""

    def __init__(self, keyboard_module) -> None:
        self._keyboard = keyboard_module
        self._lock = threading.Lock()
        self._held: set[str] = set()
        self._last_action = 0
        self._last_key = ""
        self._last_event = 0.0
        self._stop = False

    @staticmethod
    def key_name(key) -> str:
        name = getattr(key, "name", None)
        if name:
            return str(name).lower()
        char = getattr(key, "char", None)
        return str(char).lower() if char else ""

    def on_press(self, key):
        name = self.key_name(key)
        if key == self._keyboard.Key.esc or name == "esc":
            with self._lock:
                self._stop = True
            return False
        action = KEY_ACTIONS.get(name)
        if action is None:
            return
        with self._lock:
            if name in self._held:
                return
            self._held.add(name)
            self._last_action = action
            self._last_key = name
            self._last_event = time.monotonic()

    def on_release(self, key):
        name = self.key_name(key)
        with self._lock:
            self._held.discard(name)

    def snapshot(self, now: float, action_window_ms: float) -> tuple[int, str, float]:
        with self._lock:
            age_ms = (now - self._last_event) * 1000.0
            if self._last_action and age_ms <= action_window_ms:
                return self._last_action, self._last_key, max(0.0, age_ms)
            return 0, "", age_ms

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop


def session_directory(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / stamp
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "frames").mkdir()
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        type=parse_region,
        required=True,
        help="game rectangle on screen: left,top,width,height",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/browser_dataset"))
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--action-window-ms", type=float, default=180.0)
    args = parser.parse_args()
    if args.duration <= 0 or args.fps <= 0 or args.action_window_ms <= 0:
        parser.error("duration, fps, and action-window-ms must be positive")

    keyboard = load_capture_dependencies()
    session = session_directory(args.output_dir)
    metadata_path = session / "metadata.jsonl"
    summary_path = session / "session.json"
    region = args.region
    print("Browser recorder: no game input will be sent by this script.")
    print("Click the game, start a run, and play normally.")
    print("Recording starts in 3 seconds; press ESC to stop and save.")
    time.sleep(3.0)

    tracker = KeyTracker(keyboard)
    listener = keyboard.Listener(on_press=tracker.on_press, on_release=tracker.on_release)
    listener.start()
    counts: Counter[str] = Counter()
    started_at = time.time()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + args.duration
    period = 1.0 / args.fps
    next_tick = started_monotonic
    frame_count = 0
    stopped_by = "duration"

    with ScreenCapturer() as capturer, metadata_path.open(
        "w", encoding="utf-8"
    ) as metadata:
        try:
            while time.monotonic() < deadline and not tracker.stop_requested:
                tick = time.monotonic()
                small = resize_nearest(capture_rgb(capturer, region))
                action, key_name, action_age_ms = tracker.snapshot(
                    tick, args.action_window_ms
                )
                frame_name = f"frames/frame_{frame_count:06d}.npy"
                frame_path = session / frame_name
                np.save(frame_path, small, allow_pickle=False)
                metadata.write(
                    json.dumps(
                        {
                            "frame": frame_name,
                            "action": action,
                            "action_name": ACTIONS[action],
                            "key": key_name,
                            "action_age_ms": round(action_age_ms, 2),
                            "captured_at": time.time(),
                            "elapsed_ms": round((tick - started_monotonic) * 1000.0, 2),
                        }
                    )
                    + "\n"
                )
                if frame_count % 15 == 0:
                    metadata.flush()
                counts[ACTIONS[action]] += 1
                frame_count += 1
                next_tick += period
                remaining = next_tick - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    next_tick = time.monotonic()
            if tracker.stop_requested:
                stopped_by = "escape"
        except KeyboardInterrupt:
            stopped_by = "ctrl_c"
        finally:
            listener.stop()
            listener.join(timeout=2.0)

    summary_path.write_text(
        json.dumps(
            {
                "started_at": started_at,
                "finished_at": time.time(),
                "frames": frame_count,
                "fps_target": args.fps,
                "region": region,
                "action_window_ms": args.action_window_ms,
                "actions": dict(counts),
                "stopped_by": stopped_by,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved {frame_count} frames to {session}")
    print(f"actions={dict(counts)} stopped_by={stopped_by}")


if __name__ == "__main__":
    main()
