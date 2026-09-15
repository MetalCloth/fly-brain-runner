"""Watch or explicitly execute a trained browser-game policy."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from browser_model import BrowserPolicy
from browser_pipeline import (
    ACTIONS,
    ScreenCapturer,
    capture_rgb,
    parse_region,
    resize_nearest,
)


ACTION_KEYS = {1: "left", 2: "right", 3: "up", 4: "down"}


def load_policy(path: Path) -> tuple[BrowserPolicy, int]:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        history = int(payload.get("history", 4))
        state_dict = payload["state_dict"]
    else:
        history = 4
        state_dict = payload
    model = BrowserPolicy(history=history)
    model.load_state_dict(state_dict)
    model.eval()
    return model, history


def load_keyboard():
    try:
        from pynput import keyboard
    except ImportError as error:
        raise SystemExit(
            "Browser control needs pynput. Run: "
            "./.venv/bin/python -m pip install -r requirements.txt"
        ) from error
    return keyboard


class StopSignal:
    def __init__(self, keyboard_module) -> None:
        self._keyboard = keyboard_module
        self._lock = threading.Lock()
        self._stop = False

    def on_press(self, key):
        if key == self._keyboard.Key.esc:
            with self._lock:
                self._stop = True
            return False

    @property
    def requested(self) -> bool:
        with self._lock:
            return self._stop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--region", type=parse_region, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--cooldown-ms", type=float, default=180.0)
    parser.add_argument("--start-delay", type=float, default=3.0)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if (
        args.steps < 1
        or args.fps <= 0
        or not 0 <= args.confidence <= 1
        or args.cooldown_ms < 0
        or args.start_delay < 0
    ):
        parser.error("invalid steps, fps, confidence, cooldown-ms, or start-delay")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    model, history = load_policy(args.checkpoint)
    keyboard = load_keyboard()
    controller = keyboard.Controller() if args.execute else None
    stop_signal = StopSignal(keyboard)
    listener = keyboard.Listener(on_press=stop_signal.on_press)
    listener.start()
    log_path = args.log
    if log_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = Path("results/browser_runs") / f"run_{stamp}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "EXECUTE" if args.execute else "WATCH ONLY"
    print(f"{mode}: {args.steps} steps, history={history}, region={args.region}")
    if args.execute:
        print("Click the game and make sure a run is active. ESC stops immediately.")
    else:
        print("No keyboard input will be sent. ESC stops the watcher.")
    time.sleep(args.start_delay)

    frames: deque[np.ndarray] = deque(maxlen=history)
    last_sent = 0.0
    period = 1.0 / args.fps
    next_tick = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log, ScreenCapturer() as capturer:
        for step in range(args.steps):
            if stop_signal.requested:
                break
            tick = time.monotonic()
            frame = resize_nearest(capture_rgb(capturer, args.region))
            frames.append(frame)
            while len(frames) < history:
                frames.appendleft(frame.copy())
            stacked = np.concatenate(tuple(frames), axis=2)
            input_tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)
            with torch.inference_mode():
                probabilities = torch.softmax(model(input_tensor.float()), dim=1)[0]
            proposed = int(probabilities.argmax().item())
            confidence = float(probabilities[proposed].item())
            action = proposed if confidence >= args.confidence else 0
            sent = False
            now = time.monotonic()
            if (
                args.execute
                and action in ACTION_KEYS
                and (now - last_sent) * 1000.0 >= args.cooldown_ms
            ):
                controller.press(getattr(keyboard.Key, ACTION_KEYS[action]))
                controller.release(getattr(keyboard.Key, ACTION_KEYS[action]))
                last_sent = now
                sent = True
            log.write(
                json.dumps(
                    {
                        "step": step,
                        "action": action,
                        "action_name": ACTIONS[action],
                        "proposed_action": proposed,
                        "confidence": confidence,
                        "sent": sent,
                        "captured_at": time.time(),
                    }
                )
                + "\n"
            )
            log.flush()
            print(
                f"step={step:04d} action={ACTIONS[action]} "
                f"confidence={confidence:.3f}{' sent' if sent else ''}"
            )
            next_tick += period
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick = time.monotonic()

    listener.stop()
    listener.join(timeout=2.0)
    print(f"saved policy log: {log_path}")


if __name__ == "__main__":
    main()
