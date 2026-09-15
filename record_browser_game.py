"""Record labeled human play from the Poki game through the browser itself."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from browser_cdp import (
    DEFAULT_URL,
    KEY_ACTIONS,
    BrowserCdpError,
    BrowserPage,
    parse_clip,
)
from browser_pipeline import ACTIONS, FRAME_SIZE, decode_png_rgb, resize_frame


def session_directory(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / stamp
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "frames").mkdir()
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--port", type=int, default=9222)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("/tmp/fly-brain-runner-browser-profile"),
    )
    parser.add_argument("--clip", type=parse_clip, help="page clip x,y,width,height; auto-detected by default")
    parser.add_argument(
        "--region",
        help="deprecated screen coordinate option; ignored, use --clip or auto-detection",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/browser_dataset_poki")
    )
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--action-window-ms", type=float, default=180.0)
    args = parser.parse_args()
    if args.duration <= 0 or args.fps <= 0 or args.action_window_ms <= 0:
        parser.error("duration, fps, and action-window-ms must be positive")
    if args.region:
        print("warning: --region is ignored by the browser recorder; use --clip if needed")

    try:
        page = BrowserPage.connect_or_launch(
            args.url,
            port=args.port,
            profile_dir=args.profile_dir,
        )
    except BrowserCdpError as error:
        raise SystemExit(str(error)) from error

    stop = False
    last_action = 0
    last_key = ""
    last_event = 0.0
    recording = False

    def handle_event(method: str, params: dict[str, object]) -> None:
        nonlocal stop, last_action, last_key, last_event
        if method != "Runtime.bindingCalled" or params.get("name") != "flyAction":
            return
        try:
            event = json.loads(str(params.get("payload", "{}")))
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return
        key = str(event.get("key", ""))
        if key == "Escape":
            stop = True
            return
        if not recording or event.get("type") != "down" or event.get("repeat"):
            return
        action = KEY_ACTIONS.get(key)
        if action is not None:
            last_action = action
            last_key = key
            last_event = time.monotonic()

    page.set_event_handler(handle_event)
    try:
        page.prepare(args.url)
        clip = args.clip or page.find_game_clip()
        print(f"controlled browser ready; game_clip={clip}")
        print("Click the game, press Play, and play normally.")
        print("Recording starts in 3 seconds; press Escape to stop and save.")
        time.sleep(3.0)
        recording = True
        last_action = 0
        last_key = ""
        last_event = 0.0

        session = session_directory(args.output_dir)
        metadata_path = session / "metadata.jsonl"
        summary_path = session / "session.json"
        counts: Counter[str] = Counter()
        period = 1.0 / args.fps
        next_tick = time.monotonic()
        started_monotonic = next_tick
        deadline = started_monotonic + args.duration
        started_at = time.time()
        frame_count = 0
        stopped_by = "duration"
        with metadata_path.open("w", encoding="utf-8") as metadata:
            try:
                while time.monotonic() < deadline and not stop:
                    tick = time.monotonic()
                    png = page.capture_png(clip)
                    # Validate/decode once here so bad browser output never enters the dataset.
                    frame = resize_frame(decode_png_rgb(png))
                    age_ms = (tick - last_event) * 1000.0
                    action = (
                        last_action
                        if last_action and age_ms <= args.action_window_ms
                        else 0
                    )
                    frame_name = f"frames/frame_{frame_count:06d}.npy"
                    np.save(session / frame_name, frame, allow_pickle=False)
                    metadata.write(
                        json.dumps(
                            {
                                "frame": frame_name,
                                "action": action,
                                "action_name": ACTIONS[action],
                                "key": last_key if action else "",
                                "action_age_ms": round(age_ms, 2),
                                "captured_at": time.time(),
                                "elapsed_ms": round((tick - started_monotonic) * 1000.0, 2),
                                "source": "poki_cdp",
                            }
                        )
                        + "\n"
                    )
                    metadata.flush()
                    counts[ACTIONS[action]] += 1
                    frame_count += 1
                    next_tick += period
                    remaining = next_tick - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    else:
                        next_tick = time.monotonic()
                if stop:
                    stopped_by = "escape"
            except KeyboardInterrupt:
                stopped_by = "ctrl_c"
        summary_path.write_text(
            json.dumps(
                {
                    "started_at": started_at,
                    "finished_at": time.time(),
                    "frames": frame_count,
                    "fps_target": args.fps,
                    "frame_size": list(FRAME_SIZE),
                    "clip": clip,
                    "action_window_ms": args.action_window_ms,
                    "actions": dict(counts),
                    "stopped_by": stopped_by,
                    "url": args.url,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"saved {frame_count} frames to {session}")
        print(f"actions={dict(counts)} stopped_by={stopped_by}")
    finally:
        page.close()


if __name__ == "__main__":
    main()
