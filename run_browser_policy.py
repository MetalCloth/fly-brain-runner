"""Watch or explicitly execute a trained policy in the Poki browser game."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from browser_cdp import (
    ACTION_KEYS,
    DEFAULT_URL,
    BrowserCdpError,
    BrowserPage,
    parse_clip,
)
from browser_model import BrowserPolicy
from browser_pipeline import ACTIONS, decode_image_rgb, is_gameplay_frame, resize_frame


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
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
    if args.region:
        print("warning: --region is ignored by the browser controller; use --clip if needed")

    try:
        page = BrowserPage.connect_or_launch(
            args.url,
            port=args.port,
            profile_dir=args.profile_dir,
        )
    except BrowserCdpError as error:
        raise SystemExit(str(error)) from error

    stop = False

    def handle_event(method: str, params: dict[str, object]) -> None:
        nonlocal stop
        if method != "Runtime.bindingCalled" or params.get("name") != "flyAction":
            return
        try:
            event = json.loads(str(params.get("payload", "{}")))
        except json.JSONDecodeError:
            return
        if isinstance(event, dict) and event.get("key") == "Escape":
            stop = True

    page.set_event_handler(handle_event)
    try:
        page.prepare(args.url)
        clip = args.clip or page.find_game_clip()
        model, history = load_policy(args.checkpoint)
        log_path = args.log
        if log_path is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            log_path = Path("results/browser_runs") / f"run_{stamp}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "EXECUTE" if args.execute else "WATCH ONLY"
        print(f"{mode}: {args.steps} steps, history={history}, game_clip={clip}")
        if args.execute:
            print("After the delay, click the game and make sure a run is active. Escape stops.")
        else:
            print("No game input will be sent. Escape stops the watcher.")
        time.sleep(args.start_delay)

        frames: deque[np.ndarray] = deque(maxlen=history)
        last_sent = 0.0
        period = 1.0 / args.fps
        next_tick = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            for step in range(args.steps):
                if stop:
                    break
                tick = time.monotonic()
                frame = resize_frame(decode_image_rgb(page.capture_image(clip)))
                if not is_gameplay_frame(frame):
                    frames.clear()
                    log.write(
                        json.dumps(
                            {
                                "step": step,
                                "action": 0,
                                "action_name": ACTIONS[0],
                                "sent": False,
                                "state": "inactive",
                                "captured_at": time.time(),
                            }
                        )
                        + "\n"
                    )
                    log.flush()
                    print(f"step={step:04d} gated state=inactive; no input sent")
                    next_tick += period
                    remaining = next_tick - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    else:
                        next_tick = time.monotonic()
                    continue
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
                    page.dispatch_key(ACTION_KEYS[action])
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
        print(f"saved policy log: {log_path}")
    finally:
        page.close()


if __name__ == "__main__":
    main()
