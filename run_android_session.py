"""Run bounded, supervised Android episodes with automatic reset and logging.

The session runner owns the boring lifecycle around the policy: relaunch the
app, start a run, stop when the screen is no longer playable, and prepare the
next episode. It is dry-run by default; ``--execute`` is required for taps,
swipes, relaunches, and resets.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
from sb3_contrib import RecurrentPPO

from android_bridge import ACTIONS, AndroidBridge, parse_crop
from male_cns_neuron_graph import MaleCNSNeuronGraphExtractor  # noqa: F401
from run_android_policy import decode_png


DEFAULT_PACKAGE = "com.kiloo.subwaysurf"
DEFAULT_COMPONENT = (
    "com.kiloo.subwaysurf/"
    "com.sybogames.chili.multidex.ChiliMultidexSupportActivity"
)
FRAME_SHAPE = (84, 84, 3)
MENU_DISTANCE_THRESHOLD = 0.10
DIMMED_MEAN_THRESHOLD = 0.34


class ScreenState(str, Enum):
    ACTIVE = "active"
    MENU = "menu"
    BLOCKED = "blocked"


def screen_signature(frame: np.ndarray) -> np.ndarray:
    """Make a cheap, resolution-independent signature for screen gating."""

    if frame.shape != FRAME_SHAPE:
        raise ValueError(f"expected frame shape {FRAME_SHAPE}, got {frame.shape}")
    return frame[::4, ::4].astype(np.float32) / 255.0


def classify_screen(
    frame: np.ndarray,
    menu_signature: np.ndarray | None = None,
) -> ScreenState:
    """Classify only the states needed to prevent blind swipes.

    The detector is deliberately conservative and game-version-specific. A
    dark or uncertain screen is blocked rather than treated as gameplay.
    """

    if frame.shape != FRAME_SHAPE:
        raise ValueError(f"expected frame shape {FRAME_SHAPE}, got {frame.shape}")
    if menu_signature is not None:
        distance = float(np.abs(screen_signature(frame) - menu_signature).mean())
        if distance <= MENU_DISTANCE_THRESHOLD:
            return ScreenState.MENU
    if float(frame.mean()) / 255.0 < DIMMED_MEAN_THRESHOLD:
        return ScreenState.BLOCKED
    return ScreenState.ACTIVE


def capture_frame(
    bridge: AndroidBridge,
    crop: tuple[int, int, int, int],
) -> tuple[bytes, np.ndarray]:
    """Capture the raw PNG and the policy-sized RGB frame."""

    png = bridge.screenshot_png()
    return png, decode_png(png, crop)


def timestamped_directory(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / stamp
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def wait_for_menu(
    bridge: AndroidBridge,
    crop: tuple[int, int, int, int],
    *,
    timeout: float,
    poll_interval: float,
) -> tuple[bytes, np.ndarray]:
    """Wait for a bright, stable post-launch screen and use it as the menu reference."""

    deadline = time.monotonic() + timeout
    previous: np.ndarray | None = None
    latest_png = b""
    latest_frame: np.ndarray | None = None
    while time.monotonic() < deadline:
        latest_png, latest_frame = capture_frame(bridge, crop)
        stable = previous is not None and float(
            np.abs(screen_signature(latest_frame) - screen_signature(previous)).mean()
        ) <= MENU_DISTANCE_THRESHOLD
        bright = float(latest_frame.mean()) / 255.0 >= DIMMED_MEAN_THRESHOLD
        if stable and bright:
            return latest_png, latest_frame
        previous = latest_frame
        time.sleep(poll_interval)
    raise RuntimeError("timed out waiting for a stable Subway Surfers menu")


def wait_for_active(
    bridge: AndroidBridge,
    crop: tuple[int, int, int, int],
    menu_signature: np.ndarray,
    *,
    timeout: float,
    poll_interval: float,
) -> tuple[bytes, np.ndarray]:
    """Wait until the menu reference is gone and the screen is not dimmed."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        png, frame = capture_frame(bridge, crop)
        if classify_screen(frame, menu_signature) is ScreenState.ACTIVE:
            return png, frame
        time.sleep(poll_interval)
    raise RuntimeError("timed out waiting for active gameplay")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_episode(
    *,
    bridge: AndroidBridge,
    model: RecurrentPPO,
    args: argparse.Namespace,
    log_directory: Path,
    episode_index: int,
) -> dict[str, object]:
    """Run one bounded episode and return its summary."""

    width, height = bridge.screen_size()
    menu_signature: np.ndarray | None = None
    if args.execute:
        bridge.force_stop(args.package)
        bridge.launch(args.component)
        _, menu_frame = wait_for_menu(
            bridge,
            args.crop,
            timeout=args.menu_timeout,
            poll_interval=args.poll_interval,
        )
        menu_signature = screen_signature(menu_frame)
        bridge.tap(
            round(width * args.play_x_fraction),
            round(height * args.play_y_fraction),
        )
        wait_for_active(
            bridge,
            args.crop,
            menu_signature,
            timeout=args.active_timeout,
            poll_interval=args.poll_interval,
        )

    frame_log_path = log_directory / f"episode_{episode_index:03d}.jsonl"
    frame_directory = log_directory / f"episode_{episode_index:03d}_frames"
    if args.save_frames:
        frame_directory.mkdir(parents=True, exist_ok=True)

    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)
    action_counts: Counter[str] = Counter()
    stop_reason = "step_limit"
    steps_run = 0
    with frame_log_path.open("w", encoding="utf-8") as frame_log:
        for step in range(args.steps):
            tick = time.monotonic()
            png, frame = capture_frame(bridge, args.crop)
            state = classify_screen(frame, menu_signature)
            if state is not ScreenState.ACTIVE:
                stop_reason = state.value
                break

            action, lstm_state = model.predict(
                frame[None, ...],
                state=lstm_state,
                episode_start=episode_start,
                deterministic=True,
            )
            action_id = int(np.asarray(action).reshape(-1)[0])
            action_name = ACTIONS[action_id]
            action_counts[action_name] += 1
            if args.save_frames:
                (frame_directory / f"frame_{step:05d}.png").write_bytes(png)
            if args.execute:
                bridge.send_action(
                    action_id,
                    width=width,
                    height=height,
                    horizontal_fraction=args.horizontal_fraction,
                    vertical_fraction=args.vertical_fraction,
                    start_y_fraction=args.swipe_y_fraction,
                    duration_ms=args.duration_ms,
                )
            frame_log.write(
                json.dumps(
                    {
                        "step": step,
                        "state": state.value,
                        "action": action_id,
                        "action_name": action_name,
                        "executed": args.execute,
                        "captured_at": time.time(),
                    }
                )
                + "\n"
            )
            frame_log.flush()
            steps_run += 1
            episode_start[:] = False
            remaining = args.interval - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)

    summary = {
        "episode": episode_index,
        "steps": steps_run,
        "stop_reason": stop_reason,
        "actions": dict(action_counts),
        "executed": args.execute,
        "screen": {"width": width, "height": height},
        "crop": args.crop,
    }
    write_json(log_directory / f"episode_{episode_index:03d}.json", summary)
    print(
        f"episode={episode_index} steps={steps_run} stop={stop_reason} "
        f"actions={dict(action_counts)}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--serial")
    parser.add_argument("--crop", type=parse_crop, required=True)
    parser.add_argument("--package", default=DEFAULT_PACKAGE)
    parser.add_argument("--component", default=DEFAULT_COMPONENT)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--interval", type=float, default=0.15)
    parser.add_argument("--poll-interval", type=float, default=0.20)
    parser.add_argument("--menu-timeout", type=float, default=10.0)
    parser.add_argument("--active-timeout", type=float, default=5.0)
    parser.add_argument("--play-x-fraction", type=float, default=0.50)
    parser.add_argument("--play-y-fraction", type=float, default=0.83)
    parser.add_argument("--horizontal-fraction", type=float, default=0.22)
    parser.add_argument("--vertical-fraction", type=float, default=0.18)
    parser.add_argument("--swipe-y-fraction", type=float, default=0.72)
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--log-dir", type=Path, default=Path("results/android_sessions"))
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow app lifecycle taps, resets, and gameplay swipes",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes < 1 or args.steps < 1:
        raise ValueError("episodes and steps must be positive")
    if args.interval <= 0 or args.poll_interval <= 0:
        raise ValueError("intervals must be positive")
    for name in ("play_x_fraction", "play_y_fraction"):
        value = getattr(args, name)
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1")

    log_directory = timestamped_directory(args.log_dir)
    write_json(
        log_directory / "session.json",
        {
            "model": str(args.model),
            "serial": args.serial,
            "package": args.package,
            "component": args.component,
            "episodes": args.episodes,
            "steps": args.steps,
            "interval": args.interval,
            "crop": args.crop,
            "execute": args.execute,
            "save_frames": args.save_frames,
            "created_at": time.time(),
        },
    )
    bridge = AndroidBridge(serial=args.serial)
    model = RecurrentPPO.load(args.model, device="cpu")
    summaries = []
    try:
        for episode_index in range(args.episodes):
            summaries.append(
                run_episode(
                    bridge=bridge,
                    model=model,
                    args=args,
                    log_directory=log_directory,
                    episode_index=episode_index,
                )
            )
    except KeyboardInterrupt:
        print("interrupted; no further phone input will be sent")
        raise
    write_json(log_directory / "summary.json", {"episodes": summaries})
    print(f"logs={log_directory}")


if __name__ == "__main__":
    main()
