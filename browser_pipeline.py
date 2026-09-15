"""Shared browser-game dataset helpers."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


ACTIONS = {0: "noop", 1: "left", 2: "right", 3: "jump", 4: "roll"}
FRAME_SIZE = (72, 128)


def parse_region(value: str) -> tuple[int, int, int, int]:
    """Parse ``left,top,width,height`` screen coordinates."""

    try:
        parts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ValueError("region must be left,top,width,height") from error
    if (
        len(parts) != 4
        or parts[0] < 0
        or parts[1] < 0
        or parts[2] < 1
        or parts[3] < 1
    ):
        raise ValueError("region must be left,top,width,height with a positive size")
    return parts


def resize_frame(frame: np.ndarray, size: tuple[int, int] = FRAME_SIZE) -> np.ndarray:
    """Resize RGB data with area averaging or bilinear edge-safe sampling."""

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got {frame.shape}")
    height, width = frame.shape[:2]
    target_height, target_width = size
    if height < 1 or width < 1 or target_height < 1 or target_width < 1:
        raise ValueError("frame and target dimensions must be positive")
    if (height, width) == (target_height, target_width):
        return frame.astype(np.uint8, copy=True)

    if (
        height >= target_height
        and width >= target_width
        and height % target_height == 0
        and width % target_width == 0
    ):
        y_scale = height // target_height
        x_scale = width // target_width
        blocks = frame.reshape(
            target_height, y_scale, target_width, x_scale, 3
        )
        return np.rint(blocks.mean(axis=(1, 3))).astype(np.uint8)

    y_positions = (np.arange(target_height, dtype=np.float32) + 0.5) * height / target_height - 0.5
    y_floor = np.floor(y_positions).astype(np.int64)
    y_weight = np.clip(y_positions - y_floor, 0.0, 1.0)
    y0 = np.clip(y_floor, 0, height - 1)
    y1 = np.clip(y_floor + 1, 0, height - 1)
    rows = (
        frame[y0].astype(np.float32) * (1.0 - y_weight)[:, None, None]
        + frame[y1].astype(np.float32) * y_weight[:, None, None]
    )

    x_positions = (np.arange(target_width, dtype=np.float32) + 0.5) * width / target_width - 0.5
    x_floor = np.floor(x_positions).astype(np.int64)
    x_weight = np.clip(x_positions - x_floor, 0.0, 1.0)
    x0 = np.clip(x_floor, 0, width - 1)
    x1 = np.clip(x_floor + 1, 0, width - 1)
    resized = (
        rows[:, x0].astype(np.float32) * (1.0 - x_weight)[None, :, None]
        + rows[:, x1].astype(np.float32) * x_weight[None, :, None]
    )
    return np.rint(resized).clip(0, 255).astype(np.uint8)


def is_gameplay_frame(frame: np.ndarray) -> bool:
    """Recognize the active game from its blue pause HUD in the top-left."""

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got {frame.shape}")
    hud = frame[: max(1, frame.shape[0] // 6), : max(1, frame.shape[1] // 7)]
    red, green, blue = (hud[:, :, channel].astype(np.int16) for channel in range(3))
    blue_pixels = (blue > 120) & (blue > red * 1.15) & (blue > green * 1.05)
    # ponytail: color gate; use a template/DOM state detector if Poki changes its HUD.
    return int(blue_pixels.sum()) >= 25


def decode_image_rgb(data: bytes) -> np.ndarray:
    """Decode a browser image quickly into an RGB NumPy array."""

    try:
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, ValueError) as error:
        raise ValueError("browser capture did not return a readable image") from error


@dataclass(frozen=True)
class BrowserSample:
    """One labeled frame with its temporal context."""

    frame_paths: tuple[Path, ...]
    action: int
    session_dir: Path
    index: int
    captured_at: float


def load_frame(path: Path) -> np.ndarray:
    """Load and validate one recorder-produced 72x128 RGB frame (H x W)."""

    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        frame = resize_frame(decode_image_rgb(path.read_bytes()))
    else:
        frame = np.load(path, allow_pickle=False)
    if frame.shape != (*FRAME_SIZE, 3) or frame.dtype != np.uint8:
        raise ValueError(f"invalid browser frame {path}: {frame.shape} {frame.dtype}")
    return frame


def load_sessions(root: Path) -> list[tuple[Path, list[dict[str, object]]]]:
    """Load all recorder sessions below a dataset root."""

    sessions: list[tuple[Path, list[dict[str, object]]]] = []
    for metadata_path in sorted(root.rglob("metadata.jsonl")):
        records: list[dict[str, object]] = []
        with metadata_path.open(encoding="utf-8") as metadata:
            for line_number, line in enumerate(metadata, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON in {metadata_path}:{line_number}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"metadata row is not an object: {metadata_path}:{line_number}")
                frame_name = record.get("frame")
                action = record.get("action")
                if not isinstance(frame_name, str) or not isinstance(action, int):
                    raise ValueError(f"missing frame/action in {metadata_path}:{line_number}")
                if action not in ACTIONS:
                    raise ValueError(f"unknown action {action} in {metadata_path}:{line_number}")
                frame_path = metadata_path.parent / frame_name
                if not frame_path.is_file():
                    raise FileNotFoundError(frame_path)
                record["_frame_path"] = str(frame_path)
                records.append(record)
        if records:
            sessions.append((metadata_path.parent, records))
    return sessions


def make_samples(
    sessions: list[tuple[Path, list[dict[str, object]]]], history: int
) -> list[list[BrowserSample]]:
    """Build temporally padded samples while keeping sessions separate."""

    if history < 1:
        raise ValueError("history must be positive")
    grouped: list[list[BrowserSample]] = []
    for session_dir, records in sessions:
        session_samples: list[BrowserSample] = []
        for index, record in enumerate(records):
            paths = tuple(
                Path(
                    records[max(0, index - history + 1 + offset)]["_frame_path"]
                )
                for offset in range(history)
            )
            session_samples.append(
                BrowserSample(
                    frame_paths=paths,
                    action=int(record["action"]),
                    session_dir=session_dir,
                    index=index,
                    captured_at=float(record.get("captured_at", 0.0)),
                )
            )
        grouped.append(session_samples)
    return grouped


def stack_sample(sample: BrowserSample) -> np.ndarray:
    """Return a sample as HxWx(history*3) uint8 data."""

    return np.concatenate([load_frame(path) for path in sample.frame_paths], axis=2)


def iter_metadata(root: Path) -> Iterator[dict[str, object]]:
    """Yield validated metadata rows for lightweight inspection."""

    for _, records in load_sessions(root):
        yield from records
