"""Shared browser-game dataset helpers."""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


ACTIONS = {0: "noop", 1: "left", 2: "right", 3: "jump", 4: "roll"}
FRAME_SIZE = (84, 84)


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


def resize_nearest(frame: np.ndarray, size: tuple[int, int] = FRAME_SIZE) -> np.ndarray:
    """Resize an RGB frame without adding an image dependency."""

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got {frame.shape}")
    height, width = frame.shape[:2]
    target_height, target_width = size
    if height < 1 or width < 1 or target_height < 1 or target_width < 1:
        raise ValueError("frame and target dimensions must be positive")
    rows = np.minimum(
        (np.arange(target_height) * height // target_height), height - 1
    )
    columns = np.minimum(
        (np.arange(target_width) * width // target_width), width - 1
    )
    return frame[rows[:, None], columns[None, :]].astype(np.uint8, copy=True)


def decode_png_rgb(data: bytes) -> np.ndarray:
    """Decode the 8-bit RGB/RGBA PNG emitted by grim."""

    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        raise ValueError("screen capture did not return a PNG")
    offset = len(signature)
    width = height = color_type = bit_depth = interlace = None
    compressed = bytearray()
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
        elif chunk_type == b"IDAT":
            compressed.extend(chunk)
        elif chunk_type == b"IEND":
            break
    if (
        width is None
        or height is None
        or bit_depth != 8
        or color_type not in (2, 6)
        or interlace != 0
    ):
        raise ValueError("grim PNG must be non-interlaced 8-bit RGB or RGBA")

    channels = 3 if color_type == 2 else 4
    row_bytes = width * channels
    raw = zlib.decompress(bytes(compressed))
    expected = height * (row_bytes + 1)
    if len(raw) != expected:
        raise ValueError(f"unexpected PNG payload size: {len(raw)} != {expected}")

    rows: list[bytes] = []
    previous = bytearray(row_bytes)
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        encoded = raw[cursor : cursor + row_bytes]
        cursor += row_bytes
        current = bytearray(row_bytes)
        for index, value in enumerate(encoded):
            left = current[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                prediction = 0
            elif filter_type == 1:
                prediction = left
            elif filter_type == 2:
                prediction = above
            elif filter_type == 3:
                prediction = (left + above) // 2
            elif filter_type == 4:
                estimate = left + above - upper_left
                distances = (
                    abs(estimate - left),
                    abs(estimate - above),
                    abs(estimate - upper_left),
                )
                prediction = (left, above, upper_left)[distances.index(min(distances))]
            else:
                raise ValueError(f"unsupported PNG filter type {filter_type}")
            current[index] = (value + prediction) & 0xFF
        rows.append(bytes(current))
        previous = current
    pixels = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, width, channels)
    return pixels[:, :, :3].copy()


@dataclass(frozen=True)
class BrowserSample:
    """One labeled frame with its temporal context."""

    frame_paths: tuple[Path, ...]
    action: int
    session_dir: Path
    index: int
    captured_at: float


def load_frame(path: Path) -> np.ndarray:
    """Load and validate one recorder-produced 84x84 RGB frame."""

    if path.suffix.lower() == ".png":
        frame = resize_nearest(decode_png_rgb(path.read_bytes()))
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
