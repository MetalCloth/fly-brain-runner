"""Validate browser recorder output and print class balance."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from browser_pipeline import ACTIONS, load_frame, load_sessions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("results/browser_dataset"))
    args = parser.parse_args()
    sessions = load_sessions(args.data_dir)
    if not sessions:
        raise SystemExit(f"no metadata.jsonl sessions found under {args.data_dir}")

    total = 0
    counts: Counter[str] = Counter()
    for session_dir, records in sessions:
        for record in records:
            load_frame(Path(str(record["_frame_path"])))
            counts[ACTIONS[int(record["action"])]] += 1
        total += len(records)
        print(f"session={session_dir} frames={len(records)}")

    print(f"total_frames={total}")
    print(f"actions={dict(counts)}")
    missing = [name for name in ACTIONS.values() if counts[name] == 0]
    if missing:
        print(f"warning: no examples for {missing}")
    if total < 100:
        print("warning: collect more data before training; this is only a smoke set")


if __name__ == "__main__":
    main()
