"""Train a browser-game policy from human-labeled screen recordings."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from browser_model import BrowserPolicy
from browser_pipeline import (
    ACTIONS,
    FRAME_SIZE,
    BrowserSample,
    load_sessions,
    make_samples,
    stack_sample,
)


class BrowserDataset(Dataset):
    def __init__(self, samples: list[BrowserSample], *, mirror: bool = False) -> None:
        self.samples = samples
        self.mirror = mirror

    def __len__(self) -> int:
        return len(self.samples) * (2 if self.mirror else 1)

    def __getitem__(self, index: int):
        mirrored = self.mirror and index >= len(self.samples)
        sample = self.samples[index % len(self.samples)]
        stack = stack_sample(sample)
        action = sample.action
        if mirrored:
            stack = stack[:, ::-1, :].copy()
            action = {1: 2, 2: 1}.get(action, action)
        tensor = torch.from_numpy(stack).permute(2, 0, 1).contiguous()
        return tensor, action


def split_sessions(
    grouped: list[list[BrowserSample]], val_fraction: float
) -> tuple[list[BrowserSample], list[BrowserSample]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    if len(grouped) >= 2:
        validation_count = max(1, round(len(grouped) * val_fraction))
        train = [sample for group in grouped[:-validation_count] for sample in group]
        validation = [sample for group in grouped[-validation_count:] for sample in group]
        return train, validation

    only = grouped[0]
    cutoff = max(1, min(len(only) - 1, round(len(only) * (1.0 - val_fraction))))
    print("warning: only one session; validation is the final temporal slice")
    return only[:cutoff], only[cutoff:]


def class_counts(samples: list[BrowserSample]) -> Counter[str]:
    return Counter(ACTIONS[sample.action] for sample in samples)


def balanced_accuracy(model: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    correct = np.zeros(len(ACTIONS), dtype=np.int64)
    totals = np.zeros(len(ACTIONS), dtype=np.int64)
    with torch.inference_mode():
        for frames, labels in loader:
            predictions = model(frames.float()).argmax(1).numpy()
            labels_np = labels.numpy()
            for action in ACTIONS:
                mask = labels_np == action
                totals[action] += int(mask.sum())
                correct[action] += int((predictions[mask] == action).sum())
    accuracy = float(correct.sum() / max(1, totals.sum()))
    valid = totals > 0
    macro = float((correct[valid] / totals[valid]).mean()) if valid.any() else 0.0
    return accuracy, macro


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for frames, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(frames.float()), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total_items += len(labels)
    return total_loss / max(1, total_items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("results/browser_dataset_poki")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/browser_bc_v1"))
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also train on horizontally mirrored frames and swap left/right",
    )
    args = parser.parse_args()
    if args.history < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("history, epochs, and batch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)

    sessions = load_sessions(args.data_dir)
    if not sessions:
        raise SystemExit(f"no recordings found under {args.data_dir}")
    grouped = make_samples(sessions, args.history)
    train_samples, validation_samples = split_sessions(grouped, args.val_fraction)
    if not train_samples or not validation_samples:
        raise SystemExit("need both training and validation samples")

    print(f"sessions={len(grouped)} train={len(train_samples)} val={len(validation_samples)}")
    print(f"train_actions={dict(class_counts(train_samples))}")
    print(f"val_actions={dict(class_counts(validation_samples))}")

    train_dataset = BrowserDataset(train_samples, mirror=args.mirror)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        BrowserDataset(validation_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    counts = np.bincount(
        np.asarray([sample.action for sample in train_samples]),
        minlength=len(ACTIONS),
    ).astype(np.float32)
    weights = np.power(len(train_samples) / np.maximum(counts, 1.0), 0.35)
    weights[counts == 0] = 0.0
    nonzero = weights > 0
    weights[nonzero] /= weights[nonzero].mean()

    model = BrowserPolicy(history=args.history)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "browser_policy_best.pt"
    best_macro = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_function)
        accuracy, macro = balanced_accuracy(model, validation_loader)
        row = {
            "epoch": epoch,
            "loss": loss,
            "val_accuracy": accuracy,
            "val_macro_accuracy": macro,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={loss:.4f} "
            f"val_accuracy={accuracy:.3f} val_macro={macro:.3f}"
        )
        if macro > best_macro:
            best_macro = macro
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "history": args.history,
                    "actions": ACTIONS,
                    "frame_size": list(FRAME_SIZE),
                    "best_val_macro_accuracy": best_macro,
                },
                checkpoint_path,
            )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "history": history,
                "train_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "train_actions": dict(class_counts(train_samples)),
                "validation_actions": dict(class_counts(validation_samples)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"saved model: {checkpoint_path}")


if __name__ == "__main__":
    main()
