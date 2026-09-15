"""Small visual policy used by the real browser-game path."""

from __future__ import annotations

import torch
from torch import nn

from browser_pipeline import ACTIONS, FRAME_SIZE


class BrowserPolicy(nn.Module):
    """A compact 72x128 CNN with a short visual history (H x W)."""

    def __init__(self, history: int = 4) -> None:
        if history < 1:
            raise ValueError("history must be positive")
        super().__init__()
        self.history = history
        self.features = nn.Sequential(
            nn.Conv2d(3 * history, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            feature_size = self.features(
                torch.zeros(1, 3 * history, *FRAME_SIZE)
            ).shape[1]
        self.head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if (
            frames.ndim != 4
            or frames.shape[1] != self.history * 3
            or tuple(frames.shape[-2:]) != FRAME_SIZE
        ):
            raise ValueError(
                f"expected N x {self.history * 3} x {FRAME_SIZE[0]} x {FRAME_SIZE[1]} "
                f"input, got {tuple(frames.shape)}"
            )
        return self.head(self.features(frames / 255.0))
