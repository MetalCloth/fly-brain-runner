"""Small visual policy used by the real browser-game path."""

from __future__ import annotations

import torch
from torch import nn

from browser_pipeline import ACTIONS


class BrowserPolicy(nn.Module):
    """A compact 84x84 CNN with a short visual history."""

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
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != self.history * 3:
            raise ValueError(
                f"expected N x {self.history * 3} x H x W input, got {tuple(frames.shape)}"
            )
        return self.head(self.features(frames / 255.0))
