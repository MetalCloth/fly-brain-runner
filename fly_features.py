"""Small sparse feature extractor used by the first fly-inspired policy."""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F


class SparseFlyExtractor(BaseFeaturesExtractor):
    """Sparse sensory projection before RecurrentPPO's memory layer.

    This is an engineering approximation, not a biological connectome. The
    fixed masks create separate body/motion and obstacle/looming pathways;
    RecurrentPPO supplies the temporal memory after this extractor.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 24) -> None:
        if len(observation_space.shape) != 1:
            raise ValueError("SparseFlyExtractor expects a flat observation vector")
        input_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim)

        hidden_dim = 32
        self.sensor = nn.Linear(input_dim, hidden_dim)
        self.integration = nn.Linear(hidden_dim, features_dim)
        self.register_buffer("sensor_mask", self._sensor_mask(hidden_dim, input_dim))
        self.register_buffer(
            "integration_mask", self._integration_mask(features_dim, hidden_dim)
        )

    @staticmethod
    def _sensor_mask(rows: int, columns: int) -> torch.Tensor:
        mask = torch.zeros(rows, columns)
        # Preserve every raw sensory channel through one sparse pathway.
        for row in range(min(columns, rows)):
            mask[row, row] = 1.0

        # Local looming units combine a lane indicator with its obstacle type.
        for group in range(9):
            row = 17 + group
            if row >= rows:
                break
            lane = group // 3
            mask[row, lane] = 1.0
            mask[row, 8 + group] = 1.0

        # A few global motion units mix obstacle urgency with body state.
        for row in range(26, rows):
            for offset in range(3):
                mask[row, 3 + ((row + offset) % 5)] = 1.0
                mask[row, 8 + ((row - 26) * 3 + offset) % 9] = 1.0
        return mask

    @staticmethod
    def _integration_mask(rows: int, columns: int) -> torch.Tensor:
        mask = torch.zeros(rows, columns)
        # Preserve the direct sensory pathway at the feature output.
        for row in range(min(rows, 17)):
            mask[row, row] = 1.0

        # Add small combinations for action selection without making the
        # feature extractor a dense generic MLP.
        for row in range(17, rows):
            group = row - 17
            mask[row, group % 17] = 1.0
            mask[row, 17 + (group % max(1, columns - 17))] = 1.0
            mask[row, (group * 3 + 17) % columns] = 1.0
        return mask

    @property
    def active_edges(self) -> int:
        return int(self.sensor_mask.sum().item() + self.integration_mask.sum().item())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sensor = F.linear(
            observations,
            self.sensor.weight * self.sensor_mask,
            self.sensor.bias,
        )
        integrated = F.linear(
            torch.tanh(sensor),
            self.integration.weight * self.integration_mask,
            self.integration.bias,
        )
        return torch.tanh(integrated)
