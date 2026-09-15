"""A small fly-inspired graph extractor for visual runner frames.

This is a sparse engineering model, not a MaleCNS-derived biological model.
It turns lane-local color signals into graph nodes, then applies fixed sparse
edges before RecurrentPPO supplies temporal memory.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F


RETINA_SIZE = 21


def _rgb_channels(observations: torch.Tensor) -> torch.Tensor:
    if observations.ndim != 4:
        raise ValueError("expected a batch of RGB frames")
    if observations.shape[1] == 3:
        channels = observations
    elif observations.shape[-1] == 3:
        channels = observations.permute(0, 3, 1, 2)
    else:
        raise ValueError("expected RGB channels")

    if float(channels.detach().amax()) > 1.0:
        channels = channels / 255.0
    return channels


def pool_retina(observations: torch.Tensor) -> torch.Tensor:
    """Pool RGB frames into the fixed 21-value visual receptor vector."""

    channels = _rgb_channels(observations)

    lane_bounds = ((5, 29), (30, 55), (56, 79))
    bands = ((26, 45), (45, 68))
    receptors: list[torch.Tensor] = []
    player_receptors: list[torch.Tensor] = []

    for left, right in lane_bounds:
        for top, bottom in bands:
            region = channels[:, :, top:bottom, left:right]
            red, green, blue = region.unbind(dim=1)
            receptors.extend(
                (
                    F.relu(red - (green + blue) / 2).mean(dim=(1, 2)),
                    F.relu(torch.minimum(red, green) - blue).mean(dim=(1, 2)),
                    F.relu(blue - (red + green) / 2 - 0.05).mean(dim=(1, 2)),
                )
            )

        bottom = channels[:, :, 58:78, left:right]
        red, green, blue = bottom.unbind(dim=1)
        player_receptors.append(
            F.relu(green - (red + blue) / 2).mean(dim=(1, 2))
        )

    return torch.stack(receptors + player_receptors, dim=1)


def _perspective_lane_mask(
    height: int,
    width: int,
    lane: int,
    top: int,
    bottom: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a mask following the rich renderer's trapezoid lanes."""

    mask = torch.zeros((height, width), device=device, dtype=dtype)
    horizon = 25.0
    depth = max(1.0, float(height - 26))
    for y in range(max(0, top), min(height, bottom)):
        progress = max(0.0, min(1.0, (float(y) - horizon) / depth))
        half_width = 18.0 + 24.0 * progress
        centers = [42.0 + (index - 1) * (6.0 + 21.0 * progress) for index in range(3)]
        left = 42.0 - half_width if lane == 0 else (centers[lane - 1] + centers[lane]) / 2.0
        right = 42.0 + half_width if lane == 2 else (centers[lane] + centers[lane + 1]) / 2.0
        left_index = max(0, int(round(left)) + 1)
        right_index = min(width, int(round(right)) - 1)
        if right_index > left_index:
            mask[y, left_index:right_index] = 1.0
    return mask


@lru_cache(maxsize=4)
def _perspective_masks(height: int, width: int) -> torch.Tensor:
    """Build the nine fixed lane/band masks once per frame size."""

    masks = [
        _perspective_lane_mask(
            height,
            width,
            lane,
            top,
            bottom,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for lane in range(3)
        for top, bottom in ((27, 49), (45, 78))
    ]
    masks.extend(
        _perspective_lane_mask(
            height,
            width,
            lane,
            59,
            81,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for lane in range(3)
    )
    return torch.stack(masks)


def pool_perspective_retina(observations: torch.Tensor) -> torch.Tensor:
    """Pool rich frames with lane regions that follow their perspective."""

    channels = _rgb_channels(observations)
    _, _, height, width = channels.shape
    red, green, blue = channels.unbind(dim=1)

    # ponytail: behavior-oriented groups keep the original 21-channel
    # contract; add per-object receptors only if held-out tests prove needed.
    solid = torch.maximum(
        F.relu(red - 1.25 * green),
        F.relu(torch.minimum(torch.minimum(red, green), blue) - 0.35),
    )
    jump = torch.maximum(
        F.relu(torch.minimum(red, green) - blue),
        F.relu(torch.minimum(red, blue) - green),
    )
    roll = F.relu(blue - (red + green) / 2 - 0.05)
    bright_neutral = F.relu(
        torch.minimum(torch.minimum(red, green), blue) - 0.35
    )
    dark = F.relu(0.16 - (red + green + blue) / 3)
    roll = torch.maximum(roll, dark)
    masks = _perspective_masks(height, width).to(
        device=channels.device, dtype=channels.dtype
    )
    denominators = masks.flatten(1).sum(dim=1).clamp_min(1.0)
    semantic = torch.stack((solid, jump, roll), dim=1)
    pooled = (
        semantic[:, None] * masks[None, :, None]
    ).sum(dim=(-1, -2)) / denominators[None, :, None]
    obstacle_receptors = pooled[:, :6].reshape(channels.shape[0], 18)
    player_signal = F.relu(green - (red + blue) / 2)[:, None]
    player_receptors = (
        player_signal * masks[None, 6:]
    ).sum(dim=(-1, -2)) / denominators[None, 6:]
    return torch.cat((obstacle_receptors, player_receptors), dim=1)


class FlyGraphExtractor(BaseFeaturesExtractor):
    """Pool a frame into receptors, then pass it through masked graph edges."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 24) -> None:
        if len(observation_space.shape) != 3 or 3 not in observation_space.shape:
            raise ValueError("FlyGraphExtractor expects an RGB image observation")
        super().__init__(observation_space, features_dim)

        hidden_dim = 36
        self.sensory = nn.Linear(RETINA_SIZE, hidden_dim)
        self.integration = nn.Linear(hidden_dim, features_dim)
        self.register_buffer(
            "sensory_mask", self._sensory_mask(hidden_dim, RETINA_SIZE)
        )
        self.register_buffer(
            "integration_mask", self._integration_mask(features_dim, hidden_dim)
        )

    @staticmethod
    def _sensory_mask(rows: int, columns: int) -> torch.Tensor:
        mask = torch.zeros(rows, columns)

        # Nine local obstacle nodes: lane x obstacle color, using near and far
        # receptor bands. The first 18 receptors are ordered the same way.
        for lane in range(3):
            for kind in range(3):
                node = lane * 3 + kind
                for band in range(2):
                    mask[node, lane * 6 + band * 3 + kind] = 1.0

        # Three body-state nodes detect the green player in each lane.
        for lane in range(3):
            mask[9 + lane, 18 + lane] = 1.0

        # Looming and cross-lane nodes mix local obstacle bands and neighboring
        # body/obstacle signals without making this layer dense.
        for row in range(12, rows):
            group = row - 12
            lane = group % 3
            kind = (group // 3) % 3
            mask[row, lane * 6 + kind] = 1.0
            mask[row, lane * 6 + 3 + kind] = 1.0
            mask[row, 18 + ((lane + 1) % 3)] = 1.0
            if group % 2 == 0:
                mask[row, ((lane + 2) % 3) * 6 + kind] = 1.0
        return mask

    @staticmethod
    def _integration_mask(rows: int, columns: int) -> torch.Tensor:
        mask = torch.zeros(rows, columns)

        # Preserve lane/type pathways and add a small neighboring-lane pathway.
        for lane in range(3):
            for kind in range(3):
                row = lane * 3 + kind
                mask[row, lane * 3 + kind] = 1.0
                mask[row, 12 + lane * 3 + kind] = 1.0
                mask[row, 21 + ((lane + 1) % 3) * 3 + kind] = 1.0

        for lane in range(3):
            row = 9 + lane
            mask[row, 9 + lane] = 1.0
            mask[row, 18 + ((lane + 1) % 3)] = 1.0

        for row in range(12, rows):
            group = row - 12
            mask[row, group % columns] = 1.0
            mask[row, (group * 5 + 12) % columns] = 1.0
            mask[row, (group * 7 + 21) % columns] = 1.0
        return mask

    @property
    def active_edges(self) -> int:
        return int(self.sensory_mask.sum().item() + self.integration_mask.sum().item())

    def _retina(self, observations: torch.Tensor) -> torch.Tensor:
        return pool_retina(observations)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        retina = self._retina(observations)
        sensory = F.linear(
            retina,
            self.sensory.weight * self.sensory_mask,
            self.sensory.bias,
        )
        integrated = F.linear(
            torch.tanh(sensory),
            self.integration.weight * self.integration_mask,
            self.integration.bias,
        )
        return torch.tanh(integrated)


class DenseFlyExtractor(FlyGraphExtractor):
    """Dense 21-to-36-to-24 control using the same fixed retina pooling."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 24) -> None:
        if len(observation_space.shape) != 3 or 3 not in observation_space.shape:
            raise ValueError("DenseFlyExtractor expects an RGB image observation")
        BaseFeaturesExtractor.__init__(self, observation_space, features_dim)
        self.sensory = nn.Linear(RETINA_SIZE, 36)
        self.integration = nn.Linear(36, features_dim)

    @property
    def active_edges(self) -> int:
        return RETINA_SIZE * 36 + 36 * self.features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        retina = self._retina(observations)
        sensory = torch.tanh(self.sensory(retina))
        return torch.tanh(self.integration(sensory))
