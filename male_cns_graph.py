"""A fixed type-level graph built from the real MaleCNS data slice."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from fly_graph import pool_retina


DEFAULT_MANIFEST = Path(__file__).resolve().parent / "data/male_cns_visual_subgraph.json"
GRAPH_SIGNAL_GAIN = 4.0
CHANNELS_PER_LANE = 7
FEATURES_DIM = 2 * 3 * CHANNELS_PER_LANE


class MaleCNSGraphExtractor(BaseFeaturesExtractor):
    """Apply fixed MaleCNS type edges to lane- and color-local receptors.

    The graph has no trainable parameters. PPO learns the recurrent/action
    readout after this extractor; the manifest controls the hidden topology.
    Three lane channels and seven signal channels are replicated across the
    type graph so the six-node type reduction does not erase lane, distance,
    or obstacle identity. The two descending nodes become the 42-value output.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = FEATURES_DIM,
        manifest_path: str | Path = DEFAULT_MANIFEST,
    ) -> None:
        if len(observation_space.shape) != 3 or 3 not in observation_space.shape:
            raise ValueError("MaleCNSGraphExtractor expects an RGB image observation")
        if features_dim != FEATURES_DIM:
            raise ValueError(f"the fixed MaleCNS graph exposes exactly {FEATURES_DIM} features")
        super().__init__(observation_space, features_dim)

        manifest = json.loads(Path(manifest_path).read_text())
        self.node_types = tuple(manifest["selection"]["types"])
        if len(self.node_types) != 6:
            raise ValueError("expected the six-node MaleCNS milestone slice")
        positions = {node_type: index for index, node_type in enumerate(self.node_types)}
        adjacency = torch.zeros((len(self.node_types), len(self.node_types)))
        for edge in manifest["edges"]:
            source = positions[edge["upstream_type"]]
            target = positions[edge["downstream_type"]]
            adjacency[target, source] += math.log1p(float(edge["total_weight"]))
        adjacency /= adjacency.sum(dim=0, keepdim=True).clamp_min(1e-8)
        self.register_buffer("adjacency", adjacency)
        self._active_edges = len(manifest["edges"])
        self._output_indices = tuple(
            positions[node_type] for node_type in ("DNc01", "DNc02")
        )

    @property
    def active_edges(self) -> int:
        return self._active_edges

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        retina = pool_retina(observations)
        obstacle_bands = retina[:, :18].reshape(-1, 3, 2, 3)
        sensory = torch.cat((obstacle_bands.reshape(-1, 3, 6), retina[:, 18:21].unsqueeze(-1)), dim=-1)

        # Type 0 is R1-R6. Apply the real type graph independently per lane
        # and per receptor channel, retaining spatial channels for readout.
        state = torch.zeros(
            (sensory.shape[0], len(self.node_types), *sensory.shape[1:]),
            device=sensory.device,
            dtype=sensory.dtype,
        )
        state[:, 0] = sensory
        for _ in range(3):
            state = torch.tanh(
                GRAPH_SIGNAL_GAIN * torch.einsum("ij,bjlc->bilc", self.adjacency, state)
            )
        return state[:, self._output_indices].reshape(state.shape[0], -1)
