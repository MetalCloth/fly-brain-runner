"""A fixed neuron-level graph built from the real MaleCNS subgraph."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from fly_graph import RETINA_SIZE, pool_perspective_retina, pool_retina


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent / "data/male_cns_neuron_subgraph.json"
)
GRAPH_SIGNAL_GAIN = 4.0


class MaleCNSNeuronGraphExtractor(BaseFeaturesExtractor):
    """Apply fixed individual-neuron edges to the pooled visual receptors.

    The graph has no trainable parameters. Selected R1-R6 neurons receive the
    pooled retina as a vector, and every graph edge is applied independently
    to each visual channel. The four selected DNc neurons become the output;
    this preserves the sensory channels without inventing neuron coordinates
    that the toy renderer does not provide.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int | None = None,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        randomized: bool = False,
        graph_seed: int = 13,
        trainable_edges: bool = False,
        retina_mode: str = "fixed",
    ) -> None:
        if len(observation_space.shape) != 3 or 3 not in observation_space.shape:
            raise ValueError("MaleCNSNeuronGraphExtractor expects an RGB image observation")
        if retina_mode not in ("fixed", "perspective"):
            raise ValueError("retina_mode must be 'fixed' or 'perspective'")

        manifest = json.loads(Path(manifest_path).read_text())
        nodes = manifest["nodes"]
        if not nodes:
            raise ValueError("the neuron manifest contains no nodes")

        positions = {int(node["body_id"]): index for index, node in enumerate(nodes)}
        edges = manifest["edges"]
        adjacency = torch.zeros((len(nodes), len(nodes)))
        if randomized:
            rng = random.Random(graph_seed)
            pairs = [(source, target) for source in range(len(nodes)) for target in range(len(nodes))]
            rng.shuffle(pairs)
            weights = [math.log1p(float(edge["weight"])) for edge in edges]
            rng.shuffle(weights)
            for (source, target), weight in zip(pairs, weights):
                adjacency[target, source] = weight
        else:
            for edge in edges:
                source = positions[int(edge["upstream_body_id"])]
                target = positions[int(edge["downstream_body_id"])]
                adjacency[target, source] += math.log1p(float(edge["weight"]))
        adjacency /= adjacency.sum(dim=0, keepdim=True).clamp_min(1e-8)
        edge_targets, edge_sources = adjacency.nonzero(as_tuple=True)
        edge_values = adjacency[edge_targets, edge_sources]

        sensory_indices = [
            index for index, node in enumerate(nodes) if node["type"] == "R1-R6"
        ]
        if not sensory_indices:
            raise ValueError("the neuron manifest contains no R1-R6 neurons")
        output_indices = [
            index
            for index, node in enumerate(nodes)
            if node["type"] in ("DNc01", "DNc02")
        ]
        if len(output_indices) != 4:
            raise ValueError("expected four DNc output neurons in this manifest")
        output_features = len(output_indices) * RETINA_SIZE
        if features_dim is not None and features_dim != output_features:
            raise ValueError(f"expected features_dim={output_features} for this neuron manifest")
        super().__init__(observation_space, output_features)
        if trainable_edges:
            self.register_buffer("edge_sources", edge_sources)
            self.register_buffer("edge_targets", edge_targets)
            self.edge_weights = nn.Parameter(edge_values.clone())
        else:
            # Keep fixed-model checkpoints compact and backwards-compatible.
            self.register_buffer("adjacency", adjacency)
        self._sensory_indices = tuple(sensory_indices)
        self._output_indices = tuple(output_indices)
        self._sensory_count = len(sensory_indices)
        self._output_features = output_features
        self._active_edges = len(edges)
        self._neuron_count = len(nodes)
        self._trainable_edges = trainable_edges
        self._effective_edges = len(edge_values)
        self._retina_mode = retina_mode

    @property
    def active_edges(self) -> int:
        return self._active_edges

    @property
    def neuron_count(self) -> int:
        return self._neuron_count

    @property
    def output_features(self) -> int:
        return self._output_features

    @property
    def retina_mode(self) -> str:
        return self._retina_mode

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        retina = (
            pool_perspective_retina(observations)
            if self._retina_mode == "perspective"
            else pool_retina(observations)
        )
        return self.forward_retina(retina)

    def forward_retina(self, retina: torch.Tensor) -> torch.Tensor:
        """Run the graph from a differentiable 21-value retina."""

        if retina.ndim != 2 or retina.shape[1] != RETINA_SIZE:
            raise ValueError(f"expected retina shape (batch, {RETINA_SIZE})")
        state = torch.zeros(
            (retina.shape[0], self._neuron_count, RETINA_SIZE),
            device=retina.device,
            dtype=retina.dtype,
        )
        state[:, self._sensory_indices] = (
            retina.unsqueeze(1) / self._sensory_count
        )
        if self._trainable_edges:
            weights = torch.zeros(
                (self._neuron_count, self._neuron_count),
                device=retina.device,
                dtype=retina.dtype,
            )
            weights.index_put_((self.edge_targets, self.edge_sources), self.edge_weights)
        else:
            weights = self.adjacency
        for _ in range(3):
            state = torch.tanh(
                GRAPH_SIGNAL_GAIN * torch.einsum("ij,bjc->bic", weights, state)
            )
        return state[:, self._output_indices].reshape(retina.shape[0], -1)
