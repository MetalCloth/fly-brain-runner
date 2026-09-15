"""Small assert-based checks for the fixed neuron-level graph."""

from __future__ import annotations

import numpy as np
import torch

from gym_env import PartialPixelGymRunnerEnv, RichPixelGymRunnerEnv
from fly_graph import pool_perspective_retina, pool_retina
from male_cns_neuron_graph import MaleCNSNeuronGraphExtractor


def main() -> None:
    env = PartialPixelGymRunnerEnv(seed=7)
    try:
        observation, _ = env.reset(seed=7)
        extractor = MaleCNSNeuronGraphExtractor(env.observation_space)
        output = extractor(torch.from_numpy(observation).unsqueeze(0))
        assert output.shape == (1, extractor.output_features)
        assert extractor.neuron_count == 196
        assert extractor.active_edges == 247
        assert torch.isfinite(output).all()
        assert sum(parameter.numel() for parameter in extractor.parameters()) == 0

        env.core.add_obstacle(0, "block", distance=3.0)
        changed = extractor(torch.from_numpy(env._pixels()).unsqueeze(0))
        assert not np.array_equal(output.detach().numpy(), changed.detach().numpy())

        randomized = MaleCNSNeuronGraphExtractor(
            env.observation_space, randomized=True, graph_seed=13
        )
        assert randomized.active_edges == extractor.active_edges
        assert not torch.equal(randomized.adjacency, extractor.adjacency)
        assert sum(parameter.numel() for parameter in randomized.parameters()) == 0

        trainable = MaleCNSNeuronGraphExtractor(
            env.observation_space, trainable_edges=True
        )
        assert sum(parameter.numel() for parameter in trainable.parameters()) == 247
        assert trainable.edge_weights.shape == (247,)
        signal = trainable(torch.from_numpy(observation).unsqueeze(0))
        signal.sum().backward()
        assert trainable.edge_weights.grad is not None
    finally:
        env.close()

    rich_env = RichPixelGymRunnerEnv(seed=7, level_variant="mixed")
    try:
        rich_observation, _ = rich_env.reset(seed=7)
        rich_extractor = MaleCNSNeuronGraphExtractor(rich_env.observation_space)
        rich_output = rich_extractor(torch.from_numpy(rich_observation).unsqueeze(0))
        assert rich_env.action_space.n == 6
        assert rich_output.shape == (1, rich_extractor.output_features)
        assert torch.isfinite(rich_output).all()
        assert torch.allclose(
            rich_output,
            rich_extractor.forward_retina(pool_retina(
                torch.from_numpy(rich_observation).unsqueeze(0)
            )),
        )

        perspective = pool_perspective_retina(
            torch.from_numpy(rich_observation).unsqueeze(0)
        )
        assert perspective.shape == (1, 21)
        perspective_extractor = MaleCNSNeuronGraphExtractor(
            rich_env.observation_space, retina_mode="perspective"
        )
        perspective_output = perspective_extractor(
            torch.from_numpy(rich_observation).unsqueeze(0)
        )
        assert perspective_extractor.retina_mode == "perspective"
        assert torch.isfinite(perspective_output).all()
        rich_env.core.add_obstacle(1, "solid", appearance="train", distance=3.0)
        changed_perspective = perspective_extractor(
            torch.from_numpy(rich_env._pixels()).unsqueeze(0)
        )
        assert not torch.equal(perspective_output, changed_perspective)
    finally:
        rich_env.close()
    print("MaleCNS neuron graph check passed: 196 neurons, 247 fixed edges")


if __name__ == "__main__":
    main()
