"""Small assert-based checks for the fixed MaleCNS graph extractor."""

from __future__ import annotations

import numpy as np
import torch

from gym_env import PartialPixelGymRunnerEnv
from male_cns_graph import MaleCNSGraphExtractor


def main() -> None:
    env = PartialPixelGymRunnerEnv(seed=7)
    try:
        observation, _ = env.reset(seed=7)
        extractor = MaleCNSGraphExtractor(env.observation_space)
        output = extractor(torch.from_numpy(observation).unsqueeze(0))
        assert output.shape == (1, 42)
        assert torch.isfinite(output).all()
        assert extractor.active_edges == 30
        assert sum(parameter.numel() for parameter in extractor.parameters()) == 0

        env.core.add_obstacle(0, "block", distance=3.0)
        changed = extractor(torch.from_numpy(env._pixels()).unsqueeze(0))
        assert not np.array_equal(output.detach().numpy(), changed.detach().numpy())
    finally:
        env.close()
    print("MaleCNS graph check passed: 6 types, 30 fixed edges, output=(1, 42)")


if __name__ == "__main__":
    main()
