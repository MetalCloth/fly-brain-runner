"""Check the sparse fly-inspired feature extractor in isolation."""

import numpy as np
import torch

from fly_features import SparseFlyExtractor
from gym_env import GymRunnerEnv


def main() -> None:
    env = GymRunnerEnv(seed=7)
    extractor = SparseFlyExtractor(env.observation_space)
    observation, _ = env.reset(seed=7)
    batch = torch.as_tensor(np.asarray([observation]), dtype=torch.float32)
    output = extractor(batch)
    assert output.shape == (1, extractor.features_dim)
    assert torch.isfinite(output).all()
    print(
        f"sparse extractor check passed: input={batch.shape[1]} "
        f"output={output.shape[1]} active_edges={extractor.active_edges}"
    )


if __name__ == "__main__":
    main()

