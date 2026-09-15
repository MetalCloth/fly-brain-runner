"""Check the reduced fly-inspired graph before PPO training."""

import numpy as np
import torch

from fly_graph import DenseFlyExtractor, FlyGraphExtractor
from gym_env import PartialPixelGymRunnerEnv


def main() -> None:
    env = PartialPixelGymRunnerEnv(render_mode="rgb_array", seed=7)
    observation, _ = env.reset(seed=7)
    extractor = FlyGraphExtractor(env.observation_space, features_dim=24)
    batch = torch.from_numpy(np.asarray(observation)).float().permute(2, 0, 1)
    features = extractor((batch / 255.0).unsqueeze(0))

    assert features.shape == (1, 24)
    assert extractor.active_edges > 0
    assert extractor.active_edges < 36 * 24

    env.core.add_obstacle(lane=0, kind="block", distance=4.0)
    obstacle_frame = env.render()
    obstacle_batch = torch.from_numpy(np.asarray(obstacle_frame)).float().permute(2, 0, 1)
    retina = extractor._retina((obstacle_batch / 255.0).unsqueeze(0))
    assert float(retina[0, 0]) > 0.05  # lane-0 far red receptor sees the block
    dense = DenseFlyExtractor(env.observation_space, features_dim=24)
    dense_features = dense((batch / 255.0).unsqueeze(0))
    assert dense_features.shape == (1, 24)
    assert dense.active_edges == 21 * 36 + 36 * 24
    print("Reduced fly-inspired graph check passed")
    print(
        f"input={observation.shape} sparse_output={tuple(features.shape)} "
        f"sparse_active_edges={extractor.active_edges} "
        f"dense_active_edges={dense.active_edges}"
    )
    env.close()


if __name__ == "__main__":
    main()
