"""Check the pixel observation boundary before training a vision policy."""

from gymnasium.utils.env_checker import check_env

from gym_env import PartialPixelGymRunnerEnv, PixelGymRunnerEnv


def main() -> None:
    env = PixelGymRunnerEnv(render_mode="rgb_array", seed=7, max_steps=20)
    check_env(env, skip_render_check=True)

    observation, _ = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert observation.shape == (84, 84, 3)
    assert observation.dtype.name == "uint8"

    for _ in range(50):
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            observation, _ = env.reset()

    frame = env.render()
    assert frame is not None
    assert frame.shape == (84, 84, 3)
    assert int(frame.max()) > int(frame.min())
    print("Pixel Gymnasium environment check passed")
    print(
        f"shape={frame.shape} dtype={frame.dtype} "
        f"range={int(frame.min())}..{int(frame.max())}"
    )
    env.close()

    partial = PartialPixelGymRunnerEnv(render_mode="rgb_array", seed=7, max_steps=20)
    check_env(partial, skip_render_check=True)
    partial.reset(seed=7)
    partial.core.add_obstacle(lane=1, kind="block", distance=4.0)
    visible = partial.render()
    partial.step(0)
    hidden = partial.render()
    assert (visible[:, :, 0] > 200).any()
    assert not (hidden[:, :, 0] > 200).any()
    print("Partial pixel environment check passed (obstacles hidden every other frame)")
    partial.close()


if __name__ == "__main__":
    main()
