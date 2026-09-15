"""Run Gymnasium's environment checks and a short random smoke test."""

from gymnasium.utils.env_checker import check_env

from gym_env import GymRunnerEnv


def main() -> None:
    env = GymRunnerEnv(render_mode="ansi", seed=7, max_steps=20)
    check_env(env, skip_render_check=True)

    observation, _ = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    for _ in range(50):
        observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            observation, _ = env.reset()

    print("Gymnasium environment check passed")
    print(env.render())
    env.close()


if __name__ == "__main__":
    main()

