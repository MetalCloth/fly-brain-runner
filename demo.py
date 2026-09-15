"""Run a simple hand-written player through the symbolic runner."""

from env import ACTIONS, RunnerEnv


def choose_action(observation: dict) -> int:
    """Use simple rules so we can test the environment before RL."""

    current_lane = observation["lane"]
    nearby = [
        obstacle
        for obstacle in observation["obstacles"]
        if obstacle["lane"] == current_lane and obstacle["distance"] <= 2.5
    ]
    if not nearby:
        return 0

    obstacle = min(nearby, key=lambda item: item["distance"])
    if obstacle["kind"] == "jump":
        return 3
    if obstacle["kind"] == "roll":
        return 4

    # A block requires a lane change. Prefer the lane with no nearby obstacle.
    nearby_lanes = {
        item["lane"]
        for item in observation["obstacles"]
        if item["distance"] <= 2.5
    }
    for target_lane in (current_lane - 1, current_lane + 1):
        if 0 <= target_lane < 3 and target_lane not in nearby_lanes:
            return 1 if target_lane < current_lane else 2
    return 0


def main() -> None:
    env = RunnerEnv(seed=7, max_steps=60, spawn_interval=8)
    observation, _ = env.reset(seed=7)
    print(env.render())

    while True:
        action = choose_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            f"action={ACTIONS[action]:5s} reward={reward:5.2f} "
            f"event={info['event']} score={info['score']}"
        )
        if terminated or truncated:
            print(env.render())
            print(f"finished: {info['event']}")
            return


if __name__ == "__main__":
    main()

