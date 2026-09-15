"""Measure rich-environment solvability with a symbolic teacher baseline."""

from __future__ import annotations

from random import Random
from statistics import mean, median

from env import LANE_COUNT, RICH_ACTIONS, RICH_ALLOWED_LEVEL_VARIANTS, RunnerEnv


PICKUP_VALUES = {
    "coin": 0.40,
    "key": 0.55,
    "jetpack": 0.85,
    "super_sneakers": 0.75,
    "coin_magnet": 0.70,
    "multiplier": 0.90,
    "pogo": 0.75,
    "speed_pad": 0.55,
    "mystery_box": 0.80,
}


def teacher_action(observation: dict, _: Random) -> int:
    """Prefer a lane with no visible hazard, then use its required motion."""

    current_lane = int(observation["lane"])
    obstacles = [
        item for item in observation["obstacles"] if item["distance"] <= 7.0
    ]
    lane_load = {
        lane: sum(1.0 / (item["distance"] + 0.1) for item in obstacles if item["lane"] == lane)
        for lane in range(LANE_COUNT)
    }
    if lane_load[current_lane] > 0:
        target_lane = min(lane_load, key=lane_load.get)
        if lane_load[target_lane] == 0 and target_lane != current_lane:
            return 1 if target_lane < current_lane else 2

    current = [item for item in obstacles if item["lane"] == current_lane]
    if not current:
        return 0
    nearest = min(current, key=lambda item: item["distance"])
    if nearest["kind"] in ("jump", "gap", "either"):
        return 3
    if nearest["kind"] == "roll":
        return 4
    if nearest["kind"] == "solid" and observation["hoverboard_charges"] > 0:
        return 5
    return 0


def _candidate_lane(observation: dict, action: int) -> int:
    lane = int(observation["lane"])
    if action == 1:
        return max(0, lane - 1)
    if action == 2:
        return min(LANE_COUNT - 1, lane + 1)
    return lane


def _candidate_motion(observation: dict, action: int) -> str:
    motion = str(observation["motion"])
    active_powerups = observation.get("active_powerups", {})
    if active_powerups.get("jetpack", 0) > 0:
        return "jetpack"
    if action == 3:
        return "jump"
    if action == 4:
        return "roll"
    return motion


def _candidate_can_pass(observation: dict, obstacle: dict, action: int) -> bool:
    """Mirror the small rich collision contract for one candidate action."""

    kind = obstacle["kind"]
    active_powerups = observation.get("active_powerups", {})
    if kind == "roof_route":
        return True
    motion = _candidate_motion(observation, action)
    if motion == "jetpack" or active_powerups.get("jetpack", 0) > 0:
        return True
    if kind == "gap":
        return motion == "jump" or observation.get("elevated_timer", 0) > 0
    if kind == "jump":
        return motion == "jump"
    if kind == "roll":
        return motion == "roll"
    if kind == "either":
        return motion in ("jump", "roll")
    if kind == "solid":
        high_jump = active_powerups.get("pogo", 0) > 0 or active_powerups.get(
            "super_sneakers", 0
        ) > 0
        return bool(obstacle.get("roofable")) and (
            observation.get("elevated_timer", 0) > 0
            or (motion == "jump" and high_jump)
        )
    return False


def _hazard_cost(observation: dict, obstacle: dict, action: int) -> float:
    if _candidate_can_pass(observation, obstacle, action):
        return 0.0

    distance = max(0.0, float(obstacle["distance"]))
    # A short, continuous danger curve makes a coin a trade-off rather than a
    # forbidden target. Far hazards are preparation signals; only an imminent
    # unpassable hazard dominates a valuable pickup.
    danger_horizon = max(3.0, float(observation.get("speed", 1.0)) * 3.0)
    urgency = max(0.0, (danger_horizon - distance) / danger_horizon)
    cost = 6.0 * urgency * urgency
    if action == 5 and observation.get("hoverboard_charges", 0) > 0:
        # A board is a valuable last-resort save, not a free lane choice.
        cost = 0.80 + cost * 0.12
    return cost


def _pickup_value(observation: dict, pickup: dict, target_lane: int) -> float:
    kind = pickup["kind"]
    magnet_active = observation.get("active_powerups", {}).get("coin_magnet", 0) > 0
    reachable = magnet_active and kind in ("coin", "key")
    reachable = reachable or int(pickup["lane"]) == target_lane
    if not reachable:
        return 0.0
    distance = max(0.0, float(pickup["distance"]))
    closeness = max(0.0, 1.0 - distance / 7.0)
    return PICKUP_VALUES.get(kind, 0.0) * (0.5 + 0.5 * closeness)


def human_teacher_action(observation: dict, _: Random) -> int:
    """Choose a route with a soft survival-versus-pickup utility.

    This is still an oracle for the simulator, but it does not use a binary
    "never collect near danger" rule. It can jump over a pickup-lane hazard,
    take a nearby coin when the risk is small, or skip the coin when the
    expected collision cost is large.
    """

    best_action = 0
    best_score = float("-inf")
    safety_action = teacher_action(observation, None)
    current_hazards = [
        item
        for item in observation.get("obstacles", ())
        if item["lane"] == observation["lane"] and float(item["distance"]) <= 7.0
    ]
    if current_hazards:
        nearest = min(current_hazards, key=lambda item: float(item["distance"]))
        if nearest["kind"] in ("jump", "gap", "either"):
            safety_action = 3
        elif nearest["kind"] == "roll":
            safety_action = 4
    for action in RICH_ACTIONS:
        if action == 5 and observation.get("hoverboard_charges", 0) <= 0:
            continue
        target_lane = _candidate_lane(observation, action)
        score = -0.04 if action in (1, 2) and target_lane != observation["lane"] else 0.0
        # Keep the verified survival policy as a soft prior. This avoids a
        # coin bonus making the teacher wait until a two-lane escape is gone,
        # while still letting pickups change choices when risk is comparable.
        if action == safety_action:
            score += 0.80
        # ponytail: one-step scoring keeps the teacher inspectable; use short
        # rollouts only if compound patterns still cause measurable failures.
        for obstacle in observation.get("obstacles", ()):
            if obstacle["lane"] != target_lane or float(obstacle["distance"]) > 7.0:
                continue
            score -= _hazard_cost(observation, obstacle, action)
        for pickup in observation.get("pickups", ()):
            if float(pickup["distance"]) <= 7.0:
                score += _pickup_value(observation, pickup, target_lane)
        if score > best_score:
            best_action, best_score = action, score
    return best_action


def random_action(_: dict, rng: Random) -> int:
    return rng.randrange(len(RICH_ACTIONS))


def run_episode(
    policy, seed: int, max_steps: int = 300, level_variant: str = "standard"
) -> dict[str, float]:
    env = RunnerEnv(
        seed=seed,
        max_steps=max_steps,
        rich_mechanics=True,
        rich_layer=2,
        level_variant=level_variant,
    )
    observation, _ = env.reset(seed=seed)
    rng = Random(seed + 100_000)
    total_reward = 0.0
    steps = 0
    near_misses = 0
    coin_opportunities = 0
    while True:
        observation, reward, terminated, truncated, info = env.step(
            policy(observation, rng)
        )
        total_reward += reward
        steps += 1
        near_misses += int(info.get("near_miss", False))
        coin_opportunities += sum(
            kind == "coin" for kind in info.get("pickup_opportunities", ())
        )
        if terminated or truncated:
            return {
                "steps": float(steps),
                "reward": total_reward,
                "score": float(info["score"]),
                "coins": float(info.get("coins", 0)),
                "coin_opportunities": float(coin_opportunities),
                "near_misses": float(near_misses),
                "collision": float(info["collision"]),
            }


def report(
    name: str, policy, episodes: int = 50, level_variant: str = "standard"
) -> None:
    results = [
        run_episode(policy, seed, level_variant=level_variant)
        for seed in range(episodes)
    ]
    print(
        f"{name:7s} episodes={episodes} "
        f"mean_steps={mean(item['steps'] for item in results):.1f} "
        f"median_steps={median(item['steps'] for item in results):.1f} "
        f"mean_score={mean(item['score'] for item in results):.2f} "
        f"mean_coins={mean(item['coins'] for item in results):.2f} "
        f"near_misses={mean(item['near_misses'] for item in results):.2f} "
        f"collision_rate={mean(item['collision'] for item in results):.2f}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level-variant", choices=RICH_ALLOWED_LEVEL_VARIANTS, default="standard"
    )
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    report("random", random_action, args.episodes, args.level_variant)
    report("teacher", teacher_action, args.episodes, args.level_variant)
    report("human", human_teacher_action, args.episodes, args.level_variant)
