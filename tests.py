"""Small assert-based checks for the Milestone 0 runner."""

from env import (
    ACTIONS,
    PICKUP_KINDS,
    RICH_ACTIONS,
    RICH_OBSTACLE_KINDS,
    RICH_OBSERVATION_SIZE,
    RICH_LEVEL_VARIANTS,
    RICH_SURPRISE_VARIANT,
    RunnerEnv,
    OBSERVATION_SIZE,
    encode_observation,
    encode_rich_observation,
)
from rich_benchmark import human_teacher_action
from random import Random


def test_seeded_runs_match() -> None:
    first = RunnerEnv(seed=1, spawn_interval=3)
    second = RunnerEnv(seed=1, spawn_interval=3)
    first.reset(seed=42)
    second.reset(seed=42)

    for action in (0, 0, 1, 3, 0, 2, 4, 0, 0):
        result_a = first.step(action)
        result_b = second.step(action)
        assert result_a == result_b


def test_lane_changes_are_clamped() -> None:
    env = RunnerEnv(spawn_interval=999)
    env.reset()
    env.step(1)
    assert env.player_lane == 0
    env.step(1)
    assert env.player_lane == 0
    env.step(2)
    assert env.player_lane == 1


def test_block_collides() -> None:
    env = RunnerEnv(spawn_interval=999)
    env.reset()
    env.add_obstacle(lane=1, kind="block", distance=1)
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == -1.0
    assert terminated is True
    assert truncated is False
    assert info["event"] == "collision"


def test_jump_passes_jump_obstacle() -> None:
    env = RunnerEnv(spawn_interval=999)
    env.reset()
    env.add_obstacle(lane=1, kind="jump", distance=1)
    _, reward, terminated, _, info = env.step(3)
    assert reward > 1.0
    assert terminated is False
    assert info["passed"] == 1


def test_roll_passes_roll_obstacle() -> None:
    env = RunnerEnv(spawn_interval=999)
    env.reset()
    env.add_obstacle(lane=1, kind="roll", distance=1)
    _, _, terminated, _, info = env.step(4)
    assert terminated is False
    assert info["passed"] == 1


def test_invalid_action_fails_loudly() -> None:
    env = RunnerEnv(spawn_interval=999)
    env.reset()
    try:
        env.step(max(ACTIONS) + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid actions must raise ValueError")


def test_numeric_observation_has_fixed_range() -> None:
    env = RunnerEnv(spawn_interval=999)
    observation, _ = env.reset()
    env.add_obstacle(lane=2, kind="jump", distance=1)
    observation, _, _, _, _ = env.step(0)
    vector = encode_observation(observation, spawn_distance=env.spawn_distance)
    assert len(vector) == OBSERVATION_SIZE
    assert all(0.0 <= value <= 1.0 for value in vector)


def test_rich_powerup_enables_high_jump_over_train() -> None:
    env = RunnerEnv(rich_mechanics=True, spawn_interval=999, speed=1.0)
    env.reset(seed=7)
    env.add_pickup(lane=1, kind="super_sneakers", distance=1)
    env.add_obstacle(
        lane=1, kind="solid", appearance="train", roofable=True, distance=2
    )

    env.step(0)
    _, reward, terminated, _, info = env.step(3)

    assert terminated is False
    assert reward > 0
    assert info["passed"] == 1
    assert env.active_powerups["super_sneakers"] > 0


def test_rich_magnet_collects_coin_from_another_lane() -> None:
    env = RunnerEnv(rich_mechanics=True, spawn_interval=999, speed=1.0)
    env.reset(seed=7)
    env.add_pickup(lane=1, kind="coin_magnet", distance=1)
    env.add_pickup(lane=0, kind="coin", distance=1)

    _, _, _, _, info = env.step(0)

    assert env.coins == 1
    assert info["active_powerups"]["coin_magnet"] > 0


def test_rich_hoverboard_absorbs_one_collision() -> None:
    env = RunnerEnv(rich_mechanics=True, spawn_interval=999, speed=1.0)
    env.reset(seed=7)
    env.add_obstacle(lane=1, kind="solid", appearance="train", distance=1)

    _, reward, terminated, _, info = env.step(5)

    assert terminated is False
    assert reward < 0
    assert info["event"] == "hoverboard_saved_collision"
    assert env.hoverboard_active is False
    assert env.hoverboard_charges == 0


def test_rich_ramp_carries_across_long_train() -> None:
    env = RunnerEnv(rich_mechanics=True, spawn_interval=999, speed=1.0)
    env.reset(seed=7)
    env.add_obstacle(lane=1, kind="roof_route", appearance="ramp", distance=2)
    for distance in (3, 4, 5):
        env.add_obstacle(
            lane=1,
            kind="solid",
            appearance="train",
            roofable=True,
            distance=distance,
        )

    results = [env.step(0) for _ in range(5)]

    assert all(result[2] is False for result in results)
    assert results[-1][4]["passed"] == 1


def test_rich_gap_requires_jump() -> None:
    env = RunnerEnv(rich_mechanics=True, spawn_interval=999, speed=1.0)
    env.reset(seed=7)
    env.add_obstacle(lane=1, kind="gap", distance=1)
    _, _, terminated, _, info = env.step(3)
    assert terminated is False
    assert info["passed"] == 1


def test_rich_oncoming_train_is_not_a_normal_jump() -> None:
    env = RunnerEnv(rich_mechanics=True, spawn_interval=999, speed=1.0)
    env.reset(seed=7)
    env.add_obstacle(
        lane=1, kind="solid", appearance="train", speed_scale=1.8, distance=1
    )
    _, _, terminated, _, info = env.step(3)
    assert terminated is True
    assert info["collision_kind"] == "solid"
    assert info["collision_appearance"] == "train"


def test_rich_schema_separates_behavior_from_appearance() -> None:
    env = RunnerEnv(rich_mechanics=True, spawn_interval=999)
    env.reset(seed=7)
    env.add_obstacle(lane=1, kind="solid", appearance="bus", distance=2)
    env.add_obstacle(lane=0, kind="roll", appearance="tunnel", distance=3)
    observation, *_ = env.step(0)

    assert RICH_OBSTACLE_KINDS == ("solid", "jump", "roll", "either", "gap", "roof_route")
    assert observation["obstacles"][0]["kind"] == "solid"
    assert observation["obstacles"][0]["appearance"] == "bus"
    assert observation["obstacles"][1]["kind"] == "roll"
    assert observation["obstacles"][1]["appearance"] == "tunnel"


def test_rich_layer_one_stays_with_core_hazards() -> None:
    env = RunnerEnv(rich_mechanics=True, rich_layer=1, spawn_interval=999)
    env.reset(seed=7)
    env._spawn_rich_pattern()

    assert env._obstacles
    assert all(item.kind in ("solid", "jump", "roll", "either") for item in env._obstacles)


def test_rich_level_variants_are_seeded_and_distinct() -> None:
    standard = RunnerEnv(rich_mechanics=True, level_variant="standard")
    dense = RunnerEnv(rich_mechanics=True, level_variant="dense")
    fast = RunnerEnv(rich_mechanics=True, level_variant="fast")
    mixed_a = RunnerEnv(rich_mechanics=True, level_variant="mixed")
    mixed_b = RunnerEnv(rich_mechanics=True, level_variant="mixed")

    _, standard_info = standard.reset(seed=7)
    _, dense_info = dense.reset(seed=7)
    _, fast_info = fast.reset(seed=7)
    _, mixed_info_a = mixed_a.reset(seed=7)
    _, mixed_info_b = mixed_b.reset(seed=7)

    assert RICH_LEVEL_VARIANTS == ("standard", "dense", "fast", "mixed")
    assert standard_info["level_variant"] == "standard"
    assert dense_info["level_variant"] == "dense"
    assert fast_info["level_variant"] == "fast"
    assert dense.spawn_interval < standard.spawn_interval
    assert fast.speed > standard.speed
    assert mixed_info_a["level_variant"] in RICH_LEVEL_VARIANTS[:-1]
    assert mixed_info_a == mixed_info_b


def test_rich_surprise_variant_is_seeded_and_held_out() -> None:
    first = RunnerEnv(rich_mechanics=True, level_variant=RICH_SURPRISE_VARIANT)
    second = RunnerEnv(rich_mechanics=True, level_variant=RICH_SURPRISE_VARIANT)
    _, first_info = first.reset(seed=7007)
    _, second_info = second.reset(seed=7007)
    assert RICH_SURPRISE_VARIANT not in RICH_LEVEL_VARIANTS
    assert first_info == second_info
    assert first_info["level_variant"] == RICH_SURPRISE_VARIANT


def test_rich_all_pickups_are_valid_and_encoded() -> None:
    env = RunnerEnv(rich_mechanics=True, spawn_interval=999)
    env.reset(seed=7)
    for index, kind in enumerate(PICKUP_KINDS):
        env.add_pickup(lane=1, kind=kind, distance=20 + index)
    observation, _, _, _, _ = env.step(0)
    vector = encode_rich_observation(observation, spawn_distance=env.spawn_distance)

    assert len(RICH_ACTIONS) == 6
    assert len(vector) == RICH_OBSERVATION_SIZE
    assert all(0.0 <= value <= 1.0 for value in vector)


def test_human_teacher_collects_coin_when_route_is_clear() -> None:
    env = RunnerEnv(rich_mechanics=True, spawn_interval=999)
    observation, _ = env.reset(seed=7)
    env.add_pickup(lane=1, kind="coin", distance=1)
    observation = env._observation()
    assert human_teacher_action(observation, Random(7)) == 0


def test_human_teacher_jumps_for_coin_over_jump_hazard() -> None:
    env = RunnerEnv(rich_mechanics=True, spawn_interval=999)
    env.reset(seed=7)
    env.add_pickup(lane=1, kind="coin", distance=1)
    env.add_obstacle(lane=1, kind="jump", distance=1)
    assert human_teacher_action(env._observation(), Random(7)) == 3


def test_human_teacher_skips_dangerous_coin() -> None:
    env = RunnerEnv(rich_mechanics=True, spawn_interval=999)
    env.reset(seed=7)
    env.add_pickup(lane=1, kind="coin", distance=1)
    env.add_obstacle(lane=1, kind="solid", distance=1)
    assert human_teacher_action(env._observation(), Random(7)) in (1, 2)


def test_rich_info_reports_pickups_and_near_miss() -> None:
    env = RunnerEnv(rich_mechanics=True, spawn_interval=999)
    env.reset(seed=7)
    env.add_pickup(lane=1, kind="coin", distance=1)
    env.add_obstacle(lane=1, kind="jump", distance=1)
    _, _, terminated, _, info = env.step(3)
    assert terminated is False
    assert info["collected"] == ("collected_coin",)
    assert info["pickup_opportunities"] == ("coin",)
    assert info["near_miss"] is True


if __name__ == "__main__":
    tests = [
        test_seeded_runs_match,
        test_lane_changes_are_clamped,
        test_block_collides,
        test_jump_passes_jump_obstacle,
        test_roll_passes_roll_obstacle,
        test_invalid_action_fails_loudly,
        test_numeric_observation_has_fixed_range,
        test_rich_powerup_enables_high_jump_over_train,
        test_rich_magnet_collects_coin_from_another_lane,
        test_rich_hoverboard_absorbs_one_collision,
        test_rich_ramp_carries_across_long_train,
        test_rich_gap_requires_jump,
        test_rich_oncoming_train_is_not_a_normal_jump,
        test_rich_schema_separates_behavior_from_appearance,
        test_rich_layer_one_stays_with_core_hazards,
        test_rich_level_variants_are_seeded_and_distinct,
        test_rich_surprise_variant_is_seeded_and_held_out,
        test_rich_all_pickups_are_valid_and_encoded,
        test_human_teacher_collects_coin_when_route_is_clear,
        test_human_teacher_jumps_for_coin_over_jump_hazard,
        test_human_teacher_skips_dangerous_coin,
        test_rich_info_reports_pickups_and_near_miss,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed")
