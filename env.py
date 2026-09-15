"""A tiny, dependency-free runner environment for Milestone 0.

It supports both clean symbolic observations and a tiny RGB renderer. The
physics stay the same in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any


ACTIONS = {
    0: "noop",
    1: "left",
    2: "right",
    3: "jump",
    4: "roll",
}

# Rich mode keeps the original five-action checkpoint compatible while adding
# the one meaningful piece of equipment that is itself an action.
RICH_ACTIONS = {**ACTIONS, 5: "hoverboard"}

OBSTACLE_KINDS = ("block", "jump", "roll")
# Rich mode separates what the player must do from what the object looks like.
# A train, bus, or wall can all be a ``solid`` behavior; the appearance is
# useful to a human and to later visual-domain randomization, but does not
# silently create another collision rule.
RICH_OBSTACLE_KINDS = (
    "solid",
    "jump",
    "roll",
    "either",
    "gap",
    "roof_route",
)
RICH_OBSTACLE_APPEARANCES = (
    "block",
    "barrier",
    "roll_bar",
    "either_hurdle",
    "train",
    "bus",
    "gap",
    "tunnel",
    "ramp",
)
RICH_OBSTACLE_ALIASES = {
    "block": ("solid", "block", 1.0, False),
    "train": ("solid", "train", 1.0, True),
    "oncoming_train": ("solid", "train", 1.8, False),
    "tunnel": ("roll", "tunnel", 1.0, False),
    "ramp": ("roof_route", "ramp", 1.0, False),
}
RICH_DEFAULT_APPEARANCES = {
    "solid": "block",
    "jump": "barrier",
    "roll": "roll_bar",
    "either": "either_hurdle",
    "gap": "gap",
    "roof_route": "ramp",
}
RICH_LEVEL_VARIANTS = ("standard", "dense", "fast", "mixed")
RICH_SURPRISE_VARIANT = "surprise"
RICH_ALLOWED_LEVEL_VARIANTS = RICH_LEVEL_VARIANTS + (RICH_SURPRISE_VARIANT,)
RICH_EVAL_LEVEL_VARIANTS = RICH_LEVEL_VARIANTS[:-1] + (RICH_SURPRISE_VARIANT,)
RICH_LEVEL_PROFILES = {
    "standard": {
        "speed_scale": 1.0,
        "spawn_interval_scale": 1.0,
        "spawn_distance_scale": 1.0,
        "pattern_weights": (42, 18, 18, 10, 12),
    },
    "dense": {
        "speed_scale": 1.0,
        "spawn_interval_scale": 0.75,
        "spawn_distance_scale": 1.0,
        "pattern_weights": (28, 24, 18, 14, 16),
    },
    "fast": {
        "speed_scale": 1.25,
        "spawn_interval_scale": 0.9,
        "spawn_distance_scale": 1.15,
        "pattern_weights": (34, 18, 18, 15, 15),
    },
    # Held out from training and the normal level loop; it is a generalization
    # check, not another curriculum level.
    "surprise": {
        "speed_scale": 1.12,
        "spawn_interval_scale": 0.82,
        "spawn_distance_scale": 1.10,
        "pattern_weights": (16, 20, 14, 28, 22),
    },
}
PICKUP_KINDS = (
    "coin",
    "key",
    "jetpack",
    "super_sneakers",
    "coin_magnet",
    "multiplier",
    "pogo",
    "speed_pad",
    "mystery_box",
)
ACTIVE_POWERUPS = (
    "jetpack",
    "super_sneakers",
    "coin_magnet",
    "multiplier",
    "pogo",
    "speed_pad",
)
LANE_COUNT = 3
MOTION_DURATION = 3
MOTION_STATES = ("ground", "jump", "roll")
RICH_MOTION_STATES = ("ground", "jump", "roll", "jetpack")
OBSERVATION_SIZE = 17
RICH_OBSERVATION_SIZE = (
    LANE_COUNT
    + len(RICH_MOTION_STATES)
    + 1  # motion timer
    + 1  # speed
    + 1  # hoverboard active
    + 1  # hoverboard charges
    + 1  # elevated state
    + len(ACTIVE_POWERUPS) * 2
    + LANE_COUNT * (len(RICH_OBSTACLE_KINDS) + len(PICKUP_KINDS))
)
PIXEL_WIDTH = 84
PIXEL_HEIGHT = 84


@dataclass
class Obstacle:
    lane: int
    kind: str
    distance: float
    appearance: str = "generic"
    speed_scale: float = 1.0
    roofable: bool = False


@dataclass
class Pickup:
    lane: int
    kind: str
    distance: float


class RunnerEnv:
    """Small three-lane endless runner with a Gymnasium-like API."""

    def __init__(
        self,
        seed: int = 0,
        max_steps: int = 300,
        speed: float = 1.0,
        spawn_interval: int = 8,
        spawn_distance: float = 6.0,
        rich_mechanics: bool = False,
        rich_layer: int = 2,
        level_variant: str = "standard",
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if speed <= 0:
            raise ValueError("speed must be positive")
        if spawn_interval < 1:
            raise ValueError("spawn_interval must be positive")
        if spawn_distance <= 0:
            raise ValueError("spawn_distance must be positive")
        if rich_layer not in (1, 2):
            raise ValueError("rich_layer must be 1 or 2")
        if not rich_mechanics and rich_layer != 2:
            raise ValueError("rich_layer requires rich_mechanics=True")
        if level_variant not in RICH_ALLOWED_LEVEL_VARIANTS:
            raise ValueError(
                f"level_variant must be one of {RICH_ALLOWED_LEVEL_VARIANTS}"
            )
        if not rich_mechanics and level_variant != "standard":
            raise ValueError("level_variant requires rich_mechanics=True")

        self.max_steps = max_steps
        self._requested_speed = speed
        self._requested_spawn_interval = spawn_interval
        self._requested_spawn_distance = spawn_distance
        self.speed = speed
        self.base_speed = speed
        self.spawn_interval = spawn_interval
        self.spawn_distance = spawn_distance
        self.rich_mechanics = rich_mechanics
        self.rich_layer = rich_layer
        self.level_variant = level_variant
        self.active_level_variant = "standard"
        self.actions = RICH_ACTIONS if rich_mechanics else ACTIONS
        self.obstacle_kinds = (
            RICH_OBSTACLE_KINDS if rich_mechanics else OBSTACLE_KINDS
        )
        self._seed = seed
        self.rng = random.Random(seed)
        self.reset(seed=seed)

    def _configure_level(self) -> None:
        variant = self.level_variant
        if variant == "mixed":
            variant = self.rng.choice(RICH_LEVEL_VARIANTS[:-1])
        profile = RICH_LEVEL_PROFILES[variant]
        self.active_level_variant = variant
        self.base_speed = self._requested_speed * profile["speed_scale"]
        self.spawn_interval = max(
            1, round(self._requested_spawn_interval * profile["spawn_interval_scale"])
        )
        self.spawn_distance = (
            self._requested_spawn_distance * profile["spawn_distance_scale"]
        )

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the run and return ``(observation, info)``."""

        if seed is not None:
            self._seed = seed
            self.rng = random.Random(seed)

        self._configure_level()
        self.step_count = 0
        self.player_lane = 1
        self.motion = "ground"
        self.motion_timer = 0
        self.score = 0
        self.speed = self.base_speed
        self._obstacles: list[Obstacle] = []
        self._pickups: list[Pickup] = []
        self.active_powerups = {kind: 0 for kind in ACTIVE_POWERUPS}
        self.hoverboard_active = False
        self.hoverboard_timer = 0
        self.hoverboard_charges = 1 if self.rich_mechanics else 0
        self.elevated_timer = 0
        self.coins = 0
        self.keys = 0
        self.terminated = False
        self.truncated = False
        self.last_event = "reset"
        return self._observation(), {
            "seed": self._seed,
            "level_variant": self.active_level_variant,
        }

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance the world by one decision step."""

        if self.terminated or self.truncated:
            raise RuntimeError("step() called after the episode ended; call reset()")
        if action not in self.actions:
            raise ValueError(
                f"unknown action {action}; expected one of {sorted(self.actions)}"
            )

        if self.rich_mechanics:
            return self._rich_step(action)

        self._apply_action(action)
        self.step_count += 1

        for obstacle in self._obstacles:
            obstacle.distance -= self.speed

        passed = 0
        collision = False
        remaining: list[Obstacle] = []
        for obstacle in self._obstacles:
            if obstacle.distance > 0:
                remaining.append(obstacle)
                continue

            if obstacle.lane == self.player_lane and not self._can_pass(obstacle.kind):
                collision = True
                continue

            passed += 1

        self._obstacles = remaining

        if collision:
            self.terminated = True
            self.last_event = "collision"
            reward = -1.0
        else:
            reward = 0.01 + passed * 1.0
            self.score += passed
            self.last_event = "passed" if passed else "survived"

            if self.step_count % self.spawn_interval == 0:
                self._spawn_random_obstacle()

            if self.step_count >= self.max_steps:
                self.truncated = True
                self.last_event = "timeout"

        self._tick_motion()
        info = {
            "event": self.last_event,
            "score": self.score,
            "passed": passed,
            "collision": collision,
        }
        return self._observation(), reward, self.terminated, self.truncated, info

    def add_obstacle(
        self,
        lane: int,
        kind: str,
        distance: float | None = None,
        *,
        appearance: str | None = None,
        speed_scale: float | None = None,
        roofable: bool | None = None,
    ) -> None:
        """Add a known obstacle for deterministic tests and debugging."""

        if lane not in range(LANE_COUNT):
            raise ValueError(f"lane must be between 0 and {LANE_COUNT - 1}")
        if distance is None:
            distance = self.spawn_distance
        if distance <= 0:
            raise ValueError("distance must be positive")

        if not self.rich_mechanics:
            if kind not in self.obstacle_kinds:
                raise ValueError(f"kind must be one of {self.obstacle_kinds}")
            if any(value is not None for value in (appearance, speed_scale, roofable)):
                raise ValueError("rich obstacle metadata requires rich_mechanics=True")
            self._obstacles.append(Obstacle(lane, kind, float(distance)))
            return

        alias = RICH_OBSTACLE_ALIASES.get(kind)
        if alias is not None:
            kind, alias_appearance, alias_speed, alias_roofable = alias
            appearance = alias_appearance if appearance is None else appearance
            speed_scale = alias_speed if speed_scale is None else speed_scale
            roofable = alias_roofable if roofable is None else roofable
        if kind not in self.obstacle_kinds:
            raise ValueError(f"behavior must be one of {self.obstacle_kinds}")
        appearance = RICH_DEFAULT_APPEARANCES[kind] if appearance is None else appearance
        speed_scale = 1.0 if speed_scale is None else float(speed_scale)
        roofable = False if roofable is None else bool(roofable)
        if appearance not in RICH_OBSTACLE_APPEARANCES:
            raise ValueError(
                f"appearance must be one of {RICH_OBSTACLE_APPEARANCES}"
            )
        if speed_scale <= 0:
            raise ValueError("speed_scale must be positive")
        if roofable and kind != "solid":
            raise ValueError("only solid obstacles can be roofable")
        self._obstacles.append(
            Obstacle(
                lane,
                kind,
                float(distance),
                appearance,
                speed_scale,
                roofable,
            )
        )

    def add_pickup(
        self, lane: int, kind: str, distance: float | None = None
    ) -> None:
        """Add a deterministic rich-mode pickup for tests and debugging."""

        if not self.rich_mechanics:
            raise ValueError("pickups require rich_mechanics=True")
        if lane not in range(LANE_COUNT):
            raise ValueError(f"lane must be between 0 and {LANE_COUNT - 1}")
        if kind not in PICKUP_KINDS:
            raise ValueError(f"kind must be one of {PICKUP_KINDS}")
        if distance is None:
            distance = self.spawn_distance
        if distance <= 0:
            raise ValueError("distance must be positive")
        self._pickups.append(Pickup(lane, kind, float(distance)))

    def render(self) -> str:
        """Return a compact text view for humans, not a model observation."""

        if self.rich_mechanics:
            return self._rich_render()

        cells = [".", ".", "."]
        for obstacle in sorted(self._obstacles, key=lambda item: item.distance):
            if 0 < obstacle.distance <= self.spawn_distance:
                symbol = {"block": "B", "jump": "J", "roll": "R"}[obstacle.kind]
                cells[obstacle.lane] = symbol

        player = [" ", " ", " "]
        player[self.player_lane] = "P"
        return (
            f"step={self.step_count:03d}  motion={self.motion:6s}  "
            f"score={self.score:02d}\n"
            f"obstacles: {' | '.join(cells)}\n"
            f"player:    {' | '.join(player)}"
        )

    def render_rgb(
        self,
        width: int = PIXEL_WIDTH,
        height: int = PIXEL_HEIGHT,
        show_obstacles: bool = True,
    ) -> bytes:
        """Return the current frame as dependency-free row-major RGB bytes."""

        if (width, height) != (PIXEL_WIDTH, PIXEL_HEIGHT):
            raise ValueError(f"frame must be {PIXEL_WIDTH}x{PIXEL_HEIGHT}")

        if self.rich_mechanics:
            return self._rich_render_rgb(show_obstacles=show_obstacles)

        # A deliberately simple scene for the original baseline.
        pixels = bytearray((18, 25, 39) * (width * height))

        def fill(
            left: int,
            top: int,
            right: int,
            bottom: int,
            color: tuple[int, int, int],
        ) -> None:
            left = max(0, min(width, left))
            top = max(0, min(height, top))
            right = max(0, min(width, right))
            bottom = max(0, min(height, bottom))
            if right <= left or bottom <= top:
                return
            row = bytes(color) * (right - left)
            for y in range(top, bottom):
                start = (y * width + left) * 3
                pixels[start : start + len(row)] = row

        # A deliberately simple scene: dark sky, road, and clear lane dividers.
        fill(0, 0, width, 24, (11, 17, 29))
        fill(5, 24, width - 5, height, (43, 50, 67))
        fill(5, 24, width - 5, 26, (104, 114, 132))
        for divider in (29, 55):
            fill(divider, 27, divider + 1, height, (92, 101, 119))

        lane_centers = (17, 42, 68)
        if show_obstacles:
            for obstacle in sorted(
                self._obstacles, key=lambda item: item.distance, reverse=True
            ):
                if not 0 < obstacle.distance <= self.spawn_distance:
                    continue
                progress = 1.0 - obstacle.distance / self.spawn_distance
                center = lane_centers[obstacle.lane]
                contact_y = int(31 + progress * 34)
                half_width = int(4 + progress * 5)

                if obstacle.kind == "block":
                    fill(
                        center - half_width,
                        contact_y - int(7 + progress * 7),
                        center + half_width + 1,
                        contact_y + 1,
                        (224, 67, 71),
                    )
                elif obstacle.kind == "jump":
                    fill(
                        center - half_width,
                        contact_y - int(3 + progress * 2),
                        center + half_width + 1,
                        contact_y + 1,
                        (242, 177, 58),
                    )
                else:  # roll obstacle: a bar above the runner's head.
                    bar_y = contact_y - int(10 + progress * 5)
                    fill(
                        center - half_width,
                        bar_y,
                        center + half_width + 1,
                        bar_y + int(3 + progress * 2),
                        (111, 177, 238),
                    )

        player_center = lane_centers[self.player_lane]
        if self.motion == "jump":
            player_top, player_bottom = 54, 65
        elif self.motion == "roll":
            player_top, player_bottom = 69, 76
        else:
            player_top, player_bottom = 63, 76
        fill(player_center - 4, player_top, player_center + 5, player_bottom, (75, 214, 139))
        fill(player_center - 3, player_top - 3, player_center + 4, player_top, (150, 235, 175))
        return bytes(pixels)

    def _apply_action(self, action: int) -> None:
        if action == 1:
            self.player_lane = max(0, self.player_lane - 1)
        elif action == 2:
            self.player_lane = min(LANE_COUNT - 1, self.player_lane + 1)
        elif action == 3:
            self.motion = "jump"
            self.motion_timer = MOTION_DURATION
        elif action == 4:
            self.motion = "roll"
            self.motion_timer = MOTION_DURATION

    def _tick_motion(self) -> None:
        if self.motion_timer > 0:
            self.motion_timer -= 1
        if self.motion_timer == 0:
            self.motion = "ground"

    def _can_pass(self, kind: str) -> bool:
        if kind == "block":
            return False
        return self.motion == kind

    # Rich mechanics -----------------------------------------------------
    # These are decision-equivalent categories, not a pixel-perfect clone.
    # Cosmetic and seasonal variants should map to one of these categories.

    def _rich_step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._apply_rich_action(action)
        self.step_count += 1

        for obstacle in self._obstacles:
            obstacle.distance -= self.speed * obstacle.speed_scale
        for pickup in self._pickups:
            pickup.distance -= self.speed

        pickup_opportunities = tuple(
            pickup.kind for pickup in self._pickups if pickup.distance <= 0
        )
        pickup_reward, collected = self._collect_pickups()
        passed = 0
        collision = False
        near_miss = False
        collision_kind = None
        collision_appearance = None
        remaining: list[Obstacle] = []
        # Process the nearest object first so a roof route can activate before
        # the train it crosses reaches the player on the same step.
        for obstacle in sorted(self._obstacles, key=lambda item: item.distance):
            if obstacle.distance > 0:
                remaining.append(obstacle)
                continue

            if obstacle.lane == self.player_lane and not self._rich_can_pass(obstacle):
                collision = True
                collision_kind = obstacle.kind
                collision_appearance = obstacle.appearance
                continue

            if obstacle.lane == self.player_lane and obstacle.kind != "roof_route":
                near_miss = True
            if obstacle.kind == "roof_route":
                self.elevated_timer = max(self.elevated_timer, 6)
            else:
                passed += 1

        self._obstacles = remaining

        if collision and self.hoverboard_active:
            self.hoverboard_active = False
            self.hoverboard_timer = 0
            self.last_event = "hoverboard_saved_collision"
            reward = pickup_reward - 0.05
        elif collision:
            self.terminated = True
            self.last_event = f"collision_{collision_kind}"
            reward = pickup_reward - 1.0
        else:
            multiplier = 2.0 if self.active_powerups["multiplier"] > 0 else 1.0
            reward = 0.01 + passed * multiplier + pickup_reward
            self.score += passed * multiplier
            if collected:
                self.last_event = collected[-1]
            else:
                self.last_event = "passed" if passed else "survived"

            if self.step_count % self.spawn_interval == 0:
                self._spawn_random_obstacle()

            if self.step_count >= self.max_steps:
                self.truncated = True
                self.last_event = "timeout"

        self._tick_rich_state()
        info = {
            "event": self.last_event,
            "score": self.score,
            "coins": self.coins,
            "keys": self.keys,
            "passed": passed,
            "collision": collision,
            "near_miss": near_miss,
            "collision_kind": collision_kind,
            "collision_appearance": collision_appearance,
            "collected": tuple(collected),
            "pickup_opportunities": pickup_opportunities,
            "active_powerups": dict(self.active_powerups),
            "hoverboard_active": self.hoverboard_active,
        }
        return self._observation(), reward, self.terminated, self.truncated, info

    def _apply_rich_action(self, action: int) -> None:
        if action == 1:
            self.player_lane = max(0, self.player_lane - 1)
        elif action == 2:
            self.player_lane = min(LANE_COUNT - 1, self.player_lane + 1)
        elif action == 3:
            if self.active_powerups["jetpack"] <= 0:
                self.motion = "jump"
                jump_duration = MOTION_DURATION
                if self.active_powerups["pogo"] > 0:
                    jump_duration = 6
                elif self.active_powerups["super_sneakers"] > 0:
                    jump_duration = 5
                self.motion_timer = max(self.motion_timer, jump_duration)
        elif action == 4:
            if self.active_powerups["jetpack"] <= 0:
                self.motion = "roll"
                self.motion_timer = max(self.motion_timer, MOTION_DURATION)
        elif action == 5 and self.hoverboard_charges > 0:
            if not self.hoverboard_active:
                self.hoverboard_charges -= 1
                self.hoverboard_active = True
                self.hoverboard_timer = 30

    def _rich_can_pass(self, obstacle: Obstacle) -> bool:
        if obstacle.kind == "roof_route":
            return True
        if self.motion == "jetpack" or self.active_powerups["jetpack"] > 0:
            return True
        if obstacle.kind == "gap":
            return self.motion == "jump" or self.elevated_timer > 0
        if obstacle.kind == "jump":
            return self.motion == "jump"
        if obstacle.kind == "roll":
            return self.motion == "roll"
        if obstacle.kind == "either":
            return self.motion in ("jump", "roll")
        if obstacle.kind == "solid":
            high_jump = self.active_powerups["pogo"] > 0 or self.active_powerups[
                "super_sneakers"
            ] > 0
            return obstacle.roofable and (
                self.elevated_timer > 0 or (self.motion == "jump" and high_jump)
            )
        return False

    def _can_collect_pickup(self, pickup: Pickup) -> bool:
        if pickup.kind in ("coin", "key") and self.active_powerups["coin_magnet"] > 0:
            return True
        return pickup.lane == self.player_lane

    def _collect_pickups(self) -> tuple[float, list[str]]:
        reward = 0.0
        collected: list[str] = []
        remaining: list[Pickup] = []
        for pickup in self._pickups:
            if pickup.distance > 0:
                remaining.append(pickup)
                continue
            if not self._can_collect_pickup(pickup):
                continue

            if pickup.kind == "coin":
                self.coins += 1
                reward += 0.05
                collected.append("collected_coin")
            elif pickup.kind == "key":
                self.keys += 1
                reward += 0.10
                collected.append("collected_key")
            else:
                activated = self._activate_powerup(pickup.kind)
                reward += 0.20
                collected.append(f"collected_{activated}")
        self._pickups = remaining
        return reward, collected

    def _activate_powerup(self, kind: str) -> str:
        if kind == "mystery_box":
            kind = self.rng.choice(
                ("jetpack", "super_sneakers", "coin_magnet", "multiplier", "pogo")
            )

        durations = {
            "jetpack": 16,
            "super_sneakers": 20,
            "coin_magnet": 20,
            "multiplier": 20,
            "pogo": 20,
            "speed_pad": 10,
        }
        if kind not in durations:
            raise ValueError(f"unknown power-up {kind}")
        self.active_powerups[kind] = max(self.active_powerups[kind], durations[kind])
        if kind == "jetpack":
            self.motion = "jetpack"
            self.motion_timer = durations[kind]
        return kind

    def _tick_rich_state(self) -> None:
        for kind in ACTIVE_POWERUPS:
            if self.active_powerups[kind] > 0:
                self.active_powerups[kind] -= 1
        if self.hoverboard_timer > 0:
            self.hoverboard_timer -= 1
        if self.hoverboard_timer == 0:
            self.hoverboard_active = False
        if self.elevated_timer > 0:
            self.elevated_timer -= 1
        if self.motion_timer > 0:
            self.motion_timer -= 1

        if self.active_powerups["jetpack"] > 0:
            self.motion = "jetpack"
        elif self.motion == "jetpack" or self.motion_timer == 0:
            self.motion = "ground"
        self.speed = self.base_speed * (
            1.5 if self.active_powerups["speed_pad"] > 0 else 1.0
        )

    def _spawn_rich_pattern(self) -> None:
        # Layer 1 teaches the core timing loop. Layer 2 then adds route
        # changes without changing the action or observation contract.
        if self.rich_layer == 1:
            pattern = "single"
        else:
            weights = RICH_LEVEL_PROFILES[self.active_level_variant][
                "pattern_weights"
            ]
            pattern = self.rng.choices(
                ("single", "lane_change", "long_train", "oncoming", "combo"),
                weights=weights,
                k=1,
            )[0]
        lane = self.rng.randrange(LANE_COUNT)

        if pattern == "long_train":
            length = self.rng.randint(2, 4)
            for segment in range(length):
                self.add_obstacle(
                    lane,
                    "solid",
                    self.spawn_distance + segment * 0.9,
                    appearance="train",
                    roofable=True,
                )
            # Smaller distance means the ramp is encountered before the front
            # of the train. The elevated timer then carries across the roof.
            self.add_obstacle(
                lane,
                "roof_route",
                max(0.5, self.spawn_distance - 0.8),
                appearance="ramp",
            )
        elif pattern == "oncoming":
            length = self.rng.randint(1, 2)
            for segment in range(length):
                self.add_obstacle(
                    lane,
                    "solid",
                    self.spawn_distance + segment * 0.8,
                    appearance="train",
                    speed_scale=1.8,
                )
        elif pattern == "lane_change":
            blocked_lanes = self.rng.sample(range(LANE_COUNT), 2)
            for blocked_lane in blocked_lanes:
                obstacle_kind = self.rng.choice(
                    ("solid", "jump", "roll", "either", "gap")
                )
                self.add_obstacle(blocked_lane, obstacle_kind)
        elif pattern == "combo":
            self.add_obstacle(lane, self.rng.choice(("jump", "roll", "either")))
            second_kind = self.rng.choice(("jump", "roll", "gap"))
            self.add_obstacle(lane, second_kind, self.spawn_distance + 1.2)
        else:
            obstacle_kinds = ("solid", "jump", "roll", "either")
            if self.rich_layer == 2:
                obstacle_kinds += ("gap",)
            obstacle_kind = self.rng.choice(obstacle_kinds)
            self.add_obstacle(lane, obstacle_kind)

        if self.rng.random() < 0.65:
            self.add_pickup(
                self.rng.randrange(LANE_COUNT),
                "coin",
                self.spawn_distance + 0.2,
            )
        if self.rng.random() < 0.18:
            pickup_kind = self.rng.choice(PICKUP_KINDS[2:])
            self.add_pickup(
                self.rng.randrange(LANE_COUNT),
                pickup_kind,
                self.spawn_distance + 0.4,
            )

    def _rich_observation(self) -> dict[str, Any]:
        return {
            "level_variant": self.active_level_variant,
            "lane": self.player_lane,
            "motion": self.motion,
            "motion_timer": self.motion_timer,
            "speed": self.speed,
            "step": self.step_count,
            "score": self.score,
            "coins": self.coins,
            "keys": self.keys,
            "hoverboard_active": self.hoverboard_active,
            "hoverboard_charges": self.hoverboard_charges,
            "elevated_timer": self.elevated_timer,
            "active_powerups": dict(self.active_powerups),
            "obstacles": [
                {
                    "lane": obstacle.lane,
                    "kind": obstacle.kind,
                    "distance": round(obstacle.distance, 3),
                    "appearance": obstacle.appearance,
                    "speed_scale": round(obstacle.speed_scale, 3),
                    "roofable": obstacle.roofable,
                }
                for obstacle in sorted(self._obstacles, key=lambda item: item.distance)
            ],
            "pickups": [
                {
                    "lane": pickup.lane,
                    "kind": pickup.kind,
                    "distance": round(pickup.distance, 3),
                }
                for pickup in sorted(self._pickups, key=lambda item: item.distance)
            ],
        }

    def _rich_render(self) -> str:
        powerups = ",".join(
            f"{kind}:{duration}"
            for kind, duration in self.active_powerups.items()
            if duration > 0
        ) or "none"
        return (
            f"step={self.step_count:03d} speed={self.speed:.1f} "
            f"score={self.score:.1f} coins={self.coins} "
            f"lane={self.player_lane} motion={self.motion}\n"
            f"powerups: {powerups}  board={self.hoverboard_active} "
            f"charges={self.hoverboard_charges}\n"
            f"obstacles: {[(item['lane'], item['kind'], item['distance']) for item in self._rich_observation()['obstacles']]}\n"
            f"pickups: {[(item['lane'], item['kind'], item['distance']) for item in self._rich_observation()['pickups']]}"
        )

    def _rich_render_rgb(
        self,
        width: int = PIXEL_WIDTH,
        height: int = PIXEL_HEIGHT,
        show_obstacles: bool = True,
    ) -> bytes:
        # Keep the 84x84 contract, but give each object a recognizable
        # silhouette. The first version used floating rectangles, which made
        # trains, pickups, and barriers look almost identical to a human.
        pixels = bytearray((8, 14, 25) * (width * height))

        def fill(
            left: int,
            top: int,
            right: int,
            bottom: int,
            color: tuple[int, int, int],
        ) -> None:
            left = max(0, min(width, left))
            top = max(0, min(height, top))
            right = max(0, min(width, right))
            bottom = max(0, min(height, bottom))
            if right <= left or bottom <= top:
                return
            row = bytes(color) * (right - left)
            for y in range(top, bottom):
                start = (y * width + left) * 3
                pixels[start : start + len(row)] = row

        def disk(center_x: int, center_y: int, radius: int, color: tuple[int, int, int]) -> None:
            for y in range(center_y - radius, center_y + radius + 1):
                for x in range(center_x - radius, center_x + radius + 1):
                    if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2:
                        fill(x, y, x + 1, y + 1, color)

        def track_x(lane: int, progress: float) -> int:
            progress = min(max(progress, 0.0), 1.0)
            # Far lanes converge near the horizon and spread toward the
            # runner, making lane ownership visible at a glance.
            return int(42 + (lane - 1) * (6 + 21 * progress))

        def draw_train(
            center: int,
            base_y: int,
            half_width: int,
            progress: float,
            color: tuple[int, int, int],
            oncoming: bool = False,
        ) -> None:
            left, right = center - half_width, center + half_width + 1
            top = base_y - int(11 + progress * 15)
            fill(left + 1, top + 2, right - 1, base_y + 1, color)
            fill(
                left,
                top,
                right,
                top + 3,
                (218, 224, 232) if not oncoming else (255, 111, 91),
            )
            window = (38, 48, 61) if not oncoming else (89, 35, 42)
            window_top = top + int(5 + progress * 2)
            fill(
                left + 2,
                window_top,
                right - 2,
                window_top + max(2, int(3 * progress)),
                window,
            )
            if oncoming:
                light_radius = max(1, int(progress * 2))
                disk(left + 3, base_y - 3, light_radius, (255, 220, 91))
                disk(right - 4, base_y - 3, light_radius, (255, 220, 91))
            else:
                fill(left + 2, base_y - 2, right - 2, base_y, (55, 64, 78))

        def draw_pickup(center: int, y: int, kind: str, color: tuple[int, int, int]) -> None:
            if kind == "coin":
                disk(center, y, 3, color)
                fill(center, y - 2, center + 1, y + 3, (255, 242, 137))
            elif kind == "key":
                disk(center - 2, y - 2, 2, color)
                fill(center, y - 1, center + 4, y + 1, color)
                fill(center + 2, y, center + 3, y + 3, color)
            elif kind == "jetpack":
                fill(center - 3, y - 4, center + 3, y + 3, color)
                fill(center - 5, y - 2, center - 3, y + 2, color)
                fill(center - 2, y + 3, center, y + 5, (255, 166, 65))
                fill(center + 1, y + 3, center + 3, y + 5, (255, 220, 91))
            elif kind == "super_sneakers":
                fill(center - 5, y - 1, center - 1, y + 3, color)
                fill(center - 4, y + 2, center + 1, y + 4, color)
                fill(center + 1, y - 3, center + 5, y + 2, color)
                fill(center + 1, y + 1, center + 6, y + 3, color)
            elif kind == "coin_magnet":
                fill(center - 4, y - 3, center - 2, y + 3, color)
                fill(center + 2, y - 3, center + 4, y + 3, color)
                fill(center - 2, y + 1, center + 2, y + 3, color)
            elif kind == "multiplier":
                fill(center - 4, y - 4, center + 4, y + 4, color)
                fill(center - 2, y - 2, center, y, (245, 238, 255))
                fill(center + 1, y + 1, center + 3, y + 3, (245, 238, 255))
            elif kind == "pogo":
                fill(center - 1, y - 5, center + 1, y + 3, color)
                fill(center - 4, y + 3, center + 4, y + 5, color)
                fill(center - 3, y - 5, center + 3, y - 3, color)
            elif kind == "speed_pad":
                fill(center - 5, y - 2, center + 5, y + 3, color)
                fill(center - 2, y - 4, center + 3, y - 2, (255, 224, 120))
            else:  # mystery_box
                fill(center - 4, y - 4, center + 4, y + 4, color)
                fill(center - 1, y - 2, center + 2, y, (111, 82, 160))
                fill(center, y + 2, center + 1, y + 3, (111, 82, 160))

        # Sky, a small skyline, and a trapezoidal track.
        fill(0, 0, width, 24, (11, 18, 31))
        for x, building_height in ((4, 5), (13, 8), (23, 4), (62, 6), (73, 9)):
            fill(x, 24 - building_height, x + 6, 24, (17, 28, 44))
        fill(0, 24, width, 25, (91, 105, 125))
        for y in range(25, height):
            progress = (y - 25) / max(1, height - 25)
            left = int(42 - (18 + 24 * progress))
            right = int(42 + (18 + 24 * progress))
            fill(left, y, right + 1, y + 1, (45, 54, 73))
            if y % 9 in (0, 1):
                fill(left, y, right + 1, y + 1, (55, 64, 82))

        # Track edges and dashed lane dividers show the three actual routes.
        for y in range(25, height):
            progress = (y - 25) / max(1, height - 25)
            left_edge = int(42 - (18 + 24 * progress))
            right_edge = int(42 + (18 + 24 * progress))
            fill(left_edge, y, left_edge + 1, y + 1, (112, 124, 143))
            fill(right_edge, y, right_edge + 1, y + 1, (112, 124, 143))
            if y % 7 < 4:
                for divider in (0, 1):
                    divider_x = int(
                        (track_x(divider, progress) + track_x(divider + 1, progress)) / 2
                    )
                    fill(divider_x, y, divider_x + 1, y + 1, (103, 113, 132))

        obstacle_colors = {
            "block": (224, 67, 71),
            "barrier": (242, 177, 58),
            "roll_bar": (111, 177, 238),
            "either_hurdle": (193, 103, 224),
            "train": (156, 163, 175),
            "bus": (171, 127, 220),
            "gap": (12, 16, 24),
            "tunnel": (28, 33, 46),
            "ramp": (75, 214, 139),
        }
        if show_obstacles:
            for obstacle in sorted(
                self._obstacles, key=lambda item: item.distance, reverse=True
            ):
                if not 0 < obstacle.distance <= self.spawn_distance + 1.0:
                    continue
                progress = 1.0 - obstacle.distance / (self.spawn_distance + 1.0)
                center = track_x(obstacle.lane, progress)
                contact_y = int(30 + progress * 43)
                half_width = int(3 + progress * 6)
                visual = obstacle.appearance
                color = obstacle_colors[visual]
                if obstacle.kind == "solid" and visual in ("train", "bus"):
                    if obstacle.speed_scale > 1.0:
                        color = (227, 87, 86)
                    draw_train(
                        center,
                        contact_y,
                        half_width,
                        progress,
                        color,
                        oncoming=obstacle.speed_scale > 1.0,
                    )
                elif obstacle.kind == "solid":
                    top = contact_y - int(7 + progress * 10)
                    fill(center - half_width, top, center + half_width + 1, contact_y + 1, color)
                    fill(center - half_width, top + 2, center + half_width + 1, top + 4, (255, 207, 82))
                    fill(center - 1, top + 5, center + 1, contact_y, (161, 42, 48))
                elif obstacle.kind == "jump":
                    bar_y = contact_y - int(4 + progress * 3)
                    fill(center - half_width, bar_y, center + half_width + 1, bar_y + int(3 + progress * 2), color)
                    fill(center - half_width + 1, bar_y + 1, center + half_width, bar_y + 2, (255, 224, 120))
                    fill(center - half_width + 1, bar_y + 3, center - half_width + 3, contact_y + 1, (116, 78, 43))
                    fill(center + half_width - 2, bar_y + 3, center + half_width, contact_y + 1, (116, 78, 43))
                elif obstacle.kind == "roll" and visual == "tunnel":
                    top = contact_y - int(17 + progress * 5)
                    fill(center - half_width, top, center + half_width + 1, top + 4, color)
                    fill(center - half_width, top + 4, center - half_width + 2, contact_y, color)
                    fill(center + half_width - 1, top + 4, center + half_width + 1, contact_y, color)
                elif obstacle.kind == "roll":
                    top = contact_y - int(16 + progress * 6)
                    fill(center - half_width, top, center + half_width + 1, top + int(3 + progress * 2), color)
                    fill(center - half_width + 1, top + 1, center + half_width, top + 2, (198, 228, 255))
                    fill(center - half_width, top + 3, center - half_width + 2, contact_y, (64, 100, 137))
                    fill(center + half_width - 1, top + 3, center + half_width + 1, contact_y, (64, 100, 137))
                elif obstacle.kind == "either":
                    top = contact_y - int(12 + progress * 6)
                    fill(center - half_width, top, center + half_width + 1, top + 3, color)
                    fill(center - half_width, contact_y - 3, center + half_width + 1, contact_y, color)
                    fill(center - 1, top + 3, center + 1, contact_y - 3, color)
                elif obstacle.kind == "gap":
                    fill(center - half_width - 2, contact_y - 1, center + half_width + 3, contact_y + 3, color)
                    fill(center - half_width - 2, contact_y - 2, center - half_width, contact_y + 1, (245, 201, 81))
                    fill(center + half_width + 1, contact_y - 2, center + half_width + 3, contact_y + 1, (245, 201, 81))
                elif obstacle.kind == "roof_route":
                    for row in range(8):
                        fill(
                            center - half_width + row // 2,
                            contact_y - row,
                            center + half_width + 1,
                            contact_y - row + 2,
                            color,
                        )

        pickup_colors = {
            "coin": (247, 211, 66),
            "key": (255, 235, 150),
            "jetpack": (89, 211, 255),
            "super_sneakers": (255, 119, 188),
            "coin_magnet": (255, 136, 68),
            "multiplier": (181, 128, 255),
            "pogo": (100, 235, 145),
            "speed_pad": (255, 116, 64),
            "mystery_box": (238, 238, 246),
        }
        for pickup in self._pickups:
            if not 0 < pickup.distance <= self.spawn_distance + 1.0:
                continue
            progress = 1.0 - pickup.distance / (self.spawn_distance + 1.0)
            center = track_x(pickup.lane, progress)
            y = int(31 + progress * 32)
            color = pickup_colors[pickup.kind]
            draw_pickup(center, y, pickup.kind, color)

        # Keep status bars in the header so they do not obscure the game
        # geometry.
        fill(3, 3, 25, 6, (25, 36, 52))
        fill(4, 4, 4 + int(20 * min(self.speed / 2.0, 1.0)), 5, (100, 230, 176))
        for index, kind in enumerate(ACTIVE_POWERUPS):
            if self.active_powerups[kind] > 0:
                fill(2 + index * 12, 2, 11 + index * 12, 6, pickup_colors[kind])
        if self.hoverboard_active:
            disk(78, 4, 3, (84, 225, 255))

        player_center = track_x(self.player_lane, 1.0)
        if self.motion == "jetpack":
            player_top, player_bottom = 42, 54
        elif self.elevated_timer > 0:
            player_top, player_bottom = 54, 67
        elif self.motion == "jump":
            player_top, player_bottom = 53, 66
        elif self.motion == "roll":
            player_top, player_bottom = 70, 77
        else:
            player_top, player_bottom = 64, 77
        fill(player_center - 5, player_bottom, player_center + 6, player_bottom + 2, (25, 31, 43))
        if self.motion == "roll":
            fill(player_center - 6, player_top + 3, player_center + 6, player_bottom, (75, 214, 139))
            disk(player_center + 2, player_top + 2, 3, (150, 235, 175))
        else:
            fill(player_center - 4, player_top + 4, player_center + 5, player_bottom, (75, 214, 139))
            disk(player_center, player_top + 2, 3, (150, 235, 175))
            fill(player_center - 6, player_top + 6, player_center - 4, player_top + 10, (75, 214, 139))
            fill(player_center + 5, player_top + 6, player_center + 7, player_top + 10, (75, 214, 139))
            fill(player_center - 3, player_bottom, player_center - 1, player_bottom + 2, (75, 214, 139))
            fill(player_center + 2, player_bottom, player_center + 4, player_bottom + 2, (75, 214, 139))
        if self.motion == "jetpack":
            fill(player_center - 6, player_bottom, player_center - 3, player_bottom + 4, (89, 211, 255))
            fill(player_center + 3, player_bottom, player_center + 6, player_bottom + 4, (255, 166, 65))
        if self.hoverboard_active:
            fill(player_center - 6, player_bottom, player_center + 7, player_bottom + 2, (84, 225, 255))
        return bytes(pixels)

    def _spawn_random_obstacle(self) -> None:
        if self.rich_mechanics:
            self._spawn_rich_pattern()
            return
        self.add_obstacle(
            lane=self.rng.randrange(LANE_COUNT),
            kind=self.rng.choice(OBSTACLE_KINDS),
        )

    def _observation(self) -> dict[str, Any]:
        if self.rich_mechanics:
            return self._rich_observation()
        obstacles = sorted(self._obstacles, key=lambda item: item.distance)
        return {
            "lane": self.player_lane,
            "motion": self.motion,
            "motion_timer": self.motion_timer,
            "speed": self.speed,
            "step": self.step_count,
            "obstacles": [
                {
                    "lane": obstacle.lane,
                    "kind": obstacle.kind,
                    "distance": round(obstacle.distance, 3),
                }
                for obstacle in obstacles
            ],
        }


def encode_observation(observation: dict[str, Any], spawn_distance: float = 6.0) -> list[float]:
    """Turn the readable observation into a fixed-size numeric vector.

    The vector is intentionally boring and inspectable. Each obstacle value is
    an urgency score: zero means no nearby obstacle of that type, one means it
    is at the player. This is the input that PPO will receive before pixels.
    """

    if spawn_distance <= 0:
        raise ValueError("spawn_distance must be positive")

    values: list[float] = []
    player_lane = int(observation["lane"])
    motion = observation["motion"]

    values.extend(1.0 if player_lane == lane else 0.0 for lane in range(LANE_COUNT))
    values.extend(1.0 if motion == state else 0.0 for state in MOTION_STATES)
    values.append(
        min(max(float(observation["motion_timer"]) / MOTION_DURATION, 0.0), 1.0)
    )
    values.append(min(max(float(observation["speed"]) / spawn_distance, 0.0), 1.0))

    obstacles = observation["obstacles"]
    for lane in range(LANE_COUNT):
        for kind in OBSTACLE_KINDS:
            distances = [
                float(item["distance"])
                for item in obstacles
                if item["lane"] == lane and item["kind"] == kind
            ]
            if not distances:
                values.append(0.0)
                continue
            nearest = min(distances)
            urgency = 1.0 - (nearest / spawn_distance)
            values.append(min(max(urgency, 0.0), 1.0))

    assert len(values) == OBSERVATION_SIZE
    return values


def encode_rich_observation(
    observation: dict[str, Any], spawn_distance: float = 6.0
) -> list[float]:
    """Encode rich mode state while keeping every value in ``[0, 1]``."""

    if spawn_distance <= 0:
        raise ValueError("spawn_distance must be positive")

    values: list[float] = []
    player_lane = int(observation["lane"])
    motion = observation["motion"]
    values.extend(1.0 if player_lane == lane else 0.0 for lane in range(LANE_COUNT))
    values.extend(1.0 if motion == state else 0.0 for state in RICH_MOTION_STATES)
    values.append(min(max(float(observation["motion_timer"]) / 16.0, 0.0), 1.0))
    values.append(min(max(float(observation["speed"]) / 2.0, 0.0), 1.0))
    values.append(1.0 if observation["hoverboard_active"] else 0.0)
    values.append(min(max(float(observation["hoverboard_charges"]) / 3.0, 0.0), 1.0))
    values.append(min(max(float(observation["elevated_timer"]) / 6.0, 0.0), 1.0))

    active_powerups = observation["active_powerups"]
    for kind in ACTIVE_POWERUPS:
        duration = float(active_powerups.get(kind, 0))
        values.append(1.0 if duration > 0 else 0.0)
        values.append(min(max(duration / 30.0, 0.0), 1.0))

    obstacles = observation["obstacles"]
    for lane in range(LANE_COUNT):
        for kind in RICH_OBSTACLE_KINDS:
            distances = [
                float(item["distance"])
                for item in obstacles
                if item["lane"] == lane and item["kind"] == kind
            ]
            if not distances:
                values.append(0.0)
                continue
            urgency = 1.0 - min(distances) / spawn_distance
            values.append(min(max(urgency, 0.0), 1.0))

    pickups = observation["pickups"]
    for lane in range(LANE_COUNT):
        for kind in PICKUP_KINDS:
            distances = [
                float(item["distance"])
                for item in pickups
                if item["lane"] == lane and item["kind"] == kind
            ]
            if not distances:
                values.append(0.0)
                continue
            urgency = 1.0 - min(distances) / spawn_distance
            values.append(min(max(urgency, 0.0), 1.0))

    assert len(values) == RICH_OBSERVATION_SIZE
    return values
