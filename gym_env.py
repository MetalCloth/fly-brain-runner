"""Gymnasium adapter for the dependency-free runner core."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env import (
    ACTIONS,
    RICH_OBSERVATION_SIZE,
    OBSERVATION_SIZE,
    PIXEL_HEIGHT,
    PIXEL_WIDTH,
    RunnerEnv,
    encode_rich_observation,
    encode_observation,
)


class GymRunnerEnv(gym.Env):
    """Expose ``RunnerEnv`` through Gymnasium's standard RL interface."""

    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(
        self, render_mode: str | None = None, pixels: bool = False, **runner_kwargs
    ) -> None:
        super().__init__()
        allowed_modes = (None, "rgb_array") if pixels else (None, "ansi")
        if render_mode not in allowed_modes:
            raise ValueError(f"render_mode must be one of {allowed_modes}")

        self.render_mode = render_mode
        self.pixels = pixels
        self.core = RunnerEnv(**runner_kwargs)
        self.rich_mechanics = self.core.rich_mechanics
        self.action_space = spaces.Discrete(len(self.core.actions))
        if pixels:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(PIXEL_HEIGHT, PIXEL_WIDTH, 3),
                dtype=np.uint8,
            )
        else:
            observation_size = (
                RICH_OBSERVATION_SIZE if self.rich_mechanics else OBSERVATION_SIZE
            )
            self.observation_space = spaces.Box(
                low=np.zeros(observation_size, dtype=np.float32),
                high=np.ones(observation_size, dtype=np.float32),
                dtype=np.float32,
            )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        observation, info = self.core.reset(seed=seed)
        return self._encode(observation), info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.core.step(int(action))
        return self._encode(observation), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self.core.render()
        if self.render_mode == "rgb_array":
            return self._pixels()
        return None

    def close(self) -> None:
        return None

    def _encode(self, observation: dict) -> np.ndarray:
        if self.pixels:
            return self._pixels()
        if self.rich_mechanics:
            return np.asarray(
                encode_rich_observation(
                    observation, spawn_distance=self.core.spawn_distance
                ),
                dtype=np.float32,
            )
        return np.asarray(
            encode_observation(observation, spawn_distance=self.core.spawn_distance),
            dtype=np.float32,
        )

    def _pixels(self) -> np.ndarray:
        return np.frombuffer(self.core.render_rgb(), dtype=np.uint8).reshape(
            (PIXEL_HEIGHT, PIXEL_WIDTH, 3)
        ).copy()


class PixelGymRunnerEnv(GymRunnerEnv):
    """Expose the same runner as an 84x84 RGB observation."""

    def __init__(self, render_mode: str | None = None, **runner_kwargs) -> None:
        super().__init__(render_mode=render_mode, pixels=True, **runner_kwargs)


class PartialPixelGymRunnerEnv(PixelGymRunnerEnv):
    """Hide obstacles on alternating frames so temporal memory can help."""

    def _pixels(self) -> np.ndarray:
        show_obstacles = self.core.step_count % 2 == 0
        return np.frombuffer(
            self.core.render_rgb(show_obstacles=show_obstacles), dtype=np.uint8
        ).reshape((PIXEL_HEIGHT, PIXEL_WIDTH, 3)).copy()


class RichGymRunnerEnv(GymRunnerEnv):
    """Gymnasium environment with the decision-relevant rich mechanics."""

    def __init__(self, render_mode: str | None = None, **runner_kwargs) -> None:
        runner_kwargs["rich_mechanics"] = True
        super().__init__(render_mode=render_mode, **runner_kwargs)


class RichPixelGymRunnerEnv(RichGymRunnerEnv):
    """Rich mechanics with an 84x84 RGB observation."""

    def __init__(self, render_mode: str | None = None, **runner_kwargs) -> None:
        super().__init__(render_mode=render_mode, pixels=True, **runner_kwargs)


class RichPartialPixelGymRunnerEnv(RichPixelGymRunnerEnv):
    """Rich pixel environment with alternating obstacle visibility."""

    def _pixels(self) -> np.ndarray:
        show_obstacles = self.core.step_count % 2 == 0
        return np.frombuffer(
            self.core.render_rgb(show_obstacles=show_obstacles), dtype=np.uint8
        ).reshape((PIXEL_HEIGHT, PIXEL_WIDTH, 3)).copy()
