#!/usr/bin/env python3
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from torchrl.envs import GymWrapper, default_info_dict_reader


TOY_PRECISION_GATE_INFO_KEYS = [
    "success",
    "inside_gate",
    "gate_region",
    "collision",
    "goal_reached",
    "distance_to_goal",
    "distance_to_gate",
    "crossed_gate",
    "y_error_at_gate",
    "progress",
    "unscaled_reward",
]


class PrecisionGateEnv(gym.Env):
    """2D point-mass task with a narrow precision bottleneck.

    The agent starts on the left side of a wall and must cross at x=0 through a
    narrow gate before reaching a goal on the right. This gives a controlled
    environment where task success depends on preserving precise, long-horizon
    information about the y-coordinate.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        gate_width: float = 0.08,
        action_scale: float = 0.08,
        goal_threshold: float = 0.08,
        spawn_y_noise: float = 0.35,
        collision_terminates: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.gate_x = 0.0
        self.gate_width = float(gate_width)
        self.action_scale = float(action_scale)
        self.goal_threshold = float(goal_threshold)
        self.spawn_y_noise = float(spawn_y_noise)
        self.collision_terminates = bool(collision_terminates)
        self.goal = np.array([1.0, 0.0], dtype=np.float32)
        self.start_x = -1.0
        self.bounds = np.array([[-1.15, 1.15], [-1.0, 1.0]], dtype=np.float32)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-1.15, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.15, 1.0, 1.0, 0.0, 0.5, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self.pos = np.zeros(2, dtype=np.float32)
        self.crossed_gate = False
        self.collided = False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        y = self._rng.uniform(-self.spawn_y_noise, self.spawn_y_noise)
        self.pos = np.array([self.start_x, y], dtype=np.float32)
        self.crossed_gate = False
        self.collided = False
        return self._obs(), self._info(reward=0.0, progress=0.0)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        prev_pos = self.pos.copy()
        prev_distance = self._distance_to_goal(prev_pos)
        self.pos = np.clip(
            self.pos + self.action_scale * action,
            self.bounds[:, 0],
            self.bounds[:, 1],
        ).astype(np.float32)

        crossed_now = prev_pos[0] < self.gate_x <= self.pos[0]
        y_at_gate = float(self.pos[1])
        if crossed_now:
            step_dx = float(self.pos[0] - prev_pos[0])
            if abs(step_dx) > 1e-12:
                gate_fraction = float((self.gate_x - prev_pos[0]) / step_dx)
                y_at_gate = float(
                    prev_pos[1] + gate_fraction * float(self.pos[1] - prev_pos[1])
                )
        inside_gate = abs(y_at_gate) <= self.gate_width
        collision = crossed_now and not inside_gate
        if crossed_now and inside_gate:
            self.crossed_gate = True
        if collision:
            self.collided = True
            self.pos[0] = np.nextafter(np.float32(self.gate_x), np.float32(-1.0))

        distance = self._distance_to_goal(self.pos)
        progress = prev_distance - distance
        goal_reached = self.crossed_gate and distance <= self.goal_threshold

        reward = 4.0 * progress - 0.01 * float(np.square(action).sum())
        reward += 0.05 * float(inside_gate and self._in_gate_region())
        reward += 2.0 * float(goal_reached)
        reward -= 1.0 * float(collision)

        terminated = bool(goal_reached or (collision and self.collision_terminates))
        truncated = False
        return self._obs(), float(reward), terminated, truncated, self._info(
            reward=reward,
            progress=progress,
            collision=collision,
            goal_reached=goal_reached,
        )

    def _obs(self):
        return np.array(
            [
                self.pos[0],
                self.pos[1],
                self.goal[0],
                self.goal[1],
                self.gate_width,
                float(self.crossed_gate),
                float(self.collided),
            ],
            dtype=np.float32,
        )

    def _info(
        self,
        reward: float,
        progress: float,
        collision: bool = False,
        goal_reached: bool = False,
    ):
        distance_to_goal = self._distance_to_goal(self.pos)
        distance_to_gate = abs(float(self.pos[0] - self.gate_x))
        inside_gate = abs(float(self.pos[1])) <= self.gate_width
        gate_region = self._in_gate_region()
        y_error_at_gate = abs(float(self.pos[1])) if gate_region else 0.0
        success = bool(goal_reached)
        return {
            "success": float(success),
            "inside_gate": float(inside_gate),
            "gate_region": float(gate_region),
            "collision": float(collision or self.collided),
            "goal_reached": float(goal_reached),
            "distance_to_goal": float(distance_to_goal),
            "distance_to_gate": float(distance_to_gate),
            "crossed_gate": float(self.crossed_gate),
            "y_error_at_gate": float(y_error_at_gate),
            "progress": float(progress),
            "unscaled_reward": float(reward),
        }

    def _distance_to_goal(self, pos):
        return float(np.linalg.norm(np.asarray(pos, dtype=np.float32) - self.goal))

    def _in_gate_region(self):
        return abs(float(self.pos[0] - self.gate_x)) <= 0.08

    def render(self):
        canvas = np.ones((128, 128, 3), dtype=np.uint8) * 255
        x_px = int(np.interp(self.pos[0], self.bounds[0], [8, 119]))
        y_px = int(np.interp(self.pos[1], self.bounds[1], [119, 8]))
        gate_x_px = int(np.interp(self.gate_x, self.bounds[0], [8, 119]))
        gate_y_low = int(np.interp(-self.gate_width, self.bounds[1], [119, 8]))
        gate_y_high = int(np.interp(self.gate_width, self.bounds[1], [119, 8]))
        canvas[:, gate_x_px : gate_x_px + 2] = np.array([40, 40, 40], dtype=np.uint8)
        canvas[gate_y_high:gate_y_low, gate_x_px : gate_x_px + 2] = 255
        canvas[max(0, y_px - 2) : y_px + 3, max(0, x_px - 2) : x_px + 3] = np.array(
            [30, 90, 220], dtype=np.uint8
        )
        return canvas


def make_env(
    seed: int = 42,
    frame_skip: int = 1,
    device: str = "cpu",
    max_episode_steps: Optional[int] = None,
    gate_width: float = 0.08,
    action_scale: float = 0.08,
    goal_threshold: float = 0.08,
    spawn_y_noise: float = 0.35,
    collision_terminates: bool = True,
):
    if max_episode_steps is None:
        max_episode_steps = 80
    env = PrecisionGateEnv(
        gate_width=gate_width,
        action_scale=action_scale,
        goal_threshold=goal_threshold,
        spawn_y_noise=spawn_y_noise,
        collision_terminates=collision_terminates,
        seed=seed,
    )
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    env = GymWrapper(env=env, frame_skip=frame_skip, device=device)
    return env.set_info_dict_reader(
        info_dict_reader=default_info_dict_reader(TOY_PRECISION_GATE_INFO_KEYS)
    )
