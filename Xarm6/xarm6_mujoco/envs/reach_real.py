import os
import numpy as np
from typing import Any, SupportsFloat
from Xarm6.xarm6_mujoco.envs.xarm6_env_reel import Xarm6Real
l

class Xarm6ReachEnvReal(Xarm6Real):
    def __init__(
        self,
        distance_threshold: float = 0.005,
        max_episode_steps: int = 400,
        **kwargs: Any,
    ):
        self.distance_threshold = distance_threshold
        self.goal = np.array([0.6, 0.0, 0.0])
        self.max_episode_steps = max_episode_steps
        
        super().__init__(
            n_substeps=20,
            block_gripper=True,
            **kwargs,
        )

    def _env_setup(self) -> None:
        self.set_joint_neutral()
        

    def step(self, action: np.ndarray) -> tuple:
        self._set_action(action)

        observation = self._get_obs().copy()
        info = {"is_success": self._is_success(observation["achieved_goal"], self.goal)}
        terminated = bool(info["is_success"])
        truncated = self.compute_truncated(observation["achieved_goal"], self.goal, info)
        reward = self.compute_reward(observation["achieved_goal"], self.goal, info)

        return observation, reward, terminated, truncated, info

    def _is_success(self, achieved_position: np.ndarray, desired_goal: np.ndarray) -> np.float32:
        distance = self.goal_distance(achieved_position, desired_goal)
        return (distance < self.distance_threshold).astype(np.float32)

    def _get_obs(self) -> dict:
        ee_position = self.get_ee_position()
        ee_velocity = self.get_ee_speed()

        return {
            "observation": np.concatenate([ee_position, ee_velocity]),
            "achieved_goal": ee_position,
            "desired_goal": self.goal,
        }

    def compute_truncated(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> bool:
        return False

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> SupportsFloat:
        distance = self.goal_distance(achieved_goal, desired_goal)
        return -distance

    def goal_distance(self, goal_a: np.ndarray, goal_b: np.ndarray) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b)
