import os
import numpy as np
from typing import Any, SupportsFloat
from xarm6_mujoco.envs.xarm6_env_sim import Xarm6


MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "reach.xml")

class Xarm6ReachEnv(Xarm6):
    def __init__(
        self,
        distance_threshold: float = 0.005,
        goal_xy_range: float = 0.3,
        goal_x_offset: float = 0.0,
        goal_z_range: float = 0.3,
        **kwargs: Any,
    ):
        self.model_path = MODEL_XML_PATH
        self.distance_threshold = distance_threshold
        self.goal_xy_range = goal_xy_range
        self.goal_x_offset = goal_x_offset
        self.goal_z_range = goal_z_range

        self.goal_range_low = np.array([
            -self.goal_xy_range / 2 + self.goal_x_offset, 
            -self.goal_xy_range / 2, 
            0
        ]) + np.array([0.6, 0, 0])
        
        self.goal_range_high = np.array([
            self.goal_xy_range / 2 + self.goal_x_offset, 
            self.goal_xy_range / 2, 
            self.goal_z_range
        ]) + np.array([0.6, 0, 0])

        super().__init__(
            model_path=self.model_path,
            n_substeps=20,
            block_gripper=True,
            **kwargs,
        )

        self.num_actuators = self.model.nu
        self.num_positions = self.model.nq
        self.num_velocities = self.model.nv
        self.control_range = self.model.actuator_ctrlrange

    def _env_setup(self, neutral_joint_values: np.ndarray) -> None:
        self.set_joint_neutral()
        self.data.ctrl[0:7] = neutral_joint_values[0:7]
        self._mujoco.mj_forward(self.model, self.data)
        self.grasp_site_pose = self.get_ee_orientation().copy()
        self._mujoco_step()

    def step(self, action: np.ndarray) -> tuple:
        if action.shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        self._set_action(action)
        self._mujoco_step(action)
        self._step_callback()

        if self.render_mode == "human":
            self.render()

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
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_velocity = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt

        return {
            "observation": np.concatenate([ee_position, ee_velocity]),
            "achieved_goal": ee_position,
            "desired_goal": self.goal,
        }

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> SupportsFloat:
        distance = self.goal_distance(achieved_goal, desired_goal)
        return -distance

    def _render_callback(self) -> None:
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._model_names.site_name2id["target"]
        self.model.site_pos[site_id] = self.goal - sites_offset[site_id]
        self._mujoco.mj_forward(self.model, self.data)

    def _sample_goal(self) -> np.ndarray:
        goal = np.array([0.0, 0.0, 0.05])
        noise = np.random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        goal_fixed = np.array([0.34, -0.3, 0.34])
        return goal_fixed

    def goal_distance(self, goal_a: np.ndarray, goal_b: np.ndarray) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b)
