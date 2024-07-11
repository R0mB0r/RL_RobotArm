import os
import numpy as np
from typing import Any, SupportsFloat
from xarm6_mujoco.envs.xarm6_env import Xarm6
import pygame


# Path to the MuJoCo model file for the reach environment
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
        """
        Initialize the Xarm6 Reach environment.

        Parameters:
        - distance_threshold (float): Distance within which the goal is considered achieved.
        - goal_xy_range (float): Range for the goal position in the XY plane.
        - goal_x_offset (float): Offset in the X direction for goal positioning.
        - goal_z_range (float): Range for the goal position in the Z direction.
        - kwargs: Additional arguments passed to the parent class.
        """
        self.model_path = MODEL_XML_PATH
        self.distance_threshold = distance_threshold
        self.goal_xy_range = goal_xy_range
        self.goal_x_offset = goal_x_offset
        self.goal_z_range = goal_z_range

        # Define goal positioning bounds
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

        # Initialize parent class with model path and environment settings
        super().__init__(
            model_path=self.model_path,
            n_substeps=20,
            block_gripper=True,
            **kwargs,
        )

        self.nu = self.model.nu  # Number of actuators
        self.nq = self.model.nq  # Number of positions
        self.nv = self.model.nv  # Number of velocities
        self.ctrl_range = self.model.actuator_ctrlrange  # Control range for actuators


    def _env_setup(self, neutral_joint_values: np.ndarray) -> None:
        """
        Set up the environment to a neutral pose.

        Parameters:
        - neutral_joint_values (np.ndarray): Neutral positions for the robot joints.
        """
        self.set_joint_neutral()
        self.data.ctrl[0:7] = neutral_joint_values[0:7]
        self._mujoco.mj_forward(self.model, self.data)
        self.grasp_site_pose = self.get_ee_orientation().copy()
        self._mujoco_step()

    def step(self, action: np.ndarray) -> tuple:
        """
        Execute a step in the environment.

        Parameters:
        - action (np.ndarray): The action to be applied.

        Returns:
        - tuple: Observation, reward, termination flag, truncation flag, and additional info.
        """

        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        self._set_action(action)
        self._mujoco_step(action)
        self._step_callback()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()
        info = {"is_success": self._is_success(obs["achieved_goal"], self.goal)}
        terminated = bool(info["is_success"])
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        return obs, reward, terminated, truncated, info

    def _is_success(self, achieved_position: np.ndarray, desired_goal: np.ndarray) -> np.float32:
        """
        Determine if the goal is achieved.

        Parameters:
        - achieved_position (np.ndarray): Current position of the end-effector.
        - desired_goal (np.ndarray): Target goal position.

        Returns:
        - np.float32: Indicates if the goal is achieved (1) or not (0).
        """
        d = self.goal_distance(achieved_position, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _get_obs(self) -> dict:
        """
        Get the current observation of the environment.

        Returns:
        - dict: Current state observation including position and velocity of the end-effector.
        """
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_velocity = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt

        return {
            "observation": np.concatenate([ee_position, ee_velocity]),
            "achieved_goal": ee_position,
            "desired_goal": self.goal,
        }

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict) -> SupportsFloat:
        """
        Calculate the reward for the current state.

        Parameters:
        - achieved_goal (np.ndarray): Position of the achieved goal.
        - desired_goal (np.ndarray): Target goal position.
        - info (dict): Additional information.

        Returns:
        - SupportsFloat: The computed reward based on distance to the goal.
        """
        d = self.goal_distance(achieved_goal, desired_goal)
        return -d

    def _render_callback(self) -> None:
        """
        Render the environment.

        Updates the target site position for visualization.
        """
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._model_names.site_name2id["target"]
        self.model.site_pos[site_id] = self.goal - sites_offset[site_id]
        self._mujoco.mj_forward(self.model, self.data)

    def _sample_goal(self) -> np.ndarray:
        """
        Sample a new goal position within the specified range.

        Returns:
        - np.ndarray: Sampled goal position.
        """
        goal = np.array([0.0, 0.0, 0.05])
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return goal
    
    def modify_goal_position(self, key):
        step_size = 0.01  # Ajustez cette valeur selon vos besoins
        current_goal = self.goal
        if pygame.K_w in key:    # Augmente la coordonnée z
            current_goal[2] += step_size
        elif pygame.K_s in key:  # Diminue la coordonnée z
            current_goal[2] -= step_size
        elif pygame.K_a in key:  # Diminue la coordonnée x
            current_goal[0] -= step_size
        elif pygame.K_d in key:  # Augmente la coordonnée x
            current_goal[0] += step_size
        elif pygame.K_q in key:  # Diminue la coordonnée y
            current_goal[1] -= step_size
        elif pygame.K_e in key:  # Augmente la coordonnée y
            current_goal[1] += step_size

        # Limiter la position du goal à l'intérieur de la plage spécifiée
        current_goal[0] = np.clip(current_goal[0], self.goal_range_low[0], self.goal_range_high[0])
        current_goal[1] = np.clip(current_goal[1], self.goal_range_low[1], self.goal_range_high[1])
        current_goal[2] = np.clip(current_goal[2], self.goal_range_low[2], self.goal_range_high[2])
        self.goal = current_goal

    def goal_distance(self, goal_a: np.ndarray, goal_b: np.ndarray) -> SupportsFloat:
        """
        Compute the distance between two goal positions.

        Parameters:
        - goal_a (np.ndarray): First goal position.
        - goal_b (np.ndarray): Second goal position.

        Returns:
        - SupportsFloat: Euclidean distance between the two goals.
        """
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b)


