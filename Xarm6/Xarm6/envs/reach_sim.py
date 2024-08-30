import os
import numpy as np
from typing import Any, SupportsFloat
from Xarm6.envs.xarm6_env_sim import Xarm6


MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "reach.xml")

class Xarm6ReachEnv(Xarm6):
    def __init__(
        self,
        distance_threshold: float = 0.01,
        goal_xy_range: float = 0.3,
        goal_x_offset: float = 0.0,
        goal_z_range: float = 0.3,
        **kwargs: Any,
    ):
        """
        Initialize the Panda Reach environment.

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

        self.success_reset = True
        self.fix = True

    
    def _initialize_simulation(self) -> None:
        """
        Initialize the MuJoCo simulation.

        Loads the MuJoCo model and sets up the initial simulation parameters.
        """
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)

        self._model_names = self._utils.MujocoModelNames(self.model)
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        self.arm_joint_names = self._model_names.joint_names[0:6]
        
        # Set initial joint positions
        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)
    
    
    def _env_setup(self, neutral_joint_values: np.ndarray) -> None:
        """
        Set up the environment to a neutral pose.

        Parameters:
        - neutral_joint_values (np.ndarray): Neutral positions for the robot joints.
        """
        self.set_joint_neutral()
        self.data.ctrl[0:6] = neutral_joint_values[0:6]
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
        
        self.is_reached = False
        obs = self._get_obs().copy()
        info = {"is_success": self._is_success(obs["achieved_goal"], self.goal)}
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)
        terminated = bool(info["is_success"])
        
        if terminated:
            self.is_reached = False


        if not self.success_reset:
            terminated = False

        if self.is_reached and self.fix:
            obs = self._get_obs().copy()
            info = {}
            reward = 0.0

        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            self._set_action(action)
            self._mujoco_step(action)
            self._step_callback()

            if self.render_mode == "human":
                self.render()

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
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_velocity = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt

        return {
            "observation": np.concatenate([ee_position, ee_velocity]),
            "achieved_goal": ee_position,
            "desired_goal": self.goal,
        }

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, log = True) -> SupportsFloat:
        """
        Calculate the reward for the current state.

        Parameters:
        - achieved_goal (np.ndarray): Position of the achieved goal.
        - desired_goal (np.ndarray): Target goal position.
        - log (bool): Whether to log the distance to a file.

        Returns:
        - SupportsFloat: The computed reward based on distance to the goal.
        """
        d = self.goal_distance(achieved_goal, desired_goal)

        if log :
            log_dir = '/home/yoshidalab/Documents/Romain/RL_RobotArm/Xarm6/Log/Mujoco_Simulation'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(os.path.join(log_dir, "distances_sim.txt"), "a") as file:
                file.write(f"{d}\n")

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
        goal_fixed = np.array([0.34, -0.30, 0.34])
        return goal_fixed  # Return a predefined goal position

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
        distance = np.linalg.norm(goal_a - goal_b, axis=-1)

        return distance
