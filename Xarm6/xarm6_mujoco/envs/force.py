import os
import numpy as np
from typing import Any
from xarm6_mujoco.envs.xarm6_env import Xarm6
import mujoco as mj
import matplotlib.pyplot as plt

def get_sensor_data(data, sensor_name: str) -> np.ndarray:
    """
    Get the sensor data from the environment.

    Parameters:
    - data: The MuJoCo data object.
    - sensor_name: The name of the sensor.

    Returns:
    - np.ndarray: The sensor data from the environment.
    """
    return data.sensor("contact_force").data

def get_force_sensor_data(data) -> np.ndarray:
    """
    Get the force sensor data from the environment.

    Parameters:
    - data: The MuJoCo data object.

    Returns:
    - np.ndarray: The force sensor data from the environment.
    """
    return get_sensor_data(data, "contact_force")

# Path to the MuJoCo model file for the reach environment
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "force.xml")

class Xarm6ForceEnv(Xarm6):
    def __init__(self, distance_threshold: float = 0.05, goal_force=np.array([50,0,0]), **kwargs: Any):
        self.model_path = MODEL_XML_PATH
        self.distance_threshold = distance_threshold
        self.goal_force = goal_force
        
        # Initialize parent class with model path and environment settings
        super().__init__(
            model_path=self.model_path,
            n_substeps=20,
            block_gripper=True,
            **kwargs,
        )

        self.nu = self.model.nu  # Number of actuators
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.ctrl_range = self.model.actuator_ctrlrange

        # Initialize variables for real-time plotting
        self.fig, self.ax = plt.subplots()
        self.force_data = []
        self.force_treshold = 10.0


    def _env_setup(self, neutral_joint_values: np.ndarray) -> None:
        """
        Set up the environment to a neutral pose.

        Parameters:
        - neutral_joint_values (np.ndarray): Neutral positions for the robot joints.
        """
        self.set_joint_neutral()
        self.data.ctrl[0:7] = neutral_joint_values[0:7]
        mj.mj_forward(self.model, self.data)
        self.grasp_site_pose = self.get_ee_orientation().copy()
        self._mujoco_step()

    def step(self, action: np.ndarray) -> tuple:
        """
        Execute a step in the environment.
        
        Parameters:
        - action (np.ndarray): The action to be applied.

        Returns:
        - tuple: Observation, reward, termination flag, and additional info.
        """
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        self._set_action(action)
        self._mujoco_step(action)
        self._step_callback()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()
        info = {"is_success": self._is_success(obs["achieved_goal"], obs["desired_goal"])}
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)
        terminated = bool(info["is_success"])
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"])

        # Update real-time plot
        self.force_data.append(obs["achieved_goal"][3:6])  # Assuming force is from index 3 to 5
        self.update_plot()


        return obs, reward,terminated, truncated, info
    
    def _get_obs(self) -> dict:
        """
        Get the current observation of the environment.

        Returns:
        - dict: Current state observation including position and velocity of the end-effector.
        """
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_velocity = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt
        ee_force = get_force_sensor_data(self.data)

        return {
            "observation": np.concatenate([ee_position, ee_velocity, ee_force]),
            "achieved_goal": np.concatenate([ee_position, ee_force]),  
            "desired_goal": np.concatenate([self.goal,self.goal_force]),
        }
    
    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, alpha=10000.0, beta=2.0, gamma=2) -> float:
        ee_position = achieved_goal[0:3]
        goal_position = desired_goal[0:3]
        

        position_error = np.linalg.norm(ee_position - goal_position)
        is_reached = position_error < self.distance_threshold

        print("Position Error:", position_error)

        if is_reached:
            ee_force = achieved_goal[3:6]
            goal_force_x = desired_goal[3]
            force_error_x = abs(ee_force[0] - goal_force_x)
            reward = - beta * force_error_x

        else:
            reward = - alpha * position_error

        return reward

    
    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.float32:
        
        return False

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
        Sample a target goal for the environment.

        Returns:
        - np.ndarray: The sampled target goal.
        """
        goal = self._utils.get_site_xpos(self.model, self.data, "target").copy()
        return goal

    def update_plot(self) -> None:
            """
            Update the real-time plot of force sensor data.
            """
            if len(self.force_data) > 0:
                force_data_np = np.array(self.force_data)
                if force_data_np.shape[1] == 3:  # Assuming force data is in 3D
                    x_vals = np.arange(len(force_data_np))
                    self.ax.clear()
                    self.ax.plot(x_vals, force_data_np[:, 0], label='Force X')
                    # self.ax.plot(x_vals, force_data_np[:, 1], label='Force Y')
                    # self.ax.plot(x_vals, force_data_np[:, 2], label='Force Z')
                    self.ax.set_xlabel('Time Step')
                    self.ax.set_ylabel('Force')
                    self.ax.set_title('Force Sensor Data')
                    self.ax.legend()
                    self.fig.canvas.draw()
                    plt.pause(0.001)