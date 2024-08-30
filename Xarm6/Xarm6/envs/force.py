import os
import numpy as np
from typing import Any
from Xarm6.envs.xarm6_env_sim import Xarm6
import mujoco as mj
import matplotlib.pyplot as plt

def get_sensor_data(data, sensor_name: str) -> np.ndarray:
    return data.sensor(sensor_name).data

def get_force_sensor_data(data) -> np.ndarray:
    return get_sensor_data(data, "contact_force")

MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "force.xml")

class Xarm6ForceEnv(Xarm6):
    def __init__(self, distance_threshold: float = 0.02, goal_force=np.array([0, 0, -40]), **kwargs: Any):
        self.model_path = MODEL_XML_PATH
        self.distance_threshold = distance_threshold
        self.goal_force = goal_force
        super().__init__(model_path=self.model_path, n_substeps=20, block_gripper=True, **kwargs)

        self.num_actuators = self.model.nu
        self.num_joints = self.model.nq
        self.num_velocities = self.model.nv
        self.ctrl_range = self.model.actuator_ctrlrange

        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1)
        self.distance_data = []
        self.force_data = []
        self.force_threshold = 5.0
        self.speed_data = []
        self.rotational_speed_data = []

    def _env_setup(self, neutral_joint_positions: np.ndarray) -> None:
        self.set_joint_neutral()
        self.data.ctrl[0:7] = neutral_joint_positions[0:7]
        mj.mj_forward(self.model, self.data)
        self.grasp_site_pose = self.get_ee_orientation().copy()
        self._mujoco_step()

    def step(self, action: np.ndarray) -> tuple:
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
        reward = self.compute_reward(obs["observation"], obs["achieved_goal"], obs["desired_goal"])

        self.update_plot()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> dict:
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_velocity = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt
        ee_rotational_velocity = self._utils.get_site_xvelr(self.model, self.data, "ee_center_site").copy() * self.dt
        ee_force = get_force_sensor_data(self.data)

        return {
            "observation": np.concatenate([ee_position, ee_velocity, ee_rotational_velocity, ee_force]),
            "achieved_goal": np.concatenate([ee_position, ee_force]),
            "desired_goal": np.concatenate([self.goal, self.goal_force]),
        }

    def compute_reward(self, observation: np.ndarray, achieved_goal: np.ndarray, desired_goal: np.ndarray, alpha=100000, beta=0.001, gamma=1000) -> float:
        ee_position = achieved_goal[0:3]
        goal_position = desired_goal[0:3]
        
        distance_to_goal = np.linalg.norm(ee_position - goal_position)
        is_reached = distance_to_goal < self.distance_threshold
        is_contact = np.linalg.norm(achieved_goal[3:6]) > 0

        ee_speed = observation[3:6]
        ee_speed_error = np.linalg.norm(ee_speed) 
        ee_rotational_speed = observation[6:9]
        ee_rotational_speed_error = np.linalg.norm(ee_rotational_speed)
        
        if is_reached and is_contact:
            ee_force = achieved_goal[3:6]
            goal_force = desired_goal[3:6]
            force_error = np.linalg.norm(ee_force - goal_force)
            reward = -beta * force_error - gamma * ee_speed_error - 1000 * ee_rotational_speed_error
        else:
            reward = -alpha * distance_to_goal
        
        self.distance_data.append(distance_to_goal)
        self.force_data.append(achieved_goal[3:6])
        self.speed_data.append(ee_speed_error)
        self.rotational_speed_data.append(ee_rotational_speed_error)

        return reward

    def _is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return False

    def _render_callback(self) -> None:
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._model_names.site_name2id["target"]
        self.model.site_pos[site_id] = self.goal - sites_offset[site_id]
        self._mujoco.mj_forward(self.model, self.data)

    def _sample_goal(self) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, "target").copy()
    
    def update_plot(self) -> None:
        if len(self.force_data) > 0:
            force_data_np = np.array(self.force_data)
            x_vals = np.arange(len(force_data_np))

            self.ax1.clear()
            if force_data_np.shape[1] == 3:
                self.ax1.plot(x_vals, force_data_np[:, 0], label='Force X', color='red')
                self.ax1.plot(x_vals, force_data_np[:, 1], label='Force Y', color='green')
                self.ax1.plot(x_vals, force_data_np[:, 2], label='Force Z', color='blue')
                self.ax1.plot(x_vals, self.goal_force[2] * np.ones(len(force_data_np)), label='Goal Force Z', linestyle='--', color='darkblue')
            self.ax1.set_xlabel('Time Step')
            self.ax1.set_ylabel('Force')
            self.ax1.set_title('Force Sensor Data')
            self.ax1.legend()

        if len(self.distance_data) > 0:
            distance_vals = np.array(self.distance_data)
            x_vals = np.arange(len(distance_vals))

            self.ax2.clear()
            self.ax2.plot(x_vals, distance_vals, label='Distance to Goal', color='red')
            self.ax2.plot(x_vals, self.distance_threshold * np.ones(len(force_data_np)), label='Threshold', linestyle='--', color='green')
            self.ax2.set_xlabel('Time Step')
            self.ax2.set_ylabel('Distance')
            self.ax2.set_title('Distance to Goal')
            self.ax2.legend()

        if len(self.speed_data) > 0:
            speed_vals = np.array(self.speed_data)
            x_vals = np.arange(len(speed_vals))

            self.ax3.clear()
            self.ax3.plot(x_vals, speed_vals, label='EE Speed', color='red')
            self.ax3.set_xlabel('Time Step')
            self.ax3.set_ylabel('Speed')
            self.ax3.set_title('End-effector Speed')
            self.ax3.legend()

        if len(self.rotational_speed_data) > 0:
            rotational_speed_vals = np.array(self.rotational_speed_data)
            x_vals = np.arange(len(rotational_speed_vals))

            self.ax4.clear()
            self.ax4.plot(x_vals, rotational_speed_vals, label='EE Rotational Speed', color='red')
            self.ax4.set_xlabel('Time Step')
            self.ax4.set_ylabel('Rotational Speed')
            self.ax4.set_title('End-effector Rotational Speed')
            self.ax4.legend()
        
        plt.pause(0.001)
