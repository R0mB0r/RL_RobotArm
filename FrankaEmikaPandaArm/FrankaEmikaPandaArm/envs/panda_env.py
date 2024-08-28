import numpy as np
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from typing import Optional
import mujoco
import time

# Default camera configuration for visualization
DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.0, 0.5, 0.0]),
}

class Panda(MujocoRobotEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        n_substeps: int = 30,
        model_path: str = "panda.xml",
        block_gripper: bool = False,
        **kwargs,
    ):
        """
        Initialize the Panda robot environment.

        Parameters:
        - n_substeps (int): Number of simulation substeps.
        - model_path (str): Path to the MuJoCo model file.
        - block_gripper (bool): Whether to prevent gripper movement.
        - **kwargs: Additional arguments passed to the parent class.
        """
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.angle_min = np.array([-2.9, -1.76, 0, -3.07, -2.9, 0, -2.9])
        self.angle_max = np.array([2.9, 1.76, 0, 0, 2.9, 3.75, 2.9])
        self.block_gripper = block_gripper
        n_actions = 7 if block_gripper else 8

        self.is_reached = False

        super().__init__(
            n_actions=n_actions,
            n_substeps=n_substeps,
            model_path=model_path,
            initial_qpos=self.neutral_joint_values,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        """
        Perform a simulation step in MuJoCo.

        Parameters:
        - action (Optional[np.ndarray]): The action to apply, if any.
        """
        self._mujoco.mj_step(self.model, self.data)

    def _set_action(self, action: np.ndarray) -> None:
        """
        Process and apply the given action to control the robot.

        Parameters:
        - action (np.ndarray): The action array containing control inputs for the arm and gripper.
        """
        action = action.copy()
        arm_joint_ctrl = action[:7]
        target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)
    
        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.set_joint_angles(target_angles)

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """
        Convert control inputs into target joint angles.

        Parameters:
        - arm_joint_ctrl (np.ndarray): Control inputs for the arm joints.

        Returns:
        - np.ndarray: The target joint angles for the robot's arm.
        """
        arm_joint_ctrl *= 0.025
        current_arm_joint_angles = np.array([self.get_joint_angle(i) for i in range(7)])
        target_arm_joint_angles = current_arm_joint_angles + arm_joint_ctrl
        
        for i in range(len(target_arm_joint_angles)):
            target_arm_joint_angles[i] = np.clip(target_arm_joint_angles[i], self.angle_min[i], self.angle_max[i])
        
        return target_arm_joint_angles

    def _reset_sim(self) -> bool:
        """
        Reset the simulation to its initial state.

        Returns:
        - bool: True if the simulation was successfully reset.
        """
        if self.is_reached:
            self.data.time = 0
            self.data.qvel[:] = 0
            if self.model.na != 0:
                self.data.act[:] = 0
            joint_angles = np.array([self.get_joint_angle(i) for i in range(7)])
            joint_velocities = np.array([0.0] * 7)
            self.set_joint_angles(joint_angles)
            self.set_joint_velocities(joint_velocities)
            self._mujoco.mj_forward(self.model, self.data)
        
        else:
            self.data.time = 0
            self.data.qvel[:] = 0
            if self.model.na != 0:
                self.data.act[:] = 0
            self.set_joint_neutral()
            self._mujoco.mj_forward(self.model, self.data)
        return True
    
    def set_joint_neutral(self) -> None:
        """
        Set the robot's joints to their neutral positions.
        """
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[:7]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        for name, value in zip(self.gripper_joint_names, self.neutral_joint_values[7:9]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def get_fingers_width(self) -> float:
        """
        Get the current width between the gripper fingers.

        Returns:
        - float: The combined width of both gripper fingers.
        """
        finger1 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint1")
        finger2 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint2")
        return finger1 + finger2

    def get_joint_angle(self, joint: int) -> float:
        """
        Get the current angle of the specified joint.

        Parameters:
        - joint (int): The index of the joint to query.

        Returns:
        - float: The current angle of the specified joint.
        """
        return self._utils.get_joint_qpos(self.model, self.data, self.arm_joint_names[joint])[0]
    

    def set_joint_angles(self, target_angles: np.ndarray) -> None:
        """
        Set the robot's joints to the specified target angles.

        Parameters:
        - target_angles (np.ndarray): The target angles for the arm and gripper joints.
        """
        for name, value in zip(self.arm_joint_names, target_angles[:7]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        for name, value in zip(self.gripper_joint_names, target_angles[7:9]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def set_joint_velocities(self, target_velocities: np.ndarray) -> None:
        """
        Set the robot's joint velocities to the specified target values.

        Parameters:
        - target_velocities (np.ndarray): The target velocities for the arm and gripper joints.
        """
        for name, value in zip(self.arm_joint_names, target_velocities[:7]):
            self._utils.set_joint_qvel(self.model, self.data, name, value)
        for name, value in zip(self.gripper_joint_names, target_velocities[7:9]):
            self._utils.set_joint_qvel(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def get_ee_orientation(self) -> np.ndarray:
        """
        Get the end-effector orientation in quaternion.

        Returns:
        - np.ndarray: The orientation of the end-effector as a quaternion.
        """
        site_mat = self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(9, 1)
        current_quat = np.empty(4)
        self._mujoco.mju_mat2Quat(current_quat, site_mat)
        return current_quat
