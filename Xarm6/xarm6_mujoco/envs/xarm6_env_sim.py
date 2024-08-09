import numpy as np
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from typing import Optional
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.0, 0.5, 0.0]),
}

class Xarm6(MujocoRobotEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        n_substeps: int = 30,
        model_path: str = "xarm.xml",
        block_gripper: bool = False,
        **kwargs,
    ):
        self.neutral_joint_positions = np.array([0., 0., 0., 0., 0., 0., 0.85, 0.839, 0.856, 0.85, 0.839, 0.856])
        self.min_angles = np.array([-6.28, -2.06, -0.192, -6.28, -1.69, -6.28, 0])
        self.max_angles = np.array([6.28, 2.09, 3.93, 6.28, 3.14, 6.28, 0])
        self.block_gripper = block_gripper
        num_actions = 6 if block_gripper else 7

        super().__init__(
            n_actions=num_actions,
            n_substeps=n_substeps,
            model_path=model_path,
            initial_qpos=self.neutral_joint_positions,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def _initialize_simulation(self) -> None:
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)

        self._model_names = self._utils.MujocoModelNames(self.model)
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        self.arm_joint_names = self._model_names.joint_names[0:6]
        self.gripper_joint_names = self._model_names.joint_names[6:12]

        self._env_setup(self.neutral_joint_positions)
        self.initial_time = self.data.time
        self.initial_velocities = np.copy(self.data.qvel)

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        self._mujoco.mj_step(self.model, self.data)

    def _set_action(self, action: np.ndarray) -> None:
        action = action.copy()
        arm_joint_ctrl = action[:6]
        target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)
        # target_arm_angles = np.zeros(6)
        
        if self.block_gripper:
            target_fingers = [0.85, 0.839, 0.856]

        target_angles = np.concatenate((target_arm_angles, target_fingers, target_fingers))
        print(target_angles)
        self.set_joint_angles(target_angles)

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        arm_joint_ctrl *= 0.025
        current_joint_angles = np.array([self.get_joint_angle(i) for i in range(12)])
        current_arm_joint_angles = current_joint_angles[:6]
        
        target_arm_joint_angles = current_arm_joint_angles + arm_joint_ctrl
        
        for i in range(len(target_arm_joint_angles)):
            target_arm_joint_angles[i] = np.clip(target_arm_joint_angles[i], self.min_angles[i], self.max_angles[i])
        
        return target_arm_joint_angles

    def _reset_sim(self) -> bool:
        self.data.time = 0
        self.data.qvel[:] = 0
        if self.model.na != 0:
            self.data.act[:] = 0
        self.set_joint_neutral()
        self._mujoco.mj_forward(self.model, self.data)
        return True
    
    def set_joint_neutral(self) -> None:
        for name, value in zip(self.arm_joint_names, self.neutral_joint_positions[:6]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        for name, value in zip(self.gripper_joint_names, self.neutral_joint_positions[6:12]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def get_joint_angle(self, joint: int) -> float:
        if joint < 6:
            return self._utils.get_joint_qpos(self.model, self.data, self.arm_joint_names[joint])[0]
        return self._utils.get_joint_qpos(self.model, self.data, self.gripper_joint_names[joint - 6])[0]

    def set_joint_angles(self, target_angles: np.ndarray) -> None:
        for name, value in zip(self.arm_joint_names, target_angles[:6]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        for name, value in zip(self.gripper_joint_names, target_angles[6:12]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def get_ee_orientation(self) -> np.ndarray:
        site_matrix = self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(9, 1)
        current_quaternion = np.empty(4)
        self._mujoco.mju_mat2Quat(current_quaternion, site_matrix)
        return current_quaternion

