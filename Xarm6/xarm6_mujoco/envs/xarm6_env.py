import numpy as np
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from typing import Optional
import mujoco

# Default camera configuration for visualization
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
        """
        Initialize the Panda robot environment.

        Parameters:
        - n_substeps (int): Number of simulation substeps.
        - model_path (str): Path to the MuJoCo model file.
        - block_gripper (bool): Whether to prevent gripper movement.
        - **kwargs: Additional arguments passed to the parent class.
        """
        self.neutral_joint_values = np.array([0., 0., 0., 0., 0., 0., 0.85, 0.839, 0.856, 0.85, 0.839, 0.856])
        self.angle_min = np.array([-6.28, -2.06, -0.192, -6.28, -1.69, -6.28,0])
        self.angle_max = np.array([6.28, 2.09, 3.93, 6.28, 3.14, 6.28,0])
        self.block_gripper = block_gripper
        n_actions = 6 if block_gripper else 7

        super().__init__(
            n_actions=n_actions,
            n_substeps=n_substeps,
            model_path=model_path,
            initial_qpos=self.neutral_joint_values,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    
    def _initialize_simulation(self) -> None:
        """
        Initialize the MuJoCo simulation.

        Loads the MuJoCo model and sets up the initial simulation parameters.
        """
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        # self.model.worldbody.find('body', "ee_center_body").add('geom', dclass='collision', size='0.05 0.05 0.01', pos='0.0 0 -0.2', type='box')
        self.data = self._mujoco.MjData(self.model)

        self._model_names = self._utils.MujocoModelNames(self.model)
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        self.arm_joint_names = self._model_names.joint_names[0:6]
        self.gripper_joint_names = self._model_names.joint_names[6:12]

        # Set initial joint positions
        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)


    
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
        arm_joint_ctrl = action[:6]
        target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)
    
        if self.block_gripper:
            target_fingers = [0.85, 0.839, 0.856]

        target_angles = np.concatenate((target_arm_angles, target_fingers, target_fingers))
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
        current_joint_angles = np.array([self.get_joint_angle(i) for i in range(12)])
        current_arm_joint_angles = current_joint_angles[:6]
        
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
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[:6]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        for name, value in zip(self.gripper_joint_names, self.neutral_joint_values[6:12]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)
        

    def get_joint_angle(self, joint: int) -> float:
        """
        Get the current angle of the specified joint.

        Parameters:
        - joint (int): The index of the joint to query.

        Returns:
        - float: The current angle of the specified joint.
        """
        if joint < 6:
            arm_joints_value = self._utils.get_joint_qpos(self.model, self.data, self.arm_joint_names[joint])[0]
        else:
            j = joint - 6
            gripper_joints_value = self._utils.get_joint_qpos(self.model, self.data, self.gripper_joint_names[j])[0]

        return arm_joints_value if joint < 6 else gripper_joints_value
    
    def set_joint_angles(self, target_angles: np.ndarray) -> None:
        """
        Set the robot's joints to the specified target angles.

        Parameters:
        - target_angles (np.ndarray): The target angles for the arm and gripper joints.
        """
        for name, value in zip(self.arm_joint_names, target_angles[:6]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        for name, value in zip(self.gripper_joint_names, target_angles[6:12]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
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
