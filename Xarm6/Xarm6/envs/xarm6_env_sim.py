import numpy as np
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from typing import Optional
import mujoco
from math import pi
from scipy.spatial.transform import Rotation as R

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.0, 0.5, 0.0]),
}

def quaternion_to_euler_degrees(q):
        return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler('zyx', degrees=True)

class Xarm6(MujocoRobotEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 25,
    }

    def __init__(
        self,
        n_substeps: int = 20,
        model_path: str = "xarm6_no_gripper.xml",
        block_gripper: bool = True,
        **kwargs,
    ):
        # self.neutral_joint_values_gripper = np.array([0., 0., 0., 0., 0., 0., 0.85, 0.839, 0.856, 0.85, 0.839, 0.856])
        # self.min_angles_gripper = np.array([-6.28, -2.06, -0.192, -6.28, -1.69, -6.28, 0])
        # self.max_angles_gripper = np.array([6.28, 2.09, 3.93, 6.28, 3.14, 6.28, 0])

        self.neutral_joint_values = np.array([0., 0., 0., 0., 0., 0.])
        self.min_angles = np.array([-6.28, -2.05, -3.9, -6.28, -1.67, -6.28])
        self.max_angles = np.array([6.28, 2.05, 0, 6.28, 3.12, 6.28])

        self.block_gripper = block_gripper
        num_actions = 6 if block_gripper else 7

        self.is_reached = False

        super().__init__(
            n_actions=num_actions,
            n_substeps=n_substeps,
            model_path=model_path,
            initial_qpos=self.neutral_joint_values,
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
    
        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_velocities = np.copy(self.data.qvel)
    
    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        """
        Perform a simulation step in MuJoCo.

        Parameters:
        - action (Optional[np.ndarray]): The action to apply, if any.
        """
        self._mujoco.mj_step(self.model, self.data)

    def _set_action(self, action: np.ndarray) -> None:
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
        current_arm_joint_angles = np.array([self.get_joint_angle(i) for i in range(6)])
        target_arm_joint_angles = current_arm_joint_angles + arm_joint_ctrl
        
        for i in range(len(target_arm_joint_angles)):
            target_arm_joint_angles[i] = np.clip(target_arm_joint_angles[i], self.min_angles[i], self.max_angles[i])
        
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
            joint_angles = np.array([self.get_joint_angle(i) for i in range(6)])
            joint_velocities = np.array([0.0] * 6)
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
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[:6]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def get_joint_angle(self, joint: int) -> float:
        if joint < 6:
            return self._utils.get_joint_qpos(self.model, self.data, self.arm_joint_names[joint])[0]
        # return self._utils.get_joint_qpos(self.model, self.data, self.gripper_joint_names[joint - 6])[0]

    def set_joint_angles(self, target_angles: np.ndarray) -> None:
        for name, value in zip(self.arm_joint_names, target_angles[:6]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def set_joint_velocities(self, velocities: np.ndarray) -> None:
        for name, value in zip(self.arm_joint_names, velocities[:6]):
            self._utils.set_joint_qvel(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)

    def get_ee_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, "ee_center_site")

    
    def get_ee_orientation(self) -> np.ndarray:
        site_matrix = self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(9, 1)
        current_quaternion = np.empty(4)
        self._mujoco.mju_mat2Quat(current_quaternion, site_matrix)
        #print(current_quaternion)
        euler_angles = quaternion_to_euler_degrees(current_quaternion)  
        return euler_angles
    
    def _get_obs(self):
        return {
            'desired_goal': np.zeros(3),    # Example desired goal
            'achieved_goal': self.get_ee_position(),
            'observation': self.get_ee_position()
        }
  
if __name__ == "__main__":
        # Créer une instance de la classe Xarm6
    env = Xarm6(model_path="/home/yoshidalab/Documents/Romain/RL_RobotArm/Xarm6/Xarm6/assets/xarm6_no_gripper.xml", render_mode="human")


    # Tester l'initialisation de la simulation
    env._initialize_simulation()
    print("Simulation initialized successfully.")

    # Tester le reset de la simulation
    reset_success = env._reset_sim()
    print(f"Simulation reset successful: {reset_success}")

    # Tester l'obtention de tous les angles des joints
    joint_angles = np.array([env.get_joint_angle(i) for i in range(6)])
    print(f"All joint angles: {joint_angles}")

    # Tester l'obtention de la position de l'end-effector
    ee_position = env.get_ee_position()
    print(f"End-effector position: {ee_position}")

    ee_orientation = env.get_ee_orientation()
    print(f"End-effector orientation: {ee_orientation}")

    goal_pose = np.array([0.34, -0.30, 0.34])
    distance = np.linalg.norm(ee_position - goal_pose)
    print(f"Distance to goal: {distance}")

    # Tester l'application des angles des joints
    target_angles = np.array([-0.58983501, -0.08572703, -0.97932936, 2.37100247, -0.10409321, -2.11260381])
    env.set_joint_angles(target_angles)
    print("Joint angles set successfully.")

    # Tester le pas de simulation (étape de Mujoco)
    env._mujoco_step()
    print("Mujoco step completed successfully.")


    # Tester l'obtention de la position de l'end-effector
    ee_position = env.get_ee_position()
    print(f"End-effector position: {ee_position}")

    ee_orientation = env.get_ee_orientation()
    print(f"End-effector orientation: {ee_orientation}")

    # Tester l'obtention de tous les angles des joints
    joint_angles = np.array([env.get_joint_angle(i) for i in range(6)])
    print(f"All joint angles: {joint_angles}")