import mujoco
import numpy as np
from gymnasium.core import ObsType
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
from typing import Optional, Any, SupportsFloat

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.0, 0.5, 0.0]),
}

class Panda(MujocoRobotEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        model_path: str = None,
        n_substeps: int = 50,
        reward_type: str = "sparse",
        block_gripper: bool = False,
        distance_threshold: float = 0.05,
        is_obj: bool = False,
        obj_xy_range: float = 0.3,
        goal_xy_range: float = 0.3,
        goal_x_offset: float = 0.4,
        goal_z_range: float = 0.2,
        **kwargs,
    ):
        # Paramètres du modèle MuJoCo
        self.model_path = model_path
        self.n_substeps = n_substeps
        
        # Paramètres du type de récompense
        self.reward_type = reward_type

        # Paramètres du bras robotique
        self.block_gripper = block_gripper
        action_size = 7 + (0 if self.block_gripper else 1)
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])

        self.is_obj = is_obj

        # Appel du constructeur parent
        super().__init__(
            n_actions=action_size,
            n_substeps=n_substeps,
            model_path=model_path,
            initial_qpos=self.neutral_joint_values,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # Paramètres de distance
        self.distance_threshold = distance_threshold

        # Paramètres pour le but
        self.goal_xy_range = goal_xy_range
        self.goal_x_offset = goal_x_offset
        self.goal_z_range = goal_z_range

        self.goal_range_low = np.array([-self.goal_xy_range / 2 + self.goal_x_offset, -self.goal_xy_range / 2, 0])
        self.goal_range_high = np.array([self.goal_xy_range / 2 + self.goal_x_offset, self.goal_xy_range / 2,self.goal_z_range])

        # Ajustement de la plage du but
        self.goal_range_low[0] += 0.6
        self.goal_range_high[0] += 0.6

        # Paramètres pour l'objet à manipuler
        self.obj_xy_range = obj_xy_range
        if self.is_obj:
            self.obj_range_low = np.array([-self.obj_xy_range / 2, -self.obj_xy_range / 2, 0])
            self.obj_range_high = np.array([self.obj_xy_range / 2, self.obj_xy_range / 2, 0])
            # Ajustement de la plage de l'objet
            self.obj_range_low[0] += 0.6
            self.obj_range_high[0] += 0.6

        # Paramètres du modèle MuJoCo
        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.ctrl_range = self.model.actuator_ctrlrange

    
    # override the methods in MujocoRobotEnv
    # -----------------------------

    # Initialize simulation
    def _initialize_simulation(self) -> None:
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)
        print(self._model_names.joint_names)

        # Set visualization dimensions
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        # Index used to distinguish arm and gripper joints
        self.arm_joint_names = self._model_names.joint_names[0:7]
        self.gripper_joint_names = self._model_names.joint_names[7:9]
        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

    # Set up environment
    def _env_setup(self, neutral_joint_values) -> None:
        self.set_joint_neutral()
        self.data.ctrl[0:7] = neutral_joint_values[0:7]

        self._mujoco.mj_forward(self.model, self.data)

        self.grasp_site_pose = self.get_ee_orientation().copy()

        self._mujoco_step()

        if self.is_obj:
            self.initial_object_height = self._utils.get_joint_qpos(self.model, self.data, "obj_joint")[2].copy()

    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        d = self.goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # Take a step in the environment
    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()

        info = {"is_success": self._is_success(obs["achieved_goal"], self.goal)}

        terminated = info["is_success"]
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        return obs, reward, terminated, truncated, info
    
    # Set action for the robot
    def _set_action(self, action) -> None:
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

    # Get observation from the environment
    def _get_obs(self):
        # Get robot end-effector position and velocity
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_velocity = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt
        if not self.block_gripper:
            fingers_width = self.get_fingers_width().copy()

        # Get object information
        object_position = self._utils.get_site_xpos(self.model, self.data, "obj_site").copy()
        object_rotation = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, "obj_site")).copy()
        object_velp = self._utils.get_site_xvelp(self.model, self.data, "obj_site").copy() * self.dt
        object_velr = self._utils.get_site_xvelr(self.model, self.data, "obj_site").copy() * self.dt

        # Construct observation dictionary
        if not self.block_gripper:
            obs = {
                "observation": np.concatenate(
                    [
                        ee_position,
                        ee_velocity,
                        fingers_width,
                        object_position,
                        object_rotation,
                        object_velp,
                        object_velr,
                    ]
                ).copy(),
                "achieved_goal": object_position.copy(),
                "desired_goal": self.goal.copy(),
            }
        else:
            obs = {
                "observation": np.concatenate(
                    [
                        ee_position,
                        ee_velocity,
                        object_position,
                        object_rotation,
                        object_velp,
                        object_velr,
                    ]
                ).copy(),
                "achieved_goal": object_position.copy(),
                "desired_goal": self.goal.copy(),
            }

        return obs

    # Check if the task is successful
    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    # Render callback function
    def _render_callback(self):
        # Visualize goal site
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._model_names.site_name2id["target"]
        self.model.site_pos[site_id] = self.goal - sites_offset[site_id]
        self._mujoco.mj_forward(self.model, self.data)

    # Reset simulation
    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self.set_joint_neutral()
        
        if self.is_obj:
            self._sample_object()
    
        self._mujoco.mj_forward(self.model, self.data)
        return True

    # Mujoco simulation step
    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # custom methods
    # -----------------------------
    
    # Calculate distance between goals
    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    # Set arm joints to neutral position
    def set_joint_neutral(self) -> None:
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[0:7]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        for name, value in zip(self.gripper_joint_names, self.neutral_joint_values[7:9]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)

    # Sample goal position
    def _sample_goal(self) -> np.ndarray:
        goal = np.array([0.0, 0.0, self.initial_object_height])
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if not self.block_gripper and self.goal_z_range > 0.0:
            if self.np_random.random() < 0.3:
                noise[2] = 0.0
        goal += noise
        return goal

    # Sample object position
    def _sample_object(self) -> None:
        object_position = np.array([0.0, 0.0, self.initial_object_height])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        object_xpos = np.concatenate([object_position, np.array([1, 0, 0, 0])])
        self._utils.set_joint_qpos(self.model, self.data, "obj_joint", object_xpos)

    # Get end-effector orientation
    def get_ee_orientation(self) -> np.ndarray:
        site_mat = self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(9, 1)
        current_quat = np.empty(4)
        self._mujoco.mju_mat2Quat(current_quat, site_mat)
        return current_quat

    # Get end-effector position
    def get_ee_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, "ee_center_site")

    # Get state of a body
    def get_body_state(self, name) -> np.ndarray:
        body_id = self._model_names.body_name2id[name]
        body_xpos = self.data.xpos[body_id]
        body_xquat = self.data.xquat[body_id]
        body_state = np.concatenate([body_xpos, body_xquat])
        return body_state

    # Get width of gripper fingers
    def get_fingers_width(self) -> np.ndarray:
        finger1 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint1")
        finger2 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint2")
        return finger1 + finger2

    # Convert arm joint control to target arm angles
    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        arm_joint_ctrl = arm_joint_ctrl*0.5
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    # Get joint angle
    def get_joint_angle(self, joint: int) -> float:
        joint_qpos = self._utils.get_joint_qpos(self.model, self.data, self.arm_joint_names[joint])[0]
        return joint_qpos
    
    # Set joint angles
    def set_joint_angles(self, target_angles: np.ndarray) -> None:
        for name, value in zip(self.arm_joint_names, target_angles):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        for name, value in zip(self.gripper_joint_names, target_angles[7:9]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)