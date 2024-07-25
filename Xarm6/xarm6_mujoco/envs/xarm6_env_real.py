import os
import sys
import time
import math
import numpy as np

from xarm.wrapper import XArmAPI

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

class Xarm6Real(XArmAPI):

    def __init__(
        self,
        block_gripper: bool = False,
        port = None,
        **kwargs,
    ):
        self.neutral_joint_positions = np.array([0., 0., 0., 0., 0., 0., 0.85, 0.839, 0.856, 0.85, 0.839, 0.856])
        self.min_angles = np.array([-6.28, -2.06, -0.192, -6.28, -1.69, -6.28, 0])
        self.max_angles = np.array([6.28, 2.09, 3.93, 6.28, 3.14, 6.28, 0])
        self.block_gripper = block_gripper
        self.speed = math.radians(50)
        
        self.p0 = self.get_ee_position()
        self.t0 = time.time()
 
        super().__init__(
            port=port,
            is_radian = True,
            **kwargs,
        )

    def _initialize_experimentation(self) -> None:
        self._env_setup(self.neutral_joint_positions)

    def _env_setup(self, neutral_joint_positions: np.ndarray) -> None:
        pass

    def _set_action(self, action: np.ndarray) -> None:
        action = action.copy()
        arm_joint_ctrl = action[:6]
        target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)
        
        if self.block_gripper:
            target_fingers = [0.85, 0.839, 0.856]

        target_angles = np.concatenate((target_arm_angles, target_fingers, target_fingers))
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
        self.set_joint_neutral()
        return True
    
    def set_joint_neutral(self) -> None:
        nb_joints = self.neutral_joint_positions.size
        for i in range(nb_joints) :
            self.set_servo_angle(servo_id=i+1, angle=self.neutral_joint_positions[i], speed=self.speed, wait=True)

    def set_joint_angles(self, target_angles: np.ndarray) -> None:
        nb_joints = target_angles.size
        for i in range(nb_joints) :
            self.set_servo_angle(servo_id=i+1, angle=target_angles[i], speed=self.speed, wait=True)       
            
    def get_joint_angle(self, joint: int) -> float:
        if joint < 6:
            return self.get_servo_angle(servo_id=joint + 1)[1]
        return self.get_servo_angle(servo_id=joint-6 + 1)[1]
    
    def get_ee_position(self) -> np.ndarray:
        return self.get_position()

    def get_ee_speed(self) -> np.ndarray:
        dt = time.time() - self.t0
        p0 = self.p0
        p1 = self.get_ee_position()
        speed = np.linalg.norm(p1 - p0) / dt
        self.t0 = time.time()
        self.p0 = p1
        return speed
        