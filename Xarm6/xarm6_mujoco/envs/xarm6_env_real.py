import os
import sys
import time
import math
import numpy as np

import datetime
import random
import traceback
import threading

try:
    from xarm.tools import utils
except:
    pass
from xarm import version
from xarm.wrapper import XArmAPI

class Xarm6Real(XArmAPI):

    def __init__(
        self,
        block_gripper: bool = False,
        port = None,
        **kwargs,
    ):
        super().__init__(
            port=port,
            is_radian = True,
            **kwargs,
        )

        self.clean_warn()
        self.clean_error()
        self.motion_enable(True)
        self.set_mode(0)
        self.set_state(0)

        self.neutral_joint_positions = np.array([0., 0., 0., 0., 0., 0.])
        self.min_angles = np.array([-6.2, -2., -3.8, -6.2, -1.6, -6.2])
        self.max_angles = np.array([6.2, 2., 0, 6.2, 3.14, 6.2])
        self.block_gripper = block_gripper
        self.speed = 0.5
        
        self.p0 = self.get_ee_position()
        self.t0 = time.time()
 
       

    def _env_setup(self) -> None:
        self.move_gohome()

    def random_action(self):
        return np.random.uniform(self.min_angles,self.max_angles)

    def _set_action(self, action: np.ndarray) -> None:
        action = action.copy()
        target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(action)
        self.set_joint_angles(target_arm_angles)

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        arm_joint_ctrl *= 0.025
        current_joint_angles = np.array([self.get_joint_angle(i) for i in range(6)])
        current_arm_joint_angles = current_joint_angles[:6]
        
        target_arm_joint_angles = current_arm_joint_angles + arm_joint_ctrl
        
        for i in range(len(target_arm_joint_angles)):
            target_arm_joint_angles[i] = np.clip(target_arm_joint_angles[i], self.min_angles[i], self.max_angles[i])
        
        return target_arm_joint_angles

    def set_joint_angles(self, target_angles: np.ndarray) -> None:
        self.set_servo_angle(angle=target_angles)    
            
    def get_joint_angle(self, joint: int) -> float:
        if joint < 6:
            return self.get_servo_angle(servo_id=joint + 1)[1]
        return self.get_servo_angle(servo_id=joint-6 + 1)[1]
    
    def get_ee_position(self) -> np.ndarray:
        ee_position_mm = self.get_position()[1][:3]
        ee_position_m = np.array([ee_position_mm[i]*10**-3 for i in range(3)])
        return ee_position_m

    def get_ee_speed(self) -> np.ndarray:
        dt = time.time() - self.t0
        p0 = self.p0
        p1 = self.get_ee_position()
        speed = (p1 - p0) / dt
        self.t0 = time.time()
        self.p0 = p1
        return speed
        
if __name__ == '__main__':
    try:
        arm = Xarm6Real(port='192.168.1.217')
    except Exception as e:
        print("Impossible to connect to the robot: " + str(e))
        exit(10)

    arm._initialize_experimentation()
    action=arm.random_action()
    print(arm.get_ee_position())
    print(arm.get_ee_speed())
    # arm.set_joint_angles(action)

    

    