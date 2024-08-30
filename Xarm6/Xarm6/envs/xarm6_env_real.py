import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from xarm.wrapper import XArmAPI
from math import pi

class Xarm6Real(gym.Env):
    def __init__(self, n_actions=6, robot_ip='192.168.1.217'):
        super(Xarm6Real, self).__init__()

        self.robot_ip = robot_ip
        self.arm = XArmAPI(self.robot_ip, is_radian=True)
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'desired_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
            'achieved_goal': spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
            'observation': spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float64)
        })

        self.min_angles = np.array([-6.2, -2., -3.8, -6.2, -1.6, -6.2])
        self.max_angles = np.array([6.2, 2., 0., 6.2, 3.14, 6.2])

        self.p0 = self.get_ee_position()
        self.t0 = time.time()

        self.speed = 1
    def reset(self):
        """Réinitialise les angles des articulations à zéro et renvoie l'observation."""
        # Réinitialisation de l'armature jusqu'à ce que les angles des articulations soient à zéro
        self.arm.reset()
        current_joint_angles = np.array([self.get_joint_angle(i) for i in range(6)])
        
        # Vérifier si tous les angles sont à zéro
        while not np.all(np.isclose(current_joint_angles, 0.0)):
            time.sleep(1)
            current_joint_angles = np.array([self.get_joint_angle(i) for i in range(6)])
        
        return self._get_obs()

    def step(self, action):
        self._set_action(action)
        obs = self._get_obs()
        reward = 0.0  # TODO: Implement reward logic
        done = False   # TODO: Implement termination logic
        info = {}      # Additional information, if needed
        return obs, reward, done, info

    def _set_action(self, action):
        action = action.copy()
        target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(action)
        self.set_joint_angles(target_arm_angles)

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl):
        arm_joint_ctrl *= 0.015
        current_joint_angles = np.array([self.get_joint_angle(i) for i in range(6)])
        current_arm_joint_angles = current_joint_angles[:6]
        
        target_arm_joint_angles = current_arm_joint_angles + arm_joint_ctrl
        
        for i in range(len(target_arm_joint_angles)):
            target_arm_joint_angles[i] = np.clip(target_arm_joint_angles[i], self.min_angles[i], self.max_angles[i])
        
        #print(target_arm_joint_angles)
        return target_arm_joint_angles

    def set_joint_angles(self, target_angles):
        self.arm.set_servo_angle(angle=target_angles, speed=self.speed)

    def get_joint_angle(self, joint):
        if joint < 6:
            return self.arm.get_servo_angle(servo_id=joint + 1)[1]
        return self.arm.get_servo_angle(servo_id=joint-6 + 1)[1]

    def get_ee_position(self):
        ee_position_mm = self.arm.get_position()[1][:3]
        offset_mm = [0,0,50]
        ee_position_mm = [a + b for a,b in zip(ee_position_mm,offset_mm)]
        ee_position_m = np.array([ee_position_mm[i] * 10 ** -3 for i in range(3)])
        return ee_position_m


    def get_ee_speed(self) -> np.ndarray:
        dt = time.time() - self.t0
        p0 = self.p0
        p1 = self.get_ee_position()
        speed = (p1 - p0) / dt
        self.t0 = time.time()
        self.p0 = p1
        return speed

    def _get_obs(self):
        return {
            'desired_goal': np.zeros(3),    # Example desired goal
            'achieved_goal': self.get_ee_position(),
            'observation': self.get_ee_position()
        }
    
    def set_ee_position(self,x,y,z):
        self.arm.set_position(x,y,z)

    
if __name__ == '__main__':
    try:
        arm = Xarm6Real()
    except Exception as e:
        print("Impossible to connect to the robot: " + str(e))
        exit(10)

    arm.reset()
    arm.set_ee_position(340,-300,340)
    time.sleep(10)
    ee_pose = arm.get_ee_position()
    goal_pose = np.array([0.34, -0.3, 0.34])
    print(ee_pose)
    print(np.linalg.norm(ee_pose - goal_pose))



