import sys
import time
import gymnasium as gym
import xarm6_mujoco
import numpy as np
from xarm6_mujoco.envs.reach_real import Xarm6ReachEnvReal

if __name__ == "__main__":
    
    try:
        env = Xarm6ReachEnvReal(port='192.168.1.217')
    except Exception as e:
        print("Impossible to connect to the robot: " + str(e))
        exit(10)

    # Réinitialise l'environnement
    observation = env._env_setup()

    running = True
    while running:
        action = env.random_action()
        observation, reward, terminated, truncated, info = env.step(action)

        # if terminated or truncated:
        #     observation, info = env.reset()

    sys.exit()

        


