import sys
import time
import gymnasium as gym
import xarm6_mujoco
import numpy as np

if __name__ == "__main__":
    
    # Crée l'environnement avec rendu MuJoCo
    env = gym.make("Xarm6Reach-v3", render_mode="human")

    # Réinitialise l'environnement
    observation, info = env.reset()

    running = True
    while running:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    sys.exit()

        


