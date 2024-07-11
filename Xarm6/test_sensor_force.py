import sys
import time
import gymnasium as gym
import xarm6_mujoco
import numpy as np

env = gym.make("Xarm6Force-v3", render_mode="human")

# Réinitialise l'environnement
observation, info = env.reset()
active_keys = set()  # Utiliser un ensemble pour éviter les doublons

running = True
while running: 
    # Mettre à jour la simulation MuJoCo
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # if terminated or truncated:
    #     observation, info = env.reset()
    
    time.sleep(0.05)
sys.exit()
