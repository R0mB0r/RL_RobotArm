import sys
import time
import gymnasium as gym
import panda_mujoco_gym_joint

import pdb

if __name__ == "__main__":
    env = gym.make("PandaReach-v3", render_mode="human")

    observation, info = env.reset()

    print("Action Space Shape:", env.action_space.shape)
    print("Action Space Sample:", env.action_space.sample())  # Random action sample

    for i in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(0.05)

    env.close()
