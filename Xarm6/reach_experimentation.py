import os
import numpy as np
import time
import argparse
import gymnasium as gym
import xarm6_mujoco

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from Xarm6.xarm6_mujoco.envs.reach_real import Xarm6ReachEnvReal

def create_env():
    """Create and wrap the environment."""
    return Xarm6ReachEnvReal()


def experimentation(model, env, test_duration=120):
    """Run a final test to visualize the agent's performance."""
    observations = env.reset()
    states = None
    episode_starts = np.array([True])

    t0 = time.time()

    while (time.time() - t0) < test_duration:
        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=True,
        )
        observations, _, _, _ = env.step(actions)

    env.close()

if __name__ == "__main__":
    
    env_name_load = "Xarm6Reach"

    exp_env = create_env()
    exp_env = VecNormalize.load(f"trainings/vec_normalize-{env_name_load}.pkl", exp_env)
    model = PPO.load(f"trainings/ppo-{env_name_load}.zip")
    experimentation(model, exp_env)
