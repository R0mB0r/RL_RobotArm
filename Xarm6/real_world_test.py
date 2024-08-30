import time
import numpy as np
import gymnasium as gym

import Xarm6

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def create_env(env_name):
    """Creates the specified environment."""
    try:
        return gym.make(env_name)
    except Exception as e:
        raise RuntimeError(f"Error while creating the environment: {e}")


def test(model, env, num_steps=1600, sleep_duration=0.2):
    """Performs a final test to visualize the agent's performance and display the rewards."""
    observations = env.reset()
    states = None
    episode_starts = np.array([True])

    test_duration = num_steps * sleep_duration
    print(f"Running the test for {test_duration} seconds...")

    predict_fn = model.predict  # Reference to avoid looking up the method at each iteration
    step_fn = env.step  # Same for the env.step method

    for _ in range(num_steps):
        actions, states = predict_fn(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=True,
        )

        observations, reward, done, info = step_fn(actions)
        episode_starts = done
        time.sleep(sleep_duration)

    env.close()


if __name__ == "__main__":
    env_name = 'Xarm6ReachRealEnv'

    # Create the environment
    test_env = DummyVecEnv([lambda: create_env(env_name)])

    # Load the environment normalization
    test_env = VecNormalize.load("Xarm6/Trainings/Training_Reach_2M_goal_fix_ee_fix/vec_normalize-Xarm6ReachEnv.pkl", test_env)

    # Load the trained model
    model = PPO.load("Xarm6/Trainings/Training_Reach_2M_goal_fix_ee_fix/ppo-Xarm6ReachEnv.zip")

    # Run the test
    test(model, test_env)

