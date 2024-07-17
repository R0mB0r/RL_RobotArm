import os
import numpy as np
import sys
import time
import argparse
import gymnasium as gym
import xarm6_mujoco

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

import pdb

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a PPO agent on the Xarm6-v3 environment.")
    parser.add_argument("--iterations", type=int, default=1_000_000,
                        help="Total number of training iterations (timesteps).")
    parser.add_argument("--show_spaces", action="store_true",
                        help="Show information about observation and action spaces.")
    parser.add_argument("--training", action="store_true",
                        help="Train the agent on the environment.")
    parser.add_argument("--final_test", action="store_true",
                        help="Perform a final test with rendering after training.")
    return parser.parse_args()

def show_spaces(env):
    """Print information about observation and action spaces."""
    print("_____OBSERVATION SPACE_____ \n")
    print("Sample observation:", env.observation_space.sample())  # Random observation sample
    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape:", env.action_space.shape)
    print("Action Space Sample:", env.action_space.sample())  # Random action sample

def create_env(env_name,render_mode=None):
    """Create and wrap the environment."""
    env = gym.make(env_name, render_mode=render_mode)
    env = Monitor(env)  # Wrap the environment with Monitor
    return env

def train_agent(env, iterations):
    """Train the PPO agent."""
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
    )
    model.learn(total_timesteps=iterations)
    model.save("ppo-xarm6force-v3")
    env.save("vec_normalize_force.pkl")
    return model

def evaluate_agent(model, env):
    """Evaluate the trained PPO agent."""
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

def final_test(model, env, test_duration=120):
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
        observations, rewards, dones, infos = env.step(actions)
        time.sleep(0.05)

    env.close()

if __name__ == "__main__":
    args = parse_args()

    if args.show_spaces:
        env = create_env()
        show_spaces(env)
        env.close()

    if args.training:
        env = make_vec_env("Xarm6Force-v3", n_envs=1)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        train_agent(env, args.iterations)

    if args.final_test:
        eval_env = DummyVecEnv([lambda: create_env("Xarm6Force-v3", render_mode="human")])
        eval_env = VecNormalize.load("vec_normalize_force.pkl", eval_env)
        model = PPO.load("ppo-xarm6force-v3")
        #evaluate_agent(model, eval_env)
        final_test(model, eval_env)