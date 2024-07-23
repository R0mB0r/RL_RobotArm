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

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a PPO agent on the Xarm6 environment.")
    parser.add_argument("--env_name", type=str, default="Xarm6Force",
                        help="Name of the environment to train the agent on.")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000,
                        help="Total number of training iterations (timesteps).")
    parser.add_argument("--show_spaces", action="store_true",
                        help="Show information about observation and action spaces.")
    parser.add_argument("--training", action="store_true",
                        help="Train the agent on the environment.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the trained agent on the environment.")
    parser.add_argument("--simulation", action="store_true",
                        help="Perform a simulation with rendering after training.")
    return parser.parse_args()

def create_env(env_name, render_mode=None):
    """Create and wrap the environment."""
    env = gym.make(env_name, render_mode=render_mode)
    return Monitor(env)

def show_spaces(env):
    """Print information about observation and action spaces."""
    print("_____OBSERVATION SPACE_____ \n")
    print("Sample observation:", env.observation_space.sample())
    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape:", env.action_space.shape)
    print("Action Space Sample:", env.action_space.sample())

def train_agent(env, total_timesteps, env_name):
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
    model.learn(total_timesteps)
    model_save_path = f"trainings/ppo-{env_name}.zip"
    vec_normalize_path = f"trainings/vec_normalize-{env_name}.pkl"
    model.save(model_save_path)
    env.save(vec_normalize_path)
    return model

def evaluate_agent(model, env):
    """Evaluate the trained PPO agent."""
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

def simulation(model, env, test_duration=120):
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
    args = parse_args()

    if args.show_spaces:
        env = create_env(args.env_name)
        show_spaces(env)
        env.close()

    if args.training:
        env = make_vec_env(args.env_name, n_envs=1)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        train_agent(env, args.total_timesteps, args.env_name)

    if args.evaluate:
        eval_env = DummyVecEnv([lambda: create_env(args.env_name, render_mode="human")])
        eval_env = VecNormalize.load(f"trainings/vec_normalize-{args.env_name}.pkl", eval_env)
        model = PPO.load(f"trainings/ppo-{args.env_name}.zip")
        evaluate_agent(model, eval_env)

    if args.simulation:
        sim_env = DummyVecEnv([lambda: create_env(args.env_name, render_mode="human")])
        sim_env = VecNormalize.load(f"trainings/vec_normalize-{args.env_name}.pkl", sim_env)
        model = PPO.load(f"trainings/ppo-{args.env_name}.zip")
        simulation(model, sim_env)
