import os
import numpy as np
import time
import argparse
import gymnasium as gym
import FrankaEmikaPandaArm

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a PPO agent on the PandaReach-v3 environment.")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000,
                        help="Total number of training timesteps.")
    parser.add_argument("--show_spaces", action="store_true",
                        help="Show information about observation and action spaces.")
    parser.add_argument("--training", action="store_true",
                        help="Train the agent on the environment.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the trained agent on the environment.")
    parser.add_argument("--simulation", action="store_true",
                        help="Perform a simulation with rendering after a training.")
    parser.add_argument("--checkpoint_freq", type=int, default=100_000,
                        help="Frequency of saving checkpoints (in timesteps).")
    parser.add_argument("--log_dir", type=str, default="Trainings",
                        help="Directory where the logs and models will be saved.")
    return parser.parse_args()

def show_spaces(env):
    """Print information about observation and action spaces."""
    print("_____OBSERVATION SPACE_____ \n")
    print("Sample observation:", env.observation_space.sample())  # Random observation sample
    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape:", env.action_space.shape)
    print("Action Space Sample:", env.action_space.sample())  # Random action sample

def create_env(render_mode=None):
    """Create and wrap the environment."""
    env = gym.make("PandaReach", render_mode=render_mode)
    env = Monitor(env)  # Wrap the environment with Monitor
    return env

def train_agent(env, total_timesteps, log_dir, checkpoint_freq):
    """Train the PPO agent with checkpointing and logging."""
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=log_dir,
        name_prefix="ppo_model"
    )

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
    
    model.learn(total_timesteps, callback=checkpoint_callback)
    model_save_path = os.path.join(log_dir, "ppo-pandareach.zip")
    vec_normalize_path = os.path.join(log_dir, "vec_normalize.pkl")
    model.save(model_save_path)
    env.save(vec_normalize_path)
    
    return model

def evaluate_agent(model, env):
    """Evaluate the trained PPO agent."""
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

def simulation(model, env, num_steps = 1_000, sleep_duration=0.05):
    """Run a final test to visualize the agent's performance."""
    observations = env.reset()
    states = None
    episode_starts = np.array([True])
    
    test_duration = num_steps*sleep_duration
    print(f"Running the simulation for {test_duration} seconds...")

    for _ in range(num_steps):
        actions, states = model.predict(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=True,
        )
        observations, rewards, dones, infos = env.step(actions)
        episode_starts = dones
        env.render()
        time.sleep(sleep_duration)

    env.close()

def load_env_and_model(log_dir):
    """Load the environment and the trained model."""
    eval_env = DummyVecEnv([lambda: create_env(render_mode="human")])
    eval_env = VecNormalize.load(os.path.join(log_dir, "vec_normalize.pkl"), eval_env)
    model = PPO.load(os.path.join(log_dir, "ppo-pandareach"))
    return model, eval_env


if __name__ == "__main__":
    args = parse_args()

    if args.show_spaces:
        env = create_env()
        show_spaces(env)
        env.close()

    if args.training:
        env = make_vec_env("PandaReach", n_envs=16)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        train_agent(env, args.total_timesteps, args.log_dir, args.checkpoint_freq)

    if args.evaluate or args.simulation:
        model, eval_env = load_env_and_model(args.log_dir)

    if args.evaluate:
        evaluate_agent(model, eval_env)

    if args.simulation:
        simulation(model, eval_env)












