import os
import numpy as np
import time
import argparse
import xarm6_mujoco

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from xarm6_mujoco.envs.reach_sim import Xarm6ReachEnv

import pdb

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a PPO agent on the Xarm6 environment.")
    parser.add_argument("--env_name", type=str, default="Xarm6ReachEnv",
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

def create_env(render_mode=None):
    """Create and wrap the environment."""
    env = Xarm6ReachEnv(render_mode=render_mode)
    return env

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
    model.save(model_save_path)
    return model

def evaluate_agent(model, env):
    """Evaluate the trained PPO agent."""
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


def load_actions_from_file(action_file):
    liste = []
    with open(action_file, 'r') as file:
        for line in file:
            # Supprimer les crochets et autres caractères non numériques
            cleaned_line = line.replace('[', '').replace(']', '').replace(',', '').strip()
            # Transformer la ligne en liste de flottants
            if cleaned_line:
                values = np.array([float(x) for x in cleaned_line.split()])
                liste.append(values)
    return liste

def simulation(env, action_file, test_duration=120):
    """Run a final test to visualize the agent's performance."""
    obs, _ = env.reset()
    actions_list = load_actions_from_file(action_file)
    action_index = 0
    t0 = time.time()

    while (time.time() - t0) < test_duration and action_index < len(actions_list):
        actions = actions_list[action_index]
        print(actions)
        action_index += 1
        obs, _, _, _, _ = env.step(actions)

    env.close()

if __name__ == "__main__":
    args = parse_args()

    if args.show_spaces:
        env = create_env(args.env_name)
        show_spaces(env)
        env.close()

    if args.training:
        env = create_env(render_mode="human")
        model = train_agent(env, args.total_timesteps, args.env_name)
        env.close()

    if args.evaluate:
        eval_env = create_env(render_mode="human")
        model = PPO.load(f"trainings/ppo-{args.env_name}")
        evaluate_agent(model, eval_env)
        eval_env.close()

    if args.simulation:
        sim_env = create_env(render_mode="human")
        action_file = "actions.txt"
        simulation(sim_env,action_file)
        sim_env.close()
