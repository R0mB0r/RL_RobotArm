import os
import numpy as np
import sys
import time
import argparse
import gymnasium as gym
import panda_mujoco_gym_joint

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a PPO agent on the PandaReach-v3 environment.")
    parser.add_argument("--iterations", type=int, default=1_000_000,
                        help="Total number of training iterations (timesteps).")
    parser.add_argument("--show_spaces", action="store_true",
                        help="Show information about observation and action spaces.")
    parser.add_argument("--training", action="store_true",
                        help="Train the agent on the environment.")
    parser.add_argument("--final_test", action="store_true",
                        help="Perform a final test with rendering after training.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create the environment
    env = gym.make("PandaReach-v3")
    env.reset()

    if args.show_spaces:
        # Print observation and action space details
        print("_____OBSERVATION SPACE_____ \n")
        print("Sample observation:", env.observation_space.sample())  # Random observation sample

        print("\n _____ACTION SPACE_____ \n")
        print("Action Space Shape:", env.action_space.shape)
        print("Action Space Sample:", env.action_space.sample())  # Random action sample

    if args.training:
        # Create a vectorized environment for parallel processing
        env = make_vec_env("PandaReach-v3", n_envs=16)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

        # Initialize the PPO model
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

        # Train the model
        model.learn(total_timesteps=args.iterations)

        # Save the trained model
        model_name = "ppo-pandareach-v3"
        model.save(model_name)
        # Optionally, save the VecNormalize statistics
        env.save("vec_normalize.pkl")

    # Evaluate the trained model
    eval_env = DummyVecEnv([lambda: gym.make("PandaReach-v3")])
    eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

    model = PPO.load("ppo-pandareach-v3")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    if args.final_test:
        # Render the environment and visualize the agent's performance
        env = gym.make("PandaReach-v3", render_mode="human")
        observation, info = env.reset()
        for i in range(1000):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            print("reward: ", reward)
            env.render()
            
            if terminated or truncated:
                print(terminated, truncated)
                observation, info = env.reset()

            time.sleep(0.05)
        
        env.close()










