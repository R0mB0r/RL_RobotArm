import os
import numpy as np
import time

from xarm6_mujoco.envs.reach_real import Xarm6ReachEnvReal


from stable_baselines3 import PPO


def experimentation(model, env, test_duration=120):
    """Run a final test to visualize the agent's performance."""
    observations = env._env_setup()
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
        print(actions)
        observations, _, _, _, _ = env.step(actions)

    env.close()

if __name__ == "__main__":
    try:
        env = Xarm6ReachEnvReal(port='192.168.1.217')
    except Exception as e:
        print("Impossible to connect to the robot: " + str(e))
        exit(10)
    model = PPO.load(f"RL_RobotArm/Xarm6/trainings/ppo-xarm6reach-v3.zip")
    experimentation(model, env)
