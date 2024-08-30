import sys
import time
import gymnasium as gym
import Xarm6
from Xarm6.envs.reach_real import Xarm6ReachRealEnv


# Constants
ENV_NAME = "Xarm6ReachRealEnv"
RENDER_MODE = "human"
NUM_STEPS = 1000
SLEEP_DURATION = 0.2

test_duration = NUM_STEPS*SLEEP_DURATION
print(f"Running the simulation for {test_duration} seconds...")


def main():
    env = gym.make(ENV_NAME, render_mode=RENDER_MODE)
    
    observation, info = env.reset()
    
    print("Action Space Shape:", env.action_space.shape)
    print("Action Space Sample:", env.action_space.sample())
    
    for _ in range(NUM_STEPS):
        
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
    
        if terminated or truncated:
            observation, info = env.reset()
        
        time.sleep(SLEEP_DURATION)
    
    env.close()


if __name__ == "__main__":
    
    try:
        env = Xarm6ReachRealEnv(port='192.168.1.217')
    except Exception as e:
        print("Impossible to connect to the robot: " + str(e))
        exit(10)

    main()

    sys.exit()

    

    