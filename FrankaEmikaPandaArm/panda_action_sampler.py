import time
import gymnasium as gym
import FrankaEmikaPandaArm

# Constants
ENV_NAME = "PandaReach"
RENDER_MODE = "human"
NUM_STEPS = 1000
SLEEP_DURATION = 0.05

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
    main()

