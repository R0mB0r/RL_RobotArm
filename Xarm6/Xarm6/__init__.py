import os
from gymnasium.envs.registration import register


ENV_IDS = []

for task in ["Reach", "Force", "ReachReal"]:
    env_id = f"Xarm6{task}Env"

    register(
        id=env_id,
        entry_point=f"Xarm6.envs:Xarm6{task}Env",
        max_episode_steps=1600,
    )

    ENV_IDS.append(env_id)
