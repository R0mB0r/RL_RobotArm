import os
from gymnasium.envs.registration import register


ENV_IDS = []

for task in ["Reach", "Force", "ReachReal"]:
    env_id = f"Xarm6{task}Env"

    register(
        id=env_id,
        entry_point=f"xarm6_mujoco.envs:Xarm6{task}Env",
        max_episode_steps=400,
    )

    ENV_IDS.append(env_id)
