import os
from gymnasium.envs.registration import register


ENV_IDS = []

for task in ["Reach"]:
    env_id = f"Xarm6{task}-v3"

    register(
        id=env_id,
        entry_point=f"xarm6_mujoco.envs:Xarm6{task}Env",
        max_episode_steps=200,
    )

    ENV_IDS.append(env_id)
