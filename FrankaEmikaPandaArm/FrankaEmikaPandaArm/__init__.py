import os
from gymnasium.envs.registration import register


ENV_IDS = []

for task in ["Reach"]:
    env_id = f"Panda{task}"

    register(
        id=env_id,
        entry_point=f"FrankaEmikaPandaArm.envs:Panda{task}Env",
        max_episode_steps=100,
    )

    ENV_IDS.append(env_id)
