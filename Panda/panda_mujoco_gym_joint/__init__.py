import os
from gymnasium.envs.registration import register


ENV_IDS = []

for task in ["Push","Reach", "Slide", "PickAndPlace"]:
    env_id = f"Panda{task}-v3"

    register(
        id=env_id,
        entry_point=f"panda_mujoco_gym_joint.envs:Panda{task}Env",
        max_episode_steps=100,
    )

    ENV_IDS.append(env_id)
