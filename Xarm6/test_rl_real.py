import os
import numpy as np
import time
import gymnasium as gym
import xarm6_mujoco

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize



def create_env(env_name):
    try:
        env = gym.make(env_name)
        print(f"Environnement {env_name} créé avec succès.")
        return env
    except Exception as e:
        print(f"Erreur lors de la création de l'environnement: {e}")
        raise


# def simulation(model, env, test_duration=120):
#     """Exécute un test final pour visualiser les performances de l'agent et sauvegarder les actions dans un fichier."""
#     print(env.reset())
#     observations = env.reset()
#     states = None
#     episode_starts = np.array([True])

#     t0 = time.time()

#     while (time.time() - t0) < test_duration:
#         actions, states = model.predict(
#             observations,
#             state=states,
#             episode_start=episode_starts,
#             deterministic=True,
#         )

#         print(actions)

#         observations, _, _, _ = env.step(actions)

#     env.close()


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

def simulation(env, action_file):
    """Run a final test to visualize the agent's performance."""
    obs = env.reset()
    actions_list = load_actions_from_file(action_file)
    action_index = 0

    while action_index < len(actions_list):
        actions = actions_list[action_index]
        print(actions)
        action_index += 1
        obs, _, _, _ = env.step(actions)

    env.close()

if __name__ == "__main__":
    env_name = 'Xarm6ReachRealEnv'
    
    # Créer l'environnement
    sim_env = DummyVecEnv([lambda: create_env(env_name)])
    
    # Charger la normalisation de l'environnement
    sim_env = VecNormalize.load("Xarm6/trainings/vec_normalize-Xarm6ReachEnv.pkl", sim_env)
    
    # Charger le modèle entraîné
    model = PPO.load("Xarm6/trainings/ppo-Xarm6ReachEnv.zip")
    
    # Exécuter la simulation
    # simulation(model, sim_env)

    action_file = "Xarm6/actions.txt"
    simulation(sim_env,action_file)
