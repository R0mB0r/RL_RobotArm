import time

import numpy as np
import gymnasium as gym

import xarm6_mujoco

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def create_env(env_name):
    try:
        return gym.make(env_name)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la création de l'environnement: {e}")

def simulation(model, env, test_duration=500):
    """Exécute un test final pour visualiser les performances de l'agent et afficher les récompenses."""
    observations = env.reset()
    states = None
    episode_starts = np.array([True])

    rewards = np.zeros(test_duration)  # Pré-allouer la mémoire pour les récompenses
    steps = np.arange(1, test_duration + 1)  # Pré-allouer les étapes

    predict_fn = model.predict  # Référence pour éviter de chercher la méthode à chaque itération
    step_fn = env.step  # Idem pour la méthode env.step

    for step in range(test_duration):
        actions, states = predict_fn(
            observations,
            state=states,
            episode_start=episode_starts,
            deterministic=True,
        )

        observations, reward, done, info = step_fn(actions)
        print("rr ", reward)
        rewards[step] = reward[0]  # Stocker la récompense

        time.sleep(0.2)

    env.close()

if __name__ == "__main__":
    print('hello')
    env_name = 'Xarm6ReachRealEnv'
    
    # Créer l'environnement
    sim_env = DummyVecEnv([lambda: create_env(env_name)])
    
    # Charger la normalisation de l'environnement
    sim_env = VecNormalize.load("Xarm6/trainings/vec_normalize-Xarm6ReachEnv.pkl", sim_env)
    
    # Charger le modèle entraîné
    model = PPO.load("Xarm6/trainings/ppo-Xarm6ReachEnv.zip")
    
    # Exécuter la simulation
    simulation(model, sim_env)
