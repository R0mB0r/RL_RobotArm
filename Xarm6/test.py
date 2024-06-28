import sys
import time
import gymnasium as gym
import xarm6_mujoco
import numpy as np
import pygame


# Initialisation de Pygame
def init_pygame(width, height, caption):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(caption)
    return screen

# Mapping des touches aux indices de direction
key_to_direction = [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_q, pygame.K_e]


# Gestion des événements
def handle_events(active_keys):
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in key_to_direction:
                active_keys.add(event.key)
        elif event.type == pygame.KEYUP:
            if event.key in key_to_direction:
                active_keys.discard(event.key)
    return running, active_keys

if __name__ == "__main__":
    
    screen = init_pygame(200, 200, "Test d'affichage de la croix directionnelle")
    # Crée l'environnement avec rendu MuJoCo
    env = gym.make("Xarm6Reach-v3", render_mode="human")

    # Réinitialise l'environnement
    observation, info = env.reset()
    active_keys = set()  # Utiliser un ensemble pour éviter les doublons
    
    clock = pygame.time.Clock()
    running = True
    while running:
        running, active_keys = handle_events(active_keys)
        if active_keys != set():
            env.modify_goal_position(active_keys)  
        # Mettre à jour la simulation MuJoCo
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        # if terminated or truncated:
        #     observation, info = env.reset()

    pygame.quit()
    sys.exit()

        


