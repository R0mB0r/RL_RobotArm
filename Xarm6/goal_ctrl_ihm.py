import pygame
import sys

# Constantes pour les directions
DIR_X_POS = 0
DIR_X_NEG = 1
DIR_Y_POS = 2
DIR_Y_NEG = 3
DIR_Z_POS = 4
DIR_Z_NEG = 5

# Initialisation de pygame
def init_pygame(width, height, caption):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(caption)
    return screen

# Fonction pour dessiner la croix directionnelle
def draw_cross(screen, active_keys):
    center = (100, 100)  # Centre de la fenêtre
    length = 50  # Longueur des segments
    thickness = 5  # Épaisseur des lignes

    # Lignes de la croix
    lines = [
        ((center[0]-30, center[1]), (center[0]-30+length, center[1])),   # +x
        ((center[0]-30, center[1]), (center[0]-30-length, center[1])),   # -x
        ((center[0]-30, center[1]), (center[0]-30, center[1]-length)),   # +y
        ((center[0]-30, center[1]), (center[0]-30, center[1]+length)),   # -y
        ((center[0]+60, center[1]), (center[0]+60, center[1]-length)),   # +z
        ((center[0]+60, center[1]), (center[0]+60, center[1]+length)),   # -z
    ]

    # Couleur par défaut et couleur active
    default_color = (255, 255, 255)  # Blanc
    active_color = (255, 0, 0)       # Rouge

    # Dessiner les lignes
    for idx, line in enumerate(lines):
        color = active_color if idx in active_keys else default_color
        pygame.draw.line(screen, color, line[0], line[1], thickness)

# Mapping des touches aux indices de direction
key_to_direction = {
    pygame.K_w: DIR_Z_POS,
    pygame.K_s: DIR_Z_NEG,
    pygame.K_a: DIR_X_NEG,
    pygame.K_d: DIR_X_POS,
    pygame.K_q: DIR_Y_NEG,
    pygame.K_e: DIR_Y_POS,
}

# Gestion des événements
def handle_events(active_keys):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key in key_to_direction:
                active_keys.add(key_to_direction[event.key])
        elif event.type == pygame.KEYUP:
            if event.key in key_to_direction:
                active_keys.discard(key_to_direction[event.key])
    return True

# Fonction principale
def main():
    screen = init_pygame(200, 200, "Test d'affichage de la croix directionnelle")
    running = True
    active_keys = set()  # Utiliser un ensemble pour éviter les doublons
    
    while running:
        running = handle_events(active_keys)
        
        # Effacer l'écran
        screen.fill((0, 0, 0))

        # Dessiner la croix directionnelle
        draw_cross(screen, active_keys)

        # Mettre à jour l'affichage
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()



