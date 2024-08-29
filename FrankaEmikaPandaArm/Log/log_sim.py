import numpy as np
import plotly.graph_objects as go

# Lire les valeurs depuis le fichier texte
with open("FrankaEmikaPandaArm/Log/distances.txt", "r") as file:
    distances = [float(line.strip()) for line in file]

# Conserver uniquement les 5000 dernières valeurs, ou toutes si moins de 1000
num_values_to_show = 1000
if len(distances) > num_values_to_show:
    distances = distances[-num_values_to_show:]

# Créer une liste d'étapes en fonction du nombre de valeurs à afficher
steps = np.arange(len(distances))

# Créer le graphique avec Plotly
fig = go.Figure()

# Ajouter les données au graphique
fig.add_trace(go.Scatter(x=steps, y=distances, mode='lines+markers', name='Distance'))
fig.add_trace(go.Scatter(x=steps, y=[0.01] * len(steps), mode='lines', name='Threshold', line=dict(color='red', dash='dash')))

# Ajouter des titres et des labels
fig.update_layout(
    title='Distance vs Step',
    xaxis_title='Step',
    yaxis_title='Distance',
    xaxis=dict(showline=True, showgrid=True),
    yaxis=dict(showline=True, showgrid=True)
)

# Afficher le graphique interactif
fig.show()

