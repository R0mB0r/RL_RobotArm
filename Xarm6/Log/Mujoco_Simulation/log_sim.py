import numpy as np
import plotly.graph_objects as go

# Read values from the text file
with open("Xarm6/Log/Mujoco_Simulation/distances_sim.txt", "r") as file:
    distances = [float(line.strip()) for line in file]

# Keep only the last 1000 values, or all if there are fewer than 1000
num_values_to_show = 5000
if len(distances) > num_values_to_show:
    distances = distances[-num_values_to_show:]

# Create a list of steps based on the number of values to display
steps = np.arange(len(distances))

# Create the plot with Plotly
fig = go.Figure()

# Add data to the plot
fig.add_trace(go.Scatter(x=steps, y=distances, mode='lines+markers', name='Distance'))
fig.add_trace(go.Scatter(x=steps, y=[0.01] * len(steps), mode='lines', name='Threshold', line=dict(color='red', dash='dash')))

# Add titles and labels
fig.update_layout(
    title='Distance vs Step',
    xaxis_title='Step',
    yaxis_title='Distance',
    xaxis=dict(showline=True, showgrid=True),
    yaxis=dict(showline=True, showgrid=True)
)

# Display the interactive plot
fig.show()


