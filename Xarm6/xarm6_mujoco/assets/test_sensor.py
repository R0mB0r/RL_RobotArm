import mujoco_py
from mujoco_py import load_model_from_xml, MjSim
import mjcpy

# Charger le modèle MuJoCo
model = mjcpy.MJCModel("Xarm6/xarm6_mujoco/assets/force.xml")

# Créer une simulation
sim = MjSim(model)

# Exécuter la simulation
for i in range(1000):
    sim.step()

    # Récupérer les données du capteur de force
    sensor_data = sim.data.sensordata

    # Afficher les données du capteur de force
    print("Force sensor data:", sensor_data)

# Fermer la simulation
sim.close()

