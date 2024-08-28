# Franka Emika Panda Arm

Development of a 3D model in the MuJoCo simulation software for the Franka Emika Panda robot, where the input is the angular position of the actuators. This robot will be controlled using the PPO reinforcement learning algorithm from the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) library. 

In this project, our goal is to train the robot to perform the `Reach` task, which involves reaching a small green box that is randomly generated within the simulation space.


## Test of the 3D model and the scene `Reach`

This test verifies whether the 3D model of the robot and the `Reach` task scene function correctly in MuJoCo without any control. The input values are chosen randomly.

To run the test:

```bash
python3 panda_action_sampler.py 
```

<div align="center">

`untrained model simulation`|`untrained model log: distance as a function of steps`|
|:------------------------:|:------------------------:|
<img src="/pictures/FrankaEmikaPandaArm/panda_test.gif" alt="" width="230"/> | <img src="/pictures/FrankaEmikaPandaArm/panda_log_untrained.png>" alt="" width="230"/>| 

</div>

## Training, Evaluation and Simulation

In the script `ppo_panda_training_pipeline.py`, we train an agent in the `Reach` environment using the PPO algorithm developed by the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) library. After training, the model is saved in the `Trainings` directory. The model is composed of two files: ppo-pandareach.zip and vec_normalize.pkl.

This model can also be loaded in this script to evaluate the trained agent and to perform a 3D simulation of the agent executing the `Reach` task.


### Explanation of Script Arguments:
    
    --show_spaces: Displays information about the observation and action spaces.
    --training: Trains the agent in the environment.
    --total_timesteps: Sets the total number of training timesteps (default: 1,000,000).
    --evaluate: Evaluates the trained agent in the environment.
    --simulation: Performs a final test with rendering after training.
    --checkpoint_freq: Specifies the frequency of saving checkpoints (in timesteps) (default: 100,000).
    --log_dir: Specifies the directory where logs and models will be saved (default: Trainings).

### Exemple Command to Execute the Script: 

```bash
python3 ppo_panda_training_pipeline.py  --show_spaces --training --total_timesteps 2000000 --evaluate --simulation 
```
<div align="center">

`trained model simulation`|`trained model log: distance as a function of steps`|  
<img src="/pictures/FrankaEmikaPandaArm/panda_simu_trained.gif" alt="" width="230"/> | <img src="/pictures/FrankaEmikaPandaArm/panda_log_trained.png" alt="" width="230"/>

</div>

