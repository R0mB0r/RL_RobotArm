# Xarm6 robot

Development of a 3D model in the MuJoCo simulation software for the Xarm6 and development of an algorithm to control in a real world the Xarm6 robot, where the input is the angular position of the actuators.
This robot will be trained using the PPO reinforcement learning algorithm from the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) library. 


In this project, our goal is to train the robot to perform the different tasks:

- `Reach` task: Reaching a small green box which have a fix position in the simulation space

- `Force` task: Aplly a defined force on a a difined position on a wall

The objective of this work is to succeed to implement simulation model trained on a real world robot and to find how we can multiply task with Reinforcement learning methods

## Test of the 3D model and the different scene

This test verifies whether the 3D model of the robot and the different task scene function correctly in MuJoCo and in a real world without any control. The input values are chosen randomly.

To run the test:

In simulation:
```bash
python3 xarm6_sim_action_sampler.py 
```
In the real world:
```bash
python3 xarm6_real_action_sampler.py
```

<div align="center">

`Untrained Reach model simulation` | `Untrained Force simulation`| `Untrained model log: distance as a function of steps`|
|:------------------------:|:------------------------:|:------------------------:|
<img src="/Pictures/Xarm6/xarm6_sim_Reach_untrained.gif" alt="" width="230"/> | <img src="/Pictures/Xarm6/xarm6_sim_Force_untrained.gif" alt="" width=""/>| <img src="/Pictures/Xarm6/xarm6_sim_Reach_log_untrained.png" />

</div> 


Real model

## Training, Evaluation and Simulation

In the script `ppo_xarm6_training_pipeline.py`, we train an agent in the chosen environment using the PPO algorithm developed by the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) library. After training, the model is saved in the `Trainings` directory. The model is composed of two files: ppo-{name of the environment}.zip and vec_normalize-{name of the environment}.pkl. If the training last more than 1_000_000 steps, a model is saved every 1_000_000 of steps.

This model can also be loaded in this script to evaluate the trained agent and to perform a 3D simulation of the agent executing the chosen task.

### Explanation of Script Arguments:
    
    --env_name: Name of the environment. (default: Xarm6ReachEnv)
    --show_spaces: Displays information about the observation and action spaces.
    --training: Trains the agent in the environment.
    --total_timesteps: Sets the total number of training timesteps (default: 1,000,000).
    --evaluate: Evaluates the trained agent in the environment.
    --simulation: Performs a final test with rendering after training.
    --checkpoint_freq: Specifies the frequency of saving checkpoints (in timesteps) (default: 100,000).
    --log_dir: Specifies the directory where logs and models will be saved (default: Trainings).

### Exemple Command to Execute the Script: 

```bash
python3 ppo_xarm6_training_pipeline.py  --show_spaces --training --total_timesteps 2000000 --evaluate --simulation 
```

### Reach Env Trained Simulation 

### Force Env Trained Simulation

### Reach Env Trained Real World