# Xarm6 robot

Development of a 3D model in the MuJoCo simulation software for the Xarm6 and the development of an algorithm to control the Xarm6 robot in the real world, where the input is the angular position of the actuators. This robot will be trained using the PPO reinforcement learning algorithm from the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) library. 


In this project, our goal is to train the robot to perform different tasks:

- Reach Task: Reaching a small green box with a fixed position in the simulation space.
- Force Task: Applying a defined force at a specific position on a wall.

The objective of this work is to successfully implement a simulation model trained on a real-world robot and explore how we can multiply tasks using reinforcement learning methods.

## Testing the 3D Model and the Different Scenes

This test verifies whether the 3D model of the robot and the different task scenes function correctly in MuJoCo and in the real world without any control. The input values are chosen randomly.

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

<div align="center">

`Untrained Reach model real log: distance as a function of steps`|
|:------------------------:|
<img src="/Pictures/Xarm6/xarm6_real_Reach_untrained.png" alt="" width=""/>|

</div>


## Training, Evaluation and Simulation

In the script `ppo_xarm6_training_pipeline.py`, we train an agent in the chosen environment using the PPO algorithm developed by the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) library. After training, the model is saved in the `Trainings` directory. The model consists of two files: `ppo-{name_of_environment}.zip` and `vec_normalize-{name_of_environment}.pkl`. If the training lasts more than 1,000,000 steps, a model is saved every 1,000,000 steps.

This model can also be loaded in this script to evaluate the trained agent and perform a 3D simulation of the agent executing the chosen task.

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

<div align="center">

`Trained Reach model simulation`|`Trained Reach model log: distance as a function of steps`|
|:------------------------:|:------------------------:| 
<img src="/Pictures/Xarm6/xarm6_sim_Reach_trained.gif" alt="" width="230"/>| <img src="/Pictures/Xarm6/xarm6_sim_Reach_log_trained.png" />

</div>

For the `Reach` task: 

- reward = - distance_beetwen_ee_effector_and_goal_position
- observation = {​  
                    "observation": (ee_position, ee_velocity),​
                    "achieved_goal": ee_position,​
                    "desired_goal": goal_position
                ​}

</div>

### Force Env Trained Simulation

<div align="center">

`Trained Force model simulation`| `Trained Force model log: Force, Distance, Speed and Rotationnal speed as function of timestep`|
|:------------------------:|:------------------------:| 
<img src="/Pictures/Xarm6/xarm6_sim_Force_trained.gif"/>| <img src="/Pictures/Xarm6/xarm6_sim_Force_log_trained.png"/>

</div>

For the `Force` task:

- Reward:   If distance < treshold and contact > 0:​
                reward = - beta*force_error​
            Else:​
                reward = - alpha*distance​

(where alpha and beta are constant chosen empirically)

- Observation = {​
                    "observation": (ee_position, ee_velocity, ee_force),​
                    "achieved_goal": (ee_position, ee_force),  ​
                    "desired_goal": (goal_position, goal_force),​
                }​

As shown in the graph, the robot successfully reaches the target position but struggles to apply the desired force.

The real challenge of this work lies in combining these two tasks effectively. One potential solution could be to find the appropriate coefficients or to design an optimal reward function that balances both tasks.

Currently, this solution remains an open question in the field of research, and much work is needed to develop a robust and permanent solution.

## Implementation on the Real-World Robot

One of the main objectives of this work is to implement a reinforcement learning trained model on a real-world robot.

To achieve this, I had to create two entire environments: one for controlling the robot (`xarm6_env_real.py`) and one for the task (`reach_real.py`), similar to the simulation models.

The script real_world_test.py allows running a model trained in simulation on the real robot.

<div align="center">

`Trained Reach model real log: distance as a function of steps`|
 |:------------------------:|
 <img src="/Pictures/Xarm6/xarm6_real_Reach_log_trained.png" alt="" width=""/>|

</div>

From this log, we can observe an error of approximately 4 cm between the position of the end effector and the target threshold.

Identifying the exact cause of this error and determining how to reduce it is challenging. One potential solution could be increasing the number of timesteps during training to allow the model more time to learn and fine-tune its actions.

However, due to time constraints during my internship, I was unable to fully address this issue.