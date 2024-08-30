# Franka Emika Panda Arm

Development of a 3D model in the MuJoCo simulation software for the Franka Emika Panda robot, where the input is the angular position of the actuators. This robot will be trained using the PPO reinforcement learning algorithm from the [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) library. 

In this project, our goal is to train the robot to perform the `Reach` task, which involves reaching a small green box that is randomly generated within the simulation space.


## Test of the 3D model and the scene `Reach`

This test verifies whether the 3D model of the robot and the `Reach` task scene function correctly in MuJoCo without any control. The input values are chosen randomly.

To run the test:

```bash
python3 panda_action_sampler.py 
```

<div align="center">

`Untrained model simulation` | `Untrained model log: distance as a function of steps`|
|:------------------------:|:------------------------:|
<img src="/Pictures/FrankaEmikaPandaArm/panda_simu_test.gif" alt="" width="230"/> | <img src="/Pictures/FrankaEmikaPandaArm/panda_log_untrained.png" alt="" width=""/>| 

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

`Trained model simulation`|`Trained model log: distance as a function of steps`|
|:------------------------:|:------------------------:|  
<img src="/Pictures/FrankaEmikaPandaArm/panda_simu_trained.gif" alt="" width="230"/> | <img src="/Pictures/FrankaEmikaPandaArm/panda_log_trained.png" alt="" width=""/>

</div>

### Model improvement suggestions : no success reset

As you can see above, the robot resets its position each time it successfully reaches the target. However, this behavior is not desirable in real-life scenarios; when a robotic arm is instructed to go to a point, we expect it to stay there. To eliminate this success reset, there are two effective strategies: either tell the robot that reaching the point is not a terminal condition, or fix its position by performing no action when it reaches the target.

These solutions are implemented in the code through the boolean variables `self.success_reset = False` and `self.fix` in the `reach.py` script.

<div align="center">

`self.success_reset = False, self.fix=True`| mean_reward = -1.16 +/- 0.83| 
|:------------------------:|:------------------------:|
<img src="/Pictures/FrankaEmikaPandaArm/panda_simu_trained_no_success_reset_fix.gif" width = "230"/> | <img src="/Pictures/FrankaEmikaPandaArm/panda_log_trained_no_success_reset_fix.png"/>

</div>



<div align="center">

`self.success_reset = False, self.fix=False`| mean_reward = -1.30 +/- 0.45|
|:------------------------:|:------------------------:|
<img src="/Pictures/FrankaEmikaPandaArm/panda_simu_trained_no_success_reset.gif" width = "230"> | <img src="/Pictures/FrankaEmikaPandaArm/pand_log_trained_no_success_reset.png"/>


</div>


With this robot and environment, there is not a significant difference between these solutions, but it could be interesting to try them in another environment. 