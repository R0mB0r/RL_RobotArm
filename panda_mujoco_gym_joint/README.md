Pour executer la simulation: 

```bash
python3 test_rl  --show_spaces --training --iterations 2000000 --final_test 
```

Explanation:
--show_spaces, Show information about observation and action spaces.
--training, Train the agent on the environment.
--iterations, Total number of training iterations (timesteps). (default: 1_000_000)
--final_test, Perform a final test with rendering after training.

