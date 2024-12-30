import pprint
import gymnasium
import highway_env
from gymnasium import register
from stable_baselines3 import DQN

# **Imports**
# `pprint` - Used for pretty-printing complex data structures (like configurations).
# `gymnasium` - The core library for reinforcement learning environments.
# `highway_env` - A custom environment for simulating highway scenarios.
# `register` - Used to define and register a custom environment.
# `DQN` - A reinforcement learning algorithm from the Stable-Baselines3 library for training and loading models.

# Registering the custom environment with Gymnasium
register(
    id='CustomRewardEnv',  # Unique identifier for the custom environment.
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',  # Specifies the module and class defining the custom environment.
)

# **Creating the Environment**
env = gymnasium.make('CustomRewardEnv', render_mode='rgb_array')
# `gymnasium.make` creates an instance of the registered environment with the given `id`.
# `render_mode='rgb_array'` specifies that the environment will render as an RGB image.

# **Configuring the Environment**
env.unwrapped.config.update({
    "lanes_count": 7,  # Number of lanes in the highway simulation.
    "vehicles_count": 150,  # Total number of vehicles in the simulation.
    "controlled_vehicles": 1,  # Number of "ego" vehicles controlled by the agent.
    "duration": 1000,  # Duration of each episode in simulation steps.
    "reward_speed_range": [23, 27],  # Speed range for rewarding the agent.
    "vehicles_density": 1.0,  # High density ensures more vehicles on the road.
    "spawn_probability": 1.0,  # Vehicles will spawn continuously.
    "target_speed": [20, 25, 25],  # Desired speed range for the ego vehicle.
    "show_trajectories": True  # Visualize trajectories of all vehicles.
})

# **Printing the Configuration**
pprint.pprint(env.unwrapped.config)
# Prints the updated configuration in a readable format using `pprint`.

# **Loading the Pre-trained Model**
model = DQN.load("C:/Users/hamed/Desktop/tenserflow/week4/dqn/algo/logs/rl_model_dqn_104000_steps.zip")
# Loads a pre-trained DQN model from the specified file path.
# This model has been trained previously and will be used to predict actions for the agent.

# **Running the Simulation**
while True:  # Infinite loop to run the simulation indefinitely.
    # Reset the environment at the start of each episode.
    done = truncated = False  # Flags indicating if the episode is over or truncated.
    obs, info = env.reset()  # Resets the environment to its initial state.
    # `obs` - The initial observation/state from the environment.
    # `info` - Additional metadata about the environment reset.

    # **Simulating an Episode**
    while not (done or truncated):  # Continue until the episode ends or is truncated.
        action, _states = model.predict(obs, deterministic=True)
        # `model.predict` - Uses the pre-trained model to predict the next action for the agent.
        # `obs` - The current observation/state.
        # `deterministic=True` - Ensures consistent predictions for the same input state.
        # `action` - The action predicted by the model.
        # `_states` - Internal states of the model (not used here).

        obs, reward, done, truncated, info = env.step(action)
        # `env.step(action)` - Applies the chosen action to the environment.
        # `obs` - The new state after the action.
        # `reward` - Reward received from the environment for the action.
        # `done` - Boolean flag indicating if the episode has ended.
        # `truncated` - Boolean flag indicating if the episode ended prematurely (e.g., timeout).
        # `info` - Additional metadata about the environment after the step.

        env.render()
        # `env.render()` - Renders the environment, typically visualizing the current state.
