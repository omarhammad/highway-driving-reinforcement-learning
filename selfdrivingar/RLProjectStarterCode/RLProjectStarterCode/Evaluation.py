import pprint

import gymnasium
import highway_env
from gymnasium import register
from stable_baselines3 import DQN

register(
    id='CustomRewardEnv',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)
env = gymnasium.make('CustomRewardEnv', render_mode='rgb_array')

# Make sure to update the environment configuration to match the training configuration!
env.config["lanes_count"] = 5  # Change the number of lanes to make environment more complex
env.config["vehicles_count"] = 60  # Increase the number of vehicles to make environment more complex
env.config[
    "duration"] = 1000  # Increase the duration of the simulation to see how the agent behaves over a longer period

pprint.pprint(env.unwrapped.config)

# Load and test saved model
model = DQN.load("highway_dqn/model")
while True:  # Run the simulation indefinitely
    done = truncated = False  # Reset the done and truncated flags to False
    obs, info = env.reset()  # Reset the environment and get the initial observation
    while not (done or truncated):  # Continue the simulation until done or truncated
        action, _states = model.predict(obs,
                                        deterministic=True)  # Get the action from the model, without exploration (deterministic)
        obs, reward, done, truncated, info = env.step(action)  # Perform the action in the environment
        env.render()  # Render the environment
