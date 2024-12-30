import pprint

import gymnasium
import highway_env
from gymnasium import register
from stable_baselines3 import DQN
#
register(
    id='CustomRewardEnv',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)
env = gymnasium.make('CustomRewardEnv', render_mode='rgb_array')
# register(
#     id='HighwayEnvFast-v0',
#     entry_point='highway_env.envs:HighwayEnv',
# )
#
# env = gymnasium.make("HighwayEnvFast-v0", render_mode='rgb_array')

# making sure to update the environment configuration to match the training configuration!
env.unwrapped.config["lanes_count"] = 4  # Increase the number of lanes
env.unwrapped.config["vehicles_count"] = 100  # Increase the number of vehicles
env.unwrapped.config["duration"] = 1000  # Extend the simulation duration
env.unwrapped.config["show_trajectories"] = True  # Update the speed reward range
env.unwrapped.config["vehicles_density"] = 1

pprint.pprint(env.unwrapped.config)

# Loaloading and test saved model
model = DQN.load("logs/history/speed_05_lane01_coll_04/finetune/rl_model_dqn_60000_steps.zip")
# model = DQN.load("logs/history/default/rl_model_dqn_55000_steps.zip")
while True:  # running the simulation indefinitely
    done = truncated = False  # reseting the done and truncated flags to False
    obs, info = env.reset()  # reseting the environment and get the initial observation
    while not (done or truncated):  # the simulation until done or truncated
        action, _states = model.predict(obs,
                                        deterministic=True)  # getting the action from the model, without exploration (deterministic)
        obs, reward, done, truncated, info = env.step(action)  # performong the action in the environment
        env.render()  # rendering the environment
