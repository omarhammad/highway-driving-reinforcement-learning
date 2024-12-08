import pprint
import gymnasium
from gymnasium import register
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from HighwayEnvCustomReward import HighwayEnvFastCustomReward  # Ensure this matches your file structure

# saving a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=1000, save_path='./logs/', name_prefix='rl_model_dqn'
)

# registering the custom environment with the custom reward function
register(
    id='CustomRewardEnv',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)

# using the custom environment with rendering
env = gymnasium.make('CustomRewardEnv', render_mode='human')

# optionally configure the environment for increased complexity
#  these lines to test with different configurations
env.unwrapped.config["lanes_count"] = 4  # Increase the number of lanes
env.unwrapped.config["vehicles_count"] = 60  # Increase the number of vehicles
env.unwrapped.config["duration"] = 1000  # Extend the simulation duration

# printing environment configuration for debugging
pprint.pprint(env.unwrapped.config)

# train a DQN model using the custom reward function
model = DQN(
    'MlpPolicy',
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=15000,
    learning_starts=200,
    batch_size=32,
    gamma=0.8,
    train_freq=1,
    gradient_steps=1,
    target_update_interval=50,
    verbose=1,
    tensorboard_log="highway_dqn/"
)

# trian the model and save checkpoints
model.learn(int(2e4), callback=checkpoint_callback)

# save the final trained model
model.save("highway_dqn/model")

# close the environment after training
env.close()
