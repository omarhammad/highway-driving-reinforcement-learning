import pprint

import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Register the custom environment
register(
    id='HighwayFastCustomReward-v0',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)

# Create the environment with rendering
env = gymnasium.make("HighwayFastCustomReward-v0", render_mode="human")

# Configure the environment
env.unwrapped.config.update({
    "lanes_count": 8,
    "vehicles_count": 50,
    "duration": 40,
    "reward_speed_range": [20, 30],
    "speed_coefficient": 1.0,
    "collision_coefficient": 1.0,
    "reward_min": -30.0,
    "reward_max": 15.0,
})
pprint.pprint(env.unwrapped.config)

# Define the PPO model
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    learning_rate=5e-4,
    gamma=0.8,
    verbose=1,
    tensorboard_log="highway_ppo/",
)

# Save checkpoints during training
checkpoint_callback = CheckpointCallback(
    save_freq=1000, save_path='./logs/', name_prefix='rl_model_ppo'
)

# Train the model and visualize
model.learn(total_timesteps=20000, callback=checkpoint_callback)

# Save the final model
model.save("highway_ppo/model")

# Close the environment after training
env.close()
