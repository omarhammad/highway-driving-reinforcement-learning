import pprint
import gymnasium
import torch
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
    "lanes_count": 4,
    "vehicles_count": 70,
    "duration": 1000,
    "reward_speed_range": [20, 30],
    "speed_coefficient": 1.2,  # Slightly increase speed influence
    "collision_coefficient": 1.2,  # Slightly increase collision penalty influence
    "reward_min": -40.0,  # Increased range for rewards
    "reward_max": 25.0,
    "ego_spacing": 5.0,
})
pprint.pprint(env.unwrapped.config)

# Define the PPO model
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    n_steps=512,  # Increased n_steps for better gradient updates
    batch_size=128,  # Increased batch size for smoother updates
    n_epochs=20,  # Increased epochs for more stable training
    learning_rate=1e-4,  # Reduced learning rate for finer adjustments
    gamma=0.9,  # Increased gamma to value future rewards
    verbose=1,
    tensorboard_log="highway_ppo/",
    device="cuda" if torch.cuda.is_available() else "cpu",  # Ensure GPU is used
)

# Save checkpoints during training
checkpoint_callback = CheckpointCallback(
    save_freq=2000, save_path='./logs/', name_prefix='rl_model_ppo'
)

# Train the model and visualize
model.learn(total_timesteps=50000, callback=checkpoint_callback)  # Increased timesteps

# Save the final model
model.save("highway_ppo/model")

# Close the environment after training
env.close()
