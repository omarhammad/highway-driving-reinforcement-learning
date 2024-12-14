import pprint
import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import time
import torch

# Check GPU availability
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Register the custom environment
register(
    id='HighwayFastCustomReward-v0',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)

# Create the environment
env = gymnasium.make("HighwayFastCustomReward-v0", render_mode="human")

# Configure the environment
env.unwrapped.config.update({
    "lanes_count": 4,  # Number of lanes
    "vehicles_count": 100,  # Ensure a high number of vehicles
    "controlled_vehicles": 1,  # Number of ego vehicles
    "duration": 1000,  # Episode duration
    "reward_speed_range": [20, 30],  # Speed reward range
    "vehicles_density": 1.0,  # Increase density for more consistent presence
    "spawn_probability": 1.0,  # Ensure continuous vehicle spawning
})

pprint.pprint(env.unwrapped.config)

# PPO Training
model_path = "logs/rl_model_ppo_100000_steps"
if os.path.exists(model_path + ".zip"):
    print(f"Loading existing model ({model_path})...")
    model = PPO.load(model_path, env=env, device="cuda")
    reset_timesteps = False
else:
    print("No saved model found. Starting training from scratch...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=128 * 12 // 6,
        batch_size=128,
        n_epochs=20,
        learning_rate=5e-4,
        gamma=0.8,
        verbose=2,
        tensorboard_log="highway_ppo/",
        device="cuda",
    )
    reset_timesteps = True

# Checkpoint Callback
checkpoint_callback = CheckpointCallback(
    save_freq=1000, save_path="./logs/", name_prefix="rl_model_ppo"
)

# Train the model
model.learn(total_timesteps=int(5e4), callback=checkpoint_callback, reset_num_timesteps=reset_timesteps)

# Save the final model
model.save("models/first_model")

# Close the environment
env.close()
