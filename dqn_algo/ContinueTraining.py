import pprint
import gymnasium
from gymnasium import register
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Register the custom environment
register(
    id='HighwayFastCustomReward-v0',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)

# Create the environment
env = gymnasium.make("HighwayFastCustomReward-v0")

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

# Check if a checkpoint exists
checkpoint_dir = "./logs/history/speed_05_lane01_coll_04/"
checkpoint_model_path = os.path.join(checkpoint_dir, "rl_model_dqn_20000_steps.zip")

# If the model already exists, load it; otherwise, create a new model
if os.path.exists(checkpoint_model_path):
    print("Loading existing model...")
    model = DQN.load(checkpoint_model_path, env=env, device=device)
else:
    print("Creating a new model...")
    model = DQN(
        "MlpPolicy",
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
        tensorboard_log="highway_dqn/",
        device=device,
    )

# Checkpoint Callback (save every 1000 steps, adjust save frequency)
checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path=checkpoint_dir+"/finetune",
    name_prefix="rl_model_dqn"
)

# Train the model (additional 20,000 steps)
model.learn(total_timesteps=int(20000), callback=checkpoint_callback)

# Save the final model (with updated name reflecting 20000+ steps)
model.save(os.path.join(checkpoint_dir+"/finetune", "rl_model_dqn_20000_steps"))

# Close the environment
env.close()
