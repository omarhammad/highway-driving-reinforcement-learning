import pprint

import gymnasium
from gymnasium import register
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Register the custom environment
register(
    id='HighwayFastCustomReward-v0',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)

# Create the environment
env = gymnasium.make("HighwayFastCustomReward-v0")

# # Register the default highway environment
# register(
#     id='HighwayFastDefault-v0',
#     entry_point='highway_env.envs:HighwayEnv',
# )
#
# # Create the environment
# env = gymnasium.make("HighwayFastDefault-v0")

# Configure the environment
# env.unwrapped.config.update({
#     "lanes_count": 4,  # Number of lanes
#     "vehicles_count": 100,  # Ensure a high number of vehicles
#     "controlled_vehicles": 1,  # Number of ego vehicles
#     "duration": 1000,  # Episode duration
#     "reward_speed_range": [20, 30],  # Speed reward range
#     "vehicles_density": 1.0,  # Increase density for more consistent presence
#     "spawn_probability": 1.0,  # Ensure continuous vehicle spawning
# })
env.unwrapped.config.update({"lanes_count": 5,  # Number of lanes
                             "vehicles_count": 100,  # Ensure a high number of vehicles
                             "controlled_vehicles": 1,  # Number of ego vehicles
                             "duration": 100,  # Episode duration
                             "reward_speed_range": [20, 30],  # Speed reward range
                             "vehicles_density": 1.4,  # Increase density for more consistent presence
                             "spawn_probability": 1.0,  # Ensure continuous vehicle spawning
                             })

pprint.pprint(env.unwrapped.config)

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
    tensorboard_log="highway_dqn/hard",
    device=device,
)
# Checkpoint Callback
checkpoint_callback = CheckpointCallback(
    save_freq=1000, save_path="./logs/history/hard", name_prefix="rl_model_dqn"
)

# Train the model
model.learn(total_timesteps=int(20000), callback=checkpoint_callback)

# Save the final model
model.save("highway_dqn/model_hard")

# Close the environment
env.close()
