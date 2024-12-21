import pprint

import gymnasium as gym
import torch
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


# Learning rate scheduler
def lr_schedule(progress_remaining):
    """
    Linear decay of learning rate.
    """
    return 1e-4 * progress_remaining


# Check GPU availability
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Register the custom environment
register(
    id='HighwayFastCustomReward-v0',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)

# Validate environment registration and creation
env = gym.make("HighwayFastCustomReward-v0", render_mode="human")

# Configure the environment with the provided parameters
env.unwrapped.config.update({
    "action": {
        "type": "DiscreteMetaAction"  # Simplifies actions into lane and speed control
    },
    "observation": {
        "type": "Kinematics"  # Kinematic observations
    },
    "lanes_count": 4,  # Number of lanes
    "vehicles_count": 60,  # Traffic density
    "controlled_vehicles": 1,  # Number of agent-controlled vehicles
    "reward_speed_range": [20, 30],  # Speed range for high-speed rewards
    "duration": 1000,  # Simulation duration in seconds
    "simulation_frequency": 15,  # Frequency of simulation in Hz
    "policy_frequency": 1,  # Frequency of applying policy (agent's actions) in Hz
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # Vehicle type
    "scaling": 5.5,  # Scaling for rendering
    "show_trajectories": False,  # Don't show the trajectories
    "render_agent": True,  # Render the agent
    "offscreen_rendering": False  # No offscreen rendering
})


pprint.pprint(env.unwrapped.config)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    gamma=0.997,  # Long-term discount factor
    ent_coef=0.02,  # Entropy coefficient for exploration-exploitation balance
    learning_rate=lr_schedule,  # Dynamic learning rate
    n_steps=4096,  # Number of steps per update
    batch_size=256,  # Batch size for gradient updates
    n_epochs=10,  # Number of optimization epochs per update
    tensorboard_log="./ppo_highway_logs/",  # TensorBoard log directory
    policy_kwargs=dict(net_arch=[256, 256])  # Neural network architecture
)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ppo_highway')

# TensorBoard instructions
print("To visualize training logs, run the following command in your terminal:")
print("tensorboard --logdir=./ppo_highway_logs/")

# Train the model
print("Starting training...")
model.learn(total_timesteps=int(3e5), callback=checkpoint_callback)

# Save the final trained model
model.save("ppo_highway_control")
print("Training complete. Model saved as 'ppo_highway_control'.")

# Close the environment
env.close()
