import pprint
import gymnasium
import torch
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


# Learning rate scheduler
def lr_schedule(progress_remaining):
    return 1e-4 * progress_remaining  # Linear decay from 1e-4 to 0


# Check GPU availability
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Register the custom environment
register(
    id='HighwayFastCustomReward-v0',
    entry_point='test_reward:HighwayEnvFastCustomReward',  # Path to your custom reward env
)

# Create the environment
env = gymnasium.make("HighwayFastCustomReward-v0", render_mode="human")

# Configure the environment
# Configure the environment
env.unwrapped.config.update({
    "lanes_count": 4,
    "vehicles_count": 60,
    "controlled_vehicles": 1,
    "reward_speed_range": [25, 35],  # Reduce the target speed range
    "duration": 1000,
    "action": {
        "type": "DiscreteMetaAction"
    },
    "initial_speed": 22,  # Reduce initial speed for all vehicles (including agent)
    "simulation_frequency": 10,  # Slightly lower frequency for slower updates
    "policy_frequency": 2,       # Allow actions at a slower interval
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "vehicles_density": 0.8,  # Balance vehicle density for easier control
    "speed": {
        "max": 32,  # Lower maximum vehicle speed
        "min": 15   # Allow minimal speeds for slower behavior
    },
    "idm_acceleration": 0.5,   # Slower acceleration for other vehicles
    "idm_deceleration": 1.5,   # Slower deceleration for smoother braking
    "idm_target_velocity": 28  # Reduce the target velocity for IDM vehicles
})


pprint.pprint(env.unwrapped.config)

# Define PPO model with updated parameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    gamma=0.997,  # Same as in the paper
    ent_coef=0.01,  # Lower entropy for better exploitation
    learning_rate=lr_schedule,  # Dynamic learning rate
    n_steps=4096,  # Same as before
    batch_size=256,  # Larger batch size for smoother updates
    n_epochs=10,  # Number of epochs
    tensorboard_log="./ppo_highway_logs/",
    policy_kwargs=dict(net_arch=[256, 256])  # Improved policy network
)

# Checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ppo_highway')

# TensorBoard instructions
print("To visualize training logs, run the following command in your terminal:")
print("tensorboard --logdir=./ppo_highway_logs/")

# Increased total timesteps for more robust learning
model.learn(total_timesteps=int(30e4), callback=checkpoint_callback)

# Save the final model
model.save("ppo_highway_control")

# Close the environment
env.close()
