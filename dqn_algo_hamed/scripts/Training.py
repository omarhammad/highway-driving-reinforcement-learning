import pprint
import gymnasium
import os
from gymnasium import register
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Register the custom environment
register(id='HighwayFastCustomReward-v0',entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',)

# Create the environment
# Create the environment without visualization
env = gymnasium.make("HighwayFastCustomReward-v0", render_mode=None)

# Configure the environment
env.unwrapped.config.update({
    "lanes_count": 7,  # Number of lanes
    "vehicles_count": 150,  # Ensure a high number of vehicles
    "controlled_vehicles": 1,  # Number of ego vehicles
    "duration": 1000,  # Episode duration
    "reward_speed_range": [23, 27],  # Speed reward range
    "vehicles_density": 1.0,  # Increase density for more consistent presence
    "spawn_probability": 1.0,  # Ensure continuous vehicle spawning
    "target_speed": [20, 25, 27],  # Target speed
})

pprint.pprint(env.unwrapped.config)
# Path to the checkpoint
model_path = "logs/rl_model_dqn_102000_steps.zip"

# Check if the model exists and load it, or start from scratch
if os.path.exists(model_path):
    print(f"Checkpoint found at {model_path}. Loading the model...")
    model = DQN.load(model_path, env=env)  # Load the checkpoint and attach the environment
    num_timesteps = 100000
    reset_num_timesteps = False  # Continue training without resetting timesteps
else:
    model = DQN(
        "mlppolicy",  # use a multilayer perceptron (mlp) policy
        env,  # attach the environment to the model
        policy_kwargs=dict(net_arch=[256, 256]),  # neural network architecture with two layers of 256 units each
        learning_rate=5e-4,  # learning rate for the model
        buffer_size=15000,  # size of the replay buffer to store past experiences
        learning_starts=200,  # number of timesteps before training begins
        batch_size=32,  # batch size for training
        gamma=0.9,  # discount factor for future rewards
        train_freq=1,  # train the model after every action
        gradient_steps=1,  # number of gradient steps per training update
        target_update_interval=50,  # update target network every 50 steps
        verbose=1,  # print training progress and statistics
        tensorboard_log="highway_dqn/",  # log training metrics to tensorboard
        exploration_initial_eps=1.0,  # initial exploration rate (epsilon)
        exploration_final_eps=0.1,  # final exploration rate
        exploration_fraction=0.5,  # fraction of training steps for linear epsilon decay
    )
    num_timesteps = 100000
    reset_num_timesteps = True
# Checkpoint Callback
checkpoint_callback = CheckpointCallback(
    save_freq=1000, save_path="./logs/", name_prefix="rl_model_dqn"
)
print(f'num_timesteps: {num_timesteps}, reset_num_timesteps: {reset_num_timesteps}')
# Train the model
model.learn(total_timesteps=num_timesteps, callback=checkpoint_callback,reset_num_timesteps=reset_num_timesteps)

# Save the final model
model.save("highway_dqn/model")

# Close the environment
env.close()
