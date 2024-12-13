import pprint
import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Register the custom environment
register(
    id='HighwayFastCustomReward-v0',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)

# Create the environment
env = gymnasium.make("HighwayFastCustomReward-v0", render_mode="human")

# Configure the environment
env.unwrapped.config.update({
    "lanes_count": 4,
    "vehicles_count": 70,
    "duration": 1000,
    "reward_speed_range": [20, 30],
    "safe_passing_range": [5.0, 15.0],
    "ego_spacing": 5.0,
})
pprint.pprint(env.unwrapped.config)

# Model path
model_path = "logs/rl_model_ppo_3000_steps"
if os.path.exists(model_path + ".zip"):
    print(f"Loading existing model({model_path.split('/')[1]})...")
    model = PPO.load(model_path, env=env, device="cuda")
    reset_timesteps = False
else:
    print("No saved model found. Starting training from scratch...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])),
        n_steps=1024,
        batch_size=512,  # Replace with the best batch size
        n_epochs=20,
        learning_rate=1e-4,  # Replace with the best learning rate
        gamma=0.98,  # Replace with the best gamma
        verbose=1,
        tensorboard_log="final_training_logs/",
        device="cuda",
    )
    reset_timesteps = True

# Save checkpoints during training
checkpoint_callback = CheckpointCallback(
    save_freq=1000, save_path="./logs/", name_prefix="rl_model_ppo"
)

# Train the model
model.learn(total_timesteps=100000, callback=checkpoint_callback, reset_num_timesteps=reset_timesteps)

# Save the final model
model.save(model_path)

# Close the environment
env.close()

# tensorboard --logdir=highway_ppo/
