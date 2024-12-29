import pprint
import gymnasium
from gymnasium import register
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import os
from stable_baselines3.common.callbacks import BaseCallback


class CustomCheckpointCallback(BaseCallback):
    """
    A custom callback that saves checkpoints with dynamic step-based naming.
    """
    def __init__(self, save_path, start_step, increment_step, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.current_step = start_step
        self.increment_step = increment_step

    def _on_step(self):
        # Check if the current step is a multiple of the increment step
        if self.n_calls % self.increment_step == 0:
            checkpoint_file = os.path.join(
                self.save_path,
                f"rl_model_dqn_{self.current_step}_steps.zip"
            )
            self.model.save(checkpoint_file)
            if self.verbose > 0:
                print(f"Checkpoint saved at step: {self.current_step} Location: {checkpoint_file}")
            self.current_step += self.increment_step
        return True


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
checkpoint_dir = "./logs/history/speed_05_lane01_coll_04/finetune"
checkpoint_model_path = os.path.join(checkpoint_dir, "rl_model_dqn_40000_steps.zip")

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

# Initialize the custom checkpoint callback
custom_checkpoint_callback = CustomCheckpointCallback(
    save_path=checkpoint_dir,
    start_step=41000,
    increment_step=1000,
    verbose=1,
)

# Train the model (additional 20,000 steps)
model.learn(total_timesteps=int(20000), callback=custom_checkpoint_callback)

# Save the final model (with updated name reflecting 20000+ steps)
model.save(os.path.join(checkpoint_dir, "rl_model_dqn_60000_steps"))

# Close the environment
env.close()
