import pprint
import gymnasium
import highway_env
from gymnasium import register
from stable_baselines3 import PPO  # Updated to PPO for evaluation

# Register the custom environment
register(
    id='CustomRewardEnv',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)

# Create the environment
env = gymnasium.make('CustomRewardEnv', render_mode='rgb_array')

# Update the environment configuration to match the training configuration
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

# Load the trained PPO model
model = PPO.load("./logs/rl_model_ppo_10000_steps.zip")  # Adjust path if needed

# Run the evaluation for a limited number of episodes
episode_count = 10
for episode in range(episode_count):
    obs, info = env.reset()  # Reset the environment and get the initial observation
    done = truncated = False  # Reset the done and truncated flags
    total_rewards = 0  # Track total rewards for the episode

    while not (done or truncated):  # Continue the simulation until done or truncated
        action, _states = model.predict(obs, deterministic=True)  # Predict action without exploration (deterministic)
        obs, reward, done, truncated, info = env.step(action)  # Perform the action in the environment
        total_rewards += reward  # Accumulate rewards for the episode
        env.render()  # Render the environment

    print(f"Episode {episode + 1}/{episode_count} - Total Reward: {total_rewards}")

# Close the environment
env.close()
