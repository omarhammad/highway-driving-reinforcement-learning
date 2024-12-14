import pprint
import gymnasium
import highway_env
from gymnasium import register
from stable_baselines3 import PPO  # Updated to PPO for evaluation

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
# Load the trained PPO model
model = PPO.load("./logs/rl_model_ppo_7000_steps.zip")  # Adjust path if needed

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
