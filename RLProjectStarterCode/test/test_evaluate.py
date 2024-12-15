import gymnasium
from stable_baselines3 import PPO
from gymnasium.envs.registration import register

# Register the custom environment
register(
    id='HighwayFastCustomReward-v0',
    entry_point='test_reward:HighwayEnvFastCustomReward',  # Path to the reward.py file
)

# Load the environment and model
env = gymnasium.make("HighwayFastCustomReward-v0", render_mode="human")

# Configure the environment for longer episodes and more vehicles
env.unwrapped.config.update({
    "lanes_count": 4,  # Set to 4 lanes
    "vehicles_count": 70,  # Increase vehicles to 70
    "controlled_vehicles": 1,
    "reward_speed_range": [35, 41],  # Speed range for high-speed rewards
    "duration": 1000,  # Increased duration for longer episodes
})

# Load the pre-trained model
model = PPO.load("models/ppo_highway_50000_steps.zip")

# Function to evaluate the model
def evaluate_model(env, model, num_episodes=10):  # Increased number of episodes for long-term evaluation
    total_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset()  # Reset environment
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)  # Predict action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()  # Render environment
            total_reward += reward
            episode_length += 1

        total_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Length: {episode_length}")

    # Summary statistics
    print("\nEvaluation Summary:")
    print(f"Average Reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Average Episode Length: {sum(episode_lengths) / len(episode_lengths):.2f}")

# Evaluate the model
evaluate_model(env, model, num_episodes=50)  # Run for more episodes, or adjust as needed

# Close the environment
env.close()
