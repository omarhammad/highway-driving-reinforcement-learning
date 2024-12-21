import pprint
import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3 import PPO

# Register the custom environment
register(
    id='HighwayFastCustomReward-v0',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',  # Path to the reward.py file
)

# Configure environment settings to match training configuration
env_config = {
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
}

# Create the evaluation environment (disable logging)
env = gymnasium.make("HighwayFastCustomReward-v0", render_mode="human")
env.unwrapped.config.update(env_config)
env.unwrapped.disable_logging()

pprint.pprint(env.unwrapped.config)

# Load the pre-trained model
model = PPO.load("models/ppo_highway_120000_steps.zip")  # Ensure correct saved model file name


# Function to evaluate the model
def evaluate_model(env, model, num_episodes=10):
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
evaluate_model(env, model, num_episodes=10)

# Close the environment
env.close()
