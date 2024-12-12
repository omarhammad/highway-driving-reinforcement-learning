from itertools import product

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium
from gymnasium.envs.registration import register
import pprint

# Register the custom environment
register(
    id='HighwayFastCustomReward-v0',
    entry_point='HighwayEnvCustomReward:HighwayEnvFastCustomReward',
)

def perform_grid_search(env, param_grid, total_timesteps=10000, eval_episodes=5):
    # Generate combinations of hyperparameters
    param_combinations = list(product(*param_grid.values()))

    best_reward = float('-inf')
    best_params = None

    for params in param_combinations:
        learning_rate, gamma, batch_size = params
        print(f"Testing params: lr={learning_rate}, gamma={gamma}, batch_size={batch_size}")

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            n_steps=512,
            batch_size=batch_size,
            n_epochs=20,
            learning_rate=learning_rate,
            gamma=gamma,
            verbose=1,
            tensorboard_log="highway_ppo/",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Train the model for a short duration to evaluate performance
        model.learn(total_timesteps=total_timesteps)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes)

        # Track best configuration
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = params

        print(f"Mean reward: {mean_reward}")

    print(f"Best params: {best_params}, Best reward: {best_reward}")
    return best_params, best_reward

if __name__ == "__main__":
    # Create the environment
    env = gymnasium.make("HighwayFastCustomReward-v0", render_mode="human")

    # Configure the environment
    env.unwrapped.config.update({
        "lanes_count": 4,
        "vehicles_count": 70,
        "duration": 1000,
        "reward_speed_range": [20, 30],
        "speed_coefficient": 1.2,
        "collision_coefficient": 1.2,
        "reward_min": -40.0,
        "reward_max": 25.0,
        "ego_spacing": 5.0,
    })
    pprint.pprint(env.unwrapped.config)

    # Define hyperparameter grid
    param_grid = {
        "learning_rate": [1e-4, 5e-4, 1e-3],
        "gamma": [0.9, 0.95, 0.99],
        "batch_size": [64, 128, 256]
    }

    # Perform grid search
    best_params, best_reward = perform_grid_search(env, param_grid)

    # Print final results
    print(f"Best hyperparameters: {best_params}")
    print(f"Best mean reward: {best_reward}")

    # Close environment
    env.close()
