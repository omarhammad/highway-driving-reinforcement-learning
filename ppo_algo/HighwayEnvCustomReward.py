import tensorflow as tf
from highway_env.envs import HighwayEnvFast


class HighwayEnvFastCustomReward(HighwayEnvFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TensorBoard writer to log the reward components
        self.tensorboard_writer = tf.summary.create_file_writer("ppo_highway_logs/rewards")
        self.step_counter = 0  # Step counter for logging

    def reset(self, **kwargs):
        self.timestep = 0
        return super().reset(**kwargs)

    def step(self, action):
        self.timestep += 1
        return super().step(action)

    def _reward(self, action):
        reward = 0.0

        # Collision penalty
        collision_penalty = -1.0 if self.vehicle.crashed else 0.0

        # Rightmost lane reward
        total_lanes = self.config["lanes_count"]
        current_lane = self.vehicle.lane_index[2]
        lane_reward = 0.5 if current_lane == (total_lanes - 1) else -0.3  # Increased rightmost lane reward

        # High-speed reward
        speed = self.vehicle.speed
        reward_speed_range = self.config["reward_speed_range"]

        # Calculate the high-speed reward, map the speed between 0 and 1
        speed_reward = min(max((speed - reward_speed_range[0]) /
                               (reward_speed_range[1] - reward_speed_range[0]), 0), 1)
        speed_reward = 0.5 * speed_reward  # Scale to max 0.5 when at full speed

        # Combine rewards
        reward = 0.3 * collision_penalty + 0.4 * lane_reward + 0.3 * speed_reward  # Adjusted weights

        # Normalize reward using min-max scaling for flexibility
        reward_min = -1  # Minimum reward observed during training
        reward_max = 2  # Maximum reward observed during training
        normalized_reward = (reward - reward_min) / (reward_max - reward_min)  # Min-Max normalization

        # Log the reward components at each step for detailed tracking
        self._log_reward_components(
            collision=collision_penalty,
            speed=speed_reward,
            lane=lane_reward,
            total=normalized_reward,
        )

        return normalized_reward

    def _log_reward_components(self, collision, speed, lane, total):
        """Log the individual reward components for each timestep"""
        with self.tensorboard_writer.as_default():
            step = self.step_counter
            tf.summary.scalar("Reward/Collision", collision, step=step)
            tf.summary.scalar("Reward/Speed", speed, step=step)
            tf.summary.scalar("Reward/Lane", lane, step=step)
            tf.summary.scalar("Reward/Total", total, step=step)
            self.step_counter += 1
