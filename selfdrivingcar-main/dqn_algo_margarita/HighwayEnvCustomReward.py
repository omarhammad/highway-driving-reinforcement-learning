import tensorflow as tf
from highway_env.envs import HighwayEnvFast


class HighwayEnvFastCustomReward(HighwayEnvFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tensorboard_writer = tf.summary.create_file_writer("ppo_highway_logs/rewards/1")
        self.step_counter = 0
        self.log_rewards = True
        self.collision_reward = -0.4
        self.high_speed_reward = 0.5
        self.lane_reward = 0.1

    def reset(self, **kwargs):
        self.timestep = 0
        return super().reset(**kwargs)

    def step(self, action):
        self.timestep += 1
        return super().step(action)

    def _reward(self, action):
        reward = 0.0

        # Collision penalty
        collision_penalty = -0.4 if self.vehicle.crashed else 0.0

        # Rightmost lane reward
        total_lanes = self.config["lanes_count"]
        current_lane = self.vehicle.lane_index[2]
        lane_reward = 0.1 if current_lane == (total_lanes - 1) else 0

        # High-speed reward
        speed = self.vehicle.speed
        reward_speed_range = self.config["reward_speed_range"]

        # Calculate the high-speed reward, map the speed between 0 and 1
        speed_reward_scale = min(max((speed - reward_speed_range[0]) /
                                     (reward_speed_range[1] - reward_speed_range[0]), 0), 1)
        speed_reward = 0.5 * speed_reward_scale  # Scale to max 0.4 when at full speed

        # Combine rewards
        reward = collision_penalty + lane_reward + speed_reward

        # Normalize reward using min-max scaling for flexibility
        reward_min = -1  # Minimum reward observed during training
        reward_max = 2  # Maximum reward observed during training
        normalized_reward = (reward - reward_min) / (reward_max - reward_min)  # Min-Max normalization

        # Log the reward components at each step for detailed tracking (only during training)
        if self.log_rewards:
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

    def disable_logging(self):
        """Disable logging for evaluation."""
        self.log_rewards = False

    def enable_logging(self):
        """Enable logging for training."""
        self.log_rewards = True