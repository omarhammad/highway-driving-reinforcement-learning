import numpy as np
import tensorflow as tf
from highway_env.envs import HighwayEnvFast

# Reward Parameters
COLLISION_PENALTY = -20.0          # Strong penalty for collisions
RIGHTMOST_LANE_REWARD = 0.2        # Small reward for staying in the rightmost lane
HIGH_SPEED_REWARD_WEIGHT = 1.0     # Moderate reward for maintaining target speed
SPEED_RANGE = [30, 36]             # Speed range for high-speed rewards
SLOW_CAR_PENALTY = -1.0            # Penalty for being stuck behind a slow car
OVERTAKE_REWARD = 3.0              # Reward for successful overtaking
RETURN_RIGHTMOST_REWARD = 0.5      # Reward for returning to rightmost lane after overtaking
SLOW_CAR_DISTANCE = 5.0            # Distance threshold for detecting slow cars

# Reward Weights
W_COLLISION = 1.2
W_LANE = 0.8
W_SPEED = 1.0
W_SLOW_CAR = 1.2
W_OVERTAKE = 2.0
W_RETURN_RIGHTMOST = 0.7


class HighwayEnvFastCustomReward(HighwayEnvFast):
    def __init__(self, *args, log_rewards=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.epsilon = 1e-8

        self.last_lane = None
        self.last_speed = None

        self.log_rewards = log_rewards  # Flag to enable/disable TensorBoard logging

        # TensorBoard writer setup
        if self.log_rewards:
            self.tensorboard_writer = tf.summary.create_file_writer("highway_tensorboard_logs")
        self.global_step = 0

    def _reward(self, action: int) -> float:
        vehicle = self.vehicle

        # Initialize components
        collision_reward = lane_reward = speed_reward = 0.0
        slow_car_reward = overtaking_reward = return_rightmost_reward = 0.0

        # Collision penalty
        if vehicle.crashed:
            collision_reward = COLLISION_PENALTY * W_COLLISION

        # Lane reward with stagnation decay
        total_lanes = self.config["lanes_count"]
        current_lane = vehicle.lane_index[2]
        lane_reward = ((current_lane + 1) / total_lanes * RIGHTMOST_LANE_REWARD) * W_LANE
        if current_lane == total_lanes - 1:
            lane_reward -= 0.05 * min(self.global_step, 50)  # Penalize stagnation after step 50

        # High-speed reward (balanced with slow car logic)
        speed = vehicle.speed
        scaled_speed = (speed - SPEED_RANGE[0]) / (SPEED_RANGE[1] - SPEED_RANGE[0])
        scaled_speed = np.clip(scaled_speed, 0.0, 1.0)
        speed_reward = scaled_speed * HIGH_SPEED_REWARD_WEIGHT * W_SPEED

        # Detect slow cars
        close_vehicles = vehicle.road.close_vehicles_to(vehicle, distance=SLOW_CAR_DISTANCE)
        front_vehicle = None
        for v in close_vehicles:
            if v.lane_index == vehicle.lane_index and v.position[0] > vehicle.position[0]:
                if front_vehicle is None or v.position[0] < front_vehicle.position[0]:
                    front_vehicle = v

        if front_vehicle:
            front_distance = front_vehicle.position[0] - vehicle.position[0]
            front_speed = front_vehicle.speed
            if front_distance < SLOW_CAR_DISTANCE and front_speed < SPEED_RANGE[0]:
                if self.global_step > 10:  # Grace period
                    slow_car_reward = SLOW_CAR_PENALTY * W_SLOW_CAR

        # Overtaking reward
        if self.last_lane is not None and self.last_speed is not None:
            if self.last_lane == total_lanes - 1 and current_lane != total_lanes - 1:
                if speed > self.last_speed + 0.5:
                    overtaking_reward = OVERTAKE_REWARD * W_OVERTAKE

            # Return to rightmost lane reward
            if current_lane == total_lanes - 1 and self.last_lane != total_lanes - 1:
                if SPEED_RANGE[0] <= speed <= SPEED_RANGE[1]:
                    return_rightmost_reward = RETURN_RIGHTMOST_REWARD * W_RETURN_RIGHTMOST
            else:
                return_rightmost_reward -= 0.1  # Small penalty for lingering in other lanes

        # Combine and clip rewards
        raw_reward = (collision_reward +
                      lane_reward +
                      speed_reward +
                      slow_car_reward +
                      overtaking_reward +
                      return_rightmost_reward)
        raw_reward = np.clip(raw_reward, -50, 50)

        # Normalize reward
        self.reward_mean = 0.99 * self.reward_mean + 0.01 * raw_reward
        self.reward_std = 0.99 * self.reward_std + 0.01 * (raw_reward - self.reward_mean) ** 2
        normalized_reward = (raw_reward - self.reward_mean) / (np.sqrt(self.reward_std) + self.epsilon)

        # Update last lane and speed
        self.last_lane = current_lane
        self.last_speed = speed

        # TensorBoard logging (if enabled)
        if self.log_rewards:
            with self.tensorboard_writer.as_default():
                tf.summary.scalar("Reward/Collision", collision_reward, step=self.global_step)
                tf.summary.scalar("Reward/Lane", lane_reward, step=self.global_step)
                tf.summary.scalar("Reward/Speed", speed_reward, step=self.global_step)
                tf.summary.scalar("Reward/Slow_Car", slow_car_reward, step=self.global_step)
                tf.summary.scalar("Reward/Overtaking", overtaking_reward, step=self.global_step)
                tf.summary.scalar("Reward/Return_Rightmost", return_rightmost_reward, step=self.global_step)
                tf.summary.scalar("Reward/Total_Raw", raw_reward, step=self.global_step)

        self.global_step += 1

        return normalized_reward

    def reset(self, **kwargs):
        self.reward_mean = 0.0
        self.reward_std = 1.0
        obs = super().reset(**kwargs)
        self.last_lane = None
        self.last_speed = None
        return obs
