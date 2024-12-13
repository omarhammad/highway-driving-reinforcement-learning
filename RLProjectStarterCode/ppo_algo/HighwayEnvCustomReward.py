import numpy as np
from highway_env.envs import HighwayEnvFast
from highway_env.envs.common.action import Action

# Adjusted reward parameters
collision_penalty = -50.0
speed_penalty_outside_range = -2.0
rightmost_lane_reward = 0.02
middle_lane_reward = 0.01
leftmost_lane_penalty = -0.01
unsafe_distance_penalty = -15.0
off_road_penalty = -20.0
safe_lane_change_reward = 7.0
slowing_down_reward = 5.0
passing_vehicle_reward = 15.0
unnecessary_lane_change_penalty = -5.0
time_efficiency_multiplier = 2.0
reward_min = -50.0
reward_max = 25.0


class HighwayEnvFastCustomReward(HighwayEnvFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0  # Initialize step counter

    def _reward(self, action: int) -> float:
        self.step_count += 1  # Increment step count
        reward = 0.0
        speed = self.vehicle.speed

        # Collision Penalty
        reward += self._collision_penalty()

        # Speed Reward
        reward += self._speed_reward(speed)

        # Lane Preference Reward
        if self.vehicle.lane_index is not None:
            lane_index = self.vehicle.lane_index[2]
            reward += self._lane_reward(lane_index)

        # Surroundings and Lane Change Evaluation
        reward += self._evaluate_surroundings(action)

        # Off-road Penalty
        if not self.vehicle.on_road:
            reward += off_road_penalty

        # Time Efficiency Reward
        reward += self._time_efficiency_reward()

        # Clip the reward
        reward = np.clip(reward, reward_min, reward_max)

        # Debugging logs
        self._log_reward_components(reward)

        return reward

    def _collision_penalty(self):
        if self.vehicle.crashed:
            collision_speed_penalty = collision_penalty * (self.vehicle.speed / self.config["reward_speed_range"][1])
            return collision_speed_penalty
        return 0.0

    def _speed_reward(self, speed):
        min_speed, max_speed = self.config["reward_speed_range"]
        if min_speed <= speed <= max_speed:
            return 4 * ((speed - min_speed) / (max_speed - min_speed))  # Amplified reward for optimal speed
        return speed_penalty_outside_range

    def _lane_reward(self, lane_index):
        if lane_index == 0:
            return rightmost_lane_reward
        elif lane_index == 1:
            return middle_lane_reward
        elif lane_index == 2:  # Slight penalty for being in leftmost lane
            return leftmost_lane_penalty
        return 0.0

    def _evaluate_surroundings(self, action):
        reward = 0.0
        safe_passing_range = self.config.get("safe_passing_range", [5.0, 15.0])  # Configurable safe range
        ego_speed = self.vehicle.speed

        for neighbor in self.vehicle.road.vehicles:
            if neighbor is not self.vehicle:
                distance = np.linalg.norm(self.vehicle.position - neighbor.position)
                same_lane = neighbor.lane_index[2] == self.vehicle.lane_index[2]
                relative_position = self.vehicle.position[0] - neighbor.position[0]  # Positive: ego is ahead
                relative_speed = ego_speed - neighbor.speed

                # Unsafe Distance Penalty
                if distance < self.config["ego_spacing"]:
                    reward += unsafe_distance_penalty
                    if same_lane and ego_speed > neighbor.speed:  # Slowing down to avoid collision
                        reward += slowing_down_reward

                # Safe Passing Reward
                if action in [1, 2] and safe_passing_range[0] <= distance <= safe_passing_range[1]:
                    if relative_position > 0:  # Ego successfully passed the neighbor
                        reward += passing_vehicle_reward * (1 + relative_speed / max(ego_speed, 1e-3))

                # Unnecessary Lane Change Penalty
                if action in [1, 2] and (same_lane or distance > safe_passing_range[1]):
                    reward += unnecessary_lane_change_penalty

        return reward

    def _time_efficiency_reward(self):
        return time_efficiency_multiplier * (1.0 - (self.time / self.config["duration"]))

    def _log_reward_components(self, reward):
        self.reward_log = {
            "collision_penalty": self._collision_penalty(),
            "speed_reward": self._speed_reward(self.vehicle.speed),
            "lane_reward": self._lane_reward(self.vehicle.lane_index[2] if self.vehicle.lane_index else None),
            "off_road_penalty": off_road_penalty if not self.vehicle.on_road else 0.0,
            "time_efficiency_reward": self._time_efficiency_reward(),
            "total_reward": reward,
            "step_count": self.step_count,  # Log step count
        }
        print(self.reward_log)
