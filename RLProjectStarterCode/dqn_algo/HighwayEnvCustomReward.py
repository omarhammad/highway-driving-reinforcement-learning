import numpy as np
from highway_env.envs import HighwayEnvFast
from highway_env.envs.common.action import Action

# Updated reward parameters
collision_penalty = -25.0
speed_penalty_outside_range = -1.0
rightmost_lane_reward = 0.01
middle_lane_reward = 0.004
leftmost_lane_penalty = 0.0
unsafe_distance_penalty = -10.0
off_road_penalty = -15.0
safe_lane_change_reward = 5.0
slowing_down_reward = 3.0
passing_vehicle_reward = 7.0
time_efficiency_reward = 1.0
reward_min = -30.0
reward_max = 15.0

class HighwayEnvFastCustomReward(HighwayEnvFast):
    def _reward(self, action: int) -> float:
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
            return (speed - min_speed) / (max_speed - min_speed)
        return speed_penalty_outside_range

    def _lane_reward(self, lane_index):
        if lane_index == 0:
            return rightmost_lane_reward
        elif lane_index == 1:
            return middle_lane_reward
        return leftmost_lane_penalty

    def _evaluate_surroundings(self, action):
        reward = 0.0
        for neighbor in self.vehicle.road.vehicles:
            if neighbor is not self.vehicle:
                distance = np.linalg.norm(self.vehicle.position - neighbor.position)
                same_lane = neighbor.lane_index[2] == self.vehicle.lane_index[2]

                # Unsafe Distance Penalty
                if distance < self.config["ego_spacing"]:
                    reward += unsafe_distance_penalty
                    # Reward slowing down to avoid a collision
                    if same_lane and self.vehicle.speed > neighbor.speed:
                        reward += slowing_down_reward

                # Safe Lane Change Reward
                if action in [1, 2]:  # Lane changes
                    if not same_lane or distance >= self.config["ego_spacing"]:
                        reward += safe_lane_change_reward

                # Passing Vehicle Reward
                if not same_lane and self.vehicle.position[0] > neighbor.position[0] and abs(distance) > self.config["ego_spacing"]:
                    reward += passing_vehicle_reward
        return reward

    def _time_efficiency_reward(self):
        return time_efficiency_reward * (1.0 - (self.time / self.config["duration"]))

    def _log_reward_components(self, reward):
        self.reward_log = {
            "collision_penalty": self._collision_penalty(),
            "speed_reward": self._speed_reward(self.vehicle.speed),
            "lane_reward": self._lane_reward(self.vehicle.lane_index[2] if self.vehicle.lane_index else None),
            "off_road_penalty": off_road_penalty if not self.vehicle.on_road else 0.0,
            "time_efficiency_reward": self._time_efficiency_reward(),
            "total_reward": reward,
        }
        print(self.reward_log)
