import numpy as np
from highway_env.envs import HighwayEnvFast
from highway_env.envs.common.action import Action

# Updated reward parameters
collision_penalty = -25.0  # Increased penalty for collisions
speed_penalty_outside_range = -1.0
rightmost_lane_reward = 0.01  # Reduced emphasis on staying in the rightmost lane
middle_lane_reward = 0.004  # Balanced reward for using the middle lane
leftmost_lane_penalty = 0.0  # Neutral for leftmost lane to encourage flexibility
unsafe_distance_penalty = -10.0  # Severe penalty for unsafe distances
off_road_penalty = -15.0  # Increased penalty for going off-road
safe_lane_change_reward = 5.0  # Encourage safe lane changes more
slowing_down_reward = 3.0  # Higher reward for slowing down in high-risk situations
passing_vehicle_reward = 7.0  # New reward for overtaking a vehicle safely
reward_min = -30.0
reward_max = 15.0  # Increased maximum reward to accommodate new passing reward

class HighwayEnvFastCustomReward(HighwayEnvFast):
    def _reward(self, action: int) -> float:
        reward = 0.0

        # Collision penalty
        if self.vehicle.crashed:
            reward += collision_penalty

        # Speed management
        speed = self.vehicle.speed
        min_speed, max_speed = self.config["reward_speed_range"]
        if min_speed <= speed <= max_speed:
            reward += (speed - min_speed) / (max_speed - min_speed)
        else:
            reward += speed_penalty_outside_range

        # Lane preference - much less emphasis on sticking to the right
        if self.vehicle.lane_index is not None:
            lane_index = self.vehicle.lane_index[2]
            if lane_index == 0:
                reward += rightmost_lane_reward
            elif lane_index == 1:
                reward += middle_lane_reward
            else:
                reward += leftmost_lane_penalty

        # Check surroundings
        for neighbor in self.vehicle.road.vehicles:
            if neighbor is not self.vehicle:
                distance = np.linalg.norm(self.vehicle.position - neighbor.position)
                same_lane = neighbor.lane_index[2] == self.vehicle.lane_index[2]

                # Unsafe distance penalty
                if distance < self.config["ego_spacing"]:
                    reward += unsafe_distance_penalty
                    # Reward slowing down to avoid a collision
                    if same_lane and speed > neighbor.speed:
                        reward += slowing_down_reward

                # Safe lane change reward
                if action in [1, 2]:  # Lane changes
                    if not same_lane or distance >= self.config["ego_spacing"]:
                        reward += safe_lane_change_reward

                # Passing vehicle reward
                if not same_lane and self.vehicle.position[0] > neighbor.position[0]:
                    reward += passing_vehicle_reward  # Reward for successfully overtaking a vehicle

        # Off-road penalty
        if not self.vehicle.on_road:
            reward += off_road_penalty

        # Clip the reward
        reward = np.clip(reward, reward_min, reward_max)

        return reward
