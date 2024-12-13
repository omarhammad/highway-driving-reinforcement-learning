import numpy as np
from highway_env.envs import HighwayEnvFast

# Adjusted reward parameters for goal-oriented behavior
collision_penalty = -40.0  # Penalty for collisions
safe_distance_reward = 10.0  # Reward for maintaining a safe distance
rightmost_lane_reward = 0.5  # Higher reward for staying in the right lane
middle_lane_reward = 0.02
leftmost_lane_penalty = -0.05  # Slight penalty for staying in the left lane
unsafe_distance_penalty = -30.0  # Penalty for being too close to another vehicle
safe_lane_change_reward = 10.0  # Reward for safe lane changes
slowing_down_reward = 10.0  # Reward for slowing down to avoid collisions
passing_vehicle_reward = 30.0  # Reward for safely overtaking
unnecessary_lane_change_penalty = -20.0  # Penalty for unnecessary lane changes
unsafe_lane_change_penalty = -30.0  # Penalty for unsafe lane changes
lane_stability_reward = 15.0  # Higher reward for staying in a stable lane
time_efficiency_multiplier = 2.0  # Reduced emphasis on speeding
reward_min = -100.0
reward_max = 50.0  # Increased max reward for better scaling


class HighwayEnvFastCustomReward(HighwayEnvFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0

    def _reward(self, action: int) -> float:
        self.step_count += 1
        reward = 0.0
        speed = self.vehicle.speed

        # Apply penalties and rewards
        collision_penalty_val = self._collision_penalty()
        speed_reward_val = self._speed_reward(speed)
        lane_reward_val = self._lane_reward(self.vehicle.lane_index[2] if self.vehicle.lane_index else None)
        surroundings_reward_val = self._evaluate_surroundings(action)
        time_efficiency_reward_val = self._time_efficiency_reward()

        # Reward for staying in a stable lane
        lane_stability_reward_val = lane_stability_reward if action == 0 else 0.0

        # Sum all rewards
        reward += (
                collision_penalty_val
                + speed_reward_val
                + lane_reward_val
                + surroundings_reward_val
                + time_efficiency_reward_val
                + lane_stability_reward_val
        )

        # Clip the reward
        reward = np.clip(reward, reward_min, reward_max)

        # Log reward components
        self._log_reward_components(
            collision_penalty_val,
            speed_reward_val,
            lane_reward_val,
            surroundings_reward_val,
            time_efficiency_reward_val,
            lane_stability_reward_val,
            reward,
        )

        return reward

    def _collision_penalty(self):
        return collision_penalty if self.vehicle.crashed else 0.0

    def _speed_reward(self, speed):
        min_speed, max_speed = self.config["reward_speed_range"]
        speed_penalty_outside_range = -2.0  # Define this value explicitly here
        if min_speed <= speed <= max_speed:
            return 5 * ((speed - min_speed) / (max_speed - min_speed))
        return speed_penalty_outside_range

    def _lane_reward(self, lane_index):
        if lane_index == 0:
            return rightmost_lane_reward
        elif lane_index == 1:
            return middle_lane_reward
        elif lane_index == 2:
            return leftmost_lane_penalty
        return 0.0

    def _evaluate_surroundings(self, action):
        reward = 0.0
        safe_passing_range = self.config.get("safe_passing_range", [5.0, 15.0])
        ego_speed = self.vehicle.speed

        for neighbor in self.vehicle.road.vehicles:
            if neighbor is not self.vehicle:
                distance = np.linalg.norm(self.vehicle.position - neighbor.position)
                same_lane = neighbor.lane_index[2] == self.vehicle.lane_index[2]

                # Penalize unsafe distance
                if distance < self.config["ego_spacing"]:
                    reward += unsafe_distance_penalty
                    if same_lane and ego_speed > neighbor.speed:
                        reward += slowing_down_reward

                # Reward safe overtaking
                if action in [1, 2]:  # Lane change actions
                    if safe_passing_range[0] <= distance <= safe_passing_range[1]:
                        reward += passing_vehicle_reward
                    else:
                        reward += unsafe_lane_change_penalty  # Unsafe lane change penalty

                # Penalize unnecessary lane changes
                if action in [1, 2] and (same_lane or distance > safe_passing_range[1]):
                    reward += unnecessary_lane_change_penalty

        # Reward for maintaining a safe distance
        if distance >= self.config["ego_spacing"]:
            reward += safe_distance_reward

        return reward

    def _time_efficiency_reward(self):
        return time_efficiency_multiplier * (1.0 - (self.time / self.config["duration"]))

    def _log_reward_components(self, collision, speed, lane, surroundings, time_efficiency, lane_stability,
                               total):
        print({
            "collision_penalty": collision,
            "speed_reward": speed,
            "lane_reward": lane,
            "surroundings_reward": surroundings,
            "time_efficiency_reward": time_efficiency,
            "lane_stability_reward": lane_stability,
            "total_reward": total,
            "vehicle_speed": self.vehicle.speed,
            "lane_index": self.vehicle.lane_index[2] if self.vehicle.lane_index else None,
            "step_count": self.step_count,
        })
