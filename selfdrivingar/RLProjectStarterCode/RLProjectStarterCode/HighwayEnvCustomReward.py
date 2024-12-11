import numpy as np
from highway_env import utils
from highway_env.envs import HighwayEnvFast
from highway_env.envs.common.action import Action


class HighwayEnvFastCustomReward(HighwayEnvFast):

    def _reward(self, action: Action) -> float:
        """
        Custom reward function for complex environments:
        - Encourage high speeds.
        - Reward staying in the rightmost lanes.
        - Penalize collisions heavily.
        - Penalize unsafe lane changes or reckless driving.
        """
        # Get vehicle state
        vehicle = self.vehicle
        speed = vehicle.speed
        lane_index = self.road.network.get_closest_lane_index(vehicle.position)  # Correct lane index
        rightmost_lane = self.config["lanes_count"] - 1  # Get lane count from config

        # Handle missing SPEED_LIMIT key
        speed_limit = self.config.get("SPEED_LIMIT", 30)  # Default speed limit is 30
        speed_reward = speed / speed_limit  # Normalize speed reward

        # Reward for rightmost lane
        lane_reward = 1.0 if lane_index == rightmost_lane else 0.5
        collision_penalty = -20.0 if self.vehicle.crashed else 0.0  # Penalize collisions
        time_penalty = -0.1  # Small penalty per step to encourage efficiency

        # Penalize unsafe lane changes
        if action == Action.LANE_CHANGE_LEFT or action == Action.LANE_CHANGE_RIGHT:
            lane_change_penalty = -1.0
        else:
            lane_change_penalty = 0.0

        # Aggregate rewards
        reward = speed_reward + lane_reward + collision_penalty + lane_change_penalty + time_penalty

        # Clip the reward to avoid extreme values
        reward = np.clip(reward, -20, 10)

        return reward

# class HighwayEnvFastCustomReward(HighwayEnvFast):
#
#
#     def _reward(self, action: Action) -> float:
#         """
#         The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
#         :param action: the last action performed
#         :return: the corresponding reward
#         """
#         # TODO: Implement a custom reward function here, and return the reward value. The value should be a single float. For example, return -10 for a collision, 1 for staying on the road, and 0 otherwise.
#         reward = 0
#         return reward
