import numpy as np
from highway_env.envs import HighwayEnvFast
from highway_env.envs.common.action import Action


class HighwayEnvFastCustomReward(HighwayEnvFast):
    def _reward(self, action: Action) -> float:
        """
        Custom reward function to encourage safety, efficiency, and optimal driving behavior.
        :param action: the last action performed
        :return: the computed reward value (float)
        """
        reward = 0.0

        # reward for safty: penalty for collisions
        if self.vehicle.crashed:
            reward -= 10  # I start with a big penalty for collisions for the beginning

        # efficiency: reward for driving at desired speeds
        speed = self.vehicle.speed
        min_speed, max_speed = self.config["reward_speed_range"]  #reward_speed_range
        if min_speed <= speed <= max_speed:
            reward += 1.0  # reward for being within the desired speed range
        else:
            reward -= 1.0  # penalty for being outside the desired range

        # lane change : Reward for driving in righ most lanes
        # using the lane index directly from the vehicle
        if self.vehicle.lane_index is not None:
            lane_index = self.vehicle.lane_index[2]  # extracting the actual lane index (tuple format: (road, lane, index))
            if lane_index == 0:  # right lane
                reward += 1.0
            elif lane_index == 1:  # middle lane
                reward += 0.5
            else:  # left lane
                reward += 0.1

        # saftey : Penalize unsafe distances to other vehicles
        for neighbor in self.vehicle.road.vehicles:
            if neighbor is not self.vehicle:
                distance = np.linalg.norm(self.vehicle.position - neighbor.position)
                if distance < self.config["ego_spacing"]:  # penalty for being too close
                    reward -= 1.0

        # Off-road driving: Penalize for leaving the road
        if not self.vehicle.on_road:
            reward -= 5.0

        return reward
