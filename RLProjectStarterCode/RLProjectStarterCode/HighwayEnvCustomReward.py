import numpy as np
from highway_env.envs import HighwayEnvFast
from highway_env.envs.common.action import Action


class HighwayEnvFastCustomReward(HighwayEnvFast):
    def _reward(self, action: Action) -> float:
        """
        custom reward function to encourage safety, efficiency, and optimal driving behavior.
        :param action: the last action performed
        :return: the computed reward value (float)
        """
        reward = 0.0

        # safety: penalty for collisions
        if self.vehicle.crashed:
            reward -= 10.0  # large penalty for collisions

        # efficiency: reward for driving at desired speeds
        speed = self.vehicle.speed
        min_speed, max_speed = self.config["reward_speed_range"]
        if min_speed <= speed <= max_speed:
            reward += (speed - min_speed) / (max_speed - min_speed)  # scaled reward for staying in the range
        else:
            reward -= 2.0  # higher penalty for being outside the range

        # lane discipline: reward for staying in rightmost lanes
        if self.vehicle.lane_index is not None:
            lane_index = self.vehicle.lane_index[2]  # extracting the actual lane index
            if lane_index == 0:  # rightmost lane
                reward += 2.0
            elif lane_index == 1:  # middle lane
                reward += 1.0
            else:  # left lane
                reward -= 0.5  # small penalty for leftmost lane to encourage right-lane preference

        # safety: penalize unsafe distances to other vehicles
        for neighbor in self.vehicle.road.vehicles:
            if neighbor is not self.vehicle:
                distance = np.linalg.norm(self.vehicle.position - neighbor.position)
                if distance < self.config["ego_spacing"]:
                    reward -= 5.0 / (distance + 1e-5)  # heavier penalty for closer proximity

        # off-road driving: penalize for leaving the road
        if not self.vehicle.on_road:
            reward -= 5.0  # penalty for being off-road

        # normalize the total reward to keep it within a consistent range
        reward = np.clip(reward, -10.0, 10.0)

        return reward
