import numpy as np
from highway_env import utils
from highway_env.envs import HighwayEnvFast
from highway_env.envs.common.action import Action


class HighwayEnvFastCustomReward(HighwayEnvFast):


    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # TODO: Implement a custom reward function here, and return the reward value. The value should be a single float. For example, return -10 for a collision, 1 for staying on the road, and 0 otherwise.
        reward = 0
        return reward
