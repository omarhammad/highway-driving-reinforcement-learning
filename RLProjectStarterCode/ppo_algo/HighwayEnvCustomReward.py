import numpy as np
from highway_env.envs import HighwayEnvFast


class HighwayEnvFastCustomReward(HighwayEnvFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _reward(self, action: int) -> float:
        reward = 0.0

        # Collision Penalty
        collision_penalty = -100.0 if self.vehicle.crashed else 0.0

        # Lane Reward
        lane_index = self.vehicle.lane_index[2] if self.vehicle.lane_index else None
        if lane_index == 0:
            lane_reward = 0.3  # Rightmost lane
        elif lane_index == 1:
            lane_reward = 0.1  # Middle lane
        elif lane_index == 2:
            lane_reward = -0.1  # Leftmost lane
        else:
            lane_reward = 0.0

        # Speed Reward
        min_speed, max_speed = self.config["reward_speed_range"]
        scaled_speed = max(0, min(1, (self.vehicle.speed - min_speed) / (max_speed - min_speed)))
        speed_reward = 0.5 * scaled_speed

        # Combine Rewards
        reward += collision_penalty + lane_reward + speed_reward

        # Normalize reward
        total_reward = np.clip(reward, -150.0, 50.0)

        # Log reward components
        self._log_reward_components(
            collision=collision_penalty,
            speed=speed_reward,
            lane=lane_reward,
            total=total_reward,
        )

        return total_reward

    def _log_reward_components(self, collision, speed, lane,
                               total):
        print({
            "collision_penalty": collision,
            "speed_reward": speed,
            "lane_reward": lane,
            "vehicle_speed": self.vehicle.speed,
            "lane_index": self.vehicle.lane_index[2] if self.vehicle.lane_index else None,
            "total_reward": total,
        })