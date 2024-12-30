import tensorflow as tf
from highway_env.envs import HighwayEnvFast

class HighwayEnvFastCustomReward(HighwayEnvFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tensorboard_writer = tf.summary.create_file_writer("dqn_highway_logs/rewards")
        self.step_counter = 0
        self.log_rewards = True

        # initialize variables to track minimum and maximum rewards
        self.reward_min = float('inf')  # initially set to a very high value
        self.reward_max = -float('inf')  # initially set to a very low value

    def reset(self, **kwargs):
        # reset the timestep counter at the start of a new episode
        self.timestep = 0
        return super().reset(**kwargs)

    def step(self, action):
        # increment the timestep counter each time step is called
        self.timestep += 1
        return super().step(action)

    def _reward(self, action):
        # calculate a reward for avoiding collisions
        no_collision_reward = 0.4 if not self.vehicle.crashed else 0.0

        # calculate a reward for driving in the rightmost lane
        total_lanes = self.config["lanes_count"]
        current_lane = self.vehicle.lane_index[2]
        lane_reward = 0.2 if current_lane == (total_lanes - 1) else 0

        # calculate a reward for maintaining a high speed within the desired range
        speed = self.vehicle.speed
        reward_speed_range = self.config["reward_speed_range"]
        speed_reward_scale = min(max((speed - reward_speed_range[0]) /
                                     ((reward_speed_range[1]) - reward_speed_range[0]), 0), 1)
        speed_reward = 0.5 * speed_reward_scale  # scale the reward to a maximum of 0.5

        # calculate a reward for overtaking vehicles ahead
        overtaking_reward = 0
        for other_vehicle in self.road.vehicles:
            if other_vehicle is not self.vehicle:  # ignore the ego-vehicle
                distance_ahead = other_vehicle.position[0] - self.vehicle.position[0]
                if 0 < distance_ahead < 15:  # check if the vehicle is ahead within a specific range
                    if self.vehicle.speed > other_vehicle.speed:  # check if overtaking
                        overtaking_reward += 0.6  # reward overtaking

        # provide an incentive for lane changes when overtaking
        lane_change_incentive = 0
        if current_lane != (total_lanes - 1):  # check if not in the rightmost lane
            lane_change_incentive = 0.1 if overtaking_reward > 0 else 0  # reward lane changes for overtaking

        # combine all reward components into a single reward
        reward = (no_collision_reward +
                  lane_reward +
                  speed_reward +
                  overtaking_reward +
                  lane_change_incentive)

        # dynamically update minimum and maximum rewards for normalization
        self.reward_min = min(self.reward_min, reward)
        self.reward_max = max(self.reward_max, reward)

        # normalize the reward using min-max scaling
        if self.reward_max - self.reward_min > 0:  # avoid division by zero
            normalized_reward = (reward - self.reward_min) / (self.reward_max - self.reward_min)
        else:
            normalized_reward = reward  # if min and max are equal, use the raw reward

        # log reward components during training if logging is enabled
        if self.log_rewards:
            self._log_reward_components(
                no_collision=no_collision_reward,
                speed=speed_reward,
                lane=lane_reward,
                overtaking=overtaking_reward,
                lane_change=lane_change_incentive,
                total=normalized_reward,
            )

        return normalized_reward

    def _log_reward_components(self, no_collision, speed, lane, overtaking, lane_change, total):
        # log individual reward components to tensorboard for detailed tracking
        with self.tensorboard_writer.as_default():
            step = self.step_counter
            tf.summary.scalar("Reward/No_Collision", no_collision, step=step)
            tf.summary.scalar("Reward/Speed", speed, step=step)
            tf.summary.scalar("Reward/Lane", lane, step=step)
            tf.summary.scalar("Reward/Overtaking", overtaking, step=step)
            tf.summary.scalar("Reward/Lane_Change", lane_change, step=step)
            tf.summary.scalar("Reward/Total", total, step=step)
            self.step_counter += 1  # increment the step counter for each log

    def disable_logging(self):
        # disable reward logging, typically used during evaluation
        self.log_rewards = False

    def enable_logging(self):
        # enable reward logging, typically used during training
        self.log_rewards = True
