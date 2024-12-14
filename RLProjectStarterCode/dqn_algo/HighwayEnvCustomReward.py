import numpy as np
import tensorflow as tf
from highway_env.envs import HighwayEnvFast

class HighwayEnvFastCustomReward(HighwayEnvFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lane_timer = 0  # Timer to track time in non-rightmost lanes
        self.tensorboard_writer = tf.summary.create_file_writer("highway_tensorboard_logs")  # TensorBoard writer
        self.step_counter = 0  # Step counter for logging

    def _reward(self, action: int) -> float:
        reward = 0.0

        # Collision Penalty
        collision_penalty = -50.0 if self.vehicle.crashed else 0.0

        # Lane Reward
        lane_index = self.vehicle.lane_index[2] if self.vehicle.lane_index else None
        rightmost_lane = self.config["lanes_count"] - 1

        if lane_index == rightmost_lane:
            lane_reward = 0.8
            self.lane_timer = 0
        else:
            self.lane_timer += 1
            lane_reward = 0.5 if self._is_overtaking_possible() else -0.3 * self.lane_timer

        # Speed Reward
        min_speed, max_speed = self.config["reward_speed_range"]
        scaled_speed = max(0, min(1, (self.vehicle.speed - min_speed) / (max_speed - min_speed)))
        speed_reward = 0.5 * scaled_speed

        # Smooth Speed Transition Penalty
        speed_change_penalty = -abs(self.vehicle.speed - getattr(self, 'previous_speed', self.vehicle.speed)) * 0.05
        self.previous_speed = self.vehicle.speed

        # Overtaking Reward
        overtaking_reward = sum(
            5.0 for vehicle in self.road.vehicles
            if vehicle != self.vehicle and vehicle.position[0] < self.vehicle.position[0]
            and abs(vehicle.position[1] - self.vehicle.position[1]) < 2
        )

        # Lane Change Reward
        lane_change_reward = 2.0 if action in [1, 2] and self._is_overtaking_possible() else -0.2

        # Awareness Penalty
        awareness_penalty = -0.5 if self._is_vehicle_changing_lane() else 0.0

        # Promising Lane Penalty
        promising_lane_penalty = -0.3 if lane_index != rightmost_lane and not self._is_overtaking_possible() else 0.0

        # Returning Penalty
        returning_penalty = -0.5 if lane_index != rightmost_lane and self.lane_timer > 5 else 0

        # Combine Rewards
        reward += (
                collision_penalty
                + lane_reward
                + speed_reward
                + overtaking_reward
                + lane_change_reward
                + returning_penalty
                + awareness_penalty
                + promising_lane_penalty
                + speed_change_penalty
        )

        # Normalize reward
        total_reward = np.clip(reward, -150.0, 50.0)

        # Log reward components to TensorBoard
        self._log_to_tensorboard(
            collision_penalty, speed_reward, lane_reward, overtaking_reward,
            lane_change_reward, returning_penalty, awareness_penalty,
            promising_lane_penalty, speed_change_penalty, total_reward
        )

        return total_reward

    def _log_to_tensorboard(self, collision, speed, lane, overtaking, lane_change,
                            returning_penalty, awareness_penalty, promising_lane_penalty,
                            speed_change_penalty, total_reward):
        self.step_counter += 1
        with self.tensorboard_writer.as_default():
            tf.summary.scalar("Reward/Collision", collision, step=self.step_counter)
            tf.summary.scalar("Reward/Speed", speed, step=self.step_counter)
            tf.summary.scalar("Reward/Lane", lane, step=self.step_counter)
            tf.summary.scalar("Reward/Overtaking", overtaking, step=self.step_counter)
            tf.summary.scalar("Reward/Lane Change", lane_change, step=self.step_counter)
            tf.summary.scalar("Reward/Returning Penalty", returning_penalty, step=self.step_counter)
            tf.summary.scalar("Reward/Awareness Penalty", awareness_penalty, step=self.step_counter)
            tf.summary.scalar("Reward/Promising Lane Penalty", promising_lane_penalty, step=self.step_counter)
            tf.summary.scalar("Reward/Speed Change Penalty", speed_change_penalty, step=self.step_counter)
            tf.summary.scalar("Reward/Total", total_reward, step=self.step_counter)
        self.tensorboard_writer.flush()

    def _is_car_close(self) -> bool:
        """Check if there is a car close in the same lane."""
        for vehicle in self.road.vehicles:
            if (
                    vehicle != self.vehicle
                    and abs(vehicle.position[1] - self.vehicle.position[1]) < 1  # Same lane
                    and 0 < (vehicle.position[0] - self.vehicle.position[0]) < 5  # Within 5 units ahead
            ):
                return True
        return False

    def _is_overtaking_possible(self) -> bool:
        """Check if overtaking is a viable option."""
        for vehicle in self.road.vehicles:
            if (
                    vehicle != self.vehicle
                    and abs(vehicle.position[1] - self.vehicle.position[1]) < 2  # Close to lane
                    and 0 < (vehicle.position[0] - self.vehicle.position[0]) < 5  # Within overtaking range
            ):
                return True
        return False

    def _is_vehicle_changing_lane(self) -> bool:
        """Check if a nearby vehicle is changing lanes."""
        for vehicle in self.road.vehicles:
            if (
                    vehicle != self.vehicle
                    and abs(vehicle.position[1] - self.vehicle.position[1]) < 2  # Nearby in adjacent lanes
                    and 0 < abs(vehicle.position[0] - self.vehicle.position[0]) < 5  # Within a buffer zone
            ):
                if abs(vehicle.velocity[1]) > 0.5:  # Threshold for lateral velocity
                    return True
        return False
