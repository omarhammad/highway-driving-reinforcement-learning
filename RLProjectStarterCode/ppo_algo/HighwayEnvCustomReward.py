import numpy as np
from highway_env.envs import HighwayEnvFast


class HighwayEnvFastCustomReward(HighwayEnvFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lane_timer = 0  # Timer to track time in non-rightmost lanes

    def _reward(self, action: int) -> float:
        reward = 0.0

        # Collision Penalty
        collision_penalty = -50.0 if self.vehicle.crashed else 0.0  # Adjusted to balance risk and proactivity

        # Lane Reward with Priority for Rightmost Lane
        lane_index = self.vehicle.lane_index[2] if self.vehicle.lane_index else None
        rightmost_lane = self.config["lanes_count"] - 1  # Get the index of the rightmost lane

        if lane_index == rightmost_lane:  # Rightmost lane (rightmost visually)
            lane_reward = 0.8  # Slightly reduced reward for staying in the rightmost lane
            self.lane_timer = 0
        else:
            self.lane_timer += 1
            if self._is_overtaking_possible():
                lane_reward = 0.5  # Reward for being in another lane for overtaking
            else:
                lane_reward = -0.3 * self.lane_timer  # Penalize for staying away from rightmost lane

        # Speed Reward (Adjust to go faster when no cars are close, penalize abrupt changes)
        min_speed, max_speed = self.config["reward_speed_range"]
        scaled_speed = max(0, min(1, (self.vehicle.speed - min_speed) / (max_speed - min_speed)))
        speed_reward = 0.5 * scaled_speed

        # Smooth Speed Transition Penalty
        if hasattr(self, 'previous_speed'):
            speed_change_penalty = -abs(self.vehicle.speed - self.previous_speed) * 0.05  # Fine-tuned for smoother transitions
        else:
            speed_change_penalty = 0
        self.previous_speed = self.vehicle.speed

        # Overtaking Reward (Encourage passing slower cars)
        overtaking_reward = 0.0
        for vehicle in self.road.vehicles:
            if vehicle != self.vehicle and vehicle.position[0] < self.vehicle.position[0] and abs(vehicle.position[1] - self.vehicle.position[1]) < 2:
                overtaking_reward += 5.0

        # Lane Change Reward (Encourage lane changes for overtaking)
        lane_change_reward = 2.0  # Reward effective lane changes
        if action in [1, 2]:  # Actions for lane changes
            if self._is_overtaking_possible():
                lane_change_reward = 2.0  # Reward effective lane changes
            else:
                lane_change_reward = -0.2  # Reduce penalty for unnecessary lane changes

        # Awareness of Cars Changing Lanes
        awareness_penalty = 0.0
        if self._is_vehicle_changing_lane():
            awareness_penalty = -0.5  # Penalize if there is a vehicle nearby changing lanes

        # Identify the Most Promising Lane for Overtaking
        promising_lane_penalty = 0.0
        if lane_index != rightmost_lane and not self._is_overtaking_possible():
            promising_lane = self._find_promising_lane()
            if promising_lane is not None and lane_index != promising_lane:
                promising_lane_penalty = -0.3  # Adjusted penalty to balance exploration of promising lane

        # Penalize Staying Too Long in Non-Rightmost Lane After Overtaking
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

        # Log reward components
        self._log_reward_components(
            collision=collision_penalty,
            speed=speed_reward,
            lane=lane_reward,
            overtaking=overtaking_reward,
            lane_change=lane_change_reward,
            returning_penalty=returning_penalty,
            awareness_penalty=awareness_penalty,
            promising_lane_penalty=promising_lane_penalty,
            speed_change_penalty=speed_change_penalty,
            total=total_reward,
        )

        return total_reward

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
                # Predict if the vehicle is moving laterally (lane change indicator)
                if abs(vehicle.velocity[1]) > 0.5:  # Threshold for lateral velocity
                    return True
        return False

    def _find_promising_lane(self):
        """Find the most promising lane for overtaking."""
        current_lane = self.vehicle.lane_index[2] if self.vehicle.lane_index else None
        rightmost_lane = self.config["lanes_count"] - 1
        lanes = list(range(self.config["lanes_count"]))

        for lane in lanes:
            if lane != current_lane and lane != rightmost_lane:
                clear_path = True
                for vehicle in self.road.vehicles:
                    if abs(vehicle.position[1] - lane) < 1 and 0 < (vehicle.position[0] - self.vehicle.position[0]) < 5:
                        clear_path = False
                        break
                if clear_path:
                    return lane
        return None

    def _log_reward_components(self, collision, speed, lane, overtaking, lane_change, returning_penalty, awareness_penalty, promising_lane_penalty, speed_change_penalty, total):
        print("| Collision | Speed | Lane | Overtaking | Lane Change | Returning Penalty | Awareness Penalty | Promising Lane Penalty | Speed Change Penalty | Lane Timer | Vehicle Speed | Lane Index | Total Reward |")
        print("|-----------|-------|------|------------|-------------|-------------------|-------------------|------------------------|---------------------|------------|---------------|------------|-------------|")
        print(f"| {collision:<9} | {speed:<5} | {lane:<4} | {overtaking:<10} | {lane_change:<11} | {returning_penalty:<17} | {awareness_penalty:<17} | {promising_lane_penalty:<22} | {speed_change_penalty:<19} | {self.lane_timer:<10} | {self.vehicle.speed:<13} | {self.vehicle.lane_index[2] if self.vehicle.lane_index else None:<10} | {total:<11} |")
