import numpy as np
from typing import List, Optional, Tuple, Union

from p3_lib.lidar import simulate_lidar


RECTANGLE_WAYPOINTS = np.array([
    [ 0.0,  0.0],
    [ 1.2,  0.0],
    [ 1.2,  0.8],
    [-1.2,  0.8],
    [-1.2, -0.7],
    [ 1.2, -0.7],
    [ 1.2,  0.0],
    [ 0.0,  0.0],
])

THREE_ROOM_WAYPOINTS = [
    np.array([ # robot 0: left room only, two loops
        [-1.0,  0.0],
        [-1.4,  0.0],
        [-1.4,  0.7],
        [-0.5,  0.7],
        [-0.5, -0.7],
        [-1.4, -0.7],
        [-1.4,  0.0],
        [-0.5,  0.0],
        [-0.5,  0.7],
        [-1.4,  0.7],
        [-1.4, -0.7],
        [-0.5, -0.7],
        [-1.0,  0.0],
    ]),
    np.array([ # robot 1: corridor -> left room -> corridor -> right room
        [ 0.0,  0.0],
        [ 0.0,  0.85],
        [-0.5,  0.85],
        [-1.0,  0.5],
        [-1.0, -0.5],
        [-0.5, -0.5],
        [-0.5,  0.85],
        [ 0.0,  0.85],
        [ 0.0, -0.85],
        [ 0.5, -0.85],
        [ 1.0, -0.5],
        [ 1.0,  0.5],
        [ 0.5,  0.5],
        [ 0.5, -0.5],
        [ 1.0, -0.5],
        [ 1.0,  0.5],
        [ 0.5,  0.5],
        [ 0.5, -0.5],
        [ 1.0, -0.5],
        [ 1.0,  0.5],
    ]),
    np.array([ # robot 2: right room only, two loops
        [ 1.0,  0.0],
        [ 1.4,  0.0],
        [ 1.4,  0.7],
        [ 0.5,  0.7],
        [ 0.5, -0.7],
        [ 1.4, -0.7],
        [ 1.4,  0.0],
        [ 0.5,  0.0],
        [ 0.5,  0.7],
        [ 1.4,  0.7],
        [ 1.4, -0.7],
        [ 0.5, -0.7],
        [ 1.0,  0.0],
    ]),
]


class WaypointController:
    """Drive robots along a predefined waypoint sequence.

    A simple unicycle controller that steers each robot toward its next waypoint
    using proportional control on distance and heading error. Based on
    rps.utilities.controllers.

    Args:
        waypoints (Union[np.ndarray, List[np.ndarray]]): (W, 2) shared path
            for all robots, or a list of (W_r, 2) per-robot paths.
        n_robots (int): Number of robots.
        linear_gain (float): Proportional gain for linear velocity.
        angular_gain (float): Proportional gain for angular velocity.
        arrival_dist (float): Switch to next waypoint when closer than this (m).
        max_linear_vel (float): Clamp on linear velocity (m/s). Robotarium max = 0.2.
        max_angular_vel (float): Clamp on angular velocity (rad/s). Robotarium max ~ 3.64.
    """

    def __init__(
        self,
        waypoints: Union[np.ndarray, List[np.ndarray]],
        n_robots: int = 1,
        linear_gain: float = 1.0,
        angular_gain: float = 4.0,
        arrival_dist: float = 0.08,
        max_linear_vel: float = 0.2,
        max_angular_vel: float = 3.0,
    ) -> None:
        if isinstance(waypoints, np.ndarray) and waypoints.ndim == 2:
            self._wp = [waypoints for _ in range(n_robots)]
        elif isinstance(waypoints, list):
            self._wp = [np.asarray(w) for w in waypoints]
        else:
            waypoints = np.asarray(waypoints)
            if waypoints.ndim == 2:
                self._wp = [waypoints for _ in range(n_robots)]
            else:
                self._wp = [np.asarray(w) for w in waypoints]

        self.n_robots = n_robots
        self.linear_gain = linear_gain
        self.angular_gain = angular_gain
        self.arrival_dist = arrival_dist
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.current_wp = np.zeros(n_robots, dtype=int)

    def step(
        self,
        poses: np.ndarray,
        all_segs: Optional[list] = None,
    ) -> np.ndarray:
        """Compute unicycle velocities to steer each robot toward its next waypoint.

        Args:
            poses (np.ndarray): (3, N) current poses [x; y; theta].
            all_segs (Optional[list]): Ignored. Kept for interface compatibility
                with RandomWalkController.

        Returns:
            np.ndarray: (2, N) unicycle velocities [linear_vel; angular_vel].
        """
        N = poses.shape[1]
        dxu = np.zeros((2, N))
        for ri in range(N):
            wp_list = self._wp[ri]
            idx = self.current_wp[ri]
            if idx >= len(wp_list):
                continue
            dx = wp_list[idx, 0] - poses[0, ri]
            dy = wp_list[idx, 1] - poses[1, ri]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < self.arrival_dist:
                self.current_wp[ri] += 1
                continue
            desired_angle = np.arctan2(dy, dx)
            angle_error = desired_angle - poses[2, ri]
            angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
            dxu[0, ri] = np.clip(
                self.linear_gain * dist * np.cos(angle_error),
                -self.max_linear_vel, self.max_linear_vel,
            )
            dxu[1, ri] = np.clip(
                self.angular_gain * dist * np.sin(angle_error),
                -self.max_angular_vel, self.max_angular_vel,
            )
        return dxu

    @property
    def done(self) -> bool:
        """True when every robot has reached its final waypoint."""
        return all(self.current_wp[ri] >= len(self._wp[ri]) for ri in range(self.n_robots))


class RandomWalkController:
    """Random walk controller for robotarium robots.

    Drives forward at constant speed and uses a front-facing LiDAR cone to
    detect walls. When an obstacle is detected within detect_dist, the robot
    picks a biased-random turn direction (toward the more open side) and spins
    for a randomized number of steps. Includes stuck detection to recover from
    corners.

    Args:
        n_robots (int): Number of robots.
        forward_speed (float): Linear velocity when driving straight (m/s).
        turn_omega (float): Angular velocity when turning (rad/s).
        detect_dist (float): Distance threshold to trigger wall avoidance (m).
        turn_steps_range (Tuple[int, int]): (min, max) sim steps to turn.
        seed (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        n_robots: int,
        forward_speed: float = 0.15,
        turn_omega: float = 1.8,
        detect_dist: float = 0.35,
        turn_steps_range: Tuple[int, int] = (10, 45),
        seed: int = 99,
    ) -> None:
        self.n_robots = n_robots
        self.forward_speed = forward_speed
        self.turn_omega = turn_omega
        self.detect_dist = detect_dist
        self.turn_steps_range = turn_steps_range

        self.front_n_rays = 9
        self.front_max_range = 0.5
        self.front_fov = np.pi / 2

        self.clear_margin = 0.05
        self.clear_steps = 3
        self.min_turn_steps = 5

        self.stuck_window = 40
        self.stuck_dist = 0.03
        self.recovery_turn_steps_range = (35, 85)
        self.stuck_cooldown_steps = 60

        np.random.seed(seed)
        self.turn_remaining = np.zeros(n_robots, dtype=int)  # countdown timer for turning
        self.turn_dir = np.ones(n_robots)
        self.turn_elapsed = np.zeros(n_robots, dtype=int)
        self.clear_count = np.zeros(n_robots, dtype=int)

        self.pos_hist = np.zeros((n_robots, self.stuck_window, 2), dtype=float)
        self.hist_idx = np.zeros(n_robots, dtype=int)
        self.hist_len = np.zeros(n_robots, dtype=int)
        self.stuck_cooldown = np.zeros(n_robots, dtype=int)

    def _pick_turn_dir_biased(
        self,
        angles: np.ndarray,
        ranges: np.ndarray,
        max_range: float,
    ) -> float:
        """Choose turn direction based on which side looks more open.

        Compares mean clearance of left-side vs right-side rays. Falls back to
        a random choice when both sides are equally clear.

        Args:
            angles (np.ndarray): (K,) ray angles in robot frame (rad).
            ranges (np.ndarray): (K,) range values (NaN = no hit).
            max_range (float): Treat NaN as this value (no hit = open space).

        Returns:
            float: +1.0 for left, -1.0 for right.
        """
        rr = np.where(np.isnan(ranges), max_range, ranges)
        left_mask = angles > 1e-9
        right_mask = angles < -1e-9
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return float(np.random.choice([-1, 1]))
        left_clear = float(rr[left_mask].mean())
        right_clear = float(rr[right_mask].mean())
        if abs(left_clear - right_clear) < 1e-6:
            return float(np.random.choice([-1, 1]))
        return 1.0 if left_clear > right_clear else -1.0

    def _is_stuck(self, ri: int, pose_xy: np.ndarray) -> bool:
        """Update ring buffer and detect if robot is stuck.

        Tracks each robot's recent positions. If the robot has moved less than
        stuck_dist over the last stuck_window steps, it is considered stuck.

        Args:
            ri (int): Robot index.
            pose_xy (np.ndarray): (2,) current [x, y] position.

        Returns:
            bool: True if the robot appears stuck.
        """
        self.pos_hist[ri, self.hist_idx[ri]] = pose_xy
        self.hist_idx[ri] = (self.hist_idx[ri] + 1) % self.stuck_window
        self.hist_len[ri] = min(self.hist_len[ri] + 1, self.stuck_window)
        if self.stuck_cooldown[ri] > 0:
            self.stuck_cooldown[ri] -= 1
            return False
        if self.hist_len[ri] < self.stuck_window:
            return False
        old_xy = self.pos_hist[ri, self.hist_idx[ri]]
        moved = float(np.linalg.norm(pose_xy - old_xy))
        return moved < self.stuck_dist

    def step(
        self,
        poses: np.ndarray,
        all_segs: list,
    ) -> np.ndarray:
        """Compute unicycle velocities for all robots.

        For each robot: cast front-facing LiDAR rays, check for walls, and
        either drive forward, initiate a turn, or trigger stuck recovery.

        Args:
            poses (np.ndarray): (3, N) current poses from robotarium [x; y; theta].
            all_segs (list): Line segments for LiDAR ray-casting.

        Returns:
            np.ndarray: (2, N) unicycle velocities [linear_vel; angular_vel].
        """
        dxu = np.zeros((2, self.n_robots))
        for ri in range(self.n_robots):
            pose_ri = poses[:, ri] 
            pose_xy = pose_ri[:2].astype(float)
            stuck = self._is_stuck(ri, pose_xy)

            angles, front_ranges, _ = simulate_lidar(
                pose_ri,
                all_segs,
                n_rays=self.front_n_rays,
                max_range=self.front_max_range,
                fov=self.front_fov,
            )
            has_hit = np.any(~np.isnan(front_ranges))
            front_min = float(np.nanmin(front_ranges)) if has_hit else np.inf

            if self.turn_remaining[ri] > 0:
                dxu[0, ri] = 0.0
                dxu[1, ri] = self.turn_dir[ri] * self.turn_omega
                self.turn_remaining[ri] -= 1
                self.turn_elapsed[ri] += 1
                if front_min > (self.detect_dist + self.clear_margin):
                    self.clear_count[ri] += 1
                else:
                    self.clear_count[ri] = 0
                if (
                    self.turn_elapsed[ri] >= self.min_turn_steps
                    and self.clear_count[ri] >= self.clear_steps
                ):
                    self.turn_remaining[ri] = 0

            elif stuck:
                self.turn_remaining[ri] = np.random.randint(*self.recovery_turn_steps_range)
                self.turn_dir[ri] = self._pick_turn_dir_biased(angles, front_ranges, self.front_max_range)
                self.turn_elapsed[ri] = 0
                self.clear_count[ri] = 0
                self.stuck_cooldown[ri] = self.stuck_cooldown_steps
                dxu[0, ri] = 0.0
                dxu[1, ri] = self.turn_dir[ri] * self.turn_omega

            elif has_hit and front_min < self.detect_dist:
                self.turn_remaining[ri] = np.random.randint(*self.turn_steps_range)
                self.turn_dir[ri] = self._pick_turn_dir_biased(angles, front_ranges, self.front_max_range)
                self.turn_elapsed[ri] = 0
                self.clear_count[ri] = 0
                dxu[0, ri] = 0.0
                dxu[1, ri] = self.turn_dir[ri] * self.turn_omega

            else:
                dxu[0, ri] = self.forward_speed
                dxu[1, ri] = 0.0
        return dxu
