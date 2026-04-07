from dataclasses import dataclass
from typing import Sequence

import gtsam
import numpy as np
from gtsam import NavState


@dataclass(frozen=True)
class Measurements:
    """Sensor measurements plus a true-state snapshot for logging only."""

    position: np.ndarray | None = None
    depth: float | None = None
    ranges: Sequence[float] | None = None
    X_true: NavState | None = None


@dataclass(frozen=True)
class Control:
    u_lm: float = 0.0
    u_rm: float = 0.0
    u_vlm: float = 0.0
    u_vrm: float = 0.0


BEACONS = (gtsam.Point3(0, 0, 0.9), gtsam.Point3(4, 0, 0), gtsam.Point3(0, 4, 0))


class Robot:
    """
    Thin wrapper around Webots controller.Robot that standardizes sensor reads and state access.

    Notes:
      - Gyro gives angular velocity in rad/s in robot (device) frame.
      - Accelerometer gives linear acceleration (m/s^2) in robot (device) frame.
      - Getting ground-truth state in gtsam NavState format.
    """

    POSITION_MEASUREMENT_CADENCE = None
    DEPTH_MEASUREMENT_CADENCE = 100
    RANGE_MEASUREMENT_CADENCE = 30

    def __init__(self, supervisor):
        supervisor = supervisor
        self.robot_node = supervisor.getSelf()
        timestep_ms = int(supervisor.getBasicTimeStep())

        # --- Init Devices ---
        self.gyro = supervisor.getDevice("gyro")
        assert self.gyro is not None, "gyro device not found"
        self.gyro.enable(timestep_ms)
        self.accel = supervisor.getDevice("accelerometer")
        assert self.accel is not None, "accelerometer device not found"
        self.accel.enable(timestep_ms)

        # --- Init Motors ---
        self.lm = supervisor.getDevice("left_motor")
        self.lm.setPosition(float("inf"))
        self.lm.setVelocity(0.0)
        self.rm = supervisor.getDevice("right_motor")
        self.rm.setPosition(float("inf"))
        self.rm.setVelocity(0.0)
        self.vlm = supervisor.getDevice("vertical_left_motor")
        self.vlm.setPosition(float("inf"))
        self.vlm.setVelocity(0.0)
        self.vrm = supervisor.getDevice("vertical_right_motor")
        self.vrm.setPosition(float("inf"))
        self.vrm.setVelocity(0.0)
        self.u_lim = 30.0  # [rad/s]

        print("Robot initialized.")

    # ---------------------------
    # "State" accessors
    # ---------------------------
    def get_state(self) -> NavState:
        """Get ground-truth NavState from Supervisor API."""
        position_true: list = self.robot_node.getPosition()
        rotation_true: list = self.robot_node.getOrientation()
        wRb_true = np.array(rotation_true, dtype=float).reshape(
            3, 3
        )  # ROW-major order, world <- body
        velocity_true = self.robot_node.getVelocity()[:3]
        return NavState(
            gtsam.Rot3(wRb_true),
            gtsam.Point3(np.asarray(position_true)),
            np.asarray(velocity_true),
        )

    # ---------------------------
    # Sensor reads
    # ---------------------------
    def read_gyro(self) -> np.ndarray:
        """Angular velocity [rad/s] (device/robot frame)."""
        return np.array(self.gyro.getValues())

    def read_accel(self) -> np.ndarray:
        """Linear acceleration [m/s^2] (device/robot frame)."""
        return np.array(self.accel.getValues())

    def sense(self, step: int) -> Measurements:
        """Return simulated measurements based on cadence settings."""
        position = None
        depth = None
        ranges = None
        X_true = self.get_state()
        if (
            self.POSITION_MEASUREMENT_CADENCE is not None
            and step % self.POSITION_MEASUREMENT_CADENCE == 0
        ):
            position = X_true.position()
        if (
            self.DEPTH_MEASUREMENT_CADENCE is not None
            and step % self.DEPTH_MEASUREMENT_CADENCE == 0
        ):
            depth = float(X_true.position()[2])
        if (
            self.RANGE_MEASUREMENT_CADENCE is not None
            and step % self.RANGE_MEASUREMENT_CADENCE == 0
        ):
            ranges = [float(X_true.range(beacon)) for beacon in BEACONS]

        return Measurements(position, depth, ranges, X_true)

    # ---------------------------
    # Motor commands
    # ---------------------------
    def set_motor_velocities(self, control: Control):
        """Set motor velocities from a Control instance."""
        # Saturate (simple clamp)
        u_lm = float(np.clip(control.u_lm, -self.u_lim, self.u_lim))
        u_rm = float(np.clip(control.u_rm, -self.u_lim, self.u_lim))
        u_vlm = float(np.clip(control.u_vlm, -self.u_lim, self.u_lim))
        u_vrm = float(np.clip(control.u_vrm, -self.u_lim, self.u_lim))
        self.lm.setVelocity(u_lm)
        self.rm.setVelocity(u_rm)
        self.vlm.setVelocity(u_vlm)
        self.vrm.setVelocity(u_vrm)
