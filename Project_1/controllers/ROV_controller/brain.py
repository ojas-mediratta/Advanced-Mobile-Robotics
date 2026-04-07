"""Control, estimation, and command logic for the ROV controller."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import gtsam
import numpy as np
from gtsam import NavState, NavStateImuEKF, Point3, PreintegrationParams
from robot import Control, Measurements

# Manual command step sizes (forces in N, torques in N*m).
FORCE_X_STEP = 1.0
FORCE_Z_STEP = 2.0
TAU_Z_STEP = 1.0
TAU_X_STEP = 1.0

# IMU noise parameters.
SIGMA_ACC = 0.004
SIGMA_GYRO = 0.05

# Preintegration params for the EKF.
PREINTEGRATION_PARAMS: PreintegrationParams = gtsam.PreintegrationParams.MakeSharedU(
    9.81
)
PREINTEGRATION_PARAMS.setAccelerometerCovariance(np.diag([SIGMA_ACC**2] * 3))
PREINTEGRATION_PARAMS.setIntegrationCovariance(np.diag([1e-3] * 3))
PREINTEGRATION_PARAMS.setGyroscopeCovariance(np.diag([SIGMA_GYRO**2] * 3))

# Measurement covariances.
POSITION_MEAS_COV = np.eye(3) * 1.0**2
DEPTH_MEAS_COV = np.eye(1) * 0.1**2
RANGE_MEAS_COV = np.eye(1) * 1.0**2

# Trajectory follower gains.
KP_DIS = 5.0
KP_YAW = 1.0
KP_Z = 10.0
KP_ROLL = 1.0


@dataclass(frozen=True)
class Wrench:
    """6-DOF force/torque command expressed in body coordinates."""

    torque: np.ndarray
    force: np.ndarray

    @staticmethod
    def torque_only(x: float, y: float, z: float) -> "Wrench":
        """Construct a pure torque wrench."""
        return Wrench(
            torque=np.array([x, y, z], dtype=float),
            force=np.zeros(3, dtype=float),
        )

    @staticmethod
    def force_only(x: float, y: float, z: float) -> "Wrench":
        """Construct a pure force wrench."""
        return Wrench(
            torque=np.zeros(3, dtype=float),
            force=np.array([x, y, z], dtype=float),
        )


B_MATRIX: np.ndarray = np.array(
    [
        [-1.0, 1.0, 0.0, 0.0],
        [0.1, 0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
        [0.0, 0.0, 0.035, 0.035],
    ],
    dtype=float,
)

INVERSE_B_MATRIX: np.ndarray = np.linalg.inv(B_MATRIX)


def wrench_to_thrusters(wrench: Wrench) -> Tuple[float, float, float, float]:
    """
    Allocate thruster commands from a wrench.

    Uses the linear map y = B u where y = [Fx, tau_z, Fz, tau_x]^T.
    Return (u_lm, u_rm, u_vlm, u_vrm).
    """
    # ---------------- TODO: Implement this function --------------- #
    return 0.0, 0.0, 0.0, 0.0
    # -------------------------------------------------------------- #


class Brain:
    def __init__(self, beacons: Sequence[Point3]):
        """Initialize the brain with known range beacons."""
        self.beacons: list[Point3] = list(beacons)
        self.ekf: Optional[NavStateImuEKF] = None

    def estimate(self, measurements: Measurements) -> Optional[NavState]:
        """Return a true-state snapshot for logging; estimation is handled elsewhere."""
        return measurements.X_est

    def initialize_EKF(self, X0: NavState, P0: np.ndarray) -> None:
        """Create the EKF with initial state and covariance."""
        # ---------------- TODO: Implement this function --------------- #
        self.ekf = None
        # -------------------------------------------------------------- #

    def EKF_predict(
        self, omega_meas: np.ndarray, acc_meas: np.ndarray, dt: float
    ) -> Tuple[Optional[NavState], Optional[np.ndarray]]:
        """Run EKF prediction and return (state, covariance)."""
        if self.ekf is None:
            return None, None
        # ---------------- TODO: Implement this function --------------- #
        return None, None
        # -------------------------------------------------------------- #

    def EKF_update(self, measurements: Measurements) -> None:
        """Apply measurement updates based on available measurements (position, depth, ranges)."""
        if self.ekf is None:
            return None
        # ---------------- TODO: Implement this function --------------- #
        return None
        # -------------------------------------------------------------- #

    def act_on_command(self, key: int, keyboard: Any) -> Control:
        """Convert a keyboard command into a thruster Control."""
        wrench = None
        # ---------------- TODO: Implement this function --------------- #
        if key == keyboard.UP:
            wrench = Wrench.force_only(0.0, 0.0, 0.0)
        elif key == keyboard.DOWN:
            wrench = Wrench.force_only(0.0, 0.0, 0.0)
        elif key == keyboard.LEFT:
            wrench = Wrench.torque_only(0.0, 0.0, 0.0)
        elif key == keyboard.RIGHT:
            wrench = Wrench.torque_only(0.0, 0.0, 0.0)
        elif key == ord("S"):
            wrench = Wrench.force_only(0.0, 0.0, 0.0)
        elif key == ord("W"):
            wrench = Wrench.force_only(0.0, 0.0, 0.0)
        elif key == ord("Q"):
            wrench = Wrench.torque_only(0.0, 0.0, 0.0)
        elif key == ord("E"):
            wrench = Wrench.torque_only(0.0, 0.0, 0.0)
        # -------------------------------------------------------------- #

        if wrench is None:
            return Control()

        print(wrench)
        u_lm, u_rm, u_vlm, u_vrm = wrench_to_thrusters(wrench)
        return Control(u_lm=u_lm, u_rm=u_rm, u_vlm=u_vlm, u_vrm=u_vrm)

    @staticmethod
    def wrap_to_pi(a: float) -> float:
        """Wrap an angle in radians to [-pi, pi)."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    def follow_step(self, t: float, X: NavState, desired_state: NavState) -> Control:
        """Trajectory-following control step from desired pose. 
        Use self.wrap_to_pi for angle wrapping. 
        Use wrench_to_thrusters to convert wrench to thruster commands."""
       
        # ---------------- TODO: Implement this function --------------- #
        uL, uR, uvL, uvR = 0.0, 0.0, 0.0, 0.0
        # -------------------------------------------------------------- #

        return Control(u_lm=uL, u_rm=uR, u_vlm=uvL, u_vrm=uvR)
