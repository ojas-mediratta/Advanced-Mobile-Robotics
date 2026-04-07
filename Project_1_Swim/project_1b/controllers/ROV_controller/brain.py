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
    # TODO (students):
    # Use the provided INVERSE_B_MATRIX to convert the wrench into thruster commands.
   
    # the wrench vector y = [Fx, tau z, Fz, tau x] transposed
    y = np.array([
        wrench.force[0], # Fx
        wrench.torque[2], # yaw
        wrench.force[2], # Fz
        wrench.torque[0] # roll
    ])
    
    u = INVERSE_B_MATRIX @ y # inverse mapping: u = B^-1 * y
    u_lm, u_rm, u_vlm, u_vrm = u[0], u[1], u[2], u[3]

    # -------------------------------------------------------------- #
    return u_lm, u_rm, u_vlm, u_vrm


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
        # TODO (students):
        # Initialize self.ekf using the provided X0 and P0, and PREINTEGRATION_PARAMS.
        
        self.ekf = NavStateImuEKF(X0, P0, PREINTEGRATION_PARAMS)
        
        # -------------------------------------------------------------- #

    def EKF_predict(
        self, omega_meas: np.ndarray, acc_meas: np.ndarray, dt: float
    ) -> Tuple[Optional[NavState], Optional[np.ndarray]]:
        """Run EKF prediction and return (state, covariance)."""
        if self.ekf is None:
            return NavState(), np.eye(9)
        # ---------------- TODO: Implement this function --------------- #
        # TODO (students):
        # Use self.ekf.predict to perform the prediction step with the provided omega_meas, acc_meas, and dt.
        # Return the updated state and covariance.
        
        self.ekf.predict(omega_meas, acc_meas, dt)
        X_est = self.ekf.state()
        P = self.ekf.covariance()
        
        return X_est, P
        # -------------------------------------------------------------- #

    def EKF_update(self, measurements: Measurements) -> None:
        """Apply measurement updates based on available measurements (position, depth, ranges)."""
        if self.ekf is None:
            return
        
        X_est = self.ekf.state()
        R_est = X_est.attitude().matrix()
        
        # ---------------- TODO: Implement this function --------------- #
        # TODO (students):
        # For each available measurement in measurements (position, depth, ranges),
        # compute the appropriate measurement matrix H and call self.ekf.updateWithVector.
        
        if measurements.position is not None:
            H = np.zeros((3, 9))
            H[0:3, 3:6] = np.eye(3)
            
            z = measurements.position
            h = X_est.position()
            
            self.ekf.updateWithVector(h, H, z, POSITION_MEAS_COV)
            X_est = self.ekf.state()

        if measurements.depth is not None:
            H = np.zeros((1, 9))
            H[0, 5] = 1.0
            
            z = np.array([measurements.depth])
            h = np.array([X_est.position()[2]])
            
            self.ekf.updateWithVector(h, H, z, DEPTH_MEAS_COV)
            X_est = self.ekf.state()

        if measurements.ranges is not None and self.beacons:
            p_est = X_est.position()
            
            for beacon, range_meas in zip(self.beacons, measurements.ranges):
                diff = p_est - beacon
                predicted_range = np.linalg.norm(diff)
                
                H = np.zeros((1, 9))
                if predicted_range > 1e-6:
                    H[0, 3:6] = diff / predicted_range
                
                z = np.array([range_meas])
                h = np.array([predicted_range])
                
                self.ekf.updateWithVector(h, H, z, RANGE_MEAS_COV)
                p_est = self.ekf.state().position()
        # -------------------------------------------------------------- #

    def act_on_command(self, key: int, keyboard: Any) -> Control:
        """Convert a keyboard command into a thruster Control."""
       
        wrench = None
        # ---------------- TODO: Implement this function --------------- #
        # TODO (students):
        # Replace the 0.0 entries below with the provided step-size constants:
        #   FORCE_X_STEP, FORCE_Z_STEP, TAU_Z_STEP, TAU_X_STEP.
        # Decide (for each key) whether it should apply a force or a torque, and put the
        # correct constant on the correct axis with the correct sign:
        #   - UP/DOWN:   +/- X force
        #   - W/S:       +/- Z force
        #   - E/Q:       +/- roll torque
        #   - LEFT/RIGHT:+/- yaw torque 
        
        if key == keyboard.UP:
            wrench = Wrench.force_only(FORCE_X_STEP, 0.0, 0.0)
        elif key == keyboard.DOWN:
            wrench = Wrench.force_only(-FORCE_X_STEP, 0.0, 0.0)
        elif key == keyboard.LEFT:
            wrench = Wrench.torque_only(0.0, 0.0, TAU_Z_STEP)
        elif key == keyboard.RIGHT:
            wrench = Wrench.torque_only(0.0, 0.0, -TAU_Z_STEP)
        elif key == ord("S"):
            wrench = Wrench.force_only(0.0, 0.0, -FORCE_Z_STEP)
        elif key == ord("W"):
            wrench = Wrench.force_only(0.0, 0.0, FORCE_Z_STEP)
        elif key == ord("Q"):
            wrench = Wrench.torque_only(-TAU_X_STEP, 0.0, 0.0)
        elif key == ord("E"):
            wrench = Wrench.torque_only(TAU_X_STEP, 0.0, 0.0) 
                 
        # -------------------------------------------------------------- #

        if wrench is None:
            return Control(0.0, 0.0, 0.0, 0.0)

        u_lm, u_rm, u_vlm, u_vrm = wrench_to_thrusters(wrench)
        return Control(u_lm=u_lm, u_rm=u_rm, u_vlm=u_vlm, u_vrm=u_vrm)

    @staticmethod
    def wrap_to_pi(a: float) -> float:
        """Wrap an angle in radians to [-pi, pi)."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    def follow_step(self, t: float, X: NavState, desired_state: NavState) -> Control:
        """Trajectory-following control step from desired pose. """
       
        # ---------------- TODO: Implement this function --------------- #
        # TODO (students):
        # Implement a simple P controller using the provided gains (KP_DIS, KP_YAW, KP_Z, KP_ROLL).
        # Compute the position error in the XY plane and Z axis, as well as the yaw and roll errors.
        # Remember to wrap the angular errors with self.wrap_to_pi.
        # Use these to compute desired forces/torques, then convert to thruster commands with wrench_to_thrusters.
        
        # get the current and desired orientations
        curr_roll, curr_pitch, curr_yaw = X.attitude().rpy()
        des_roll, des_pitch, des_yaw = desired_state.attitude().rpy()
        
        # Compute angular errors with wrapping
        yaw_error = self.wrap_to_pi(des_yaw - curr_yaw)
        roll_error = self.wrap_to_pi(des_roll - curr_roll)  # desired roll is 0
        
        # compute position and distance error 
        pos_error_world = desired_state.position() - X.position()
        distance_error = np.linalg.norm(pos_error_world[:2]) # this makes the robot go forward/backward in the XY plane
        z_error = pos_error_world[2]
        
        # Apply proportional control
        Fx = KP_DIS * distance_error  # Force in X (forward)
        Fz = KP_Z * z_error  # Force in Z (vertical)
        tau_z = KP_YAW * yaw_error  # Torque around Z (yaw)
        tau_x = KP_ROLL * roll_error  # Torque around X (roll)
        
        wrench = Wrench(
            force=np.array([Fx, 0.0, Fz], dtype=float),
            torque=np.array([tau_x, 0.0, tau_z], dtype=float)
        )

        uL, uR, uvL, uvR = wrench_to_thrusters(wrench)
        
        # -------------------------------------------------------------- #
        
        # default compensation
        uvL -= 0.9  
        uvR += 0.9 
        return Control(u_lm=uL, u_rm=uR, u_vlm=uvL, u_vrm=uvR)
