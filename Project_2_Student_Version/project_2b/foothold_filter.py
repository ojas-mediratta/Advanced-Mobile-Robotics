"""Foothold-augmented NavState EKF with fixed number of feet.

State layout:
- Body NavState: 9 DoF tangent coordinates [dphi, dp, dv]
- N foothold landmarks in world frame: 3N DoF

This class uses fixed N feet. Feet in swing are handled by skipping updates.
When contact is detected for a foot, call reinitialize_foothold(...).
"""

from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np
import gtsam


FootKey = Union[int, str]


class FootholdNavStateImuEKF:
    """Fixed-N foothold augmented EKF with public state fields.

    Public fields intentionally exposed for simplicity:
    - x: gtsam.NavState
    - P: covariance matrix over [body(9), footholds(3N)]
    - footholds: array shape (N, 3)
    - num_feet, foot_names, foot_to_index

    Error-state ordering used throughout this file:
    [dphi(3), dp(3), dv(3), df_0(3), ..., df_{N-1}(3)]
    """

    def __init__(
        self,
        x0: gtsam.NavState,
        P0: np.ndarray,
        num_feet: int,
        foot_names: Sequence[str] | None = None,
        gravity: np.ndarray = np.array([0.0, 0.0, -9.81]),
        sigma_gyro: float = 8e-4,
        sigma_integration: float = 1e-3,
        sigma_acc: float = 1e-2,
        foothold_process_sigma: float = 1e-3,
        foothold_init_sigma: float = 5e-1,
        sigma_meas: float = 5e-3,
        sigma_height_prior: float = 5e-2,
    ) -> None:
        self.x = x0
        self.num_feet = int(num_feet)
        if self.num_feet <= 0:
            raise ValueError("num_feet must be >= 1")

        if foot_names is None:
            self.foot_names = [f"foot_{i}" for i in range(self.num_feet)]
        else:
            if len(foot_names) != self.num_feet:
                raise ValueError("foot_names length must equal num_feet")
            self.foot_names = list(foot_names)

        self.foot_to_index: Dict[str, int] = {
            name: i for i, name in enumerate(self.foot_names)
        }

        self.gravity = np.asarray(gravity, dtype=float)
        self.sigma_gyro = float(sigma_gyro)
        self.sigma_integration = float(sigma_integration)
        self.sigma_acc = float(sigma_acc)
        self.foothold_process_sigma = float(foothold_process_sigma)
        self.foothold_init_sigma = float(foothold_init_sigma)
        if self.foothold_init_sigma <= 0.0:
            raise ValueError("foothold_init_sigma must be > 0")
        self.sigma_meas = float(sigma_meas)
        self.sigma_height_prior = float(sigma_height_prior)

        self.footholds = np.zeros((self.num_feet, 3), dtype=float)

        dim = 9 + 3 * self.num_feet
        P0 = np.asarray(P0, dtype=float)
        if P0.shape != (dim, dim):
            raise ValueError(f"P0 must be shape {(dim, dim)}, got {P0.shape}")
        self.P = 0.5 * (P0 + P0.T)

    @staticmethod
    def gravity_increment(gravity: np.ndarray, dt: float) -> gtsam.NavState:
        """W increment: gravity-only world-frame left factor."""
        g = np.asarray(gravity, dtype=float)
        return gtsam.NavState(gtsam.Rot3(), 0.5 * g * (dt**2), g * dt)

    @staticmethod
    def imu_increment(
        omega_body: np.ndarray, specific_force_body: np.ndarray, dt: float
    ) -> gtsam.NavState:
        """U increment: first-order IMU right factor.

        Uses the approximation from the assignment note:
            U ≈ [Exp(omega*dt), f*dt, 0.5*f*dt^2]
        mapped to NavState constructor ordering (R, t, v):
            t = 0.5*f*dt^2, v = f*dt.
        """
        omega_body = np.asarray(omega_body, dtype=float)
        specific_force_body = np.asarray(specific_force_body, dtype=float)
        dt = float(dt)

        dR = gtsam.Rot3.Expmap(omega_body * dt)
        dv_b = specific_force_body * dt
        dp_b = 0.5 * specific_force_body * (dt**2)
        return gtsam.NavState(dR, dp_b, dv_b)

    @staticmethod
    def autonomous_flow(X: gtsam.NavState, dt: float) -> gtsam.NavState:
        """phi(X): velocity acts on position for dt."""
        return gtsam.NavState(
            X.attitude(),
            X.position() + X.velocity() * dt,
            X.velocity(),
        )

    @staticmethod
    def autonomous_flow_jacobian(dt: float) -> np.ndarray:
        """Phi = dphi|_e in [dR, dP, dV] coordinates."""
        Phi = np.eye(9, dtype=float)
        Phi[3:6, 6:9] = np.eye(3) * dt
        return Phi

    @classmethod
    def body_transition_jacobian(cls, U: gtsam.NavState, dt: float) -> np.ndarray:
        """A = Ad_{U^{-1}} * Phi for left-linear NavState dynamics."""
        Phi = cls.autonomous_flow_jacobian(dt)
        Ad_U_inv = U.inverse().AdjointMap()
        return np.asarray(Ad_U_inv, dtype=float) @ Phi

    @classmethod
    def predict_mean(
        cls,
        gravity: np.ndarray,
        X: gtsam.NavState,
        omega_body: np.ndarray,
        specific_force_body: np.ndarray,
        dt: float,
    ) -> tuple[gtsam.NavState, gtsam.NavState, gtsam.NavState]:
        """Mean propagation using W * phi(X) * U."""
        W = cls.gravity_increment(gravity, dt)
        U = cls.imu_increment(omega_body, specific_force_body, dt)
        phiX = cls.autonomous_flow(X, dt)
        X_next = W * phiX * U
        return X_next, W, U

    def resolve_index(self, foot: FootKey) -> int:
        if isinstance(foot, int):
            i = int(foot)
            if not (0 <= i < self.num_feet):
                raise IndexError(f"foot index out of range: {i}")
            return i
        if foot not in self.foot_to_index:
            raise KeyError(f"unknown foot id '{foot}'")
        return self.foot_to_index[foot]

    def process_contact_measurements(
        self,
        old_contacts: Dict[str, np.ndarray],
        new_contacts: Dict[str, np.ndarray],
        z_world: float = 0.0,
        R_meas: np.ndarray | None = None,
        apply_height_prior: bool = True,
    ) -> None:
        """Apply contact updates from explicit old/new contact maps.

        Update order:
        - update all `old_contacts` (continuing stance feet),
        - reinitialize all `new_contacts` (swing -> stance),
        - update all `new_contacts`.
        """
        unknown = [
            foot
            for foot in set(old_contacts.keys()) | set(new_contacts.keys())
            if foot not in self.foot_to_index
        ]
        if unknown:
            raise KeyError(f"unknown foot ids in contact maps: {unknown}")
        overlap = set(old_contacts.keys()) & set(new_contacts.keys())
        if overlap:
            raise ValueError(
                f"old_contacts and new_contacts must be disjoint, got overlap: {sorted(overlap)}"
            )

        # 1) Continuing contacts: update immediately with current contact vectors.
        for foot in sorted(old_contacts):
            z_meas = np.asarray(old_contacts[foot], dtype=float)
            self.update_foothold_measurement(foot, z_meas, R_meas=R_meas)
            if apply_height_prior:
                self.update_foothold_height_prior(foot, z_world=z_world)

        # 2) New contacts: reinitialize foothold first, then apply measurement update.
        # This ensures each newly-contacted foot starts from a physically plausible point
        # before using the residual to refine body/landmark estimates.
        for foot in sorted(new_contacts):
            self.reinitialize_foothold(foot, np.asarray(new_contacts[foot], dtype=float))

            z_meas = np.asarray(new_contacts[foot], dtype=float)
            self.update_foothold_measurement(foot, z_meas, R_meas=R_meas)
            if apply_height_prior:
                self.update_foothold_height_prior(foot, z_world=z_world)

    def foot_block(self, i: int) -> slice:
        start = 9 + 3 * i
        return slice(start, start + 3)

    def state_dim(self) -> int:
        return 9 + 3 * self.num_feet

    def _apply_state_delta(self, dx: np.ndarray) -> None:
        """Retract body state and add landmark deltas from a solved EKF increment."""
        dx = np.asarray(dx, dtype=float).reshape(-1)
        self.x = self.x.retract(dx[0:9])
        for j in range(self.num_feet):
            self.footholds[j] += dx[self.foot_block(j)]

    def _apply_linear_update(
        self,
        H: np.ndarray,
        r: np.ndarray,
        R_meas: np.ndarray,
    ) -> None:
        """Shared EKF linear-correction step for both contact and height updates."""
        dim = self.state_dim()
        H = np.asarray(H, dtype=float)
        r = np.asarray(r, dtype=float).reshape(-1)
        R_meas = np.asarray(R_meas, dtype=float)

        S = H @ self.P @ H.T + R_meas
        S = 0.5 * (S + S.T) + 1e-9 * np.eye(S.shape[0])
        K = (self.P @ H.T) @ np.linalg.inv(S)

        dx = K @ r
        self._apply_state_delta(dx)

        I = np.eye(dim)
        KH = K @ H
        # Joseph form keeps covariance PSD/symmetric in finite precision.
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def predict(
        self, omega_body: np.ndarray, specific_force_body: np.ndarray, dt: float
    ) -> gtsam.NavState:
        """IMU prediction for body + foothold dynamics.

        Foothold dynamics application:
        - Foothold means remain constant in world frame in prediction.
        - Foothold covariance blocks receive random-walk inflation each step.
        - Swing/stance is handled by measurement gating outside this method.
        """
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self.x, _, U = self.predict_mean(
            gravity=self.gravity,
            X=self.x,
            omega_body=omega_body,
            specific_force_body=specific_force_body,
            dt=dt,
        )
        A_body = self.body_transition_jacobian(U, dt)

        dim = self.state_dim()
        F = np.eye(dim)
        F[0:9, 0:9] = A_body

        Q = np.zeros((dim, dim))
        # Match NavStateImuEKF convention: continuous-time noise scaled by dt.
        Q_body_cont = np.zeros((9, 9), dtype=float)
        Q_body_cont[0:3, 0:3] = (self.sigma_gyro**2) * np.eye(3)
        Q_body_cont[3:6, 3:6] = (self.sigma_integration**2) * np.eye(3)
        Q_body_cont[6:9, 6:9] = (self.sigma_acc**2) * np.eye(3)
        Q[0:9, 0:9] = Q_body_cont * dt

        qf = (self.foothold_process_sigma**2) * max(dt, 1e-9) * np.eye(3)
        for i in range(self.num_feet):
            s = self.foot_block(i)
            Q[s, s] = qf

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

        return self.x

    def reinitialize_foothold(
        self,
        foot: FootKey,
        z_meas_body: np.ndarray,
    ) -> None:
        """Reinitialize one foot landmark from current body pose + contact vector."""
        i = self.resolve_index(foot)
        f_world = self.x.pose().transformFrom(z_meas_body)
        self.footholds[i] = f_world

        # Clear stale correlations for this foot block first.
        s = self.foot_block(i)
        self.P[s, :] = 0.0
        self.P[:, s] = 0.0
        self.P[s, s] = (self.foothold_init_sigma**2) * np.eye(3)

    def update_foothold_measurement(
        self,
        foot: FootKey,
        z_meas_body: np.ndarray,
        R_meas: np.ndarray | None = None,
    ) -> gtsam.NavState:
        """EKF update with measurement model z = R^T(f_i - p) + v.

        Intuition:
        - If the predicted body pose + foothold already explain the measured contact
          vector, residual is near zero and correction is small.
        - Otherwise, correction distributes across body and foothold according to P.
        """
        i = self.resolve_index(foot)
        z_meas = np.asarray(z_meas_body, dtype=float)
        if R_meas is None:
            R_meas = (self.sigma_meas**2) * np.eye(3)
        else:
            R_meas = np.asarray(R_meas, dtype=float).reshape(3, 3)

        f_i = self.footholds[i]
        jacobian_pose = np.zeros((3, 6), order="F")
        Hf = np.zeros((3, 3), order="F")
        z_hat = self.x.pose().transformTo(f_i, jacobian_pose, Hf)

        Hx = np.zeros((3, 9), dtype=float)
        Hx[:, 0:6] = jacobian_pose

        dim = self.state_dim()
        H = np.zeros((3, dim), dtype=float)
        H[:, 0:9] = Hx
        H[:, self.foot_block(i)] = Hf

        r = z_meas - z_hat
        self._apply_linear_update(H, r, R_meas)

        return self.x

    def update_foothold_height_prior(
        self,
        foot: FootKey,
        z_world: float = 0.0,
        sigma_z: float | None = None,
    ) -> gtsam.NavState:
        """Weak absolute prior on one foothold's world height.

        Measurement model:
            h(x) = f_i,z
            z_meas = z_world + v,  v ~ N(0, sigma_z^2)
        """
        i = self.resolve_index(foot)
        if sigma_z is None:
            sigma_z = self.sigma_height_prior
        sigma_z = float(sigma_z)
        if sigma_z <= 0.0:
            raise ValueError("sigma_z must be > 0")

        dim = self.state_dim()
        H = np.zeros((1, dim), dtype=float)
        s = self.foot_block(i)
        H[0, s.start + 2] = 1.0

        z_hat = float(self.footholds[i, 2])
        r = np.array([float(z_world) - z_hat], dtype=float)
        R_meas = np.array([[sigma_z**2]], dtype=float)
        self._apply_linear_update(H, r, R_meas)

        return self.x
