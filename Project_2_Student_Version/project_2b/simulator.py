"""Synthetic IMU/contact simulator for the 2B EKF notebook.

This module separates simulation from filter logic:
- Constructor builds the full synthetic dataset.
- replay(callback) streams measurements to any filter callback.
- calculate_metrics(...) computes trajectory/landmark metrics after a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import gtsam
import numpy as np

try:
    from .measurement import Measurement
except ImportError:
    from measurement import Measurement


@dataclass
class SimulationMetrics:
    p_true_hist: np.ndarray
    p_dr_hist: np.ndarray
    p_ekf_hist: np.ndarray
    err_dr: np.ndarray
    err_ekf: np.ndarray
    rmse_dr: float
    rmse_ekf: float
    final_dr_error: float
    final_ekf_error: float
    landmark_gt: List[Tuple[str, np.ndarray]]
    landmark_est: List[Tuple[str, np.ndarray]]
    travel_distance: float
    steps: int
    footstep_starts: int


class Simulator:
    """Build synthetic trajectory, IMU, and contact measurements once."""

    def __init__(
        self,
        dt: float = 0.05,
        num_steps: int = 360,
        seed: int = 6,
        speed: float = 0.60,
        sigma_gyro: float = 8e-4,
        sigma_acc: float = 1e-2,
        sigma_meas: float = 5e-3,
        stance_duration: int = 80,
        contact_updates_on_start_only: bool = True,
    ) -> None:
        self.dt = float(dt)
        self.N = int(num_steps)
        self.seed = int(seed)
        self.speed = float(speed)
        self.sigma_gyro = float(sigma_gyro)
        self.sigma_acc = float(sigma_acc)
        self.sigma_meas = float(sigma_meas)
        self.stance_duration = int(stance_duration)
        self.contact_updates_on_start_only = bool(contact_updates_on_start_only)

        self.rng = np.random.default_rng(self.seed)

        self.foot_offsets: Dict[str, np.ndarray] = {
            "LF": np.array([0.20, 0.16, -0.40]),
            "RF": np.array([0.20, -0.16, -0.40]),
            "LH": np.array([-0.15, 0.16, -0.40]),
            "RH": np.array([-0.15, -0.16, -0.40]),
        }
        self.foot_cycle = ["LF", "RF", "LH", "RH"]
        self.foot_names = sorted(set(self.foot_cycle))

        # Schedule footstep start events across the full simulation horizon.
        # Keep the same cadence as before (every 25 steps, first at k=20), but
        # extend until near the end of the trajectory instead of stopping at 320.
        self.start_indices = list(range(20, self.N - 39, 25))
        self.start_events: Dict[int, Tuple[str, np.ndarray]] = {}
        self.end_events: Dict[int, str] = {}
        for i, k_start in enumerate(self.start_indices):
            foot_id = self.foot_cycle[i % len(self.foot_cycle)]
            self.start_events[k_start] = (foot_id, self.foot_offsets[foot_id])
            self.end_events[k_start + self.stance_duration] = foot_id

        # Ground-truth trajectory and synthetic IMU stream.
        self.R_true_mat_hist = np.zeros((self.N, 3, 3))
        self.p_true_hist = np.zeros((self.N, 3))
        self.v_true_hist = np.zeros((self.N, 3))
        self.a_true_hist = np.zeros((self.N, 3))
        self.omega_meas_hist = np.zeros((self.N, 3))
        self.f_meas_hist = np.zeros((self.N, 3))

        for k in range(self.N):
            t = k * self.dt
            R_true, p_true, v_true, a_true = self.gt_state_straight(t)

            omega_true = np.zeros(3)
            f_true = R_true.matrix().T @ (a_true - np.array([0.0, 0.0, -9.81]))

            omega_meas = omega_true + self.rng.normal(0.0, self.sigma_gyro, 3)
            f_meas = f_true + self.rng.normal(0.0, self.sigma_acc, 3)

            self.R_true_mat_hist[k] = R_true.matrix()
            self.p_true_hist[k] = p_true
            self.v_true_hist[k] = v_true
            self.a_true_hist[k] = a_true
            self.omega_meas_hist[k] = omega_meas
            self.f_meas_hist[k] = f_meas

        self.travel_distance = float(self.p_true_hist[-1, 0] - self.p_true_hist[0, 0])
        self.footstep_starts = len(self.start_events)

        # Shared initial perturbation used by both DR and EKF runs.
        R0, p0, v0, _ = self.gt_state_straight(0.0)
        self.R0 = R0
        self.p0 = p0
        self.v0 = v0
        self.p0_init = p0 + self.rng.normal(0.0, 0.02, 3)
        self.v0_init = v0 + self.rng.normal(0.0, 0.02, 3)

        # Unified measurement stream and ground-truth foothold markers at contact starts.
        self.measurements: List[Measurement] = []
        # Backward compatibility for older notebooks/tests.
        self.samples = self.measurements
        self.landmark_gt: List[Tuple[str, np.ndarray]] = []
        self._build_replay_samples()

    def __repr__(self) -> str:
        return (
            "Simulator(\n"
            f"  dt={self.dt:.3f}, steps={self.N}, speed={self.speed:.2f}, seed={self.seed},\n"
            f"  sigma_gyro={self.sigma_gyro:.3e}, sigma_acc={self.sigma_acc:.3e},\n"
            f"  sigma_meas={self.sigma_meas:.3e},\n"
            f"  stance_duration={self.stance_duration}, feet={self.foot_names},\n"
            f"  travel_distance={self.travel_distance:.2f}, footstep_starts={self.footstep_starts}\n"
            ")"
        )

    @staticmethod
    def gt_state_straight(t: float, speed: float = 0.60):
        p = np.array([speed * t, 0.0, 0.40])
        v = np.array([speed, 0.0, 0.0])
        a = np.array([0.0, 0.0, 0.0])
        R = gtsam.Rot3.Yaw(0.0)
        return R, p, v, a

    def _build_replay_samples(self) -> None:
        active_contacts: Dict[str, Tuple[str, np.ndarray]] = {}

        for k in range(self.N):
            p_true = self.p_true_hist[k]
            R_true_mat = self.R_true_mat_hist[k]
            previous_contact_feet = set(active_contacts.keys())

            # Match notebook semantics: process start first.
            if k in self.start_events:
                foot_id, body_offset = self.start_events[k]
                event_label = f"{foot_id}_{k}"
                foothold_world_ref = p_true + R_true_mat @ body_offset
                foothold_world_ref[2] = 0.0

                active_contacts[foot_id] = (event_label, foothold_world_ref.copy())
                self.landmark_gt.append((event_label, foothold_world_ref.copy()))

            # Then process swing-end event.
            if k in self.end_events:
                foot_id = self.end_events[k]
                active_contacts.pop(foot_id, None)

            # Create body-frame contact measurements for all active contacts.
            current_contact_feet = set(active_contacts.keys())
            old_contact_feet = sorted(previous_contact_feet & current_contact_feet)
            new_contact_feet = sorted(current_contact_feet - previous_contact_feet)

            measurements_all: Dict[str, np.ndarray] = {}
            for foot_id in sorted(active_contacts):
                _, foothold_world_ref = active_contacts[foot_id]
                z_true = R_true_mat.T @ (foothold_world_ref - p_true)
                z_meas = z_true + self.rng.normal(0.0, self.sigma_meas, 3)
                measurements_all[foot_id] = z_meas

            old_contacts = {
                foot_id: measurements_all[foot_id].copy() for foot_id in old_contact_feet
            }
            new_contacts = {
                foot_id: measurements_all[foot_id].copy() for foot_id in new_contact_feet
            }

            measurement = Measurement(
                k=k,
                dt=self.dt,
                omega_meas=self.omega_meas_hist[k].copy(),
                f_meas=self.f_meas_hist[k].copy(),
                old_contacts=old_contacts,
                new_contacts=new_contacts,
            )
            self.measurements.append(measurement)

    def replay(self, callback: Callable[[Measurement], None]) -> None:
        """Deliver synthetic measurements step-by-step to a callback."""
        for measurement in self.measurements:
            callback(measurement)

    def calculate_metrics(
        self,
        p_dr_hist: np.ndarray,
        p_ekf_hist: np.ndarray,
        landmark_est: List[Tuple[str, np.ndarray]],
    ) -> SimulationMetrics:
        """Compute trajectory and landmark metrics after filter runs."""
        p_dr_hist = np.asarray(p_dr_hist, dtype=float)
        p_ekf_hist = np.asarray(p_ekf_hist, dtype=float)

        if p_dr_hist.shape != self.p_true_hist.shape:
            raise ValueError(
                f"p_dr_hist shape mismatch: expected {self.p_true_hist.shape}, got {p_dr_hist.shape}"
            )
        if p_ekf_hist.shape != self.p_true_hist.shape:
            raise ValueError(
                f"p_ekf_hist shape mismatch: expected {self.p_true_hist.shape}, got {p_ekf_hist.shape}"
            )

        err_dr = np.linalg.norm(p_dr_hist - self.p_true_hist, axis=1)
        err_ekf = np.linalg.norm(p_ekf_hist - self.p_true_hist, axis=1)

        rmse_dr = float(np.sqrt(np.mean(err_dr**2)))
        rmse_ekf = float(np.sqrt(np.mean(err_ekf**2)))

        return SimulationMetrics(
            p_true_hist=self.p_true_hist.copy(),
            p_dr_hist=p_dr_hist,
            p_ekf_hist=p_ekf_hist,
            err_dr=err_dr,
            err_ekf=err_ekf,
            rmse_dr=rmse_dr,
            rmse_ekf=rmse_ekf,
            final_dr_error=float(err_dr[-1]),
            final_ekf_error=float(err_ekf[-1]),
            landmark_gt=[(name, p.copy()) for name, p in self.landmark_gt],
            landmark_est=[(name, np.asarray(p, dtype=float).copy()) for name, p in landmark_est],
            travel_distance=self.travel_distance,
            steps=self.N,
            footstep_starts=self.footstep_starts,
        )
