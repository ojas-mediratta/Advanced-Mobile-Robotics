from __future__ import annotations

"""Shared data containers and configuration for the LIO package."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import gtsam
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ImuSample:
    """Single IMU message after parsing and timestamp normalization."""

    timestamp_sec: float
    linear_acceleration: np.ndarray
    angular_velocity: np.ndarray
    stream_index: int


@dataclass(frozen=True)
class LidarFrame:
    """Single LiDAR frame represented as an ``Nx3`` point array."""

    timestamp_sec: float
    points: np.ndarray
    stream_index: int


@dataclass(frozen=True)
class CombinedMeasurement:
    """Tagged union used to iterate over the mixed IMU/LiDAR message stream."""

    kind: Literal["imu", "lidar"]
    timestamp_sec: float
    stream_index: int
    imu: ImuSample | None = None
    lidar: LidarFrame | None = None


@dataclass
class KeyframeState:
    """State stored for each LiDAR keyframe in the factor graph."""

    keyframe_index: int
    timestamp_sec: float
    state_key: int
    bias_key: int
    navstate: gtsam.NavState
    bias: gtsam.imuBias.ConstantBias
    cloud: np.ndarray


@dataclass
class LioSlamConfig:
    """Configuration shared by both teaching backends.

    The defaults are tuned for the included Bruin Plaza bag and favor a
    stable demonstration run over aggressive IMU weighting.
    """

    bag_path: Path = ROOT / "data" / "bruin_plaza" / "bruin_plaza.bag"
    ground_truth_path: Path = ROOT / "data" / "bruin_plaza_rko_lio" / "bruin_plaza_0_tum.txt"
    imu_topic: str = "/aquila1/mpu6050/imu"
    lidar_topic: str = "/aquila1/os_cloud_node/points"
    use_preintegration: bool = True
    stream_start_index: int = 25791
    stream_end_index: int = 32190
    use_ground_truth_slice_initialization: bool = True
    imu_time_offset_sec: float = -0.05
    lidar_time_offset_sec: float = 0.0
    gravity_mps2: float = 9.81
    accel_noise_sigma: float = 1.5
    gyro_noise_sigma: float = 0.3
    integration_noise_sigma: float = 1e-2
    accel_bias_rw_sigma: float = 0.1
    gyro_bias_rw_sigma: float = 0.02
    initialization_imu_sample_count: int = 200
    stationary_accel_tolerance_mps2: float = 1.0
    stationary_gyro_tolerance_rps: float = 0.05
    prior_pose_sigmas: tuple[float, float, float, float, float, float] = (
        0.1,
        0.1,
        0.1,
        0.5,
        0.5,
        0.5,
    )
    prior_velocity_sigmas: tuple[float, float, float] = (20.0, 20.0, 20.0)
    prior_bias_sigmas: tuple[float, float, float, float, float, float] = (
        2.0,
        2.0,
        2.0,
        0.5,
        0.5,
        0.5,
    )
    bias_between_sigmas: tuple[float, float, float, float, float, float] = (
        0.01,
        0.01,
        0.01,
        0.002,
        0.002,
        0.002,
    )
    lidar_pose_sigmas: tuple[float, float, float, float, float, float] = (
        0.2,
        0.2,
        0.2,
        0.5,
        0.5,
        0.5,
    )
    baselink2imu_t: tuple[float, float, float] = (0.006253, -0.011775, 0.007645)
    baselink2imu_R: tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    baselink2lidar_t: tuple[float, float, float] = (0.0, 0.0, 0.0)
    baselink2lidar_R: tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    voxel_size_m: float = 0.25
    max_points_per_scan: int = 6000
    min_preintegration_dt_sec: float = 1e-4
    enable_gicp: bool = True
    batch_max_iterations: int = 50
    loop_closure_search_radius_m: float = 1.5
    loop_closure_min_keyframe_separation: int = 20
    loop_closure_max_candidates_per_keyframe: int = 1
    numerical_jacobian_eps: float = 1e-6
    loop_closure_interval: int = 5


@dataclass
class LioSlamResult:
    """Return value shared by the batch and ISAM2 drivers."""

    keyframes: list[KeyframeState] = field(default_factory=list)
    update_times_sec: list[float] = field(default_factory=list)
    ground_truth_tum: np.ndarray | None = None


__all__ = [
    "CombinedMeasurement",
    "ImuSample",
    "KeyframeState",
    "LidarFrame",
    "LioSlamConfig",
    "LioSlamResult",
    "ROOT",
]