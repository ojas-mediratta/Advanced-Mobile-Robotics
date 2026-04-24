from __future__ import annotations

"""Trajectory evaluation and map-building helpers for the LIO package.
"""

from pathlib import Path
from typing import Callable

import numpy as np

from .lio_math import nearest_timestamp_indices, quaternion_xyzw_to_rotation_matrix
from .lio_types import KeyframeState


def load_ground_truth_tum(tum_path: Path) -> np.ndarray:
    """Load and sort a TUM-format reference trajectory by timestamp."""
    tum = np.loadtxt(tum_path, dtype=float)
    if tum.ndim == 1:
        tum = tum[None, :]
    return tum[np.argsort(tum[:, 0])]


def build_global_map(
    keyframes: list[KeyframeState],
    lidar_pose_from_navstate: Callable,
) -> np.ndarray:
    """Transform all keyframe clouds into the world frame and stack them."""
    if not keyframes:
        return np.zeros((0, 3), dtype=float)
    
    # STUDENT TODO START: implement map-stitching with optimized poses.
    # Each keyframe cloud lives in the LiDAR frame at that keyframe.
    # Use lidar_pose_from_navstate(keyframe.navstate) to get the world pose,
    # transform every cloud into the world frame, and return one stacked Nx3 array.
    raise NotImplementedError()
    # STUDENT TODO END: implement map-stitching with optimized poses.


def evaluate_trajectory(
    keyframes: list[KeyframeState],
    ground_truth_tum: np.ndarray | None,
) -> dict:
    """Align an estimated trajectory to a reference trajectory and compute RMSE metrics."""
    if ground_truth_tum is None or len(ground_truth_tum) == 0 or not keyframes:
        return {}

    estimated_timestamps = np.array([kf.timestamp_sec for kf in keyframes], dtype=float)
    estimated_positions = np.vstack([kf.navstate.position() for kf in keyframes])

    gt_timestamps = ground_truth_tum[:, 0]
    gt_positions = ground_truth_tum[:, 1:4]
    gt_orientations = ground_truth_tum[:, 4:]
    matched_indices = nearest_timestamp_indices(gt_timestamps, estimated_timestamps)
    matched_gt_positions = gt_positions[matched_indices]
    estimated_positions_raw = estimated_positions.copy()

    try:
        first_est_pos = estimated_positions[0]
        first_gt_pos = matched_gt_positions[0]
        first_est_rot = keyframes[0].navstate.pose().rotation().matrix()

        gt_orient_first = gt_orientations[matched_indices[0]] if gt_orientations.size else None
        if gt_orient_first is None:
            raise RuntimeError("No ground-truth orientation available")
        first_gt_rot = quaternion_xyzw_to_rotation_matrix(gt_orient_first)

        alignment_R = first_gt_rot @ first_est_rot.T
        centered = (estimated_positions - first_est_pos).T
        rotated = (alignment_R @ centered).T
        estimated_positions = rotated + first_gt_pos
        alignment_t = first_gt_pos - (alignment_R @ first_est_pos)
    except Exception:
        try:
            first_est = estimated_positions[0]
            first_gt = matched_gt_positions[0]
            alignment_t = first_gt - first_est
            estimated_positions = estimated_positions + alignment_t
            alignment_R = np.eye(3, dtype=float)
        except Exception:
            alignment_t = np.zeros(3, dtype=float)
            alignment_R = np.eye(3, dtype=float)

    position_errors = estimated_positions - matched_gt_positions
    ate_rmse = float(np.sqrt(np.mean(np.sum(position_errors**2, axis=1))))
    xy_rmse = float(np.sqrt(np.mean(np.sum(position_errors[:, :2] ** 2, axis=1))))

    return {
        "estimated_timestamps": estimated_timestamps,
        "estimated_positions": estimated_positions,
        "estimated_positions_raw": estimated_positions_raw,
        "ground_truth_timestamps": gt_timestamps[matched_indices],
        "ground_truth_positions": matched_gt_positions,
        "position_errors": position_errors,
        "alignment_R": alignment_R,
        "alignment_t": alignment_t,
        "ate_rmse": ate_rmse,
        "xy_rmse": xy_rmse,
    }


__all__ = ["build_global_map", "evaluate_trajectory", "load_ground_truth_tum"]