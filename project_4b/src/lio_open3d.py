from __future__ import annotations

"""Open3D-backed point-cloud registration helpers used by the LIO front-end."""

from dataclasses import dataclass
from types import SimpleNamespace

import gtsam
import numpy as np
import open3d as o3d


@dataclass
class GICPConfig:
    max_iterations: int = 50
    max_correspondence_distance: float = 1.0
    voxel_size: float | None = None


def _to_open3d_pcd(
    points: np.ndarray,
    voxel_size: float | None = None,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    if points.dtype != np.float64:
        points = points.astype(np.float64)
    pcd.points = o3d.utility.Vector3dVector(points)
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    return pcd


def _ensure_normals(pcd: o3d.geometry.PointCloud, voxel_size: float | None = None) -> None:
    radius = (voxel_size * 2.0) if voxel_size else 0.5
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    pcd.normalize_normals()


def _matrix_to_pose3(transform: np.ndarray) -> gtsam.Pose3:
    rotation = gtsam.Rot3(transform[:3, :3])
    translation = np.asarray(transform[:3, 3], dtype=float)
    return gtsam.Pose3(rotation, translation)


def _pose3_to_matrix(pose: gtsam.Pose3) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(pose.rotation().matrix(), dtype=float)
    transform[:3, 3] = np.asarray(pose.translation(), dtype=float)
    return transform


def gicp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_bTa=None,
    config: GICPConfig | None = None,
):
    """Register ``source_points`` onto ``target_points`` with Open3D ICP/GICP."""
    cfg = config or GICPConfig()

    src_pcd = _to_open3d_pcd(source_points, cfg.voxel_size)
    tgt_pcd = _to_open3d_pcd(target_points, cfg.voxel_size)
    _ensure_normals(src_pcd, cfg.voxel_size)
    _ensure_normals(tgt_pcd, cfg.voxel_size)

    if initial_bTa is None:
        init_transform = np.eye(4, dtype=float)
    elif isinstance(initial_bTa, gtsam.Pose3):
        init_transform = _pose3_to_matrix(initial_bTa)
    else:
        init_transform = np.asarray(initial_bTa, dtype=float)
        if init_transform.shape != (4, 4):
            raise ValueError("initial_bTa must be a gtsam.Pose3 or 4x4 matrix")

    try:
        from open3d.pipelines.registration import (
            ICPConvergenceCriteria,
            TransformationEstimationForGeneralizedICP,
            registration_generalized_icp,
        )

        result = registration_generalized_icp(
            src_pcd,
            tgt_pcd,
            cfg.max_correspondence_distance,
            init_transform,
            TransformationEstimationForGeneralizedICP(),
            ICPConvergenceCriteria(max_iteration=cfg.max_iterations),
        )
        transform = np.asarray(result.transformation, dtype=float)
    except Exception:
        from open3d.pipelines.registration import (
            ICPConvergenceCriteria,
            TransformationEstimationPointToPlane,
            registration_icp,
        )

        result = registration_icp(
            src_pcd,
            tgt_pcd,
            cfg.max_correspondence_distance,
            init_transform,
            TransformationEstimationPointToPlane(),
            ICPConvergenceCriteria(max_iteration=cfg.max_iterations),
        )
        transform = np.asarray(result.transformation, dtype=float)

    return SimpleNamespace(bTa=_matrix_to_pose3(transform))


__all__ = ["GICPConfig", "gicp"]