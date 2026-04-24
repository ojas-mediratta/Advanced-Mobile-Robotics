from __future__ import annotations

"""Small math helpers shared across the LIO modules."""

import numpy as np


def rotation_matrix_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest-arc rotation that maps ``source`` onto ``target``."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)

    cross = np.cross(source, target)
    dot = float(np.dot(source, target))
    if np.linalg.norm(cross) < 1e-9:
        if dot > 0.0:
            return np.eye(3, dtype=float)
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(source[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        axis -= source * np.dot(source, axis)
        axis /= np.linalg.norm(axis)
        return -np.eye(3, dtype=float) + 2.0 * np.outer(axis, axis)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3, dtype=float) + skew + skew @ skew * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))


def quaternion_xyzw_to_rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Convert a quaternion in ``[qx, qy, qz, qw]`` order to a rotation matrix."""
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=float)
    quaternion_xyzw /= np.linalg.norm(quaternion_xyzw)
    qx, qy, qz, qw = quaternion_xyzw
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    xw = qx * qw
    yz = qy * qz
    yw = qy * qw
    zw = qz * qw
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


def nearest_timestamp_indices(
    reference_timestamps: np.ndarray,
    query_timestamps: np.ndarray,
) -> np.ndarray:
    """Return indices of the nearest reference timestamps for each query timestamp."""
    insertion_indices = np.searchsorted(reference_timestamps, query_timestamps)
    insertion_indices = np.clip(insertion_indices, 0, len(reference_timestamps) - 1)
    previous_indices = np.clip(insertion_indices - 1, 0, len(reference_timestamps) - 1)

    use_previous = np.abs(reference_timestamps[previous_indices] - query_timestamps) <= np.abs(
        reference_timestamps[insertion_indices] - query_timestamps
    )
    return np.where(use_previous, previous_indices, insertion_indices)


__all__ = [
    "nearest_timestamp_indices",
    "quaternion_xyzw_to_rotation_matrix",
    "rotation_matrix_between_vectors",
]