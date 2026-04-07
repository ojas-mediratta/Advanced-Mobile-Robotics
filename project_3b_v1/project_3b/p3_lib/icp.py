import gtsam
from scipy.spatial import KDTree
import numpy as np
from typing import Optional, Tuple


def apply_pose2(pose2: gtsam.Pose2, points: np.ndarray) -> np.ndarray:
    """Apply a gtsam.Pose2 rigid transform to an (N, 2) point array.

    Args:
        pose2 (gtsam.Pose2): The transform to apply (rotation + translation).
        points (np.ndarray): (N, 2) array of 2D points.

    Returns:
        np.ndarray: (N, 2) transformed points.
    """
    c, s = np.cos(pose2.theta()), np.sin(pose2.theta())
    R = np.array([[c, -s], [s, c]])
    return points @ R.T + np.array([pose2.x(), pose2.y()])


def world_to_local(points_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Convert (N, 2) points from world frame to the robot's local frame.

    Args:
        points_world (np.ndarray): (N, 2) points in world coordinates.
        pose (np.ndarray): (3,) robot pose [x, y, theta] in world frame.

    Returns:
        np.ndarray: (N, 2) points in the robot's local frame.
    """
    rx, ry, rtheta = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = np.cos(rtheta), np.sin(rtheta)
    R = np.array([[c, -s], [s, c]])
    return (points_world - np.array([rx, ry])) @ R


def find_correspondences(
    source_points: np.ndarray,
    target_tree: KDTree,
    max_corr_dist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find nearest-neighbor correspondences between source points and a target KDTree.

    For each source point, find the closest target point. Reject pairs where the
    distance is >= max_corr_dist (these are likely bad matches from non-overlapping
    regions of the two scans).

    Args:
        source_points (np.ndarray): (N, 2) source point cloud (NaN already filtered).
        target_tree (KDTree): scipy KDTree built on the target points.
        max_corr_dist (float): Maximum correspondence distance in meters.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (src_matched, tgt_matched), each (K, 2).
            Only the K pairs with distance < max_corr_dist are returned.
    """
    dists, idx = target_tree.query(source_points)
    mask = dists < max_corr_dist
    src_matched = source_points[mask]
    tgt_matched = target_tree.data[idx[mask]]
    return src_matched, tgt_matched


def compute_transform(
    src_matched: np.ndarray,
    tgt_matched: np.ndarray,
) -> Optional[gtsam.Pose2]:
    """Compute the best-fit rigid transform from matched source to target points.

    Uses gtsam.Pose2.Align() which solves via SVD. The returned Pose2 maps
    source points into the target frame: target = aTb * source.

    Args:
        src_matched (np.ndarray): (K, 2) matched source points.
        tgt_matched (np.ndarray): (K, 2) matched target points (same ordering).

    Returns:
        Optional[gtsam.Pose2]: The rigid transform mapping source -> target,
            or None if alignment fails (e.g. collinear points).
    """
    return gtsam.Pose2.Align(tgt_matched.T, src_matched.T)


def icp(
    source: np.ndarray,
    target: np.ndarray,
    tolerance: float = 1e-5,
    max_corr_dist: float = 0.3,
    initial_guess: Optional[gtsam.Pose2] = None,
) -> Tuple[gtsam.Pose2, float]:
    """Iterative Closest Point: align source point cloud onto target.

    Alternates between finding nearest-neighbor correspondences and computing
    the best-fit rigid transform until convergence. Handles NaN points
    (filter them out before processing).

    Args:
        source (np.ndarray): (N, 2) point cloud in frame B (to be aligned). May contain NaN.
        target (np.ndarray): (M, 2) point cloud in frame A (reference). May contain NaN.
        tolerance (float): Convergence threshold. Stop when both the translation and
            rotation of the step transform are smaller than this.
        max_corr_dist (float): Passed to find_correspondences.
        initial_guess (Optional[gtsam.Pose2]): If provided, pre-transform source
            by this before iterating (warm start).

    Returns:
        Tuple[gtsam.Pose2, float]:
            aTb: Total rigid transform from frame B into frame A.
            mean_error: Mean nearest-neighbor distance at convergence.
    """
    maskA = ~np.isnan(target).any(axis=1)
    maskB = ~np.isnan(source).any(axis=1)
    pointsA = target[maskA].astype(float)
    pointsB = source[maskB].copy().astype(float)
    if initial_guess is not None:
        pointsB = apply_pose2(initial_guess, pointsB)
        aTb_total = initial_guess
    else:
        aTb_total = gtsam.Pose2()
    tree = KDTree(pointsA)
    final_err = np.inf

    while True:
        src_matched, tgt_matched = find_correspondences(pointsB, tree, max_corr_dist)
        if len(src_matched) < 3:
            break
        aTb_step = compute_transform(src_matched, tgt_matched)
        if aTb_step is None:
            break
        pointsB = apply_pose2(aTb_step, pointsB)
        aTb_total = aTb_step.compose(aTb_total)
        final_err = np.mean(np.linalg.norm(src_matched - tgt_matched, axis=1))
        delta_t = np.sqrt(aTb_step.x()**2 + aTb_step.y()**2)
        delta_r = abs(aTb_step.theta())
        if delta_t < tolerance and delta_r < tolerance:
            break

    return aTb_total, final_err
