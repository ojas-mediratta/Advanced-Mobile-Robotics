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
    src_matched, tgt_matched = None, None
    # ------- start solution ---------------------------------
    nn_dists, nn_ids = target_tree.query(source_points)
    kept_src = []
    kept_tgt = []

    for pt_idx in range(len(source_points)):
        if nn_dists[pt_idx] < max_corr_dist:
            kept_src.append(source_points[pt_idx])
            kept_tgt.append(target_tree.data[nn_ids[pt_idx]])

    src_matched = np.array(kept_src)
    tgt_matched = np.array(kept_tgt)
    # ------- end solution -----------------------------------
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
    aTb = None
    # ------- start solution ---------------------------------
    try:
        best_pose = gtsam.Pose2.Align(tgt_matched.T, src_matched.T)
        aTb = best_pose
    except Exception:
        aTb = None
    # ------- end solution -----------------------------------
    return aTb


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
    aTb_total = gtsam.Pose2()
    mean_error = np.inf
    # ------- start solution ---------------------------------
    clean_source = []
    clean_target = []

    for row_num in range(len(source)):
        if not np.isnan(source[row_num]).any():
            clean_source.append(source[row_num])

    for row_num in range(len(target)):
        if not np.isnan(target[row_num]).any():
            clean_target.append(target[row_num])

    source_ok = np.array(clean_source)
    target_ok = np.array(clean_target)

    if source_ok.size == 0 or target_ok.size == 0:
        return aTb_total, mean_error

    if initial_guess is not None:
        moving_pts = apply_pose2(initial_guess, source_ok)
        aTb_total = initial_guess
    else:
        moving_pts = np.array(source_ok)

    tree = KDTree(target_ok)

    while True:
        src_matched, tgt_matched = find_correspondences(moving_pts, tree, max_corr_dist)

        if len(src_matched) < 3:
            break

        step_pose = compute_transform(src_matched, tgt_matched)
        if step_pose is None:
            break

        moving_pts = apply_pose2(step_pose, moving_pts)
        aTb_total = step_pose.compose(aTb_total)
        shifted_match = apply_pose2(step_pose, src_matched)
        err_sum = 0.0
        for match_idx in range(len(shifted_match)):
            x_gap = tgt_matched[match_idx][0] - shifted_match[match_idx][0]
            y_gap = tgt_matched[match_idx][1] - shifted_match[match_idx][1]
            err_sum += np.sqrt(x_gap * x_gap + y_gap * y_gap)

        mean_error = err_sum / len(shifted_match)

        move_size = np.sqrt(step_pose.x() * step_pose.x() + step_pose.y() * step_pose.y())
        turn_size = abs(step_pose.theta())
        if move_size < tolerance and turn_size < tolerance:
            break
    # ------- end solution -----------------------------------
    return aTb_total, mean_error
