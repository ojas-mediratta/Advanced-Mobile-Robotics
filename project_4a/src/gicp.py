from dataclasses import dataclass
import gtsam
import numpy as np
from gtsam import symbol_shorthand
from scipy.spatial import KDTree


X = symbol_shorthand.X
diag_cov = np.array([
    [1e-3,  0.0,    0.0],
    [0.0,   1.0,    0.0],
    [0.0,   0.0,    1.0]
])

@dataclass
class GICPConfig:
    max_iterations: int = 30
    correspondence_distance_threshold: float = 1.5
    covariance_neighbor_count: int = 20
    min_valid_correspondences: int = 10
    bTa_tolerance: float = 1e-5
    loss_tolerance: float = 1e-5
    inner_max_iterations: int = 25
    # GTSAM Pose3 tangent vectors are ordered [wx, wy, wz, tx, ty, tz].
    prior_sigmas: tuple[float, float, float, float, float, float] = (
        0.5,
        0.5,
        0.5,
        5.0,
        5.0,
        5.0,
    )


@dataclass
class GICPResult:
    bTa: gtsam.Pose3
    bTa_matrix: np.ndarray
    losses: list[float]
    tangent_updates: list[np.ndarray]
    transformations: list[gtsam.Pose3]
    source_covariances: np.ndarray
    target_covariances: np.ndarray
    transformed_sources_history: list[np.ndarray]
    correspondences_history: list[np.ndarray]
    valid_mask_history: list[np.ndarray]


def _as_point_cloud(points: np.ndarray) -> np.ndarray:
    """Convert numpy array to appropriate 3D point cloud representation and validate.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected a point cloud with shape (N, 3).")
    return points


def as_pose3(transform: gtsam.Pose3 | np.ndarray | None) -> gtsam.Pose3:
    """Construct and validate a GTSAM Pose3 object.
    """
    if transform is None:
        return gtsam.Pose3.Identity()
    if isinstance(transform, gtsam.Pose3):
        return gtsam.Pose3(transform)
    transform_matrix = np.asarray(transform, dtype=float)
    if transform_matrix.shape != (4, 4):
        raise ValueError("Expected a gtsam.Pose3 or a 4x4 homogeneous transform.")
    return gtsam.Pose3(transform_matrix)


def apply_transformation(points: np.ndarray, transformation: gtsam.Pose3 | np.ndarray) -> np.ndarray:
    """Apply a given SE(3) transformation to a point cloud.
    """
    points = _as_point_cloud(points)
    transform = as_pose3(transformation)
    return transform.transformFrom(points.T).T


def compute_covariance_matrix_single_point(
    neighbors: np.ndarray
) -> np.ndarray:
    """Estimate a local covariance by aligning a chosen variance with the neighborhood's principal 
    directions.

    Args:
        neighbors (np.ndarray): (N, 3) array of points representing the neighborhood.
    
    Returns:
        np.ndarray: (3, 3) array representing the surface-aligned covariance matrix for a 
            neighborhood.
    
    HINT: Reference the footnote on pg. 4 of the Generalized-ICP paper for intended function logic.
    HINT: Use the diag_cov np.ndarray already defined for you at the top of the file.
    """
    # ------- start solution ---------------------------------
    raise NotImplementedError("Implement compute_covariance_matrix_single_point() in gicp.py")
    # ------- end solution -----------------------------------


def compute_covariance_matrices(
    points: np.ndarray,
    k_neighbors: int
) -> np.ndarray:
    """Construct surface-aligned covariance matrices for points of a cloud using their neighborhood.
    """
    points = _as_point_cloud(points)
    if len(points) == 0:
        return np.zeros((0, 3, 3))

    tree = KDTree(points)
    covariances = np.repeat(np.eye(3)[None, :, :], len(points), axis=0)
    k = min(max(k_neighbors, 3), len(points))

    for index, point in enumerate(points):
        _, neighbor_indices = tree.query(point, k=k)
        neighbor_indices = np.atleast_1d(neighbor_indices)
        neighbors = points[neighbor_indices]
        if len(neighbors) < 3:
            continue
        try:
            covariances[index] = compute_covariance_matrix_single_point(neighbors)
        except np.linalg.LinAlgError:
            covariances[index] = np.eye(3)

    return covariances


def transform_covariances(covariances: np.ndarray, transform: gtsam.Pose3) -> np.ndarray:
    """Apply transformation to a set of covariance matrices.
    
    Args:
        covariances (np.ndarray): (N, 3, 3) collection of covariance matrices.
        transform (gtsam.Pose3): desired SE(3) transformation matrix.

    Returns:
        np.ndarray: (N, 3, 3) collection of covariance matrices after the applied transformation.

    HINT: Consider np.einsum() for straightforward operation over different-dimensioned arrays.
    HINT: How do you transform a covariance matrix from one frame to another?
    """
    # ------- start solution ---------------------------------
    raise NotImplementedError("Implement transform_covariance() in gicp.py")
    # ------- end solution -----------------------------------


def find_correspondences(
    transformed_source_points: np.ndarray,
    target_points: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find valid correspondences between point clouds using nearest-neighbors with distance 
    filtering.
    """
    transformed_source_points = _as_point_cloud(transformed_source_points)
    target_points = _as_point_cloud(target_points)

    tree = KDTree(target_points)
    distances, indices = tree.query(transformed_source_points)
    valid_mask = distances <= max_distance

    corresponding_targets = np.zeros_like(transformed_source_points)
    corresponding_targets[valid_mask] = target_points[indices[valid_mask]]
    return corresponding_targets, indices, valid_mask


def build_information_matrices(
    source_covariances_target_frame: np.ndarray,
    target_covariances: np.ndarray,
    target_indices: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Construct information/weight matrices for optimization objective.

    Args:
        source_covariances_target_frame (np.ndarray): (N, 3, 3) collection of source cloud 
            covariance matrices in the target frame.
        target_covariances (np.ndarray): (N, 3, 3) collection of target cloud covariance matrices in
            the target frame.
        target_indices (np.ndarray): (N, 1) array of indices representing element-wise 
            correspondences between transformed source cloud and target cloud.
        valid_mask (np.ndarray): (N, 1) array of bools representing validity of correspondences
            between transformed source cloud and target cloud.

    Returns:
        np.ndarray: (N, 3, 3) collection of information matrices for optimization objective.
    
    HINT: Reference Eq. 2 of the Generalized-ICP paper for the structure of information matrices.
    HINT: If unsure on information matrices, research it in the context of the Mahalanobis distance.
    """
    # ------- start solution ---------------------------------
    raise NotImplementedError("Implement build_information_matrices() in gicp.py")
    # ------- end solution -----------------------------------


def make_gicp_factor(
    key: int,
    source_point: np.ndarray,
    target_point: np.ndarray,
    information: np.ndarray,
) -> gtsam.CustomFactor:
    """Create a factor that measures the relative transform error between the predicted and target 
    point correspondence. The noise model uses the pre-computed information matrix.
    """
    noise_model = gtsam.noiseModel.Gaussian.Information(information)

    def error_function(
        _factor: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: list[np.ndarray] | None,
    ) -> np.ndarray:
        bTa = values.atPose3(key)
        H_bTa = np.zeros((3, 6), order="F")
        H_point = np.zeros((3, 3), order="F")
        predicted = bTa.transformFrom(source_point, H_bTa, H_point)
        if jacobians is not None:
            jacobians[0] = H_bTa
        return predicted - target_point

    return gtsam.CustomFactor(noise_model, [key], error_function)


def build_inner_graph(
    source_points: np.ndarray,
    corresponding_targets: np.ndarray,
    information_matrices: np.ndarray,
    valid_mask: np.ndarray,
    current_bTa: gtsam.Pose3,
    config: GICPConfig,
) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
    """Create a one-variable nonlinear factor graph with a factor for each point correspondence.
    """
    graph = gtsam.NonlinearFactorGraph()

    # A weak prior keeps the one-transform optimization well-conditioned for
    # degenerate scan geometry while still allowing meaningful motion updates.
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(config.prior_sigmas))
    graph.add(gtsam.PriorFactorPose3(X(0), current_bTa, prior_noise))

    valid_indices = np.flatnonzero(valid_mask)
    for source_index in valid_indices:
        graph.add(
            make_gicp_factor(
                key=X(0),
                source_point=source_points[source_index],
                target_point=corresponding_targets[source_index],
                information=information_matrices[source_index],
            )
        )

    initial_values = gtsam.Values()
    initial_values.insert(X(0), current_bTa)
    return graph, initial_values


def optimize_transform(
    source_points: np.ndarray,
    corresponding_targets: np.ndarray,
    information_matrices: np.ndarray,
    valid_mask: np.ndarray,
    current_bTa: gtsam.Pose3,
    config: GICPConfig,
) -> tuple[gtsam.Pose3, float]:
    """Build a nonlinear factor graph representing scan registration and optimize over it with 
    Levenberg-Marquardt.
    """
    graph, initial_values = build_inner_graph(
        source_points=source_points,
        corresponding_targets=corresponding_targets,
        information_matrices=information_matrices,
        valid_mask=valid_mask,
        current_bTa=current_bTa,
        config=config,
    )

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(config.inner_max_iterations)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    optimized_values = optimizer.optimize()
    optimized_bTa = optimized_values.atPose3(X(0))
    return optimized_bTa, float(graph.error(optimized_values))


def gicp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_bTa: gtsam.Pose3 | np.ndarray | None = None,
    config: GICPConfig | None = None,
) -> GICPResult:
    """
    Run a Generalized-ICP loop with a GTSAM nonlinear factor graph backend.

    The outer loop updates correspondences and local covariances. Each inner transform solve is a GTSAM 
    nonlinear graph with one Pose3 variable and one custom factor per active point correspondence, 
    solved with Levenberg-Marquardt on Pose3.

    Args:
        source_points (np.ndarray): (N, 3) array of points from source point cloud.
        target_points (np.ndarray): (N, 3) array of points from target point cloud.
        initial_bTa (gtsam.Pose3 | np.ndarray | None): initial a-to-b SE(3) transformation.
        config (GICPConfig | None): program parameters.
    
    Returns:
        GICPResult: results of the GICP loop.
    """
    config = config or GICPConfig()
    source_points = _as_point_cloud(source_points)
    target_points = _as_point_cloud(target_points)
    current_bTa = as_pose3(initial_bTa)

    source_covariances = compute_covariance_matrices(
        source_points,
        k_neighbors=config.covariance_neighbor_count
    )
    target_covariances = compute_covariance_matrices(
        target_points,
        k_neighbors=config.covariance_neighbor_count
    )

    transformations = [gtsam.Pose3(current_bTa)]
    losses: list[float] = []
    tangent_updates: list[np.ndarray] = []
    transformed_sources_history: list[np.ndarray] = []
    correspondences_history: list[np.ndarray] = []
    valid_mask_history: list[np.ndarray] = []

    for iteration in range(config.max_iterations):
        transformed_source_points = apply_transformation(source_points, current_bTa)
        transformed_sources_history.append(transformed_source_points.copy())

        corresponding_targets, target_indices, valid_mask = find_correspondences(
            transformed_source_points,
            target_points,
            max_distance=config.correspondence_distance_threshold,
        )
        correspondences_history.append(corresponding_targets.copy())
        valid_mask_history.append(valid_mask.copy())

        if np.count_nonzero(valid_mask) < config.min_valid_correspondences:
            raise RuntimeError(
                "Too few valid correspondences to continue GICP optimization."
            )

        source_covariances_target_frame = transform_covariances(source_covariances, current_bTa)
        information_matrices = build_information_matrices(
            source_covariances_target_frame,
            target_covariances,
            target_indices,
            valid_mask,
        )

        optimized_bTa, loss = optimize_transform(
            source_points=source_points,
            corresponding_targets=corresponding_targets,
            information_matrices=information_matrices,
            valid_mask=valid_mask,
            current_bTa=current_bTa,
            config=config,
        )

        tangent_update = current_bTa.localCoordinates(optimized_bTa)
        tangent_updates.append(tangent_update)
        losses.append(loss)
        current_bTa = optimized_bTa
        transformations.append(gtsam.Pose3(current_bTa))

        if np.linalg.norm(tangent_update) < config.bTa_tolerance:
            break
        if iteration > 0 and abs(losses[-2] - losses[-1]) < config.loss_tolerance:
            break

    return GICPResult(
        bTa=current_bTa,
        bTa_matrix=current_bTa.matrix(),
        losses=losses,
        tangent_updates=tangent_updates,
        transformations=transformations,
        source_covariances=source_covariances,
        target_covariances=target_covariances,
        transformed_sources_history=transformed_sources_history,
        correspondences_history=correspondences_history,
        valid_mask_history=valid_mask_history,
    )
