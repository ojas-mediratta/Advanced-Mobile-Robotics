import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.highlevel import AnyReader
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gicp import GICPConfig, apply_transformation, gicp


def pointcloud2_to_xyz(msg) -> np.ndarray:
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<f4", "<f4", "<f4"],
            "offsets": [0, 4, 8],
            "itemsize": msg.point_step,
        }
    )
    count = msg.width * msg.height
    structured = np.frombuffer(msg.data, dtype=dtype, count=count)
    points = np.column_stack([structured["x"], structured["y"], structured["z"]])
    return points[np.all(np.isfinite(points), axis=1)]


def load_scan_by_index(bag_path: Path, topic: str, frame_index: int) -> tuple[np.ndarray, int]:
    with AnyReader([bag_path]) as reader:
        connection = next(c for c in reader.connections if c.topic == topic)
        for current_index, (conn, timestamp, rawdata) in enumerate(
            reader.messages(connections=[connection])
        ):
            if current_index == frame_index:
                msg = reader.deserialize(rawdata, conn.msgtype)
                return pointcloud2_to_xyz(msg), timestamp
    raise IndexError(f"Frame index {frame_index} is out of range.")


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0.0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def preprocess_scan(
    points: np.ndarray,
    voxel_size: float,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    downsampled = voxel_downsample(points, voxel_size)
    if len(downsampled) <= max_points:
        return downsampled
    indices = rng.choice(len(downsampled), size=max_points, replace=False)
    return downsampled[indices]


def nearest_neighbor_rmse(source_points: np.ndarray, target_points: np.ndarray) -> float:
    if len(source_points) == 0 or len(target_points) == 0:
        return float("inf")
    distances, _ = KDTree(target_points).query(source_points)
    return float(np.sqrt(np.mean(distances**2)))


def evaluate_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    transform: np.ndarray,
) -> tuple[float, float]:
    before_rmse = nearest_neighbor_rmse(source_points, target_points)
    aligned_source = apply_transformation(source_points, transform)
    after_rmse = nearest_neighbor_rmse(aligned_source, target_points)
    return before_rmse, after_rmse


def plot_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    gicp_transform: np.ndarray,
    title: str,
) -> None:
    source_aligned_gicp = apply_transformation(source_points, gicp_transform)

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    rows = [
        ("Before", source_points),
        ("GTSAM GICP", source_aligned_gicp)
    ]
    projections = [
        ("XY", (0, 1)),
        ("XZ", (0, 2)),
    ]

    for row_index, (row_name, source_variant) in enumerate(rows):
        for col_index, (proj_name, dims) in enumerate(projections):
            ax = axes[row_index, col_index]
            ax.scatter(
                target_points[:, dims[0]],
                target_points[:, dims[1]],
                s=2,
                c="tab:blue",
                alpha=0.5,
                label="target",
            )
            ax.scatter(
                source_variant[:, dims[0]],
                source_variant[:, dims[1]],
                s=2,
                c="tab:red",
                alpha=0.5,
                label="source",
            )
            ax.set_title(f"{row_name} ({proj_name})")
            ax.set_aspect("equal", adjustable="box")
            if row_index == 0 and col_index == 0:
                ax.legend(loc="upper right")

    fig.suptitle(title)
    fig.tight_layout()


def main(
        bag: Path = Path("data/court_of_sciences.bag"),
        source_index: int = 1000,
        target_index: int = 1015,
        max_points: int = 5000
) -> None:
    """Visualize scan-to-scan registration on real LiDAR data from a rosbag.

    Args:
        bag (Path): Path to the rosbag file.
        source_index (int): Source LiDAR scan index.
        target_index (int): Target LiDAR scan index.
        max_points (int): Upper threshold of points in the cloud (will downsample if needed).
    """
    topic = "/aquila1/os_cloud_node/points"
    voxel_size = 0.25
    seed = 21

    rng = np.random.default_rng(seed)
    source_raw, source_ts = load_scan_by_index(bag, topic, source_index)
    target_raw, target_ts = load_scan_by_index(bag, topic, target_index)

    source_points = preprocess_scan(source_raw, voxel_size, max_points, rng)
    target_points = preprocess_scan(target_raw, voxel_size, max_points, rng)

    gicp_result = gicp(
        source_points,
        target_points,
        config=GICPConfig(
            correspondence_distance_threshold=2.0,
            covariance_neighbor_count=12,
            min_valid_correspondences=40,
            inner_max_iterations=30,
            prior_sigmas=(1.0, 1.0, 1.0, 2.0, 2.0, 2.0),
        ),
    )

    gicp_before_rmse, gicp_after_rmse = evaluate_transform(
        source_points, target_points, gicp_result.bTa_matrix
    )

    tag = f"src{source_index:04d}_tgt{target_index:04d}"
    plot_registration(
        source_points=source_points,
        target_points=target_points,
        gicp_transform=gicp_result.bTa_matrix,
        title=(
            f"Real Scan Registration {tag}\n"
            f"timestamps: {source_ts} -> {target_ts}\n"
            f"gtsam_gicp_rmse: {gicp_before_rmse:.3f} -> {gicp_after_rmse:.3f} | "
        ),
    )

    summary_lines = [
        f"bag={bag}",
        f"topic={topic}",
        f"source_index={source_index}",
        f"target_index={target_index}",
        f"source_timestamp={source_ts}",
        f"target_timestamp={target_ts}",
        f"source_points_raw={len(source_raw)}",
        f"target_points_raw={len(target_raw)}",
        f"source_points_used={len(source_points)}",
        f"target_points_used={len(target_points)}",
        "",
        "gtsam_gicp_transform=",
        np.array2string(gicp_result.bTa_matrix, precision=5),
        f"gtsam_gicp_rmse_before={gicp_before_rmse:.6f}",
        f"gtsam_gicp_rmse_after={gicp_after_rmse:.6f}",
        ""
    ]
    print("\n".join(summary_lines) + "\n")