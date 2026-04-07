from __future__ import annotations

# Extract 2D LiDAR scans from a Google Cartographer rosbag.
# Downloads the bag (~500 MB), extracts MultiEchoLaserScan messages,
# converts to 2D point clouds, and saves as .npz.
#
# Usage:
#   python scripts/extract_real_lidar.py              # auto-downloads bag
#   python scripts/extract_real_lidar.py path/to.bag  # use local bag

import struct
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from rosbags.rosbag1 import Reader

BAG_URL = "https://storage.googleapis.com/cartographer-public-data/bags/backpack_2d/b2-2016-04-05-14-44-52.bag"
BAG_NAME = "b2-2016-04-05-14-44-52.bag"

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "real_lidar"
PLOT_DIR = DATA_DIR / "plots"
BAG_PATH = DATA_DIR / BAG_NAME
OUTPUT_PATH = DATA_DIR / "deutsches_museum_scans.npz"

TOPIC = "horizontal_laser_2d"
SUBSAMPLE_EVERY = 25  # take every Nth scan (~0.68s gap at 36.6 Hz)
RANGE_CAP_M = 30.0    # discard returns beyond this (indoor cap)

ScanRecord = dict[str, np.ndarray | float]


def download_bag() -> None:
    """Download the bag file if not already present."""
    if BAG_PATH.exists():
        size_mb = BAG_PATH.stat().st_size / 1e6
        print(f"Bag already exists: {BAG_PATH} ({size_mb:.0f} MB)")
        return

    print(f"Downloading {BAG_NAME}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(BAG_URL, str(BAG_PATH))
    print("Download complete.")


def inspect_bag() -> float:
    """Print bag metadata and return duration in seconds."""
    print(f"\nBag inspection: {BAG_NAME}")
    with Reader(BAG_PATH) as reader:
        duration = reader.duration / 1e9
        print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"Message count: {reader.message_count}")
        print(f"\nTopics:")
        for topic, info in reader.topics.items():
            print(f"\t{topic}: {info.msgtype} ({info.msgcount} msgs)")
    return duration


def parse_multi_echo_laser_scan(rawdata: bytes) -> dict[str, float | np.ndarray]:
    """Parse sensor_msgs/MultiEchoLaserScan from raw ROS1 bytes.

    Each beam may have multiple echoes (e.g. hitting glass then the wall behind it).
    We take the first echo per beam, i.e. the nearest surface.
    """
    offset = 0

    # std_msgs/Header
    seq = struct.unpack_from("<I", rawdata, offset)[0]; offset += 4
    stamp_sec = struct.unpack_from("<I", rawdata, offset)[0]; offset += 4
    stamp_nsec = struct.unpack_from("<I", rawdata, offset)[0]; offset += 4
    frame_id_len = struct.unpack_from("<I", rawdata, offset)[0]; offset += 4
    offset += frame_id_len

    # scan metadata
    angle_min = struct.unpack_from("<f", rawdata, offset)[0]; offset += 4
    angle_max = struct.unpack_from("<f", rawdata, offset)[0]; offset += 4
    angle_increment = struct.unpack_from("<f", rawdata, offset)[0]; offset += 4
    _time_increment = struct.unpack_from("<f", rawdata, offset)[0]; offset += 4
    _scan_time = struct.unpack_from("<f", rawdata, offset)[0]; offset += 4
    range_min = struct.unpack_from("<f", rawdata, offset)[0]; offset += 4
    range_max = struct.unpack_from("<f", rawdata, offset)[0]; offset += 4

    # ranges: array of MultiEchoLaserEcho
    n_beams = struct.unpack_from("<I", rawdata, offset)[0]; offset += 4
    ranges = np.zeros(n_beams, dtype=np.float32)
    for i in range(n_beams):
        n_echoes = struct.unpack_from("<I", rawdata, offset)[0]; offset += 4
        if n_echoes > 0:
            ranges[i] = struct.unpack_from("<f", rawdata, offset)[0]
        offset += n_echoes * 4

    timestamp = stamp_sec + stamp_nsec * 1e-9

    return {
        "angle_min": angle_min, "angle_max": angle_max,
        "angle_increment": angle_increment,
        "range_min": range_min, "range_max": range_max,
        "ranges": ranges,
        "timestamp": timestamp,
    }


def polar_to_cartesian(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    range_cap: float,
) -> np.ndarray:
    """Convert polar ranges to (N, 2) cartesian points, filtering invalid returns."""
    n = len(ranges)
    angles = angle_min + np.arange(n) * angle_increment
    mask = (ranges > 0) & (ranges < range_cap)
    r, a = ranges[mask], angles[mask]
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def extract_all_scans() -> list[ScanRecord]:
    """Read the bag and extract all horizontal_laser_2d scans."""
    print(f"\nExtracting scans from {TOPIC}")
    all_scans: list[ScanRecord] = []

    with Reader(BAG_PATH) as reader:
        connections = [c for c in reader.connections if c.topic == TOPIC]
        if not connections:
            raise ValueError(f"Topic '{TOPIC}' not found in bag")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = parse_multi_echo_laser_scan(rawdata)
            points = polar_to_cartesian(msg["ranges"], msg["angle_min"], msg["angle_increment"], RANGE_CAP_M)
            all_scans.append({"points": points, "timestamp": msg["timestamp"]})

    print(f"Extracted {len(all_scans)} scans")
    pts_per_scan = [s["points"].shape[0] for s in all_scans]
    print(f"Points per scan: min={min(pts_per_scan)}, max={max(pts_per_scan)}, mean={np.mean(pts_per_scan):.0f}")
    return all_scans


def subsample_scans(all_scans: list[ScanRecord], every_n: int) -> list[ScanRecord]:
    """Take every Nth scan from the full sequence."""
    subsampled = all_scans[::every_n]
    if len(subsampled) > 1:
        dts = np.diff([s["timestamp"] for s in subsampled])
        print(f"Subsampled: {len(subsampled)} scans (every {every_n}th), avg dt={dts.mean():.2f}s")
    return subsampled


def save_npz(scans: list[ScanRecord], output_path: Path) -> None:
    """Save scans to .npz (object array of point clouds + timestamps)."""
    print(f"\nSaving scans as numpy arrays")
    points_list = [s["points"] for s in scans]
    timestamps = np.array([s["timestamp"] for s in scans])

    scans_arr = np.empty(len(points_list), dtype=object)
    for i, pts in enumerate(points_list):
        scans_arr[i] = pts

    metadata = {
        "source_bag": BAG_NAME,
        "topic": TOPIC,
        "subsample_every": str(SUBSAMPLE_EVERY),
        "range_cap_m": str(RANGE_CAP_M),
        "n_scans": str(len(scans)),
    }

    np.savez(
        output_path,
        scans=scans_arr,
        timestamps=timestamps,
        metadata_keys=np.array(list(metadata.keys())),
        metadata_vals=np.array(list(metadata.values())),
    )
    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved {output_path} ({size_mb:.1f} MB)")


def plot_single_scan(scans: list[ScanRecord], output_dir: Path) -> None:
    """Plot a single scan."""
    scan = scans[len(scans) // 5]
    pts = scan["points"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(pts[:, 0], pts[:, 1], s=1, c="steelblue")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title(f"Single scan ({pts.shape[0]} points, <{RANGE_CAP_M}m cap)")

    path = output_dir / "single_scan.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_consecutive_pairs(scans: list[ScanRecord], output_dir: Path) -> None:
    """Plot 3 consecutive scan pairs from different parts of the sequence."""
    n = len(scans)
    indices = [0, n // 2, n - 2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, i in zip(axes, indices):
        a, b = scans[i]["points"], scans[i + 1]["points"]
        ax.scatter(a[:, 0], a[:, 1], s=1, alpha=0.6, label=f"Scan {i}")
        ax.scatter(b[:, 0], b[:, 1], s=1, alpha=0.6, label=f"Scan {i+1}")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.legend(markerscale=5)
        ax.set_title(f"Pair ({i}, {i+1})")

    plt.suptitle("Consecutive Scan Pairs")
    plt.tight_layout()
    path = output_dir / "consecutive_pairs.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract 2D LiDAR scans from a Cartographer rosbag")
    parser.add_argument("bag", nargs="?", default=None,
                        help="Path to .bag file. If omitted, auto-downloads from Google storage.")
    args = parser.parse_args()

    if args.bag is not None:
        BAG_PATH = Path(args.bag).resolve()
        if not BAG_PATH.exists():
            print(f"Error: bag file not found: {BAG_PATH}")
            raise SystemExit(1)
    else:
        download_bag()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    inspect_bag()
    all_scans = extract_all_scans()
    scans = subsample_scans(all_scans, SUBSAMPLE_EVERY)

    print(f"\nGenerating plots in {PLOT_DIR}")
    plot_single_scan(scans, PLOT_DIR)
    plot_consecutive_pairs(scans, PLOT_DIR)

    save_npz(scans, OUTPUT_PATH)

    print(f"\nVerifying saved file")
    data = np.load(OUTPUT_PATH, allow_pickle=True)
    loaded_scans = data["scans"]
    loaded_ts = data["timestamps"]
    print(f"Loaded {len(loaded_scans)} scans, {len(loaded_ts)} timestamps")
    assert len(loaded_scans) == len(scans), "Scan count mismatch"
    assert np.allclose(loaded_ts[0], scans[0]["timestamp"]), "Timestamp mismatch"
    assert loaded_scans[0].shape == scans[0]["points"].shape, "Shape mismatch"
    print("Verification passed.")
