import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

Segment = Tuple[Tuple[float, float], Tuple[float, float]]


ARENA: List[Segment] = [
    ((-1.6, -1.0), ( 1.6, -1.0)),
    (( 1.6, -1.0), ( 1.6,  1.0)),
    (( 1.6,  1.0), (-1.6,  1.0)),
    ((-1.6,  1.0), (-1.6, -1.0)),
]

OBSTACLES: List[Segment] = [
    ((-0.5, -0.5), (-0.5,  0.5)),
    (( 0.3, -0.3), ( 0.9, -0.3)),
    (( 0.2,  0.2), ( 0.7,  0.6)),
]

ALL_SEGMENTS: List[Segment] = ARENA + OBSTACLES

OBSTACLES_ROOM = [
    ((-0.2, -1.0), (-0.2,  0.3)),
    (( 0.2, -0.4), ( 0.2,  1.0)),
]

ALL_SEGMENTS_ROOM = ARENA + OBSTACLES_ROOM


def simulate_lidar(
    pose: np.ndarray,
    obstacles: List[Segment],
    n_rays: int = 60,
    max_range: float = 1.5,
    fov: float = 2 * np.pi,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a 2D LiDAR by ray-casting against line segments.

    Casts n_rays evenly spaced over the field of view from the robot's pose.
    Each ray finds the closest intersection with the given segments. Rays that
    don't hit anything (or exceed max_range) return NaN.

    Args:
        pose (np.ndarray): (3,) robot pose [x, y, theta] in world frame.
        obstacles (List[Segment]): Line segments to cast against.
        n_rays (int): Number of rays.
        max_range (float): Maximum sensor range in meters.
        fov (float): Field of view in radians (default: full 360).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            angles: (n_rays,) ray angles in robot's local frame (rad).
            ranges: (n_rays,) hit distances (NaN if no hit).
            points: (n_rays, 2) hit points in world frame (NaN if no hit).
    """
    rx, ry, rtheta = float(pose[0]), float(pose[1]), float(pose[2])
    p = np.array([rx, ry])

    include_endpoint = (fov < 2 * np.pi - 1e-9)
    angles = np.linspace(-fov / 2, fov / 2, n_rays, endpoint=include_endpoint)
    world_angles = angles + rtheta

    ranges = np.full(n_rays, np.nan)
    points = np.full((n_rays, 2), np.nan)

    for i, wangle in enumerate(world_angles):
        d = np.array([np.cos(wangle), np.sin(wangle)])
        best_t = max_range
        hit = False

        for seg in obstacles:
            a = np.asarray(seg[0], dtype=float)
            b = np.asarray(seg[1], dtype=float)
            ba = b - a

            # ray-segment intersection: p + t*d = a + s*ba
            d_x_ba = d[0] * ba[1] - d[1] * ba[0]
            if abs(d_x_ba) < 1e-9:
                continue

            ap = a - p
            t = (ap[0] * ba[1] - ap[1] * ba[0]) / d_x_ba
            s = (ap[0] * d[1] - ap[1] * d[0]) / d_x_ba

            if t >= 0 and t < best_t and 0 <= s <= 1:
                best_t = t
                hit = True

        if hit:
            ranges[i] = best_t
            points[i] = p + best_t * d

    return angles, ranges, points
