from __future__ import annotations

"""Loop-closure candidate selection helpers for the LIO package."""

import numpy as np

from .lio_types import KeyframeState


def find_loop_closure_candidate_pairs(
    keyframes: list[KeyframeState],
    min_keyframe_separation: int,
    search_radius_m: float,
    max_candidates_per_keyframe: int,
) -> list[tuple[int, int]]:
    """Return nearby, well-separated keyframe index pairs for loop-closure attempts."""
    if len(keyframes) < 2:
        return []

    positions = np.vstack([keyframe.navstate.position() for keyframe in keyframes])
    candidates: list[tuple[int, int]] = []

    for current_index in range(len(keyframes)):
        candidate_distances: list[tuple[float, int]] = []
        for reference_index in range(current_index):
            separation = current_index - reference_index
            if separation < min_keyframe_separation:
                continue
            distance = float(np.linalg.norm(positions[current_index] - positions[reference_index]))
            if distance <= search_radius_m:
                candidate_distances.append((distance, reference_index))

        candidate_distances.sort(key=lambda item: item[0])
        for _, reference_index in candidate_distances[:max_candidates_per_keyframe]:
            candidates.append((reference_index, current_index))

    return candidates


__all__ = ["find_loop_closure_candidate_pairs"]