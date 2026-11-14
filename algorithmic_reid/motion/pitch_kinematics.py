"""
algorithmic_reid.motion.pitch_kinematics - 2D Pitch-Metric Kinematics via Homography (Direction A).

Projects player foot coordinates to FIFA pitch coordinates (105m x 68m) using pitch homography.
Applies:
  1. Human physiological sprint speed barrier: v_max = 10.5 m/s (~38 km/h).
  2. Scale- and perspective-invariant metric distance cost C_pitch = min(1.0, dist_meters / 4.0).
"""

import numpy as np
from typing import List, Tuple, Optional, Any
from pitch_homo import safe_project


def bboxes_to_feet(bboxes: np.ndarray) -> np.ndarray:
    """Extract bottom-center foot position [(x1+x2)/2, y2] from bboxes."""
    if len(bboxes) == 0:
        return np.empty((0, 2), dtype=np.float64)
    feet_x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    feet_y = bboxes[:, 3]
    return np.column_stack([feet_x, feet_y])


def compute_pitch_cost_and_gating(
    tracks: List[Any],
    det_boxes: np.ndarray,
    pitch_homography: Optional[np.ndarray],
    fps: float = 25.0,
    max_sprint_mps: float = 10.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise metric pitch cost matrix (M, N) and boolean violation mask (M, N).
    """
    M = len(tracks)
    N = len(det_boxes)
    cost_pitch = np.zeros((M, N), dtype=np.float64)
    violation_mask = np.zeros((M, N), dtype=bool)

    if M == 0 or N == 0 or pitch_homography is None:
        return cost_pitch, violation_mask

    # Project detections to pitch
    det_feet = bboxes_to_feet(det_boxes)
    det_pitch, det_valid = safe_project(pitch_homography, det_feet)

    # Project tracks to pitch
    track_boxes = np.array([t.bbox for t in tracks], dtype=np.float64)
    track_feet = bboxes_to_feet(track_boxes)
    track_pitch, track_valid = safe_project(pitch_homography, track_feet)

    for i, t in enumerate(tracks):
        if not track_valid[i]:
            continue

        p_t = track_pitch[i]
        dt = max(1, t.time_since_update) / float(fps)
        max_reach_m = max(2.5, max_sprint_mps * dt)

        for j in range(N):
            if not det_valid[j]:
                continue

            p_d = det_pitch[j]
            dist_m = float(np.hypot(p_d[0] - p_t[0], p_d[1] - p_t[1]))
            cost_pitch[i, j] = min(1.0, dist_m / 4.0)

            # Physiological human speed violation
            if dist_m > max_reach_m:
                violation_mask[i, j] = True

    return cost_pitch, violation_mask

