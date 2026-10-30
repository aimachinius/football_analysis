"""
algorithmic_reid.motion.motion_predictor - Camera Motion Compensation (CMC).
Applies Stage 3 inter-frame Homography G_t to warp predicted tracklet states.
"""

import numpy as np
from typing import List, Any
from .kalman_filter import z_to_bbox, bbox_to_z


def apply_camera_motion_to_tracks(tracks: List[Any], homography: np.ndarray) -> None:
    """
    Warp tracklet predicted positions and bboxes using 3x3 camera homography.
    homography: G_t mapping points from frame t-1 to frame t (or current to reference).
    """
    if homography is None or len(tracks) == 0:
        return

    # Check validity of homography
    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        return

    for track in tracks:
        # Extract bbox from mean [xc, yc, a, h]
        bbox = z_to_bbox(track.mean[:4])
        # 4 corners of bbox
        corners = np.array([
            [bbox[0], bbox[1], 1.0],
            [bbox[2], bbox[1], 1.0],
            [bbox[2], bbox[3], 1.0],
            [bbox[0], bbox[3], 1.0],
        ], dtype=np.float64).T

        warped = np.dot(homography, corners)
        w_norm = warped[2, :]
        valid_mask = np.abs(w_norm) > 1e-5
        if not np.all(valid_mask):
            continue

        warped_pts = warped[:2, :] / w_norm
        new_x1 = np.min(warped_pts[0, :])
        new_x2 = np.max(warped_pts[0, :])
        new_y1 = np.min(warped_pts[1, :])
        new_y2 = np.max(warped_pts[1, :])

        new_bbox = np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float64)
        new_z = bbox_to_z(new_bbox)

        # Update mean position components
        track.mean[:4] = new_z
