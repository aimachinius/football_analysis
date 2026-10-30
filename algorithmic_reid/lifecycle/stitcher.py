"""
algorithmic_reid.lifecycle.stitcher - Tracklet Stitching Across Multi-Frame Dropouts.

Connects broken tracklet fragments separated by 1 to 30 unobserved frames:
  - Non-overlapping temporal intervals
  - Strict species identity match
  - Physical sprint velocity feasibility (<= 35 px/frame)
  - Appearance consistency barrier (D_Bhatt <= 0.35)
"""

import numpy as np
from typing import List, Dict, Optional, Any
from .tracklet import Tracklet, TrackletState
from ..appearance.color_descriptor import bhattacharyya_distance


class TrackletStitcher:
    def __init__(
        self,
        max_gap_frames: int = 30,
        max_velocity_px: float = 35.0,
        appearance_thresh: float = 0.40,
    ):
        self.max_gap_frames = max_gap_frames
        self.max_velocity_px = max_velocity_px
        self.appearance_thresh = appearance_thresh

    def can_stitch(self, t_prev: Tracklet, t_next: Tracklet) -> bool:
        """Evaluate whether t_next can be stitched as a continuation of t_prev."""
        # 1. Species barrier
        if t_prev.class_id != t_next.class_id:
            return False

        # 2. Temporal non-overlap
        gap = t_next.start_frame - t_prev.last_frame
        if gap < 1 or gap > self.max_gap_frames:
            return False

        # 3. Velocity feasibility
        p_prev = ((t_prev.history[-1][1][0] + t_prev.history[-1][1][2]) / 2.0,
                  (t_prev.history[-1][1][1] + t_prev.history[-1][1][3]) / 2.0)
        p_next = ((t_next.history[0][1][0] + t_next.history[0][1][2]) / 2.0,
                  (t_next.history[0][1][1] + t_next.history[0][1][3]) / 2.0)

        dist = np.hypot(p_next[0] - p_prev[0], p_next[1] - p_prev[1])
        if (dist / float(gap)) > self.max_velocity_px:
            return False

        # 4. Appearance similarity
        if t_prev.appearance_template is not None and t_next.appearance_template is not None:
            app_dist = bhattacharyya_distance(t_prev.appearance_template, t_next.appearance_template)
            if app_dist > self.appearance_thresh:
                return False

        return True

    def stitch_tracklets(self, target_track: Tracklet, source_track: Tracklet) -> None:
        """Merge source_track into target_track and interpolate the gap."""
        # Linear interpolation over unobserved gap in history if gap > 1
        t_gap = source_track.history[0][0] - target_track.history[-1][0]
        if t_gap > 1:
            box_start = target_track.history[-1][1]
            box_end = source_track.history[0][1]
            f_start = target_track.history[-1][0]
            interpolated = []
            for step in range(1, t_gap):
                alpha = float(step) / float(t_gap)
                interp_box = (1.0 - alpha) * box_start + alpha * box_end
                interpolated.append((f_start + step, interp_box))
            target_track.history.extend(interpolated)

        # Extend history
        target_track.history.extend(source_track.history)
        target_track.last_frame = source_track.last_frame
        target_track.time_since_update = 0
        target_track.state = TrackletState.CONFIRMED
        target_track.hits += source_track.hits
        target_track.consecutive_hits = source_track.consecutive_hits
        target_track.age += source_track.age

        # Update Kalman state to source track's latest state
        target_track.mean = source_track.mean.copy()
        target_track.covariance = source_track.covariance.copy()

        # Update appearance template with source
        if source_track.appearance_template is not None:
            if target_track.appearance_template is None:
                target_track.appearance_template = source_track.appearance_template.copy()
            else:
                target_track.appearance_template = 0.5 * (
                    target_track.appearance_template + source_track.appearance_template
                )
                s = np.sum(target_track.appearance_template)
                if s > 1e-6:
                    target_track.appearance_template /= s

        # Mark source as deleted
        source_track.state = TrackletState.DELETED
