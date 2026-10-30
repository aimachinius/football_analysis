"""
tracker/ball_tracker.py - Modular Ball Trajectory Stabilizer & Adaptive Interpolation

Implements Milestone P1 architecture:
  1. Coordinate space decoupling with CameraMotion (reference frame dynamics).
  2. Kinematic gating and outlier rejection (speed bounds, 2-frame consistency, foot proximity).
  3. Adaptive 3-tier dropout handling:
     - Tier 1 (1-4 frames): PCHIP monotonic interpolation.
     - Tier 2 (5-10 frames): Kinematic consistency check (velocity alignment <= 60 deg).
     - Tier 3 (>10 frames): Strict no-op (0 fake frames).
  4. Confidence-weighted bilateral smoothing with strict <= 2.5 px clamping.
  5. Downstream data contract reconciliation for PitchMapper and MinimapRenderer.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.interpolate import PchipInterpolator


class BallTrajectoryStabilizer:
    """
    Stabilizes ball detections, filters spurious outliers, and provides adaptive
    interpolation across occlusions without introducing phantom trajectories.
    """

    def __init__(
        self,
        v_max: float = 50.0,                    # Max physical ball speed in px/frame (~126 km/h)
        max_acc: float = 30.0,                  # Max acceleration spike without confirmation
        foot_proximity_thresh: float = 60.0,    # Max distance (px) to player foot for kick allowance
        tier1_max_gap: int = 4,                 # Tier 1 max gap (1-4 frames) -> PCHIP
        tier2_max_gap: int = 10,                # Tier 2 max gap (5-10 frames) -> Kinematic check
        tier2_max_angle_deg: float = 60.0,      # Max direction change for Tier 2 interpolation
        max_smoothing_deviation: float = 2.5,   # Max deviation (px) clamped against raw detection
        interp_confidence: float = 0.30,        # Confidence assigned to interpolated frames
    ):
        self.v_max = float(v_max)
        self.max_acc = float(max_acc)
        self.foot_proximity_thresh = float(foot_proximity_thresh)
        self.tier1_max_gap = int(tier1_max_gap)
        self.tier2_max_gap = int(tier2_max_gap)
        self.tier2_max_angle_deg = float(tier2_max_angle_deg)
        self.max_smoothing_deviation = float(max_smoothing_deviation)
        self.interp_confidence = float(interp_confidence)

    def process(
        self,
        ball_tracks: List[Dict[int, Dict[str, Any]]],
        camera_motion: Optional[Any] = None,
        player_tracks: Optional[List[Dict[int, Dict[str, Any]]]] = None,
    ) -> List[Dict[int, Dict[str, Any]]]:
        """
        Process and stabilize full ball trajectory across all video frames.

        Args:
            ball_tracks: List of per-frame ball dictionaries [{1: {"bbox": ...}}, ...].
            camera_motion: CameraMotion object providing `.to_reference(t, pts)` or `.Ms`.
            player_tracks: Optional list of player bounding boxes for kick context gating.

        Returns:
            List of stabilized ball dictionaries preserving data contracts.
        """
        num_frames = len(ball_tracks)
        if num_frames == 0:
            return []

        # ---------------------------------------------------------------------
        # Step 1: Extract raw centers, dimensions, and confidences
        # ---------------------------------------------------------------------
        raw_centers = [None] * num_frames
        raw_sizes = [None] * num_frames
        raw_confs = [None] * num_frames
        raw_interpolated = [False] * num_frames

        for t in range(num_frames):
            frame_data = ball_tracks[t]
            if 1 in frame_data:
                bbox = frame_data[1]["bbox"]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                w = max(1.0, float(bbox[2] - bbox[0]))
                h = max(1.0, float(bbox[3] - bbox[1]))
                conf = float(frame_data[1].get("confidence", 0.80))
                is_interp = bool(frame_data[1].get("is_interpolated", False))

                raw_centers[t] = np.array([cx, cy], dtype=np.float64)
                raw_sizes[t] = (w, h)
                raw_confs[t] = conf
                raw_interpolated[t] = is_interp

        # ---------------------------------------------------------------------
        # Step 2: Transform to Reference Coordinate Space (Camera Decoupling)
        # ---------------------------------------------------------------------
        ref_centers = [None] * num_frames
        for t in range(num_frames):
            if raw_centers[t] is not None:
                ref_centers[t] = self._to_reference(camera_motion, t, raw_centers[t])

        # ---------------------------------------------------------------------
        # Step 3: Outlier Rejection & Kinematic Gating
        # ---------------------------------------------------------------------
        valid_mask = [ref_centers[t] is not None for t in range(num_frames)]
        detected_indices = [t for t in range(num_frames) if valid_mask[t]]

        # 3.1 Boundary Outlier Gating
        if len(detected_indices) >= 2:
            # Check initial detection (i=0)
            t_0 = detected_indices[0]
            t_1 = detected_indices[1]
            dt_0 = t_1 - t_0
            v_01 = (ref_centers[t_1] - ref_centers[t_0]) / max(1, dt_0)
            speed_0 = np.linalg.norm(v_01)
            is_near_foot_0 = self._is_near_player(
                player_tracks, t_0, raw_centers[t_0]
            ) or self._is_near_player(
                player_tracks, t_1, raw_centers[t_1]
            )

            is_initial_outlier = False
            if speed_0 > self.v_max:
                if speed_0 > 2.0 * self.v_max:
                    is_initial_outlier = True
                elif len(detected_indices) >= 3:
                    t_2 = detected_indices[2]
                    dt_1 = t_2 - t_1
                    v_12 = (ref_centers[t_2] - ref_centers[t_1]) / max(1, dt_1)
                    acc_0 = np.linalg.norm(v_01 - v_12)
                    if acc_0 > self.max_acc and not is_near_foot_0:
                        is_initial_outlier = True
                elif not is_near_foot_0:
                    is_initial_outlier = True

            if is_initial_outlier:
                valid_mask[t_0] = False

            # Check trailing detection (i=len-1)
            t_n = detected_indices[-1]
            t_prev = detected_indices[-2]
            dt_n = t_n - t_prev
            v_prev_n = (ref_centers[t_n] - ref_centers[t_prev]) / max(1, dt_n)
            speed_n = np.linalg.norm(v_prev_n)
            is_near_foot_n = self._is_near_player(
                player_tracks, t_n, raw_centers[t_n]
            ) or self._is_near_player(
                player_tracks, t_prev, raw_centers[t_prev]
            )

            is_trailing_outlier = False
            if speed_n > self.v_max:
                if speed_n > 2.0 * self.v_max:
                    is_trailing_outlier = True
                elif len(detected_indices) >= 3:
                    t_prev2 = detected_indices[-3]
                    dt_prev = t_prev - t_prev2
                    v_prev2_prev = (ref_centers[t_prev] - ref_centers[t_prev2]) / max(1, dt_prev)
                    acc_n = np.linalg.norm(v_prev_n - v_prev2_prev)
                    if acc_n > self.max_acc and not is_near_foot_n:
                        is_trailing_outlier = True
                elif not is_near_foot_n:
                    is_trailing_outlier = True

            if is_trailing_outlier:
                valid_mask[t_n] = False

        # 3.2 Multi-Frame Outlier Burst Gating (Lookahead Chords)
        detected_indices = [t for t in range(num_frames) if valid_mask[t]]
        if len(detected_indices) > 2:
            i = 1
            prev_valid_idx = 0
            k_max = 4

            while i < len(detected_indices):
                prev_t = detected_indices[prev_valid_idx]
                curr_t = detected_indices[i]
                dt_prev = curr_t - prev_t
                pos_prev = ref_centers[prev_t]
                pos_curr = ref_centers[curr_t]
                speed_prev = np.linalg.norm(pos_curr - pos_prev) / max(1, dt_prev)

                if speed_prev > self.v_max:
                    # Candidate at index i deviates from previous valid detection by speed > v_max
                    # Search ahead up to k_max = 4 candidate detections
                    found_reconnection = False
                    reconnect_k = -1

                    for k in range(1, k_max + 1):
                        if i + k < len(detected_indices):
                            next_t = detected_indices[i + k]
                            pos_next = ref_centers[next_t]
                            chord_dt = next_t - prev_t
                            chord_speed = np.linalg.norm(pos_next - pos_prev) / max(1, chord_dt)

                            if chord_speed <= self.v_max:
                                # Check if intermediate burst frames are explained by foot proximity kicks
                                any_kick_explained = False
                                for m in range(i, i + k):
                                    cand_m_t = detected_indices[m]
                                    cand_m_prev_t = prev_t if m == i else detected_indices[m - 1]
                                    m_speed = np.linalg.norm(ref_centers[cand_m_t] - ref_centers[cand_m_prev_t]) / max(1, cand_m_t - cand_m_prev_t)
                                    m_near_foot = self._is_near_player(player_tracks, cand_m_t, raw_centers[cand_m_t])
                                    if m_near_foot and m_speed <= 2.0 * self.v_max:
                                        any_kick_explained = True
                                        break

                                if not any_kick_explained:
                                    found_reconnection = True
                                    reconnect_k = k
                                    break

                    if found_reconnection:
                        # Mark all intermediate burst frames {i, ..., i+k-1} as outliers
                        for m in range(i, i + reconnect_k):
                            burst_t = detected_indices[m]
                            valid_mask[burst_t] = False

                        # Advance past the burst: candidate i + reconnect_k is confirmed valid
                        prev_valid_idx = i + reconnect_k
                        i = i + reconnect_k + 1
                    else:
                        # No chord reconnection within k_max (legitimate forward transition)
                        prev_valid_idx = i
                        i += 1
                else:
                    # speed_prev <= v_max: candidate is kinematically valid
                    prev_valid_idx = i
                    i += 1

        # ---------------------------------------------------------------------
        # Step 4: Adaptive 3-Tier Dropout Handling
        # ---------------------------------------------------------------------
        confirmed_indices = [t for t in range(num_frames) if valid_mask[t]]
        stabilized_ref = [None] * num_frames
        is_interpolated_mask = [False] * num_frames
        track_states = ["LOST"] * num_frames

        # Copy confirmed detections
        for t in confirmed_indices:
            stabilized_ref[t] = ref_centers[t].copy()
            track_states[t] = "DETECTED"

        # Interpolate between confirmed pairs
        for i in range(len(confirmed_indices) - 1):
            t_a = confirmed_indices[i]
            t_b = confirmed_indices[i + 1]
            gap = t_b - t_a - 1

            if gap <= 0:
                continue

            if gap <= self.tier1_max_gap:
                # Tier 1 (1-4 frames): PCHIP monotonic interpolation
                self._interpolate_segment(
                    stabilized_ref,
                    is_interpolated_mask,
                    track_states,
                    t_a,
                    t_b,
                    confirmed_indices,
                )

            elif gap <= self.tier2_max_gap:
                # Tier 2 (5-10 frames): Kinematic consistency check
                p_a = stabilized_ref[t_a]
                p_b = stabilized_ref[t_b]
                avg_speed = np.linalg.norm(p_b - p_a) / float(gap + 1)

                # Calculate velocity directions before t_a and after t_b
                v_in = self._get_velocity_at(stabilized_ref, t_a, direction="in")
                v_out = self._get_velocity_at(stabilized_ref, t_b, direction="out")

                direction_ok = True
                if v_in is not None and v_out is not None:
                    norm_in = np.linalg.norm(v_in)
                    norm_out = np.linalg.norm(v_out)
                    if norm_in > 1.0 and norm_out > 1.0:
                        cos_angle = np.dot(v_in, v_out) / (norm_in * norm_out)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(cos_angle))
                        if angle_deg > self.tier2_max_angle_deg:
                            direction_ok = False

                if avg_speed <= self.v_max and direction_ok:
                    self._interpolate_segment(
                        stabilized_ref,
                        is_interpolated_mask,
                        track_states,
                        t_a,
                        t_b,
                        confirmed_indices,
                    )
                else:
                    # Direction change or excessive speed during medium gap -> REJECT
                    pass

            else:
                # Tier 3 (> 10 frames): STRICT NO-OP (remain None / LOST)
                pass

        # ---------------------------------------------------------------------
        # Step 5: Confidence-Weighted Bilateral Smoothing (Clamped <= 2.5 px)
        # ---------------------------------------------------------------------
        smoothed_ref = [None] * num_frames

        # Find continuous populated segments
        segments = []
        curr_seg = []
        for t in range(num_frames):
            if stabilized_ref[t] is not None:
                curr_seg.append(t)
            else:
                if curr_seg:
                    segments.append(curr_seg)
                    curr_seg = []
        if curr_seg:
            segments.append(curr_seg)

        for seg in segments:
            seg_len = len(seg)
            for idx, t in enumerate(seg):
                pt = stabilized_ref[t].copy()

                # If this frame was interpolated, PCHIP is already smooth
                if is_interpolated_mask[t]:
                    smoothed_ref[t] = pt
                    continue

                if seg_len < 3:
                    smoothed_pt = pt
                else:
                    # Edge-aware polynomial regression over window W=4
                    # Interior points (>= 2 left and >= 2 right) use quadratic degree 2 (zero ballistic curvature bias)
                    # Boundary points use linear degree 1 (avoids edge oscillation / Runge phenomenon)
                    W = 4
                    offsets = []
                    positions = []
                    weights = []
                    for offset in range(-W, W + 1):
                        neighbor_idx = idx + offset
                        if 0 <= neighbor_idx < seg_len:
                            nt = seg[neighbor_idx]
                            offsets.append(offset)
                            positions.append(stabilized_ref[nt])
                            conf = raw_confs[nt] if raw_confs[nt] is not None else 0.8
                            spatial = np.exp(-0.5 * (offset / 3.0) ** 2)
                            weights.append(conf * spatial)

                    offsets = np.array(offsets, dtype=float)
                    positions = np.array(positions, dtype=float)
                    weights = np.array(weights, dtype=float)

                    has_left = np.sum(offsets < 0) >= 2
                    has_right = np.sum(offsets > 0) >= 2
                    deg = 2 if (has_left and has_right and len(offsets) >= 5) else 1

                    if deg == 2:
                        A = np.column_stack([offsets ** 2, offsets, np.ones_like(offsets)])
                        W_mat = np.diag(weights)
                        try:
                            coeffs = np.linalg.lstsq(W_mat @ A, W_mat @ positions, rcond=None)[0]
                            smoothed_pt = coeffs[2]
                        except np.linalg.LinAlgError:
                            smoothed_pt = pt
                    else:
                        A = np.column_stack([offsets, np.ones_like(offsets)])
                        W_mat = np.diag(weights)
                        try:
                            coeffs = np.linalg.lstsq(W_mat @ A, W_mat @ positions, rcond=None)[0]
                            smoothed_pt = coeffs[1]
                        except np.linalg.LinAlgError:
                            smoothed_pt = pt

                # STRICT CLAMP: Deviation from raw detection <= max_smoothing_deviation
                raw_ref = ref_centers[t] if ref_centers[t] is not None else pt
                dev_vec = smoothed_pt - raw_ref
                dev_dist = float(np.linalg.norm(dev_vec))
                if dev_dist > self.max_smoothing_deviation:
                    smoothed_pt = raw_ref + dev_vec * (self.max_smoothing_deviation / dev_dist)

                smoothed_ref[t] = smoothed_pt

        # ---------------------------------------------------------------------
        # Step 6: Map Back to Frame Coordinates & Build Output Track Dictionaries
        # ---------------------------------------------------------------------
        output_tracks = []
        default_size = (20.0, 20.0)

        # Estimate default size from detections if available
        valid_sizes = [s for s in raw_sizes if s is not None]
        if valid_sizes:
            default_size = (
                float(np.median([s[0] for s in valid_sizes])),
                float(np.median([s[1] for s in valid_sizes])),
            )

        for t in range(num_frames):
            if smoothed_ref[t] is None:
                output_tracks.append({})
                continue

            # Convert reference coordinates back to frame pixel space
            img_center = self._to_frame(camera_motion, t, smoothed_ref[t])
            cx, cy = float(img_center[0]), float(img_center[1])

            # Size: preserve raw size or interpolate
            w, h = raw_sizes[t] if raw_sizes[t] is not None else default_size
            half_w, half_h = w / 2.0, h / 2.0
            bbox = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]

            is_interp = is_interpolated_mask[t]
            conf = self.interp_confidence if is_interp else (
                raw_confs[t] if raw_confs[t] is not None else 0.85
            )

            entry = {
                "bbox": bbox,
                "confidence": float(conf),
                "is_interpolated": bool(is_interp),
                "track_state": track_states[t],
            }

            # Preserve existing metadata fields if present in original input
            if 1 in ball_tracks[t]:
                for k, v in ball_tracks[t][1].items():
                    if k not in entry:
                        entry[k] = v

            output_tracks.append({1: entry})

        return output_tracks

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _to_reference(self, camera_motion: Optional[Any], t: int, pt: np.ndarray) -> np.ndarray:
        if camera_motion is None:
            return pt.copy()
        if hasattr(camera_motion, "to_reference"):
            return camera_motion.to_reference(t, pt.reshape(1, 2))[0]
        return pt.copy()

    def _to_frame(self, camera_motion: Optional[Any], t: int, pt_ref: np.ndarray) -> np.ndarray:
        if camera_motion is None:
            return pt_ref.copy()
        if hasattr(camera_motion, "to_frame"):
            return camera_motion.to_frame(t, pt_ref.reshape(1, 2))[0]
        if hasattr(camera_motion, "Ms") and t < len(camera_motion.Ms):
            M = camera_motion.Ms[t]
            try:
                M_inv = np.linalg.inv(M)
                q = M_inv @ np.array([pt_ref[0], pt_ref[1], 1.0], dtype=np.float64)
                if abs(q[2]) > 1e-8:
                    return q[:2] / q[2]
            except np.linalg.LinAlgError:
                pass
        return pt_ref.copy()

    def _is_near_player(
        self,
        player_tracks: Optional[List[Dict[int, Dict[str, Any]]]],
        t: int,
        ball_center: Optional[np.ndarray],
    ) -> bool:
        """Check if ball is within foot_proximity_thresh of any player's feet."""
        if player_tracks is None or ball_center is None or t >= len(player_tracks):
            return False

        players = player_tracks[t]
        for _, p_info in players.items():
            bbox = p_info.get("bbox")
            if bbox is None:
                continue
            # Foot coordinates: bottom corners and bottom center
            foot_center = np.array([(bbox[0] + bbox[2]) / 2.0, bbox[3]])
            foot_left = np.array([bbox[0], bbox[3]])
            foot_right = np.array([bbox[2], bbox[3]])

            d_center = np.linalg.norm(ball_center - foot_center)
            d_left = np.linalg.norm(ball_center - foot_left)
            d_right = np.linalg.norm(ball_center - foot_right)

            if min(d_center, d_left, d_right) <= self.foot_proximity_thresh:
                return True

        return False

    def _get_velocity_at(
        self,
        positions: List[Optional[np.ndarray]],
        t: int,
        direction: str = "in",
    ) -> Optional[np.ndarray]:
        """Estimate velocity vector entering (in) or exiting (out) frame t."""
        if direction == "in":
            # Backward velocity
            for prev_t in range(t - 1, max(-1, t - 4), -1):
                if positions[prev_t] is not None:
                    return (positions[t] - positions[prev_t]) / float(t - prev_t)
        else:
            # Forward velocity
            for next_t in range(t + 1, min(len(positions), t + 4)):
                if positions[next_t] is not None:
                    return (positions[next_t] - positions[t]) / float(next_t - t)
        return None

    def _interpolate_segment(
        self,
        positions: List[Optional[np.ndarray]],
        is_interp_mask: List[bool],
        track_states: List[str],
        t_a: int,
        t_b: int,
        confirmed_indices: List[int],
    ):
        """Monotonic PCHIP interpolation across frames between t_a and t_b."""
        # Collect local context points around t_a and t_b for smooth polynomial matching
        context_t = []
        context_pts = []

        # Find up to 3 points before t_a
        before = [t for t in confirmed_indices if t <= t_a][-3:]
        # Find up to 3 points after t_b
        after = [t for t in confirmed_indices if t >= t_b][:3]

        all_ctx_t = sorted(list(set(before + after)))
        for ct in all_ctx_t:
            context_t.append(ct)
            context_pts.append(positions[ct])

        context_pts = np.array(context_pts)

        if len(context_t) >= 3:
            # Fit PCHIP on x and y coordinates independently
            pchip_x = PchipInterpolator(context_t, context_pts[:, 0])
            pchip_y = PchipInterpolator(context_t, context_pts[:, 1])

            for fill_t in range(t_a + 1, t_b):
                ix = float(pchip_x(fill_t))
                iy = float(pchip_y(fill_t))
                positions[fill_t] = np.array([ix, iy], dtype=np.float64)
                is_interp_mask[fill_t] = True
                track_states[fill_t] = "INTERPOLATED"
        else:
            # Linear fallback if strictly 2 points available
            p_a = positions[t_a]
            p_b = positions[t_b]
            for fill_t in range(t_a + 1, t_b):
                alpha = float(fill_t - t_a) / float(t_b - t_a)
                positions[fill_t] = (1.0 - alpha) * p_a + alpha * p_b
                is_interp_mask[fill_t] = True
                track_states[fill_t] = "INTERPOLATED"
