"""
algorithmic_reid.lifecycle.track_manager - Multi-Stage Cascade Association and Track Lifecycle Manager.

Executes 4-Stage Matching Cascade:
  Stage 1: Active Confirmed Tracks <-> High-Conf Detections (Multi-Cue Fused Cost)
  Stage 2: Remaining Active Confirmed Tracks <-> Low-Conf Detections (DIoU + Motion Recovery)
  Stage 3: Lost Confirmed Tracks <-> Remaining Detections (Re-ID Reactivation via Appearance + DIoU)
  Stage 4: Tentative Tracks <-> Remaining Detections (Confirmation Gateway)
  Spawning: Unmatched Detections with conf >= init_conf_thresh spawn Tentative Tracks.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from .tracklet import Tracklet, TrackletState
from .stitcher import TrackletStitcher
from ..config import ReIDConfig
from ..motion.kalman_filter import DampedKalmanFilter, bbox_to_z, z_to_bbox
from ..appearance.temporal_aggregator import AppearanceAggregator
from ..appearance.color_descriptor import extract_multizone_histogram
from ..association.cost_matrix import compute_fused_cost_matrix, compute_diou_matrix
from ..association.gating import apply_gating_barriers
from ..association.hungarian import min_cost_matching


class TrackManager:
    def __init__(self, config: Optional[ReIDConfig] = None):
        self.config = config or ReIDConfig()

        self.kf = DampedKalmanFilter(
            std_weight_position=self.config.std_weight_position,
            std_weight_velocity=self.config.std_weight_velocity,
            velocity_damping_gamma=self.config.velocity_damping_gamma,
            max_velocity_px=self.config.max_velocity_px,
        )

        self.aggregator = AppearanceAggregator(
            alpha=self.config.ema_alpha,
            min_conf=0.50,
            max_overlap_iou=self.config.overlap_occlusion_iou,
        )

        self.stitcher = TrackletStitcher(
            max_gap_frames=self.config.reactivation_window,
            max_velocity_px=self.config.max_velocity_px,
            appearance_thresh=0.40,
        )

        # Direction B: Deep Re-ID Extractor
        self.deep_extractor = None
        if getattr(self.config, "use_deep_reid", False):
            from ..appearance.deep_reid import DeepReIDExtractor
            self.deep_extractor = DeepReIDExtractor()

        # Direction C: Global Tracklet Associator
        self.gta = None
        if getattr(self.config, "use_gta", False):
            from .gta_postprocess import GlobalTrackletAssociator
            self.gta = GlobalTrackletAssociator()

        self.tracks: List[Tracklet] = []
        self.next_id: int = 0
        self.frame_count: int = 0

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0

    def _allocate_id(self) -> int:
        tid = self.next_id
        self.next_id += 1
        return tid

    def _stitch_candidates(self, matched_track_ids: set) -> None:
        """
        Evaluate and stitch newly confirmed active tracks into recently lost confirmed tracks.
        """
        lost_candidates = [
            t for t in self.tracks
            if t.state in (TrackletState.LOST, TrackletState.CONFIRMED)
            and t.time_since_update > 0
            and len(t.history) >= self.config.n_init
        ]
        if not lost_candidates:
            return

        active_candidates = [
            t for t in self.tracks
            if t.state == TrackletState.CONFIRMED
            and t.time_since_update == 0
            and t.start_frame > 0
        ]
        if not active_candidates:
            return

        for new_track in active_candidates:
            best_target = None
            best_cost = float("inf")

            for lost_track in lost_candidates:
                if self.stitcher.can_stitch(lost_track, new_track):
                    p_prev = ((lost_track.history[-1][1][0] + lost_track.history[-1][1][2]) / 2.0,
                              (lost_track.history[-1][1][1] + lost_track.history[-1][1][3]) / 2.0)
                    p_next = ((new_track.history[0][1][0] + new_track.history[0][1][2]) / 2.0,
                              (new_track.history[0][1][1] + new_track.history[0][1][3]) / 2.0)
                    dist = np.hypot(p_next[0] - p_prev[0], p_next[1] - p_prev[1])
                    gap = max(1, new_track.start_frame - lost_track.last_frame)
                    vel = dist / float(gap)

                    app_cost = 0.0
                    if lost_track.appearance_template is not None and new_track.appearance_template is not None:
                        from ..appearance.color_descriptor import bhattacharyya_distance
                        app_cost = bhattacharyya_distance(lost_track.appearance_template, new_track.appearance_template)

                    stitch_cost = app_cost + 0.01 * vel
                    if stitch_cost < best_cost:
                        best_cost = stitch_cost
                        best_target = lost_track

            if best_target is not None:
                self.stitcher.stitch_tracklets(best_target, new_track)
                lost_candidates.remove(best_target)
                matched_track_ids.discard(new_track.track_id)
                matched_track_ids.add(best_target.track_id)

    def step(
        self,
        det_boxes: np.ndarray,
        det_confs: np.ndarray,
        det_classes: np.ndarray,
        frame: Optional[np.ndarray] = None,
        camera_homography: Optional[np.ndarray] = None,
        det_teams: Optional[np.ndarray] = None,
        pitch_homography: Optional[np.ndarray] = None,
    ) -> List[Tracklet]:
        """
        Process single frame detections through multi-stage association cascade.
        Returns list of active confirmed tracklets (plus coasting tracklets) for current frame.
        """
        self.frame_count += 1
        t_now = self.frame_count

        # 1. Predict all existing tracks
        for track in self.tracks:
            track.predict(camera_homography)

        num_dets = len(det_boxes)
        if num_dets == 0:
            for track in self.tracks:
                track.mark_missed()
            self.tracks = [t for t in self.tracks if t.state != TrackletState.DELETED]
            max_coast = getattr(self.config, "max_coast_frames", 15)
            return [t for t in self.tracks if t.state == TrackletState.CONFIRMED and t.time_since_update <= max_coast]

        # Extract appearance features for detections
        det_features: List[Optional[np.ndarray]] = []
        for i in range(num_dets):
            if frame is not None:
                feat = extract_multizone_histogram(
                    frame,
                    det_boxes[i],
                    bins_h=self.config.hist_bins_h,
                    bins_s=self.config.hist_bins_s,
                    bins_v=self.config.hist_bins_v,
                )
            else:
                feat = None
            det_features.append(feat)

        # Direction B: Deep semantic embeddings
        det_deep_features = None
        if self.deep_extractor is not None and frame is not None and num_dets > 0:
            det_deep_features = self.deep_extractor.extract_crops(frame, det_boxes)

        det_measurements = np.array([bbox_to_z(b) for b in det_boxes], dtype=np.float64)

        # Precompute inter-detection overlaps to protect appearance update
        inter_diou = compute_diou_matrix(det_boxes, det_boxes)
        np.fill_diagonal(inter_diou, 0.0)
        max_inter_ious = np.max(inter_diou, axis=1) if num_dets > 1 else np.zeros(num_dets)

        # Partition detections
        high_mask = det_confs >= self.config.high_conf_thresh
        low_mask = (det_confs >= self.config.low_conf_thresh) & (~high_mask)

        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

        # Partition tracks
        # Stage 1 includes active confirmed tracks AND recently lost tracks (<= co_association_window)
        # to allow simultaneous co-association of crossing players and eliminate crossing ID switches
        co_assoc_win = getattr(self.config, "co_association_window", 15)
        stage1_tracks = [
            t for t in self.tracks
            if (t.state == TrackletState.CONFIRMED and t.time_since_update == 0)
            or (t.state in (TrackletState.CONFIRMED, TrackletState.LOST)
                and 1 <= t.time_since_update <= co_assoc_win
                and (getattr(t, "in_occlusion", False) or t.hits >= self.config.n_init))
        ]

        confirmed_lost = [
            t for t in self.tracks
            if t.state in (TrackletState.CONFIRMED, TrackletState.LOST)
            and 1 <= t.time_since_update <= self.config.reactivation_window
            and t not in stage1_tracks
        ]
        tentative_tracks = [t for t in self.tracks if t.state == TrackletState.TENTATIVE]

        matched_det_indices = set()
        matched_track_ids = set()

        # =====================================================================
        # STAGE 1: Stage 1 Tracks (Active + Recently Lost Co-Association) <-> High-Conf Detections
        # =====================================================================
        if len(stage1_tracks) > 0 and len(high_indices) > 0:
            sub_boxes = det_boxes[high_indices]
            sub_feats = [det_features[k] for k in high_indices]
            sub_meas = det_measurements[high_indices]
            sub_classes = det_classes[high_indices]
            sub_teams = det_teams[high_indices] if det_teams is not None else None
            sub_deep_feats = [det_deep_features[k] for k in high_indices] if det_deep_features is not None else None

            # Direction A: Pitch metric kinematics
            pitch_cost_mat = None
            pitch_viols = None
            if getattr(self.config, "use_pitch_kinematics", False) and pitch_homography is not None:
                from ..motion.pitch_kinematics import compute_pitch_cost_and_gating
                pitch_cost_mat, pitch_viols = compute_pitch_cost_and_gating(
                    stage1_tracks, sub_boxes, pitch_homography, fps=self.config.fps
                )

            sub_inter_ious = max_inter_ious[high_indices]
            cost_mat = compute_fused_cost_matrix(
                stage1_tracks,
                sub_boxes,
                sub_feats,
                sub_meas,
                self.kf,
                w_diou=self.config.w_diou,
                w_mot=self.config.w_mot,
                w_app=self.config.w_app if frame is not None else 0.0,
                w_dir=self.config.w_dir,
                use_appearance=(frame is not None),
                chi2_gate=self.config.mahalanobis_gate,
                det_inter_ious=sub_inter_ious,
                det_deep_features=sub_deep_feats,
                w_deep=self.config.w_deep if getattr(self.config, "use_deep_reid", False) else 0.0,
                pitch_cost_matrix=pitch_cost_mat,
                w_pitch=self.config.w_pitch if getattr(self.config, "use_pitch_kinematics", False) else 0.0,
            )

            gated_cost = apply_gating_barriers(
                cost_mat,
                stage1_tracks,
                sub_boxes,
                sub_classes,
                sub_feats,
                sub_meas,
                self.kf,
                chi2_gate=self.config.mahalanobis_gate,
                appearance_gate=self.config.appearance_gate,
                max_velocity_px=self.config.max_velocity_px,
                check_appearance=(frame is not None),
                det_inter_ious=sub_inter_ious,
                det_teams=sub_teams,
                pitch_violations=pitch_viols,
            )

            matched_pairs, unmatched_t_idx, unmatched_d_idx = min_cost_matching(
                gated_cost, max_cost=self.config.max_cost_stage1
            )

            for t_idx, d_sub_idx in matched_pairs:
                t = stage1_tracks[t_idx]
                real_d_idx = high_indices[d_sub_idx]
                d_team = int(det_teams[real_d_idx]) if det_teams is not None else 0
                d_deep = det_deep_features[real_d_idx] if det_deep_features is not None else None
                t.update(
                    det_boxes[real_d_idx],
                    det_confs[real_d_idx],
                    det_classes[real_d_idx],
                    feature=det_features[real_d_idx],
                    max_inter_iou=float(max_inter_ious[real_d_idx]),
                    frame_idx=t_now,
                    assigned_team=d_team,
                    deep_feature=d_deep,
                )
                matched_track_ids.add(t.track_id)
                matched_det_indices.add(real_d_idx)

        # =====================================================================
        # STAGE 2: Remaining Active Confirmed Tracks <-> Low-Confidence Detections
        # =====================================================================
        remaining_active = [t for t in stage1_tracks if t.track_id not in matched_track_ids and t.time_since_update == 0]
        if len(remaining_active) > 0 and len(low_indices) > 0:
            sub_boxes = det_boxes[low_indices]
            sub_feats = [det_features[k] for k in low_indices]
            sub_meas = det_measurements[low_indices]
            sub_classes = det_classes[low_indices]

            # In low-conf, rely on DIoU + motion without appearance penalty
            cost_mat = compute_fused_cost_matrix(
                remaining_active,
                sub_boxes,
                sub_feats,
                sub_meas,
                self.kf,
                w_diou=0.70,
                w_mot=0.30,
                w_app=0.0,
                w_dir=0.0,
                use_appearance=False,
                chi2_gate=self.config.mahalanobis_gate,
            )

            gated_cost = apply_gating_barriers(
                cost_mat,
                remaining_active,
                sub_boxes,
                sub_classes,
                sub_feats,
                sub_meas,
                self.kf,
                chi2_gate=self.config.mahalanobis_gate,
                appearance_gate=self.config.appearance_gate,
                max_velocity_px=self.config.max_velocity_px,
                check_appearance=False,
            )

            matched_pairs, _, _ = min_cost_matching(
                gated_cost, max_cost=self.config.max_cost_stage2
            )

            for t_idx, d_sub_idx in matched_pairs:
                t = remaining_active[t_idx]
                real_d_idx = low_indices[d_sub_idx]
                t.update(
                    det_boxes[real_d_idx],
                    det_confs[real_d_idx],
                    det_classes[real_d_idx],
                    feature=det_features[real_d_idx],
                    max_inter_iou=float(max_inter_ious[real_d_idx]),
                    frame_idx=t_now,
                )
                matched_track_ids.add(t.track_id)
                matched_det_indices.add(real_d_idx)

        # =====================================================================
        # STAGE 3: Lost Confirmed Tracks (Re-ID Reactivation) <-> Remaining High-Conf
        # =====================================================================
        remaining_high = [k for k in high_indices if k not in matched_det_indices]
        if len(confirmed_lost) > 0 and len(remaining_high) > 0:
            sub_boxes = det_boxes[remaining_high]
            sub_feats = [det_features[k] for k in remaining_high]
            sub_meas = det_measurements[remaining_high]
            sub_classes = det_classes[remaining_high]
            sub_teams = det_teams[remaining_high] if det_teams is not None else None

            # In Stage 3 Re-ID: appearance + DIoU are key
            sub_inter_ious = max_inter_ious[remaining_high]
            cost_mat = compute_fused_cost_matrix(
                confirmed_lost,
                sub_boxes,
                sub_feats,
                sub_meas,
                self.kf,
                w_diou=0.45,
                w_mot=0.15,
                w_app=0.40 if frame is not None else 0.0,
                w_dir=0.0,
                use_appearance=(frame is not None),
                chi2_gate=self.config.mahalanobis_gate,
                det_inter_ious=sub_inter_ious,
            )

            gated_cost = apply_gating_barriers(
                cost_mat,
                confirmed_lost,
                sub_boxes,
                sub_classes,
                sub_feats,
                sub_meas,
                self.kf,
                chi2_gate=self.config.mahalanobis_gate,
                appearance_gate=self.config.appearance_gate,
                max_velocity_px=self.config.max_velocity_px,
                check_appearance=(frame is not None),
                det_inter_ious=sub_inter_ious,
                det_teams=sub_teams,
            )

            matched_pairs, _, _ = min_cost_matching(
                gated_cost, max_cost=self.config.max_cost_stage3
            )

            for t_idx, d_sub_idx in matched_pairs:
                t = confirmed_lost[t_idx]
                real_d_idx = remaining_high[d_sub_idx]
                d_team = int(det_teams[real_d_idx]) if det_teams is not None else 0
                t.update(
                    det_boxes[real_d_idx],
                    det_confs[real_d_idx],
                    det_classes[real_d_idx],
                    feature=det_features[real_d_idx],
                    max_inter_iou=float(max_inter_ious[real_d_idx]),
                    frame_idx=t_now,
                    assigned_team=d_team,
                )
                matched_track_ids.add(t.track_id)
                matched_det_indices.add(real_d_idx)

        # =====================================================================
        # STAGE 4: Tentative Tracks <-> Remaining Detections
        # =====================================================================
        unmatched_all_dets = [k for k in range(num_dets) if k not in matched_det_indices]
        if len(tentative_tracks) > 0 and len(unmatched_all_dets) > 0:
            sub_boxes = det_boxes[unmatched_all_dets]
            sub_feats = [det_features[k] for k in unmatched_all_dets]
            sub_meas = det_measurements[unmatched_all_dets]
            sub_classes = det_classes[unmatched_all_dets]

            cost_mat = compute_fused_cost_matrix(
                tentative_tracks,
                sub_boxes,
                sub_feats,
                sub_meas,
                self.kf,
                w_diou=0.70,
                w_mot=0.30,
                w_app=0.0,
                w_dir=0.0,
                use_appearance=False,
                chi2_gate=self.config.mahalanobis_gate,
            )

            gated_cost = apply_gating_barriers(
                cost_mat,
                tentative_tracks,
                sub_boxes,
                sub_classes,
                sub_feats,
                sub_meas,
                self.kf,
                chi2_gate=self.config.mahalanobis_gate,
                appearance_gate=self.config.appearance_gate,
                max_velocity_px=self.config.max_velocity_px,
                check_appearance=False,
            )

            matched_pairs, _, _ = min_cost_matching(
                gated_cost, max_cost=self.config.max_cost_tentative
            )

            for t_idx, d_sub_idx in matched_pairs:
                t = tentative_tracks[t_idx]
                real_d_idx = unmatched_all_dets[d_sub_idx]
                t.update(
                    det_boxes[real_d_idx],
                    det_confs[real_d_idx],
                    det_classes[real_d_idx],
                    feature=det_features[real_d_idx],
                    max_inter_iou=float(max_inter_ious[real_d_idx]),
                    frame_idx=t_now,
                )
                matched_track_ids.add(t.track_id)
                matched_det_indices.add(real_d_idx)

        # =====================================================================
        # SPAWN NEW TENTATIVE TRACKS
        # =====================================================================
        # Protect against spawning duplicate/rogue tracklets (e.g. ID 63 during collisions)
        # by checking overlap with all confirmed tracks (active and coasting up to 15 frames)
        max_coast = getattr(self.config, "max_coast_frames", 15)
        active_boxes = [
            t.bbox for t in self.tracks
            if t.bbox is not None and (
                t.time_since_update == 0 or
                (t.state == TrackletState.CONFIRMED and t.time_since_update <= max_coast)
            )
        ]
        active_boxes_arr = np.array(active_boxes, dtype=np.float64) if len(active_boxes) > 0 else np.empty((0, 4), dtype=np.float64)

        for k in range(num_dets):
            if k not in matched_det_indices:
                if det_confs[k] >= self.config.init_conf_thresh:
                    # Spawn Guard: Suppress duplicate track spawning if overlapping with an existing track
                    if len(active_boxes_arr) > 0:
                        b_k = det_boxes[k]
                        x1 = np.maximum(b_k[0], active_boxes_arr[:, 0])
                        y1 = np.maximum(b_k[1], active_boxes_arr[:, 1])
                        x2 = np.minimum(b_k[2], active_boxes_arr[:, 2])
                        y2 = np.minimum(b_k[3], active_boxes_arr[:, 3])
                        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
                        area_k = max(1e-4, (b_k[2] - b_k[0]) * (b_k[3] - b_k[1]))
                        area_act = np.maximum(1e-4, (active_boxes_arr[:, 2] - active_boxes_arr[:, 0]) * (active_boxes_arr[:, 3] - active_boxes_arr[:, 1]))
                        union = area_k + area_act - inter
                        ious = np.where(union > 0, inter / union, 0.0)

                        cx_k = (b_k[0] + b_k[2]) / 2.0
                        cy_k = (b_k[1] + b_k[3]) / 2.0
                        cx_act = (active_boxes_arr[:, 0] + active_boxes_arr[:, 2]) / 2.0
                        cy_act = (active_boxes_arr[:, 1] + active_boxes_arr[:, 3]) / 2.0
                        center_dists = np.hypot(cx_k - cx_act, cy_k - cy_act)

                        spawn_iou_cap = getattr(self.config, "max_spawn_overlap_iou", 0.35)
                        if np.max(ious) > spawn_iou_cap or (len(center_dists) > 0 and np.min(center_dists) < 40.0):
                            continue

                    new_tid = self._allocate_id()
                    new_track = Tracklet(
                        track_id=new_tid,
                        bbox=det_boxes[k],
                        confidence=det_confs[k],
                        class_id=det_classes[k],
                        feature=det_features[k],
                        kf=self.kf,
                        aggregator=self.aggregator,
                        n_init=self.config.n_init,
                        max_age=self.config.max_age,
                        start_frame=t_now,
                    )
                    self.tracks.append(new_track)
                    # Dynamically append to active_boxes to protect against same-frame duplicate detections
                    active_boxes.append(det_boxes[k])
                    active_boxes_arr = np.array(active_boxes, dtype=np.float64)

        # =====================================================================
        # TRACKLET STITCHING
        # =====================================================================
        self._stitch_candidates(matched_track_ids)

        # =====================================================================
        # MARK MISSED & CLEANUP DELETED
        # =====================================================================
        for track in self.tracks:
            if track.track_id not in matched_track_ids and track.time_since_update > 0 or (track.last_frame != t_now):
                track.mark_missed()

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if t.state != TrackletState.DELETED]

        # Return confirmed tracks that were active on this frame OR coasting (time_since_update <= max_coast_frames)
        # to ensure rock-solid continuous tracking without flickering / blinking
        max_coast = getattr(self.config, "max_coast_frames", 15)
        confirmed_output = [
            t for t in self.tracks
            if t.state == TrackletState.CONFIRMED and t.time_since_update <= max_coast
        ]
        return confirmed_output
