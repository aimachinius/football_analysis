"""
algorithmic_reid.association.gating - Hard Gating Barriers.

Enforces:
  1. Species Isolation Barrier: class(track) != class(det) -> cost = 1e5
  2. Appearance Barrier: D_Bhatt > appearance_gate -> cost = 1e5
  3. Spatial Motion Barrier: d_M^2 > chi2_gate and Euclidean distance > max_dist -> cost = 1e5
"""

import numpy as np
from typing import List, Optional, Any
from ..appearance.color_descriptor import batch_bhattacharyya_distance
from ..motion.mahalanobis import compute_mahalanobis_distance_matrix
from ..motion.kalman_filter import DampedKalmanFilter, z_to_bbox
from .cost_matrix import compute_diou_matrix


def apply_gating_barriers(
    cost_matrix: np.ndarray,
    tracks: List[Any],
    det_boxes: np.ndarray,
    det_classes: np.ndarray,
    det_features: List[Optional[np.ndarray]],
    det_measurements: np.ndarray,
    kf: DampedKalmanFilter,
    chi2_gate: float = 13.28,
    appearance_gate: float = 0.55,
    max_velocity_px: float = 35.0,
    check_appearance: bool = True,
    inf_cost: float = 1e5,
    det_inter_ious: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Applies hard gating barriers in-place (or returning copy) to cost_matrix.
    Any violation sets cost to inf_cost (1e5).
    """
    M = len(tracks)
    N = len(det_boxes)
    if M == 0 or N == 0:
        return cost_matrix

    gated_cost = cost_matrix.copy()

    # 1. Species Isolation Barrier (Player != Referee != Goalkeeper)
    for i, t in enumerate(tracks):
        t_cls = t.class_id
        mismatch_mask = (det_classes != t_cls)
        if np.any(mismatch_mask):
            gated_cost[i, mismatch_mask] = inf_cost

    # 2. Appearance Barrier with Occlusion Awareness
    if check_appearance:
        track_templates = [t.appearance_template for t in tracks]
        app_dist = batch_bhattacharyya_distance(track_templates, det_features)

        # Precompute inter-detection overlap if not provided
        if det_inter_ious is None:
            if N > 1:
                x1 = np.maximum(det_boxes[:, 0][:, None], det_boxes[:, 0][None, :])
                y1 = np.maximum(det_boxes[:, 1][:, None], det_boxes[:, 1][None, :])
                x2 = np.minimum(det_boxes[:, 2][:, None], det_boxes[:, 2][None, :])
                y2 = np.minimum(det_boxes[:, 3][:, None], det_boxes[:, 3][None, :])
                inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
                areas = np.maximum(1e-4, (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1]))
                union = areas[:, None] + areas[None, :] - inter
                iou_mat = np.where(union > 0, inter / union, 0.0)
                np.fill_diagonal(iou_mat, 0.0)
                inter_ious = np.max(iou_mat, axis=1)
            else:
                inter_ious = np.zeros(N, dtype=np.float64)
        else:
            inter_ious = det_inter_ious

        # Spatial pairwise DIoU
        track_boxes = np.array([z_to_bbox(t.mean[:4]) for t in tracks], dtype=np.float64)
        diou_mat = compute_diou_matrix(track_boxes, det_boxes)

        # Track-level occlusion overlap
        if M > 1:
            tr_x1 = np.maximum(track_boxes[:, 0][:, None], track_boxes[:, 0][None, :])
            tr_y1 = np.maximum(track_boxes[:, 1][:, None], track_boxes[:, 1][None, :])
            tr_x2 = np.minimum(track_boxes[:, 2][:, None], track_boxes[:, 2][None, :])
            tr_y2 = np.minimum(track_boxes[:, 3][:, None], track_boxes[:, 3][None, :])
            tr_inter = np.maximum(0.0, tr_x2 - tr_x1) * np.maximum(0.0, tr_y2 - tr_y1)
            tr_areas = np.maximum(1e-4, (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1]))
            tr_union = tr_areas[:, None] + tr_areas[None, :] - tr_inter
            tr_iou_mat = np.where(tr_union > 0, tr_inter / tr_union, 0.0)
            np.fill_diagonal(tr_iou_mat, 0.0)
            is_occluded_track = np.max(tr_iou_mat, axis=1) > 0.15
        else:
            is_occluded_track = np.zeros(M, dtype=bool)

        is_occluded_det = (inter_ious > 0.15)
        in_occlusion = is_occluded_det[None, :] | is_occluded_track[:, None]

        # Relax appearance gate to 0.70 during occlusion
        effective_gate = np.where(in_occlusion, max(appearance_gate, 0.70), appearance_gate)

        # Heading alignment check
        heading_align = np.ones((M, N), dtype=bool)
        for i, t in enumerate(tracks):
            vx, vy = float(t.mean[4]), float(t.mean[5])
            v_speed = np.hypot(vx, vy)
            if v_speed >= 2.0:
                c_x, c_y = float(t.mean[0]), float(t.mean[1])
                cand_x = (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0
                cand_y = (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0
                dx = cand_x - c_x
                dy = cand_y - c_y
                dist = np.hypot(dx, dy)
                non_zero = dist > 2.0
                if np.any(non_zero):
                    cos_theta = np.ones(N, dtype=np.float64)
                    cos_theta[non_zero] = (vx * dx[non_zero] + vy * dy[non_zero]) / (v_speed * dist[non_zero])
                    heading_align[i, :] = (cos_theta >= -0.2)

        # Bypass appearance gating if in occlusion context and spatial DIoU >= 0.10 and heading aligns
        bypass_app = in_occlusion & (diou_mat >= 0.10) & heading_align

        app_violation = (app_dist > effective_gate) & ~bypass_app
        gated_cost[app_violation] = inf_cost

    # 3. Spatial / Mahalanobis Barrier
    d_m_mat = compute_mahalanobis_distance_matrix(kf, tracks, det_measurements)
    d_m_sq = np.square(d_m_mat)

    for i, t in enumerate(tracks):
        time_lost = max(1, t.time_since_update)
        max_dist = max(2.0 * max_velocity_px * time_lost, 75.0)

        # Euclidean center distance
        c_x, c_y = float(t.mean[0]), float(t.mean[1])
        cand_x = (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0
        cand_y = (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0
        dist = np.hypot(cand_x - c_x, cand_y - c_y)

        # Gate if Mahalanobis exceeds chi2 AND spatial distance exceeds max_dist
        spatial_violation = (d_m_sq[i, :] > chi2_gate) & (dist > max_dist)
        gated_cost[i, spatial_violation] = inf_cost

    return gated_cost
