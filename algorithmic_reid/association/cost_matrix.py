"""
algorithmic_reid.association.cost_matrix - Fused Multi-Cue Cost Matrix.

Fuses:
  1. Distance-IoU (DIoU) cost
  2. Mahalanobis motion innovation cost
  3. Appearance Bhattacharyya distance cost
  4. Heading/direction cosine consistency cost
"""

import numpy as np
from typing import List, Optional, Any
from ..appearance.color_descriptor import batch_bhattacharyya_distance
from ..motion.mahalanobis import compute_mahalanobis_distance_matrix
from ..motion.kalman_filter import DampedKalmanFilter, z_to_bbox


def compute_diou_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Distance-IoU (DIoU) matrix between bboxes1 (M, 4) and bboxes2 (N, 4).
    DIoU = IoU - (rho^2 / c^2)
    Returns: matrix of shape (M, N) with values in [-1.0, 1.0].
    """
    M = len(bboxes1)
    N = len(bboxes2)
    if M == 0 or N == 0:
        return np.empty((M, N), dtype=np.float64)

    # Intersection box
    x1 = np.maximum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :])
    y1 = np.maximum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :])
    x2 = np.minimum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    y2 = np.minimum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    areas1 = np.maximum(1e-4, (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1]))
    areas2 = np.maximum(1e-4, (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1]))
    union = areas1[:, None] + areas2[None, :] - inter
    iou = np.where(union > 0, inter / union, 0.0)

    # Center points
    c1_x = (bboxes1[:, 0] + bboxes1[:, 2]) / 2.0
    c1_y = (bboxes1[:, 1] + bboxes1[:, 3]) / 2.0
    c2_x = (bboxes2[:, 0] + bboxes2[:, 2]) / 2.0
    c2_y = (bboxes2[:, 1] + bboxes2[:, 3]) / 2.0

    rho2 = (c1_x[:, None] - c2_x[None, :]) ** 2 + (c1_y[:, None] - c2_y[None, :]) ** 2

    # Smallest enclosing box
    enc_x1 = np.minimum(bboxes1[:, 0][:, None], bboxes2[:, 0][None, :])
    enc_y1 = np.minimum(bboxes1[:, 1][:, None], bboxes2[:, 1][None, :])
    enc_x2 = np.maximum(bboxes1[:, 2][:, None], bboxes2[:, 2][None, :])
    enc_y2 = np.maximum(bboxes1[:, 3][:, None], bboxes2[:, 3][None, :])

    c2 = np.maximum(1e-4, (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2)

    diou = iou - (rho2 / c2)
    return diou


def compute_fused_cost_matrix(
    tracks: List[Any],
    det_boxes: np.ndarray,
    det_features: List[Optional[np.ndarray]],
    det_measurements: np.ndarray,
    kf: DampedKalmanFilter,
    w_diou: float = 0.40,
    w_mot: float = 0.20,
    w_app: float = 0.30,
    w_dir: float = 0.10,
    use_appearance: bool = True,
    chi2_gate: float = 13.28,
    det_inter_ious: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute normalized fused cost matrix between tracks and detections.
    Cost components:
      - C_diou = (1.0 - diou) / 2.0 in [0, 1]
      - C_mot  = min(1.0, d_M / sqrt(chi2_gate)) in [0, 1]
      - C_app  = D_Bhatt in [0, 1]
      - C_dir  = (1.0 - cos_theta) / 2.0 in [0, 1]
    """
    M = len(tracks)
    N = len(det_boxes)
    if M == 0 or N == 0:
        return np.empty((M, N), dtype=np.float64)

    # 1. DIoU Cost
    track_boxes = np.array([z_to_bbox(t.mean[:4]) for t in tracks], dtype=np.float64)
    diou_mat = compute_diou_matrix(track_boxes, det_boxes)
    c_diou = (1.0 - diou_mat) / 2.0
    c_diou = np.clip(c_diou, 0.0, 1.0)

    # 2. Mahalanobis Motion Cost
    d_m_mat = compute_mahalanobis_distance_matrix(kf, tracks, det_measurements)
    c_mot = np.clip(d_m_mat / np.sqrt(chi2_gate), 0.0, 1.0)

    # 3. Appearance Cost
    if use_appearance:
        track_templates = [t.appearance_template for t in tracks]
        c_app = batch_bhattacharyya_distance(track_templates, det_features)
        if det_inter_ious is not None and np.any(det_inter_ious > 0.15):
            occ_mask = det_inter_ious > 0.15
            c_app[:, occ_mask] = np.minimum(c_app[:, occ_mask], 0.60)
    else:
        c_app = np.zeros((M, N), dtype=np.float64)
        w_app = 0.0

    # 4. Heading / Motion Direction Cost
    c_dir = np.zeros((M, N), dtype=np.float64)
    for i, t in enumerate(tracks):
        vx, vy = float(t.mean[4]), float(t.mean[5])
        v_speed = np.hypot(vx, vy)
        if v_speed < 2.0:
            continue
        c_x, c_y = float(t.mean[0]), float(t.mean[1])
        cand_x = (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0
        cand_y = (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0
        dx = cand_x - c_x
        dy = cand_y - c_y
        dist = np.hypot(dx, dy)
        valid_dist = dist > 2.0
        if np.any(valid_dist):
            cos_theta = (vx * dx[valid_dist] + vy * dy[valid_dist]) / (v_speed * dist[valid_dist])
            c_dir[i, valid_dist] = (1.0 - np.clip(cos_theta, -1.0, 1.0)) / 2.0

    # Normalize active weights
    tot_weight = w_diou + w_mot + w_app + w_dir
    if tot_weight < 1e-4:
        tot_weight = 1.0

    fused_cost = (w_diou * c_diou + w_mot * c_mot + w_app * c_app + w_dir * c_dir) / tot_weight
    return fused_cost
