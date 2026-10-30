"""
algorithmic_reid.appearance.color_descriptor - Multi-Zone HSV Color Representation.

Extracts spatial-color histograms from two distinct zones:
  - Upper Torso (Jersey): y in [0.15h, 0.50h], x in [0.20w, 0.80w]
  - Lower Body (Shorts/Socks): y in [0.55h, 0.85h], x in [0.20w, 0.80w]
Excludes border pixels (pitch grass background) and computes Bhattacharyya distance.
"""

import cv2
import numpy as np
from typing import Optional, List


def extract_zone_hist(
    roi: np.ndarray,
    bins_h: int = 16,
    bins_s: int = 8,
    bins_v: int = 8,
) -> np.ndarray:
    """Extract joint 2D H-S and 1D V histogram from an image patch."""
    if roi is None or roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
        return np.ones(bins_h * bins_s + bins_v, dtype=np.float32) / (bins_h * bins_s + bins_v)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 2D H-S histogram
    hist_hs = cv2.calcHist([hsv], [0, 1], None, [bins_h, bins_s], [0, 180, 0, 256])
    hist_hs = hist_hs.flatten()

    # 1D V histogram
    hist_v = cv2.calcHist([hsv], [2], None, [bins_v], [0, 256])
    hist_v = hist_v.flatten()

    combined = np.concatenate([hist_hs, hist_v]).astype(np.float32)
    s = np.sum(combined)
    if s > 1e-6:
        combined /= s
    else:
        combined = np.ones_like(combined) / len(combined)
    return combined


def extract_multizone_histogram(
    frame: np.ndarray,
    bbox: np.ndarray,
    bins_h: int = 16,
    bins_s: int = 8,
    bins_v: int = 8,
) -> Optional[np.ndarray]:
    """
    Extract multi-zone (upper + lower) color descriptor from person bbox in frame.
    Returns concatenated unit-normalized vector of shape (2 * (bins_h*bins_s + bins_v),).
    """
    if frame is None or frame.size == 0:
        return None

    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, min(int(bbox[0]), w_frame - 1))
    y1 = max(0, min(int(bbox[1]), h_frame - 1))
    x2 = max(0, min(int(bbox[2]), w_frame))
    y2 = max(0, min(int(bbox[3]), h_frame))

    w = x2 - x1
    h = y2 - y1
    if w < 6 or h < 10:
        return None

    # Upper Torso: y in [0.15h, 0.50h], x in [0.20w, 0.80w]
    uy1 = y1 + int(0.15 * h)
    uy2 = y1 + int(0.50 * h)
    ux1 = x1 + int(0.20 * w)
    ux2 = x1 + int(0.80 * w)
    upper_roi = frame[uy1:uy2, ux1:ux2]

    # Lower Body: y in [0.55h, 0.85h], x in [0.20w, 0.80w]
    ly1 = y1 + int(0.55 * h)
    ly2 = y1 + int(0.85 * h)
    lx1 = x1 + int(0.20 * w)
    lx2 = x1 + int(0.80 * w)
    lower_roi = frame[ly1:ly2, lx1:lx2]

    h_upper = extract_zone_hist(upper_roi, bins_h, bins_s, bins_v)
    h_lower = extract_zone_hist(lower_roi, bins_h, bins_s, bins_v)

    multizone = np.concatenate([h_upper, h_lower]).astype(np.float32)
    s = np.sum(multizone)
    if s > 1e-6:
        multizone /= s
    return multizone


def bhattacharyya_distance(f1: Optional[np.ndarray], f2: Optional[np.ndarray]) -> float:
    """
    Compute Bhattacharyya distance between two normalized histograms.
      D_Bhatt = sqrt(1 - sum(sqrt(f1 * f2)))
    Range: [0.0, 1.0]. Returns 0.5 if either feature is None.
    """
    if f1 is None or f2 is None:
        return 0.5

    bc = np.sum(np.sqrt(np.maximum(0.0, f1 * f2)))
    bc = np.clip(bc, 0.0, 1.0)
    return float(np.sqrt(1.0 - bc))


def batch_bhattacharyya_distance(
    track_features: List[Optional[np.ndarray]],
    det_features: List[Optional[np.ndarray]],
) -> np.ndarray:
    """
    Compute pairwise Bhattacharyya distance matrix of shape (M, N).
    """
    M = len(track_features)
    N = len(det_features)
    if M == 0 or N == 0:
        return np.empty((M, N), dtype=np.float64)

    dist_mat = np.full((M, N), 0.5, dtype=np.float64)
    for i, tf in enumerate(track_features):
        if tf is None:
            continue
        for j, df in enumerate(det_features):
            if df is None:
                continue
            dist_mat[i, j] = bhattacharyya_distance(tf, df)
    return dist_mat
