"""
algorithmic_reid.motion package - Motion modeling, Kalman filtering and spatial prediction.
"""

from .kalman_filter import DampedKalmanFilter
from .mahalanobis import compute_mahalanobis_distance_matrix
from .motion_predictor import apply_camera_motion_to_tracks

__all__ = [
    "DampedKalmanFilter",
    "compute_mahalanobis_distance_matrix",
    "apply_camera_motion_to_tracks",
]
