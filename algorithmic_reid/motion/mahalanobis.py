r"""
algorithmic_reid.motion.mahalanobis - Mahalanobis Distance Calculation and Gating.
Computes innovation Mahalanobis distance with chi-squared test:
  d_M^2 = (z - \hat{z})^T S^{-1} (z - \hat{z})
"""

import numpy as np
from typing import List, Tuple, Any
from .kalman_filter import DampedKalmanFilter


def compute_mahalanobis_distance_matrix(
    kf: DampedKalmanFilter,
    tracks: List[Any],
    measurements: np.ndarray,
    only_position: bool = False,
) -> np.ndarray:
    """
    Compute Mahalanobis distance matrix of shape (num_tracks, num_measurements).
    tracks: list of tracklets with (mean, covariance)
    measurements: array of shape (N, 4) in [xc, yc, a, h] format
    """
    if len(tracks) == 0 or len(measurements) == 0:
        return np.empty((len(tracks), len(measurements)), dtype=np.float64)

    dists = np.zeros((len(tracks), len(measurements)), dtype=np.float64)

    for i, track in enumerate(tracks):
        mean, cov = track.mean, track.covariance
        proj_mean, S = kf.project(mean, cov)

        if only_position:
            proj_mean = proj_mean[:2]
            S = S[:2, :2]
            diff = measurements[:, :2] - proj_mean
        else:
            diff = measurements - proj_mean

        try:
            chol = np.linalg.cholesky(S)
            # Solve L * y = diff.T -> y = L^{-1} diff.T
            y = np.linalg.solve(chol, diff.T)
            # d_M^2 = sum(y^2, axis=0)
            d2 = np.sum(np.square(y), axis=0)
            dists[i, :] = np.sqrt(np.maximum(0.0, d2))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            S_inv = np.linalg.pinv(S)
            d2 = np.sum(np.dot(diff, S_inv) * diff, axis=1)
            dists[i, :] = np.sqrt(np.maximum(0.0, d2))

    return dists
