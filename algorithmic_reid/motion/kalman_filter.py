r"""
algorithmic_reid.motion.kalman_filter - Kalman Filter with Damped Aspect Ratio and Velocity Decay.

State vector:
  x = [xc, yc, a, h, vx, vy, 0, vh]^T
where:
  (xc, yc) is center position
  a = w / h (aspect ratio, with fixed rate \dot{a} = 0)
  h is box height
  (vx, vy, vh) are respective velocities

During unobserved frames (dropouts), velocity components decay exponentially:
  v_{pred} = v * gamma (default gamma=0.90)
preventing tracks from flying off the pitch.
"""

import numpy as np
from typing import Tuple


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to measurement [xc, yc, a, h]."""
    w = max(1.0, float(bbox[2] - bbox[0]))
    h = max(1.0, float(bbox[3] - bbox[1]))
    xc = float(bbox[0]) + w / 2.0
    yc = float(bbox[1]) + h / 2.0
    a = w / h
    return np.array([xc, yc, a, h], dtype=np.float64)


def z_to_bbox(z: np.ndarray) -> np.ndarray:
    """Convert [xc, yc, a, h] to bbox [x1, y1, x2, y2]."""
    xc, yc, a, h = z[:4]
    w = a * h
    return np.array([xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0], dtype=np.float64)


class DampedKalmanFilter:
    def __init__(
        self,
        std_weight_position: float = 1.0 / 20.0,
        std_weight_velocity: float = 1.0 / 160.0,
        velocity_damping_gamma: float = 0.90,
        max_velocity_px: float = 35.0,
    ):
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity
        self.gamma = velocity_damping_gamma
        self.max_velocity_px = max_velocity_px

        # Measurement matrix H (4 x 8)
        self._update_mat = np.eye(4, 8, dtype=np.float64)

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize state vector mean (8,) and covariance matrix (8, 8) from initial bbox measurement.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray, is_observed: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman prediction step.
        If is_observed is False, apply exponential velocity damping (gamma).
        """
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # State transition matrix F with aspect ratio damping (\dot{a} = 0)
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = 1.0  # xc += vx
        F[1, 5] = 1.0  # yc += vy
        F[3, 7] = 1.0  # h += vh
        # Fixed aspect ratio: F[2, 6] = 0.0

        if not is_observed:
            # Apply velocity damping during unobserved dropout
            decay = self.gamma
            F[4, 4] = decay
            F[5, 5] = decay
            F[7, 7] = decay

        # Bound velocities to max sprint speed
        v_norm = np.hypot(mean[4], mean[5])
        if v_norm > self.max_velocity_px and v_norm > 1e-4:
            scale = self.max_velocity_px / v_norm
            mean[4] *= scale
            mean[5] *= scale

        mean = np.dot(F, mean)
        covariance = np.linalg.multi_dot((F, covariance, F.T)) + motion_cov
        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space [xc, yc, a, h].
        Returns: (projected_mean, innovation_covariance S)
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        R = np.diag(np.square(std))

        H = self._update_mat
        projected_mean = np.dot(H, mean)
        S = np.linalg.multi_dot((H, covariance, H.T)) + R
        return projected_mean, S

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman update step with new measurement [xc, yc, a, h].
        """
        projected_mean, S = self.project(mean, covariance)

        # Kalman gain K = P H^T S^{-1}
        H = self._update_mat
        chol_factor = np.linalg.cholesky(S)
        K = np.linalg.solve(chol_factor.T, np.linalg.solve(chol_factor, np.dot(H, covariance))).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, K.T)
        new_covariance = covariance - np.linalg.multi_dot((K, S, K.T))
        return new_mean, new_covariance
