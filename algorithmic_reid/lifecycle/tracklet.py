"""
algorithmic_reid.lifecycle.tracklet - Rich Tracklet Data Structure and State Machine.
"""

import numpy as np
from enum import Enum
from typing import Optional, List, Tuple
from ..motion.kalman_filter import DampedKalmanFilter, z_to_bbox, bbox_to_z
from ..appearance.temporal_aggregator import AppearanceAggregator


class TrackletState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    DELETED = 4


class Tracklet:
    def __init__(
        self,
        track_id: int,
        bbox: np.ndarray,
        confidence: float,
        class_id: int,
        feature: Optional[np.ndarray],
        kf: DampedKalmanFilter,
        aggregator: AppearanceAggregator,
        n_init: int = 2,
        max_age: int = 70,
        start_frame: int = 0,
    ):
        self.track_id = track_id
        self.class_id = class_id
        self.confidence = float(confidence)
        self.kf = kf
        self.aggregator = aggregator
        self.n_init = n_init
        self.max_age = max_age
        self.start_frame = start_frame
        self.last_frame = start_frame

        # Initialize Kalman state
        z = bbox_to_z(bbox)
        self.mean, self.covariance = self.kf.initiate(z)

        # Initialize appearance template
        self.appearance_template = feature.copy() if feature is not None else None

        # Lifecycle state
        self.state = TrackletState.TENTATIVE
        self.hits = 1
        self.consecutive_hits = 1
        self.age = 1
        self.time_since_update = 0
        self.is_predicted = False

        # Stable clean appearance template & pre-occlusion momentum
        self.clean_template = feature.copy() if feature is not None else None
        self.clean_velocity: Tuple[float, float] = (0.0, 0.0)
        self.in_occlusion = False

        # Deep Re-ID embedding templates (Direction B)
        self.deep_template: Optional[np.ndarray] = None
        self.clean_deep_template: Optional[np.ndarray] = None

        # Team identity and role stabilization
        self.assigned_team: int = 0
        self.role_history: List[int] = [class_id]

        # Trajectory history: list of (frame_idx, bbox)
        self.history: List[Tuple[int, np.ndarray]] = [(start_frame, bbox.copy())]

    @property
    def bbox(self) -> np.ndarray:
        """Current bounding box [x1, y1, x2, y2]."""
        return z_to_bbox(self.mean[:4])

    def predict(self, homography: Optional[np.ndarray] = None) -> None:
        """
        Predict state for current frame. Apply velocity damping if unobserved.
        """
        is_observed = (self.time_since_update == 0)
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance, is_observed=is_observed)
        self.age += 1
        self.is_predicted = True

        if homography is not None:
            from ..motion.motion_predictor import apply_camera_motion_to_tracks
            apply_camera_motion_to_tracks([self], homography)

    def get_assigned_class(self, default_class: int = 2) -> int:
        """
        [Tầng 2] Temporal role majority voting with hysteresis.
        Prevents 1-frame detector classification noise (Player <-> Referee <-> Goalkeeper).
        """
        if not self.role_history:
            return self.class_id
        total = len(self.role_history)
        if total < 3:
            return self.role_history[-1]

        # 3: Referee, 1: Goalkeeper, 2: Player
        ref_count = self.role_history.count(3)
        gk_count = self.role_history.count(1)

        if ref_count / total >= 0.25:
            return 3
        if gk_count / total >= 0.30:
            return 1
        return 2

    def update(
        self,
        bbox: np.ndarray,
        confidence: float,
        class_id: int,
        feature: Optional[np.ndarray] = None,
        max_inter_iou: float = 0.0,
        frame_idx: int = 0,
        assigned_team: int = 0,
        deep_feature: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update tracklet with matched detection.
        """
        z = bbox_to_z(bbox)
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, z)

        # Appearance update with occlusion freeze guard
        if feature is not None:
            self.appearance_template = self.aggregator.update_template(
                self.appearance_template,
                feature,
                confidence=confidence,
                bbox=bbox,
                max_inter_iou=max_inter_iou,
            )

        # Deep Re-ID update (Direction B)
        if deep_feature is not None:
            if self.deep_template is None:
                self.deep_template = deep_feature.copy()
            elif not self.in_occlusion and confidence >= 0.45:
                up_deep = 0.85 * self.deep_template + 0.15 * deep_feature
                norm = np.linalg.norm(up_deep)
                self.deep_template = up_deep / norm if norm > 1e-6 else up_deep

        # Clean template preservation & momentum tracking before entering occlusion
        self.in_occlusion = (max_inter_iou > 0.10)
        if not self.in_occlusion and feature is not None and confidence >= 0.45:
            if self.clean_template is None:
                self.clean_template = feature.copy()
            else:
                updated = 0.85 * self.clean_template + 0.15 * feature
                s = np.sum(updated)
                self.clean_template = updated / s if s > 1e-6 else updated

            if deep_feature is not None:
                self.clean_deep_template = deep_feature.copy()

            vx, vy = float(self.mean[4]), float(self.mean[5])
            if np.hypot(vx, vy) >= 1.5:
                self.clean_velocity = (vx, vy)

        if assigned_team > 0:
            self.assigned_team = assigned_team

        self.confidence = float(confidence)
        self.class_id = class_id
        self.role_history.append(int(class_id))
        self.hits += 1
        self.consecutive_hits += 1
        self.time_since_update = 0
        self.is_predicted = False
        self.last_frame = frame_idx
        self.history.append((frame_idx, bbox.copy()))

        if self.state == TrackletState.TENTATIVE:
            if self.consecutive_hits >= self.n_init:
                self.state = TrackletState.CONFIRMED
        elif self.state == TrackletState.LOST:
            self.state = TrackletState.CONFIRMED

    def mark_missed(self) -> None:
        """Mark tracklet as missed in current frame."""
        self.time_since_update += 1
        self.consecutive_hits = 0

        if self.state == TrackletState.TENTATIVE:
            # Tentative track that misses an update is deleted
            self.state = TrackletState.DELETED
        elif self.state == TrackletState.CONFIRMED:
            if self.time_since_update > 0:
                self.state = TrackletState.LOST

        if self.time_since_update > self.max_age:
            self.state = TrackletState.DELETED
