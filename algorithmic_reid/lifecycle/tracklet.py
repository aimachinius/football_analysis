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

        if homography is not None:
            from ..motion.motion_predictor import apply_camera_motion_to_tracks
            apply_camera_motion_to_tracks([self], homography)

    def update(
        self,
        bbox: np.ndarray,
        confidence: float,
        class_id: int,
        feature: Optional[np.ndarray] = None,
        max_inter_iou: float = 0.0,
        frame_idx: int = 0,
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

        self.confidence = float(confidence)
        self.class_id = class_id
        self.hits += 1
        self.consecutive_hits += 1
        self.time_since_update = 0
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
