"""
algorithmic_reid package - Modular Algorithmic Re-Identification and Tracking Stability.
Provides multi-cue association (DIoU, Kalman motion, Mahalanobis gating, multi-zone color appearance,
species isolation, tracklet lifecycle management, and tracklet stitching).
"""

from .config import ReIDConfig
from .reid_tracker_facade import ReIDTrackerFacade

__all__ = ["ReIDConfig", "ReIDTrackerFacade"]
