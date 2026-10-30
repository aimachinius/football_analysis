"""
algorithmic_reid.lifecycle package - Tracklet state machine, tracklet stitching and cascade manager.
"""

from .tracklet import Tracklet, TrackletState
from .stitcher import TrackletStitcher
from .track_manager import TrackManager

__all__ = [
    "Tracklet",
    "TrackletState",
    "TrackletStitcher",
    "TrackManager",
]
