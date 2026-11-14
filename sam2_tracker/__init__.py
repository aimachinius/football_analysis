"""
sam2_tracker/__init__.py
Module theo dõi đối tượng độc lập dựa trên SAM 2 (Segment Anything 2 - Meta AI).
Không can thiệp vào pipeline chính (main.py). Chỉ gọi độc lập khi cần.
"""

from .sam2_bidirectional_tracker import SAM2BidirectionalTracker
from .sam2_refiner import refine_tracks_with_sam2_backward
from .sam2_forward_tracker import SAM2ForwardTracker
from .sam2_ball_refiner import refine_ball_candidates_with_sam2

__all__ = [
    "SAM2BidirectionalTracker",
    "refine_tracks_with_sam2_backward",
    "SAM2ForwardTracker",
    "refine_ball_candidates_with_sam2",
]

