"""
algorithmic_reid.appearance package - Multi-zone color descriptors and temporal appearance aggregation.
"""

from .color_descriptor import extract_multizone_histogram, bhattacharyya_distance, batch_bhattacharyya_distance
from .temporal_aggregator import AppearanceAggregator

__all__ = [
    "extract_multizone_histogram",
    "bhattacharyya_distance",
    "batch_bhattacharyya_distance",
    "AppearanceAggregator",
]
