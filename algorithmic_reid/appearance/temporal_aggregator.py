"""
algorithmic_reid.appearance.temporal_aggregator - Occlusion-Aware EMA Feature Aggregator.

Maintains appearance templates over time with update freeze guards:
  F_t = (1 - alpha) * F_{t-1} + alpha * F_{det}
Update is frozen if:
  - Detection confidence < 0.50
  - Occlusion alert: IoU with another detection > 0.15
  - Aspect ratio anomaly (h/w < 1.4 or h/w > 4.0)
"""

import numpy as np
from typing import Optional


class AppearanceAggregator:
    def __init__(self, alpha: float = 0.15, min_conf: float = 0.50, max_overlap_iou: float = 0.15):
        self.alpha = alpha
        self.min_conf = min_conf
        self.max_overlap_iou = max_overlap_iou

    def should_freeze_update(
        self,
        confidence: float,
        bbox: np.ndarray,
        max_inter_iou: float = 0.0,
    ) -> bool:
        """Evaluate whether appearance update must be frozen."""
        if confidence < self.min_conf:
            return True
        if max_inter_iou > self.max_overlap_iou:
            return True

        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        aspect = h / w
        if aspect < 1.3 or aspect > 4.2:
            return True

        return False

    def update_template(
        self,
        current_template: Optional[np.ndarray],
        new_feature: Optional[np.ndarray],
        confidence: float,
        bbox: np.ndarray,
        max_inter_iou: float = 0.0,
    ) -> Optional[np.ndarray]:
        """
        Update appearance template via EMA if update freeze guards pass.
        """
        if new_feature is None:
            return current_template

        if current_template is None:
            # First initialization
            return new_feature.copy()

        if self.should_freeze_update(confidence, bbox, max_inter_iou):
            # Freeze template
            return current_template

        updated = (1.0 - self.alpha) * current_template + self.alpha * new_feature
        s = np.sum(updated)
        if s > 1e-6:
            updated /= s
        return updated
