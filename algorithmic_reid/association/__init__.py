"""
algorithmic_reid.association package - Fused cost computation, gating barriers and Hungarian matching.
"""

from .cost_matrix import compute_fused_cost_matrix, compute_diou_matrix
from .gating import apply_gating_barriers
from .hungarian import min_cost_matching

__all__ = [
    "compute_fused_cost_matrix",
    "compute_diou_matrix",
    "apply_gating_barriers",
    "min_cost_matching",
]
