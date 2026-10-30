"""
algorithmic_reid.association.hungarian - Gated Linear Sum Assignment.

Solves bipartite assignment via scipy.optimize.linear_sum_assignment,
filtering out any match where cost > max_cost.
"""

import numpy as np
from typing import Tuple, List
from scipy.optimize import linear_sum_assignment


def min_cost_matching(
    cost_matrix: np.ndarray,
    max_cost: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve minimum cost matching with cost threshold gating.
    Returns:
      (matched_indices, unmatched_track_indices, unmatched_det_indices)
    where:
      matched_indices: list of (track_idx, det_idx)
      unmatched_track_indices: list of track_idx
      unmatched_det_indices: list of det_idx
    """
    M, N = cost_matrix.shape
    if M == 0:
        return [], [], list(range(N))
    if N == 0:
        return [], list(range(M)), []

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched = []
    unmatched_tracks = set(range(M))
    unmatched_dets = set(range(N))

    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost <= max_cost:
            matched.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

    return matched, sorted(list(unmatched_tracks)), sorted(list(unmatched_dets))
