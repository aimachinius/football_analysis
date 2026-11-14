"""
algorithmic_reid.lifecycle.gta_postprocess - Global Tracklet Association (Direction C).

Offline post-processing after Pass 1:
  1. Anomaly Split: Detects and splits tracklets that suffered impossible kinematics / erroneous merges.
  2. Global Stitch: Formulates tracklet fragment continuation as a global bipartite matching problem,
     re-linking broken fragments of the same player across long occlusions.
"""

import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Any, Optional


class GlobalTrackletAssociator:
    def __init__(
        self,
        max_jump_px: float = 140.0,
        max_stitch_gap: int = 60,
        max_velocity_px: float = 35.0,
        stitch_cost_thresh: float = 0.50,
    ):
        self.max_jump_px = max_jump_px
        self.max_stitch_gap = max_stitch_gap
        self.max_velocity_px = max_velocity_px
        self.stitch_cost_thresh = stitch_cost_thresh

    def split_erroneous_tracklets(self, tracks: Dict[str, List[Dict[int, Any]]]) -> Dict[str, List[Dict[int, Any]]]:
        """
        Phase 1: Split tracklets where an impossible teleportation or jump occurred.
        """
        for cat in ("players", "referees"):
            cat_list = tracks.get(cat, [])
            if not cat_list:
                continue

            # Find all track IDs and their active frames
            id_frames = defaultdict(list)
            for f_idx, f_data in enumerate(cat_list):
                for tid in f_data.keys():
                    id_frames[tid].append(f_idx)

            max_existing_id = max(id_frames.keys()) if id_frames else 0
            next_new_id = max(1000, max_existing_id + 1)

            for tid, f_indices in list(id_frames.items()):
                if len(f_indices) < 4:
                    continue

                split_point = None
                for k in range(len(f_indices) - 1):
                    f1 = f_indices[k]
                    f2 = f_indices[k + 1]
                    gap = f2 - f1
                    if gap <= 3:
                        b1 = cat_list[f1][tid]["bbox"]
                        b2 = cat_list[f2][tid]["bbox"]
                        c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
                        c2 = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
                        jump = np.hypot(c2[0] - c1[0], c2[1] - c1[1])

                        if jump > self.max_jump_px * gap:
                            split_point = f2
                            break

                if split_point is not None:
                    # Split at split_point: reassign all future frames to next_new_id
                    new_id = next_new_id
                    next_new_id += 1
                    for f_idx in f_indices:
                        if f_idx >= split_point:
                            entry = cat_list[f_idx].pop(tid)
                            cat_list[f_idx][new_id] = entry

        return tracks

    def stitch_global_tracklets(self, tracks: Dict[str, List[Dict[int, Any]]]) -> Dict[str, List[Dict[int, Any]]]:
        """
        Phase 2: Global bipartite matching across temporal gaps to merge broken tracklet fragments.
        """
        for cat in ("players", "referees"):
            cat_list = tracks.get(cat, [])
            if not cat_list:
                continue

            # Identify all tracklets
            id_obs = defaultdict(list)
            for f_idx, f_data in enumerate(cat_list):
                for tid, entry in f_data.items():
                    id_obs[tid].append((f_idx, entry))

            tracklets = []
            for tid, obs in id_obs.items():
                if len(obs) < 2:
                    continue
                start_f = obs[0][0]
                end_f = obs[-1][0]
                start_b = obs[0][1]["bbox"]
                end_b = obs[-1][1]["bbox"]
                team = obs[0][1].get("team", 0)
                is_gk = obs[0][1].get("is_goalkeeper", False)
                tracklets.append({
                    "id": tid,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "start_box": start_b,
                    "end_box": end_b,
                    "team": team,
                    "is_gk": is_gk,
                })

            if len(tracklets) < 2:
                continue

            # Build pairwise stitching cost matrix
            M = len(tracklets)
            cost_mat = np.full((M, M), 1e5, dtype=np.float64)

            for i, t_tail in enumerate(tracklets):
                for j, t_head in enumerate(tracklets):
                    if i == j:
                        continue
                    gap = t_head["start_frame"] - t_tail["end_frame"]
                    if 1 <= gap <= self.max_stitch_gap:
                        # Team barrier
                        if t_tail["team"] > 0 and t_head["team"] > 0 and t_tail["team"] != t_head["team"]:
                            continue
                        if t_tail["is_gk"] != t_head["is_gk"]:
                            continue

                        # Kinematic velocity check
                        c_tail = ((t_tail["end_box"][0] + t_tail["end_box"][2]) / 2.0,
                                  (t_tail["end_box"][1] + t_tail["end_box"][3]) / 2.0)
                        c_head = ((t_head["start_box"][0] + t_head["start_box"][2]) / 2.0,
                                  (t_head["start_box"][1] + t_head["start_box"][3]) / 2.0)
                        dist = np.hypot(c_head[0] - c_tail[0], c_head[1] - c_tail[1])
                        vel = dist / float(gap)

                        if vel <= self.max_velocity_px:
                            c_mot = vel / self.max_velocity_px
                            c_time = float(gap) / float(self.max_stitch_gap)
                            cost_mat[i, j] = 0.70 * c_mot + 0.30 * c_time

            # Disjoint Set (Union-Find) to resolve multi-hop stitches safely
            parent = {t["id"]: t["id"] for t in tracklets}

            def find_root(u):
                curr = u
                while parent.get(curr, curr) != curr:
                    parent[curr] = parent.get(parent[curr], parent[curr])
                    curr = parent[curr]
                return curr

            row_ind, col_ind = linear_sum_assignment(cost_mat)
            for r, c in zip(row_ind, col_ind):
                if cost_mat[r, c] <= self.stitch_cost_thresh:
                    tail_id = tracklets[r]["id"]
                    head_id = tracklets[c]["id"]
                    root_tail = find_root(tail_id)
                    root_head = find_root(head_id)
                    if root_tail != root_head:
                        parent[root_head] = root_tail

            # Reassign stitched tracklet IDs safely across all frames
            for f_idx, f_data in enumerate(cat_list):
                curr_ids = list(f_data.keys())
                for tid in curr_ids:
                    canonical_id = find_root(tid)
                    if canonical_id != tid:
                        entry = f_data.pop(tid)
                        if canonical_id not in f_data:
                            f_data[canonical_id] = entry
                        else:
                            f_data[tid] = entry

        return tracks

    def process(self, tracks: Dict[str, List[Dict[int, Any]]]) -> Dict[str, List[Dict[int, Any]]]:
        """Run two-phase Global Tracklet Association."""
        tracks = self.split_erroneous_tracklets(tracks)
        tracks = self.stitch_global_tracklets(tracks)
        return tracks
