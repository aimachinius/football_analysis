"""
tracker/viterbi_ball.py - Viterbi Dynamic Programming Top-K Ball Selector.

Thuat toan:
1. Pass 1: Luu tru top-K ung vien bong conf >= floor cho moi frame vao bo nho.
2. Sau Pass 1: Chay quy hoach dong (DP) bac nhat trong he quy chieu camera (to_reference).
   - Reset tai moi camera cut / doi segment.
   - Trang thai moi frame: K ung vien + MISSING.
   - Chi phi Emission:
       * -log(max(conf, 1e-3))
       * Thuong nho (-w_foot) neu ung vien nam trong vung 25% duoi bbox cau thu.
       * Phat (+w_upper) neu ung vien nam o nua tren bbox cau thu.
       * Phat (+w_static) neu ung vien tinh > N frame xa moi cau thu (vet son san / diem phat den).
   - Chi phi Chuyen trang thai (Transition):
       * Cho phep skip tu frame t - g sang t (g <= G, mac dinh G=30).
       * Chi phi missing: (g - 1) * missing_cost (neu g=1 thi khong ton phi missing trung gian).
       * Phat van toc: quadratic penalty theo (v - v_max) neu v > v_max (v_max = 50 px/f).
       * Chan cung (hard barrier = inf) neu v > 2 * v_max.
       * KHONG dung gia toc trong DP (giu nguyen bac 1). Do muot duoc xu ly boi Stabilizer.
3. Xuat tracks["ball"] tuan thu Data Contract:
   {1: {'bbox': bbox, 'confidence': conf, 'is_interpolated': False}} hoac {} khi LOST.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ViterbiConfig:
    floor: float = 0.05
    top_k: int = 5
    max_skip_g: int = 30
    missing_cost: float = 1.8
    birth_cost: float = 1.5
    v_max: float = 50.0  # px/frame in reference frame
    w_speed: float = 2.0
    w_foot_bonus: float = 0.5
    w_upper_penalty: float = 2.0
    w_static_penalty: float = 3.5
    static_n_frames: int = 5
    static_dist_thresh: float = 4.0  # px in reference frame
    isolated_player_dist: float = 120.0  # px
    # Chỉnh sửa (a): Phạt ứng viên cô lập
    enable_isolated_penalty: bool = False
    isolated_time_window: int = 3
    isolated_radius_px: float = 35.0
    w_isolated_penalty: float = 4.0
    # Chỉnh sửa (b): Giới hạn tầm với theo gap
    enable_gap_reach_limit: bool = False
    v_cruise: float = 30.0
    reach_margin: float = 40.0


class ViterbiBallSelector:
    """
    Bo chon bong Viterbi DP toan cuc hoat dong sau Pass 1.
    """

    def __init__(self, config: Optional[ViterbiConfig] = None):
        self.cfg = config or ViterbiConfig()
        # candidate_store[t] = list of candidate dicts
        self.candidate_store: List[List[Dict[str, Any]]] = []

    def reset(self):
        """Reset bo nho ung vien cho video moi."""
        self.candidate_store.clear()

    def add_frame_candidates(
        self,
        frame_index: int,
        ball_detections,
        ids: Optional[Dict[str, int]] = None
    ):
        """
        Thu thap top-K ung vien bong conf >= floor trong Pass 1.
        ball_detections co the la sv.Detections hoac list cac dict/bbox.
        """
        while len(self.candidate_store) <= frame_index:
            self.candidate_store.append([])

        cands = []
        if ball_detections is None:
            return

        # Neu la sv.Detections
        if hasattr(ball_detections, "xyxy") and hasattr(ball_detections, "confidence"):
            for xyxy, conf in zip(ball_detections.xyxy, ball_detections.confidence):
                if conf >= self.cfg.floor:
                    cands.append({
                        "bbox": xyxy.tolist(),
                        "confidence": float(conf),
                        "center": ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0),
                    })
        elif isinstance(ball_detections, list):
            for item in ball_detections:
                if isinstance(item, dict):
                    conf = item.get("confidence", item.get("conf", 1.0))
                    bbox = item.get("bbox")
                    if conf >= self.cfg.floor and bbox is not None:
                        cands.append({
                            "bbox": list(bbox),
                            "confidence": float(conf),
                            "center": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
                        })

        # Sap xep giam dan theo confidence va giu top-K
        cands.sort(key=lambda x: x["confidence"], reverse=True)
        self.candidate_store[frame_index] = cands[:self.cfg.top_k]

    def select_trajectory(
        self,
        total_frames: int,
        camera_motion=None,
        player_tracks: Optional[List[Dict[int, Dict[str, Any]]]] = None,
    ) -> List[Dict[int, Dict[str, Any]]]:
        """
        Chay thuat toan Viterbi DP bac nhat de tim quy dao bong toi uu toan chuoi.
        Tra ve list tracks['ball'] co do dai total_frames.
        """
        cfg = self.cfg
        T = total_frames
        while len(self.candidate_store) < T:
            self.candidate_store.append([])

        # Lay thong tin segment tu camera_motion
        segs = np.zeros(T, dtype=int)
        if camera_motion is not None:
            if hasattr(camera_motion, "seg"):
                motion_segs = camera_motion.seg
                if isinstance(motion_segs, (list, tuple, np.ndarray)):
                    for t in range(min(T, len(motion_segs))):
                        segs[t] = int(motion_segs[t])
                elif isinstance(motion_segs, (int, np.integer)):
                    segs[:] = int(motion_segs)
            elif hasattr(camera_motion, "segs"):
                motion_segs = camera_motion.segs
                if isinstance(motion_segs, (list, tuple, np.ndarray)):
                    for t in range(min(T, len(motion_segs))):
                        segs[t] = int(motion_segs[t])

        # Chuyen toa do tat ca ung vien sang he quy chieu camera (to_reference)
        # va tinh emission cost so bo
        processed_candidates: List[List[Dict[str, Any]]] = []
        for t in range(T):
            frame_cands = []
            p_boxes = []
            if player_tracks is not None and t < len(player_tracks):
                p_boxes = [p["bbox"] for p in player_tracks[t].values() if "bbox" in p]

            for c in self.candidate_store[t]:
                cx, cy = c["center"]
                if camera_motion is not None and hasattr(camera_motion, "to_reference"):
                    ref_pt = camera_motion.to_reference(t, np.array([[cx, cy]]))[0]
                else:
                    ref_pt = np.array([cx, cy], dtype=float)

                conf = max(c["confidence"], 1e-3)
                e_conf = -float(np.log(conf))

                # Kiem tra tuong quan voi cau thu
                in_foot = False
                in_upper = False
                min_p_dist = 999.0
                for pb in p_boxes:
                    pw = pb[2] - pb[0]
                    ph = pb[3] - pb[1]
                    # Vung 25% duoi bbox (foot zone)
                    if (pb[0] - 0.20 * pw <= cx <= pb[2] + 0.20 * pw) and (
                        pb[1] + 0.75 * ph <= cy <= pb[3] + 0.15 * ph
                    ):
                        in_foot = True
                    # Vung nua tren bbox (upper body / head / shirt)
                    if (pb[0] <= cx <= pb[2]) and (
                        pb[1] - 0.05 * ph <= cy <= pb[1] + 0.50 * ph
                    ):
                        in_upper = True

                    foot_x = (pb[0] + pb[2]) / 2.0
                    foot_y = pb[3]
                    dist_p = float(np.hypot(cx - foot_x, cy - foot_y))
                    if dist_p < min_p_dist:
                        min_p_dist = dist_p

                e_context = 0.0
                if in_upper:
                    e_context += cfg.w_upper_penalty
                elif in_foot:
                    e_context -= cfg.w_foot_bonus

                # Danh gia static artifact
                is_isolated = min_p_dist > cfg.isolated_player_dist

                frame_cands.append({
                    "bbox": c["bbox"],
                    "confidence": c["confidence"],
                    "center": (cx, cy),
                    "ref": ref_pt,
                    "emission": e_conf + e_context,
                    "is_isolated": is_isolated,
                })
            processed_candidates.append(frame_cands)

        # Kiem tra ung vien tinh > N frames xa cau thu (Static pitch marking filter)
        for t in range(cfg.static_n_frames, T):
            for c_curr in processed_candidates[t]:
                if not c_curr["is_isolated"]:
                    continue
                # Kiem tra co ung vien nao o cung toa do ref trong N frames truoc khong
                static_count = 0
                for past_t in range(t - cfg.static_n_frames, t):
                    for c_past in processed_candidates[past_t]:
                        if np.linalg.norm(c_curr["ref"] - c_past["ref"]) <= cfg.static_dist_thresh:
                            static_count += 1
                            break
                if static_count >= cfg.static_n_frames - 1:
                    c_curr["emission"] += cfg.w_static_penalty

        # Chỉnh sửa (a): Phạt ứng viên cô lập (không có ứng viên nào trong bán kính R ở ±3 frame lân cận)
        if cfg.enable_isolated_penalty:
            for t in range(T):
                for c_curr in processed_candidates[t]:
                    ref_curr = c_curr["ref"]
                    has_temporal_neighbor = False
                    t_start = max(0, t - cfg.isolated_time_window)
                    t_end = min(T, t + cfg.isolated_time_window + 1)
                    for nb_t in range(t_start, t_end):
                        if nb_t == t:
                            continue
                        for c_nb in processed_candidates[nb_t]:
                            if np.linalg.norm(ref_curr - c_nb["ref"]) <= cfg.isolated_radius_px:
                                has_temporal_neighbor = True
                                break
                        if has_temporal_neighbor:
                            break
                    if not has_temporal_neighbor:
                        c_curr["emission"] += cfg.w_isolated_penalty

        # =====================================================================
        # THUAT TOAN VITERBI DP THEO TỪNG CAMERA SEGMENT
        # =====================================================================
        selected_ball_tracks: List[Dict[int, Dict[str, Any]]] = [{} for _ in range(T)]

        # Tìm các khoảng segment liên tục [seg_start, seg_end]
        seg_intervals = []
        cur_start = 0
        for t in range(1, T):
            if segs[t] != segs[t - 1]:
                seg_intervals.append((cur_start, t - 1))
                cur_start = t
        seg_intervals.append((cur_start, T - 1))

        for seg_start, seg_end in seg_intervals:
            seg_len = seg_end - seg_start + 1
            if seg_len <= 0:
                continue

            dp_missing = np.full(seg_len, np.inf)
            dp_cands = [np.full(len(processed_candidates[seg_start + i]), np.inf) for i in range(seg_len)]

            parent_missing: List[Optional[Tuple[int, int]]] = [None] * seg_len
            parent_cands: List[List[Optional[Tuple[int, int]]]] = [
                [None] * len(processed_candidates[seg_start + i]) for i in range(seg_len)
            ]

            # Khởi tạo tại frame đầu tiên của segment (i = 0)
            t_first = seg_start
            dp_missing[0] = cfg.missing_cost
            for c_idx in range(len(processed_candidates[t_first])):
                dp_cands[0][c_idx] = processed_candidates[t_first][c_idx]["emission"] + cfg.birth_cost

            # Lặp qua các frame trong segment
            for i in range(1, seg_len):
                t_curr = seg_start + i

                # 1. Chuyển trạng thái sang MISSING tại i
                best_m = dp_missing[i - 1] + cfg.missing_cost
                best_p_m = (i - 1, -1)
                for j in range(len(processed_candidates[t_curr - 1])):
                    cost_from_cand = dp_cands[i - 1][j] + cfg.missing_cost
                    if cost_from_cand < best_m:
                        best_m = cost_from_cand
                        best_p_m = (i - 1, j)

                dp_missing[i] = best_m
                parent_missing[i] = best_p_m

                # 2. Chuyển trạng thái sang các Candidate tại i
                for c_idx in range(len(processed_candidates[t_curr])):
                    c_curr = processed_candidates[t_curr][c_idx]
                    ref_curr = c_curr["ref"]

                    # Lựa chọn 1: Birth khởi tạo track mới từ MISSING
                    best_cost_c = dp_missing[i - 1] + cfg.birth_cost + c_curr["emission"]
                    best_p_c = (i - 1, -1)

                    # Lựa chọn 2: Transition từ ứng viên ở i - g (1 <= g <= G)
                    max_g = min(i, cfg.max_skip_g)
                    for g in range(1, max_g + 1):
                        t_prev = t_curr - g
                        prev_i = i - g
                        for j in range(len(processed_candidates[t_prev])):
                            prev_cost = dp_cands[prev_i][j]
                            if not np.isfinite(prev_cost):
                                continue

                            ref_prev = processed_candidates[t_prev][j]["ref"]
                            dist = float(np.linalg.norm(ref_curr - ref_prev))
                            v = dist / g

                            # Chỉnh sửa (b): Giới hạn tầm với theo gap: R(g) = v_cruise * g + reach_margin
                            if cfg.enable_gap_reach_limit:
                                r_max = cfg.v_cruise * float(g) + cfg.reach_margin
                                if dist > r_max:
                                    continue

                            if v > 2.0 * cfg.v_max:
                                continue

                            if v > cfg.v_max:
                                v_penalty = cfg.w_speed * ((v - cfg.v_max) / cfg.v_max) ** 2
                            else:
                                v_penalty = 0.0

                            trans_cost = (g - 1) * cfg.missing_cost + v_penalty
                            tot_cost = prev_cost + trans_cost + c_curr["emission"]

                            if tot_cost < best_cost_c:
                                best_cost_c = tot_cost
                                best_p_c = (prev_i, j)

                    dp_cands[i][c_idx] = best_cost_c
                    parent_cands[i][c_idx] = best_p_c

            # Backtracking cho segment này
            last_i = seg_len - 1
            best_end_cost = dp_missing[last_i]
            best_end_state = (last_i, -1)
            for j in range(len(processed_candidates[seg_end])):
                if dp_cands[last_i][j] < best_end_cost:
                    best_end_cost = dp_cands[last_i][j]
                    best_end_state = (last_i, j)

            curr = best_end_state
            while curr is not None:
                cur_i, state_idx = curr
                t_actual = seg_start + cur_i
                if state_idx >= 0:
                    c = processed_candidates[t_actual][state_idx]
                    bbox = c["bbox"]
                    center_pos = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
                    selected_ball_tracks[t_actual][1] = {
                        "bbox": bbox,
                        "position": center_pos,
                        "confidence": float(c["confidence"]),
                        "is_interpolated": False,
                    }
                    curr = parent_cands[cur_i][state_idx]
                else:
                    curr = parent_missing[cur_i]

        return selected_ball_tracks
