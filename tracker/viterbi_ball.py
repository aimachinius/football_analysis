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
import os
import cv2
import numpy as np


@dataclass
class ViterbiConfig:
    floor: float = 0.01
    top_k: int = 10
    max_skip_g: int = 30
    missing_cost: float = 1.8
    birth_cost: float = 1.5
    v_max: float = 50.0  # px/frame in reference frame
    w_speed: float = 2.0
    w_foot_bonus: float = 0.5
    w_upper_penalty: float = 8.0
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
    enable_gap_reach_limit: bool = True
    v_cruise: float = 30.0
    reach_margin: float = 40.0
    # Rào cản ranh giới sân động học (Pitch Boundary Filter)
    enable_pitch_boundary_filter: bool = True
    pitch_polygon: Optional[np.ndarray] = None
    pitch_margin_px: float = 25.0
    # Máy trạng thái sở hữu bóng & Rào cản chuyển dịch (Possession State Machine)
    enable_possession_barrier: bool = True
    possession_radius_px: float = 40.0
    v_dribble_max: float = 18.0
    w_possession_retain_bonus: float = 1.0
    gk_isolation_dist_px: float = 450.0
    w_spurious_jump_penalty: float = 15.0
    # Rào cản góc lệch hướng bay (Collinear Flight Barrier)
    enable_flight_collinearity: bool = True
    max_flight_deflection_deg: float = 35.0
    w_flight_deflection: float = 6.0
    # Chặn Viterbi Re-birth xuyên sân (Cross-Pitch Re-birth Barrier)
    enable_cross_pitch_birth_barrier: bool = True
    w_cross_pitch_birth_penalty: float = 20.0
    max_cross_pitch_birth_dist: float = 200.0
    cross_pitch_lookback_frames: int = 15
    # SAM 2 Visual Candidate Producer
    enable_sam2_refiner: bool = True
    sam2_min_gap: int = 6
    sam2_max_gap: int = 40
    sam2_anchor_conf: float = 0.25


def is_point_inside_pitch(
    p_ref: np.ndarray,
    pitch_polygon: Optional[np.ndarray],
    margin_px: float = 25.0,
) -> bool:
    """
    Kiểm định tổng quát điểm trong/ngoài sân dựa trên bao lồi hệ quy chiếu tham chiếu.
    dist > 0: Trong sân
    dist == 0: Trên đường biên
    dist >= -margin_px: Trong sân hoặc sát lề biên cho phép (sai số phối cảnh/độ cao)
    """
    if pitch_polygon is None:
        return True
    poly = np.asarray(pitch_polygon, dtype=np.float32)
    if len(poly) < 3:
        return True
    dist = cv2.pointPolygonTest(poly, (float(p_ref[0]), float(p_ref[1])), measureDist=True)
    return dist >= -margin_px


def get_pitch_reference_polygon(
    calib_path: str = "calib/points.json",
    length: float = 105.0,
    width: float = 68.0,
) -> Optional[np.ndarray]:
    """
    Tính toán đa giác 4 góc sân FIFA chuẩn (105m x 68m) trên hệ quy chiếu Frame 0 pixel.
    Tự động suy diễn từ ma trận hiệu chuẩn sân (không hardcode toạ độ).
    """
    if not os.path.exists(calib_path):
        return None
    try:
        from pitch_homo import PitchCalibration, load_points
        pts = load_points(calib_path)
        if not pts:
            return None
        calib = PitchCalibration.fit(pts, motion=None)
        H0 = calib.H_by_seg.get(0)
        if H0 is None or not np.all(np.isfinite(H0)):
            return None
        H0_inv = np.linalg.inv(H0)
        fifa_corners = np.array([
            [0.0, 0.0],
            [length, 0.0],
            [length, width],
            [0.0, width]
        ], dtype=np.float64)
        homo = np.hstack([fifa_corners, np.ones((4, 1))])
        ref_corners = (H0_inv @ homo.T).T
        return (ref_corners[:, :2] / ref_corners[:, 2:3]).astype(np.float32)
    except Exception:
        return None


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
        ids: Optional[Dict[str, int]] = None,
        camera_motion: Optional[Any] = None,
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

        def _is_valid_point(cx: float, cy: float) -> bool:
            if not self.cfg.enable_pitch_boundary_filter or self.cfg.pitch_polygon is None:
                return True
            if camera_motion is not None and hasattr(camera_motion, "to_reference"):
                try:
                    ref_p = camera_motion.to_reference(frame_index, np.array([[cx, cy]]))[0]
                except Exception:
                    ref_p = np.array([cx, cy], dtype=float)
            else:
                ref_p = np.array([cx, cy], dtype=float)
            return is_point_inside_pitch(ref_p, self.cfg.pitch_polygon, self.cfg.pitch_margin_px)

        # Neu la sv.Detections
        if hasattr(ball_detections, "xyxy") and hasattr(ball_detections, "confidence"):
            for xyxy, conf in zip(ball_detections.xyxy, ball_detections.confidence):
                if conf >= self.cfg.floor:
                    cx = float((xyxy[0] + xyxy[2]) / 2.0)
                    cy = float((xyxy[1] + xyxy[3]) / 2.0)
                    if not _is_valid_point(cx, cy):
                        continue
                    cands.append({
                        "bbox": xyxy.tolist(),
                        "confidence": float(conf),
                        "center": (cx, cy),
                    })
        elif isinstance(ball_detections, list):
            for item in ball_detections:
                if isinstance(item, dict):
                    conf = item.get("confidence", item.get("conf", 1.0))
                    bbox = item.get("bbox")
                    if conf >= self.cfg.floor and bbox is not None:
                        cx = float((bbox[0] + bbox[2]) / 2.0)
                        cy = float((bbox[1] + bbox[3]) / 2.0)
                        if not _is_valid_point(cx, cy):
                            continue
                        cands.append({
                            "bbox": list(bbox),
                            "confidence": float(conf),
                            "center": (cx, cy),
                        })

        # Sap xep giam dan theo confidence va giu top-K
        cands.sort(key=lambda x: x["confidence"], reverse=True)
        self.candidate_store[frame_index] = cands[:self.cfg.top_k]

    def select_trajectory(
        self,
        total_frames: int,
        camera_motion=None,
        player_tracks: Optional[List[Dict[int, Dict[str, Any]]]] = None,
        video_path: Optional[str] = None,
        frames: Optional[List[np.ndarray]] = None,
    ) -> List[Dict[int, Dict[str, Any]]]:
        """
        Chay thuat toan Viterbi DP bac nhat de tim quy dao bong toi uu toan chuoi.
        Tra ve list tracks['ball'] co do dai total_frames.
        """
        cfg = self.cfg
        T = total_frames
        while len(self.candidate_store) < T:
            self.candidate_store.append([])

        # Tinh match_centroid_x cho tung frame de nhan dien cau thu co lap
        match_centroid_xs: List[Optional[float]] = [None] * T
        if player_tracks is not None:
            for t in range(min(T, len(player_tracks))):
                outfield_xs = [
                    (p["bbox"][0] + p["bbox"][2]) / 2.0
                    for p in player_tracks[t].values()
                    if "bbox" in p and not p.get("is_goalkeeper", False)
                ]
                if len(outfield_xs) >= 4:
                    match_centroid_xs[t] = float(np.median(outfield_xs))

        # SAM 2 Visual Candidate Producer: Tinh chỉnh các khoảng ngắt (Gaps 6-40 frames)
        if cfg.enable_sam2_refiner and (video_path is not None or (frames is not None and len(frames) > 0)):
            try:
                from sam2_tracker.sam2_ball_refiner import refine_ball_candidates_with_sam2
                refine_ball_candidates_with_sam2(
                    candidate_store=self.candidate_store,
                    video_path=video_path,
                    frames=frames,
                    cfg=cfg,
                    camera_motion=camera_motion,
                    match_centroid_xs=match_centroid_xs,
                    player_tracks=player_tracks,
                )
            except Exception as e:
                pass

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
            p_entries = []
            if player_tracks is not None and t < len(player_tracks):
                for pid, p in player_tracks[t].items():
                    if "bbox" in p:
                        pb = p["bbox"]
                        is_gk = bool(p.get("is_goalkeeper", False))
                        p_entries.append((pid, pb, is_gk))

            match_centroid_x = match_centroid_xs[t]

            for c in self.candidate_store[t]:
                cx, cy = c["center"]
                if camera_motion is not None and hasattr(camera_motion, "to_reference"):
                    ref_pt = camera_motion.to_reference(t, np.array([[cx, cy]]))[0]
                else:
                    ref_pt = np.array([cx, cy], dtype=float)

                # Rào cản ranh giới sân động học (Pitch Boundary Guard)
                if cfg.enable_pitch_boundary_filter and cfg.pitch_polygon is not None:
                    if not is_point_inside_pitch(ref_pt, cfg.pitch_polygon, cfg.pitch_margin_px):
                        continue

                conf = max(c["confidence"], 1e-3)
                e_conf = -float(np.log(conf))

                # Kiem tra tuong quan voi cau thu
                in_foot = False
                in_upper = False
                min_p_dist = 999.0
                closest_p_info = None
                for pid, pb, is_gk in p_entries:
                    pw = pb[2] - pb[0]
                    ph = pb[3] - pb[1]
                    foot_x = (pb[0] + pb[2]) / 2.0
                    foot_y = pb[3]
                    dist_p = float(np.hypot(cx - foot_x, cy - foot_y))
                    if dist_p < min_p_dist:
                        min_p_dist = dist_p
                        closest_p_info = (pid, pb, is_gk, foot_x, foot_y)

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

                possessor_id = None
                is_cand_gk = False
                if closest_p_info is not None and min_p_dist <= cfg.possession_radius_px:
                    possessor_id = closest_p_info[0]
                    is_cand_gk = bool(closest_p_info[2])

                e_context = 0.0
                if in_upper:
                    e_context += cfg.w_upper_penalty
                elif in_foot:
                    is_isolated_spurious = False
                    if match_centroid_x is not None and closest_p_info is not None:
                        c_pid, c_pb, c_is_gk, c_fx, c_fy = closest_p_info
                        if abs(c_fx - match_centroid_x) > cfg.gk_isolation_dist_px:
                            is_isolated_spurious = True
                    if is_isolated_spurious:
                        e_context += cfg.w_spurious_jump_penalty
                        in_foot = False
                        possessor_id = None
                    else:
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
                    "is_isolated_static": False,
                    "possessor_id": possessor_id,
                    "is_gk": is_cand_gk,
                    "in_foot": in_foot,
                    "match_centroid_x": match_centroid_x,
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
                    c_curr["is_isolated_static"] = True

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
                        c_curr["is_isolated_static"] = True

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
            last_seen_missing: List[Optional[Tuple[int, np.ndarray]]] = [None] * seg_len

            # Khởi tạo tại frame đầu tiên của segment (i = 0)
            t_first = seg_start
            dp_missing[0] = cfg.missing_cost
            last_seen_missing[0] = None
            for c_idx in range(len(processed_candidates[t_first])):
                c_init = processed_candidates[t_first][c_idx]
                if cfg.enable_possession_barrier and c_init.get("is_isolated_static", False):
                    dp_cands[0][c_idx] = np.inf
                else:
                    dp_cands[0][c_idx] = c_init["emission"] + cfg.birth_cost

            # Lặp qua các frame trong segment
            for i in range(1, seg_len):
                t_curr = seg_start + i
                mc_x = match_centroid_xs[t_curr] if t_curr < len(match_centroid_xs) else None

                # 1. Chuyển trạng thái sang MISSING tại i
                best_m = dp_missing[i - 1] + cfg.missing_cost
                best_p_m = (i - 1, -1)
                best_last_seen = last_seen_missing[i - 1]
                for j in range(len(processed_candidates[t_curr - 1])):
                    cost_from_cand = dp_cands[i - 1][j] + cfg.missing_cost
                    if cost_from_cand < best_m:
                        best_m = cost_from_cand
                        best_p_m = (i - 1, j)
                        best_last_seen = (t_curr - 1, processed_candidates[t_curr - 1][j]["ref"])

                dp_missing[i] = best_m
                parent_missing[i] = best_p_m
                last_seen_missing[i] = best_last_seen

                # 2. Chuyển trạng thái sang các Candidate tại i
                for c_idx in range(len(processed_candidates[t_curr])):
                    c_curr = processed_candidates[t_curr][c_idx]

                    # Lựa chọn 1: Birth khởi tạo track mới từ MISSING
                    if cfg.enable_possession_barrier and c_curr.get("is_isolated_static", False):
                        best_cost_c = float("inf")
                        best_p_c = (i - 1, -1)
                    else:
                        # Rào cản Re-birth xuyên sân (Cross-Pitch Re-birth Barrier)
                        birth_penalty = 0.0
                        if cfg.enable_cross_pitch_birth_barrier and last_seen_missing[i - 1] is not None:
                            t_last, ref_last = last_seen_missing[i - 1]
                            dt = t_curr - t_last
                            if 1 <= dt <= cfg.cross_pitch_lookback_frames:
                                birth_dist = float(np.linalg.norm(c_curr["ref"] - ref_last))
                                birth_v = birth_dist / float(dt)
                                if birth_dist > cfg.max_cross_pitch_birth_dist:
                                    birth_penalty += cfg.w_cross_pitch_birth_penalty * min(3.0, birth_dist / cfg.max_cross_pitch_birth_dist)
                                if birth_v > cfg.v_max:
                                    if birth_v > 1.8 * cfg.v_max:
                                        birth_penalty = float("inf")
                                    else:
                                        birth_penalty += cfg.w_cross_pitch_birth_penalty

                        if np.isinf(birth_penalty):
                            best_cost_c = float("inf")
                            best_p_c = (i - 1, -1)
                        else:
                            best_cost_c = dp_missing[i - 1] + cfg.birth_cost + birth_penalty + c_curr["emission"]
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

                            c_prev = processed_candidates[t_prev][j]

                            # Lấy c_prev_prev từ parent_cands[prev_i][j] nếu có
                            c_prev_prev = None
                            g_prev = 1
                            p_info = parent_cands[prev_i][j]
                            if p_info is not None and p_info[1] >= 0:
                                prev2_i, prev2_j = p_info
                                t_prev2 = seg_start + prev2_i
                                if prev2_j < len(processed_candidates[t_prev2]):
                                    c_prev_prev = processed_candidates[t_prev2][prev2_j]
                                    g_prev = max(1, t_prev - t_prev2)

                            trans_cost = self._compute_transition_cost(
                                c_prev=c_prev,
                                c_curr=c_curr,
                                g=g,
                                c_prev_prev=c_prev_prev,
                                g_prev=g_prev,
                                match_centroid_x=mc_x,
                            )
                            if not np.isfinite(trans_cost):
                                continue

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

    def _compute_transition_cost(
        self,
        c_prev: Dict[str, Any],
        c_curr: Dict[str, Any],
        g: int,
        c_prev_prev: Optional[Dict[str, Any]] = None,
        g_prev: int = 1,
        match_centroid_x: Optional[float] = None,
    ) -> float:
        """
        Tính toán chi phí chuyển trạng thái giữa 2 ứng viên từ frame t - g sang t.
        Bao gồm:
          1. Chi phí missing tích lũy: (g - 1) * missing_cost
          2. Rào cản tầm với theo gap (Gap Reach Barrier): R(g) = v_cruise * g + reach_margin
          3. Rào cản vận tốc vật lý (Kinematic Speed Barrier): quadratic penalty khi v > v_max, chặn cứng > 1.8 * v_max
          4. Rào cản Máy trạng thái sở hữu (Possession State Machine Rules):
             - Dẫn bóng cùng cầu thủ: thưởng w_possession_retain_bonus khi v <= v_dribble_max
             - Chuyển sang cầu thủ khác: cấm teleport 1-frame xuyên sân (> 80px),
               cấm nhảy vào giày thủ môn đứng cách xa tâm bóng > gk_isolation_dist_px
             - Rời chân cầu thủ: cấm biến thành đốm tĩnh cô lập
          5. Rào cản góc lệch hướng bay (Collinear Flight Barrier):
             - Khi bóng bay tự do, phạt hoặc cấm bẻ ngoặt quỹ đạo vào đốm trắng giả (cos(theta) < cos(35 deg))
        """
        cfg = self.cfg
        ref_prev = c_prev["ref"]
        ref_curr = c_curr["ref"]
        dist = float(np.linalg.norm(ref_curr - ref_prev))
        v = dist / float(g)

        # 1. Rào cản tầm với theo gap: R(g) = v_cruise * g + reach_margin
        if cfg.enable_gap_reach_limit:
            r_max = cfg.v_cruise * float(g) + cfg.reach_margin
            if dist > r_max:
                return float("inf")

        # 2. Rào cản vận tốc vật lý
        if v > 1.8 * cfg.v_max:
            return float("inf")
        v_penalty = cfg.w_speed * (((v - cfg.v_max) / cfg.v_max) ** 2) if v > cfg.v_max else 0.0

        # 3. Rào cản Máy trạng thái sở hữu bóng
        c_poss = 0.0
        if cfg.enable_possession_barrier:
            p_prev_id = c_prev.get("possessor_id")
            p_curr_id = c_curr.get("possessor_id")

            if p_prev_id is not None:
                # Đang ở chân cầu thủ A
                if p_curr_id == p_prev_id:
                    # Cùng cầu thủ A dẫn bóng (dribble)
                    if v <= cfg.v_dribble_max:
                        c_poss -= cfg.w_possession_retain_bonus
                    else:
                        c_poss += cfg.w_speed * (((v - cfg.v_dribble_max) / cfg.v_dribble_max) ** 2)
                elif p_curr_id is not None:
                    # Chuyển sang cầu thủ B
                    if g == 1 and dist > 80.0:
                        return float("inf")  # Cấm dịch chuyển tức thời 1-frame xuyên sân
                    # Kiểm tra thủ môn đứng xa (Goalkeeper Goalmouth Guard)
                    mc_x = match_centroid_x if match_centroid_x is not None else c_curr.get("match_centroid_x")
                    if c_curr.get("is_gk", False) and mc_x is not None:
                        if abs(c_curr["center"][0] - mc_x) > cfg.gk_isolation_dist_px:
                            return float("inf")
                else:
                    # Chuyển từ chân cầu thủ ra khoảng trống
                    if c_curr.get("is_isolated_static", False):
                        return float("inf")

        # 4. Rào cản góc lệch hướng bay (Collinear Flight Barrier)
        c_flight = 0.0
        if cfg.enable_flight_collinearity and c_prev_prev is not None and g_prev > 0:
            if c_prev.get("possessor_id") is None and not c_prev.get("in_foot", False):
                ref_prev_prev = c_prev_prev["ref"]
                v_in = (ref_prev - ref_prev_prev) / float(g_prev)
                v_out = (ref_curr - ref_prev) / float(g)
                norm_in = float(np.linalg.norm(v_in))
                norm_out = float(np.linalg.norm(v_out))
                if norm_in > 1.5 and norm_out > 1.5:
                    cos_theta = float(np.dot(v_in, v_out) / (norm_in * norm_out))
                    cos_theta = max(-1.0, min(1.0, cos_theta))
                    cos_thresh = float(np.cos(np.radians(cfg.max_flight_deflection_deg)))
                    cos_hard = float(np.cos(np.radians(50.0)))
                    if cos_theta < cos_hard:
                        return float("inf")
                    elif cos_theta < cos_thresh:
                        c_flight = cfg.w_flight_deflection * ((1.0 - cos_theta) ** 2)

        return (g - 1) * cfg.missing_cost + v_penalty + c_poss + c_flight
