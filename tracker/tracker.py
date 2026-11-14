from ultralytics import YOLO
from trackers import ByteTrackTracker
from trackers import BoTSORTTracker
import supervision as sv 
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from utils import get_center_of_box, get_bbox_width, get_foot_position


class Tracker:
    def __init__(
        self,
        model_path,
        fps,
        player_conf=0.35,
        goalkeeper_conf=0.30,
        referee_conf=0.30,
        ball_conf=0.01,
        track_thresh=0.2,
        batch_size=8,
        det_conf=None,  # Hỗ trợ tương thích ngược
        device=None,
        reid_min_conf=None,
        draw_interpolated_ball=True,
        ball_selector="viterbi",
        ball_floor=0.01,
        ball_top_k=10,
        ball_max_skip_g=30,
        ball_missing_cost=2.3,
        ball_birth_cost=1.5,
        ball_v_max=50.0,
        ball_enable_gap_reach_limit=True,
        ball_imgsz=1920,
        reid_config=None,
        calib_path="calib/points.json",
    ):
        import torch
        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.reid_min_conf = reid_min_conf
        self.draw_interpolated_ball = draw_interpolated_ball
        self.ball_selector = ball_selector
        self.ball_imgsz = ball_imgsz

        from tracker.viterbi_ball import ViterbiBallSelector, ViterbiConfig, get_pitch_reference_polygon
        pitch_polygon = get_pitch_reference_polygon(calib_path=calib_path)
        viterbi_cfg = ViterbiConfig(
            floor=ball_floor,
            top_k=ball_top_k,
            max_skip_g=ball_max_skip_g,
            missing_cost=ball_missing_cost,
            birth_cost=ball_birth_cost,
            v_max=ball_v_max,
            enable_gap_reach_limit=ball_enable_gap_reach_limit,
            pitch_polygon=pitch_polygon,
            pitch_margin_px=50.0,
        )
        self.viterbi_selector = ViterbiBallSelector(viterbi_cfg)

        self.model = YOLO(model_path)   
        self.tracker = ByteTrackTracker(
            lost_track_buffer=70,
            frame_rate=fps,
            track_activation_threshold=track_thresh,   
            high_conf_det_threshold=track_thresh,     
            minimum_iou_threshold=0.2,                
            # minimum_consecutive_frames=3,
        )
        self.BoTTracker = BoTSORTTracker(
            lost_track_buffer=70,
            frame_rate=fps,
            track_activation_threshold=track_thresh,   
            high_conf_det_threshold=track_thresh,     
            minimum_iou_threshold_first_assoc=0.3,       
            minimum_iou_threshold_second_assoc=0.5,     
            minimum_iou_threshold_unconfirmed_assoc=0.3, 
            enable_cmc=True,                             
            cmc_method='sparseOptFlow',                  
            cmc_downscale=2,                             
            instant_first_frame_activation=True,         
        )

        from algorithmic_reid.reid_tracker_facade import ReIDTrackerFacade
        from algorithmic_reid.config import ReIDConfig
        if reid_config is None:
            reid_cfg = ReIDConfig(
                fps=fps,
                high_conf_thresh=player_conf,
                low_conf_thresh=track_thresh,
                init_conf_thresh=player_conf,
            )
        else:
            reid_cfg = reid_config
        self.reid_tracker = ReIDTrackerFacade(config=reid_cfg, fps=fps)

        # Thiết lập ngưỡng confidence riêng biệt cho 4 class
        if det_conf is not None:
            self.player_conf = det_conf
            self.goalkeeper_conf = det_conf
            self.referee_conf = det_conf
        else:
            self.player_conf = player_conf
            self.goalkeeper_conf = goalkeeper_conf
            self.referee_conf = referee_conf

        self.ball_conf = ball_conf
        self.det_conf = self.player_conf
        self.batch_size = batch_size

        # Lịch sử vai trò của từng tracklet để bỏ phiếu thời gian (Tầng 2)
        self.track_role_history = defaultdict(list)

        # Ngưỡng thấp nhất để YOLO predict không bỏ sót class nào
        low_c = getattr(self.reid_tracker.config, "low_conf_thresh", 0.20) if hasattr(self, "reid_tracker") and hasattr(self.reid_tracker, "config") else 0.20
        min_confs = [self.player_conf, self.goalkeeper_conf, self.referee_conf, self.ball_conf, low_c]
        if self.reid_min_conf is not None:
            min_confs.append(self.reid_min_conf)
        if self.ball_selector == "viterbi":
            min_confs.append(self.viterbi_selector.cfg.floor)
            self.min_predict_conf = max(0.01, min(min_confs))
        else:
            self.min_predict_conf = max(0.10, min(min_confs) - 0.05)

    def _suppress_cross_class_overlaps(self, det, ids, iou_thresh=0.45, min_player_conf=None, min_gk_conf=None, min_ref_conf=None):
        """
        [TẦNG 1]: Khử trùng lặp đa nhãn giữa Cầu thủ - Thủ môn - Trọng tài.
        Quy tắc ưu tiên (Priority Suppression):
          Referee (3) > Goalkeeper (2) > Player (1)
        Khi 2 box đè nhau (IoU >= iou_thresh):
          - Nếu trùng Referee & Player -> Xóa Player, giữ Referee.
          - Nếu trùng Goalkeeper & Player -> Xóa Player, giữ Goalkeeper.
          - Nếu cùng loại -> Giữ box có confidence cao hơn.
        """
        p_c = self.player_conf if min_player_conf is None else min_player_conf
        gk_c = self.goalkeeper_conf if min_gk_conf is None else min_gk_conf
        ref_c = self.referee_conf if min_ref_conf is None else min_ref_conf
        is_player = (det.class_id == ids["player"]) & (det.confidence >= p_c)
        is_gk = (det.class_id == ids["goalkeeper"]) & (det.confidence >= gk_c)
        is_ref = (det.class_id == ids["referee"]) & (det.confidence >= ref_c)

        valid_idx = np.where(is_player | is_gk | is_ref)[0]
        if len(valid_idx) <= 1:
            return det[valid_idx]

        boxes = det.xyxy[valid_idx]
        classes = det.class_id[valid_idx]
        confs = det.confidence[valid_idx]

        # Tính ma trận IoU
        x1 = np.maximum(boxes[:, 0][:, None], boxes[:, 0][None, :])
        y1 = np.maximum(boxes[:, 1][:, None], boxes[:, 1][None, :])
        x2 = np.minimum(boxes[:, 2][:, None], boxes[:, 2][None, :])
        y2 = np.minimum(boxes[:, 3][:, None], boxes[:, 3][None, :])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = areas[:, None] + areas[None, :] - inter
        ious = np.where(union > 0, inter / union, 0.0)

        ref_id = ids["referee"]
        gk_id = ids["goalkeeper"]

        priorities = np.array([
            3 if c == ref_id else (2 if c == gk_id else 1)
            for c in classes
        ])

        suppressed = set()
        n = len(valid_idx)
        for i in range(n):
            if i in suppressed:
                continue
            for j in range(i + 1, n):
                if j in suppressed:
                    continue
                if ious[i, j] >= iou_thresh:
                    if priorities[i] > priorities[j]:
                        suppressed.add(j)
                    elif priorities[j] > priorities[i]:
                        suppressed.add(i)
                        break
                    else:
                        if confs[i] >= confs[j]:
                            suppressed.add(j)
                        else:
                            suppressed.add(i)
                            break

        keep_local = [i for i in range(n) if i not in suppressed]
        return det[valid_idx[keep_local]]

    def add_position_to_track(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object_name == 'ball':
                        position = get_center_of_box(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_name][frame_num][track_id]['position'] = position

    def interpolate_ball_position(self, ball_positions, camera_motion=None, player_tracks=None, *args, **kwargs):
        """
        [Milestone P1] Ổn định quỹ đạo bóng và nội suy thích nghi 3 tầng.
        Sử dụng BallTrajectoryStabilizer:
        - Lọc ngoại lai động học (v_max = 50 px/f).
        - Bộ lọc tính nhất quán 2-frame & bù trừ chuyển động camera.
        - Nội suy PCHIP cho dropout ngắn (1-4 frames).
        - Kiểm tra động học cho dropout vừa (5-10 frames).
        - Rào chắn STRICT NO-OP cho dropout dài (>10 frames).
        - Làm mượt hồi quy ballistic có chặn sai lệch <= 2.5 px.
        """
        from tracker.ball_tracker import BallTrajectoryStabilizer
        stabilizer = BallTrajectoryStabilizer()
        return stabilizer.process(
            ball_tracks=ball_positions,
            camera_motion=camera_motion,
            player_tracks=player_tracks,
        )

    def interpolate_player_trajectories(self, tracks, max_gap=15):
        """
        [P1 Gap-Fill]: Smooth and linearly interpolate player and referee bounding boxes
        across short occlusion / detection dropouts (1 <= gap <= max_gap frames).
        Ensures continuous bounding box representation without flickering.
        """
        for cat in ("players", "referees"):
            cat_list = tracks.get(cat, [])
            total_frames = len(cat_list)
            if total_frames < 2:
                continue

            id_obs = defaultdict(list)
            for f_idx, frame_data in enumerate(cat_list):
                for tid, entry in frame_data.items():
                    if not entry.get("is_interpolated", False):
                        id_obs[tid].append((f_idx, entry))

            for tid, obs_list in id_obs.items():
                if len(obs_list) < 2:
                    continue
                for i in range(len(obs_list) - 1):
                    f1, e1 = obs_list[i]
                    f2, e2 = obs_list[i + 1]
                    gap = f2 - f1
                    if 1 < gap <= max_gap:
                        b1 = np.array(e1["bbox"], dtype=float)
                        b2 = np.array(e2["bbox"], dtype=float)
                        # Rào chắn động học: cấm nội suy bước nhảy vận tốc phi thực tế (> 35 px/f)
                        c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
                        c2 = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
                        vel = np.hypot(c2[0] - c1[0], c2[1] - c1[1]) / float(gap)
                        if vel > 35.0:
                            continue
                        t1 = e1.get("team") if e1.get("team") is not None else e2.get("team")
                        tc1 = e1.get("team_color") if e1.get("team_color") is not None else e2.get("team_color")
                        is_gk = e1.get("is_goalkeeper", False) or e2.get("is_goalkeeper", False)
                        for f in range(f1 + 1, f2):
                            alpha = float(f - f1) / float(gap)
                            interp_box = (1.0 - alpha) * b1 + alpha * b2
                            interp_box_list = interp_box.tolist()
                            interp_entry = {
                                "bbox": interp_box_list,
                                "position": get_foot_position(interp_box_list),
                                "confidence": 0.5,
                                "is_interpolated": True,
                            }
                            if t1 is not None:
                                interp_entry["team"] = t1
                            if tc1 is not None:
                                interp_entry["team_color"] = tc1
                            if is_gk:
                                interp_entry["is_goalkeeper"] = True
                            cat_list[f][tid] = interp_entry

    def detect_frame(self, list_frame):
        batch_size = self.batch_size
        device = self.device
        batch = []

        separate_ball = (self.ball_imgsz is not None and self.ball_imgsz != 1536)

        for frame in list_frame:
            batch.append(frame)

            if len(batch) == batch_size:
                results = list(self.model.predict(
                    source=batch,
                    stream=False,
                    verbose=False,
                    conf=self.min_predict_conf,
                    classes=None,
                    device=device,
                    imgsz=1536,
                    agnostic_nms=False
                ))
                if separate_ball:
                    ball_results = list(self.model.predict(
                        source=batch,
                        stream=False,
                        verbose=False,
                        conf=min(self.ball_conf, 0.01),
                        classes=[0],
                        device=device,
                        imgsz=self.ball_imgsz,
                        agnostic_nms=False
                    ))
                    batch = []
                    for r, br in zip(results, ball_results):
                        yield (r, br)
                else:
                    batch = []
                    yield from results

        if batch:
            results = list(self.model.predict(
                source=batch,
                stream=False,
                verbose=False,
                conf=self.min_predict_conf,
                classes=None,
                device=device,
                imgsz=1536,
                agnostic_nms=False
            ))
            if separate_ball:
                ball_results = list(self.model.predict(
                    source=batch,
                    stream=False,
                    verbose=False,
                    conf=min(self.ball_conf, 0.01),
                    classes=[0],
                    device=device,
                    imgsz=self.ball_imgsz,
                    agnostic_nms=False
                ))
                batch = []
                for r, br in zip(results, ball_results):
                    yield (r, br)
            else:
                batch = []
                yield from results

    def process_pass1(
        self,
        list_frame,
        camera_motion=None,
        team_assigner=None,
        read_from_stub=False,
        stub_path=None,
        camera_stub_path=None,
        choose_Al=2,
        pitch_mapper=None,
        video_path=None,
    ):
        """
        Pass 1 thống nhất với Hệ thống 3 Tầng:
          1. Lọc độc lập 4 class & Khử trùng lặp đa nhãn (Tầng 1).
          2. BoT-SORT / ByteTrack tracking.
          3. Bỏ phiếu chuỗi thời gian khóa cố định vai trò (Tầng 2).
          4. Ước lượng chuyển động Camera Motion.
          5. Team Assigner trực tiếp từ frame (Phương án A).
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            motion = None
            if camera_motion is not None and camera_stub_path is not None and os.path.exists(camera_stub_path):
                with open(camera_stub_path, 'rb') as f:
                    motion = pickle.load(f)
            return tracks, motion

        tracks = {"players": [], "referees": [], "ball": []}
        ids = None
        ball_state = {
            "last_center": None,
            "last_frame": None,
            "velocity": np.zeros(2, dtype=float),
        }

        self.track_role_history.clear()
        self.reid_tracker.reset()
        if self.ball_selector == "viterbi":
            self.viterbi_selector.reset()
        if camera_motion is not None and hasattr(camera_motion, "reset"):
            camera_motion.reset()

        prev_boxes = []
        buffered_early_frames = []

        for frame_index, det_item in enumerate(self.detect_frame(list_frame)):
            if isinstance(det_item, tuple):
                d_frame, b_res = det_item
            else:
                d_frame, b_res = det_item, None

            frame = d_frame.orig_img
            if ids is None:
                ids = {v: k for k, v in d_frame.names.items()}

            det = sv.Detections.from_ultralytics(d_frame)

            # 1. Quả bóng (lọc riêng)
            if b_res is not None:
                ball_det = sv.Detections.from_ultralytics(b_res)
                # Loại bỏ bóng khỏi det chính để tránh trùng lặp
                is_not_ball = det.class_id != ids["ball"]
                det = det[is_not_ball]
            else:
                is_ball = det.class_id == ids["ball"]
                ball_det = det[is_ball]

            # 2. Camera Motion step (dùng prev_boxes từ frame trước)
            G_t = None
            if camera_motion is not None:
                camera_motion.step(frame, prev_boxes)
                G_t = getattr(camera_motion, "G_prev", None)

            if self.ball_selector == "viterbi":
                self.viterbi_selector.add_frame_candidates(frame_index, ball_det, ids, camera_motion=camera_motion)

            # 3. Tracking Engine Selection
            if choose_Al == 0:
                people = self._suppress_cross_class_overlaps(det, ids, iou_thresh=0.45)
                tracked = self.tracker.update(people)
                tracked = tracked[tracked.tracker_id != -1]
            elif choose_Al == 1:
                people = self._suppress_cross_class_overlaps(det, ids, iou_thresh=0.45)
                tracked = self.BoTTracker.update(people, frame=frame)
                tracked = tracked[tracked.tracker_id != -1]
            else:
                # choose_Al == 2: Algorithmic Re-ID Tracker (P0)
                if self.reid_min_conf is not None:
                    p_conf = self.reid_min_conf
                    gk_conf = min(self.reid_min_conf, self.goalkeeper_conf)
                    ref_conf = min(self.reid_min_conf, self.referee_conf)
                else:
                    low_c = getattr(self.reid_tracker.config, "low_conf_thresh", 0.20) if hasattr(self, "reid_tracker") and hasattr(self.reid_tracker, "config") else 0.20
                    p_conf = min(self.player_conf, low_c)
                    gk_conf = min(self.goalkeeper_conf, low_c)
                    ref_conf = min(self.referee_conf, low_c)

                is_person = (
                    ((det.class_id == ids["player"]) & (det.confidence >= p_conf)) |
                    ((det.class_id == ids["goalkeeper"]) & (det.confidence >= gk_conf)) |
                    ((det.class_id == ids["referee"]) & (det.confidence >= ref_conf))
                )
                person_det = det[is_person]
                people = self._suppress_cross_class_overlaps(person_det, ids, iou_thresh=0.45,
                                                             min_player_conf=p_conf,
                                                             min_gk_conf=gk_conf,
                                                             min_ref_conf=ref_conf)
                H_pitch = None
                if pitch_mapper is not None:
                    try:
                        H_pitch = pitch_mapper.Ht(frame_index)
                    except Exception:
                        H_pitch = None
                elif camera_motion is not None and hasattr(camera_motion, "relocalizer") and getattr(camera_motion.relocalizer, "anchors", None):
                    first_anchor = next(iter(camera_motion.relocalizer.anchors.values()))
                    if first_anchor.H_pitch is not None and getattr(camera_motion, "M", None) is not None:
                        try:
                            H_pitch = first_anchor.H_pitch @ camera_motion.M
                        except Exception:
                            H_pitch = None

                tracked = self.reid_tracker.update(
                    people,
                    frame=frame,
                    camera_homography=G_t,
                    team_assigner=team_assigner,
                    pitch_homography=H_pitch,
                )

            players, referees, ball = {}, {}, {}

            if choose_Al == 2:
                players = tracked["players"]
                referees = tracked["referees"]

                # [Cross-Class Deduplication]: Khử trùng lặp triệt để giữa Player và Referee (IoU >= 0.35)
                # Đảm bảo 1 cơ thể người tuyệt đối không bao giờ bị gắn song song 2 ID
                for rid, rdata in list(referees.items()):
                    r_box = rdata.get("bbox")
                    if r_box is None:
                        continue
                    for pid, pdata in list(players.items()):
                        p_box = pdata.get("bbox")
                        if p_box is not None:
                            x1, y1 = max(r_box[0], p_box[0]), max(r_box[1], p_box[1])
                            x2, y2 = min(r_box[2], p_box[2]), min(r_box[3], p_box[3])
                            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                            a1 = (r_box[2] - r_box[0]) * (r_box[3] - r_box[1])
                            a2 = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                            iou = inter / (a1 + a2 - inter + 1e-6)
                            if iou >= 0.35:
                                # Trùng lặp: Ưu tiên Referee nếu tự tin hơn hoặc giữ 1 ID duy nhất
                                if rdata.get("confidence", 0) >= pdata.get("confidence", 0):
                                    players.pop(pid, None)
                                else:
                                    referees.pop(rid, None)
                                break
            else:
                # [TẦNG 2] Bỏ phiếu chuỗi thời gian khóa vai trò theo từng ID (Legacy BoT-SORT / ByteTrack)
                for xyxy, cls_id, tid in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
                    entry = {"bbox": xyxy.tolist()}
                    int_tid = int(tid)

                    self.track_role_history[int_tid].append(cls_id)
                    history = self.track_role_history[int_tid]
                    total_obs = len(history)

                    ref_count = history.count(ids["referee"])
                    gk_count = history.count(ids["goalkeeper"])

                    if total_obs >= 3:
                        if ref_count / total_obs >= 0.25:
                            final_role = ids["referee"]
                        elif gk_count / total_obs >= 0.30:
                            final_role = ids["goalkeeper"]
                        else:
                            final_role = ids["player"]
                    else:
                        if ids["referee"] in history:
                            final_role = ids["referee"]
                        elif ids["goalkeeper"] in history:
                            final_role = ids["goalkeeper"]
                        else:
                            final_role = cls_id

                    if final_role == ids["referee"]:
                        referees[int_tid] = entry
                    elif final_role == ids["goalkeeper"]:
                        entry["is_goalkeeper"] = True
                        players[int_tid] = entry
                    else:
                        players[int_tid] = entry

            # --- Chọn ball bằng gating & vận tốc ---
            selected_ball = self._select_ball_detection(
                ball_det=ball_det,
                frame_index=frame_index,
                frame_shape=frame.shape,
                state=ball_state,
            )
            if selected_ball is not None:
                ball[1] = selected_ball

            tracks["players"].append(players)
            tracks["referees"].append(referees)
            tracks["ball"].append(ball)

            prev_boxes = (
                [v["bbox"] for v in players.values()]
                + [v["bbox"] for v in referees.values()]
                + [v["bbox"] for v in ball.values()]
            )

            # --- 2. Team Assigner step (Phương án A: bảo toàn nguyên bản 100%) ---
            if team_assigner is not None and referees:
                for r_id, r_track in referees.items():
                    if r_track.get("confidence", 0) >= self.referee_conf:
                        team_assigner.record_referee_color(frame, r_track["bbox"])

            if team_assigner is not None:
                # Chỉ lấy các cầu thủ outfield (loại trừ thủ môn nếu có) để không làm lệch K-Means
                outfield_players = {
                    pid: p for pid, p in players.items() if not p.get("is_goalkeeper", False)
                }
                if not outfield_players:
                    outfield_players = players

                # Bảo vệ K-Means: Chỉ dùng cầu thủ có detection tin cậy (conf >= player_conf) để fit màu áo
                fit_players = {
                    pid: p for pid, p in outfield_players.items()
                    if p.get("confidence", 1.0) >= self.player_conf
                }
                if not fit_players:
                    fit_players = outfield_players

                if 1 not in team_assigner.team_colors or 2 not in team_assigner.team_colors:
                    buffered_early_frames.append((frame_index, frame, fit_players))
                    if frame_index == 3 or (frame_index >= 3 and len(fit_players) >= 2):
                        team_assigner.assign_team_color(frame, fit_players)
                        if 1 in team_assigner.team_colors and 2 in team_assigner.team_colors:
                            for b_idx, b_frame, b_players in buffered_early_frames:
                                for p_id, p_track in b_players.items():
                                    team = team_assigner.get_player_team(b_frame, p_track['bbox'], p_id)
                                    tracks["players"][b_idx][p_id]['team'] = team
                                    tracks["players"][b_idx][p_id]['team_color'] = team_assigner.team_colors[team]
                            buffered_early_frames.clear()
                else:
                    for p_id, p_track in outfield_players.items():
                        # Chỉ bỏ phiếu màu áo nếu detection đủ tin cậy, tránh nhiễu do che khuất
                        if p_track.get("confidence", 1.0) >= self.player_conf:
                            team = team_assigner.get_player_team(frame, p_track['bbox'], p_id)
                        else:
                            team = team_assigner.player_team_dict.get(p_id, 1)
                        tracks["players"][frame_index][p_id]['team'] = team
                        tracks["players"][frame_index][p_id]['team_color'] = team_assigner.team_colors[team]

            if (frame_index + 1) % 50 == 0:
                print(f"   -> [Pass 1/2 Tracking] Đã xử lý {frame_index + 1} frames...")

        # Xử lý trường hợp video cực ngắn (< 4 frames) nếu buffer chưa xả
        if team_assigner is not None and buffered_early_frames:
            best_f = max(buffered_early_frames, key=lambda x: len(x[2]))
            team_assigner.assign_team_color(best_f[1], best_f[2])
            for b_idx, b_frame, b_players in buffered_early_frames:
                for p_id, p_track in b_players.items():
                    team = team_assigner.get_player_team(b_frame, p_track['bbox'], p_id) if 1 in team_assigner.team_colors else 1
                    tracks["players"][b_idx][p_id]['team'] = team
                    tracks["players"][b_idx][p_id]['team_color'] = team_assigner.team_colors.get(team, (0, 0, 255))
            buffered_early_frames.clear()

        # Sau Pass 1: Nếu dùng Viterbi DP, chọn lại quỹ đạo bóng toàn cục trên top-K ứng viên trong RAM
        if self.ball_selector == "viterbi":
            tracks["ball"] = self.viterbi_selector.select_trajectory(
                total_frames=len(tracks["players"]),
                camera_motion=camera_motion,
                player_tracks=tracks.get("players"),
                video_path=video_path,
                frames=list_frame if isinstance(list_frame, list) else None,
            )

        # [Direction C: Global Tracklet Association]: Chia tách tracklet bất thường & ghép nối toàn cục
        if getattr(self.reid_tracker.config, "use_gta", False):
            print("   -> [Post-Process] Đang kích hoạt Direction C: Global Tracklet Association (GTA)...")
            tracks = self.reid_tracker.postprocess_gta(tracks)

        # [P1 Gap-Fill]: Nội suy mượt các khoảng ngắt ngắn (1 đến 15 frames) để bám sát liên tục không nhấp nháy
        self.interpolate_player_trajectories(tracks, max_gap=15)

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        motion = camera_motion.get_result(camera_stub_path) if camera_motion is not None else None
        return tracks, motion

    def get_object_track(
        self,
        list_frame,
        read_from_stub=False,
        stub_path=None,
        choose_Al=0,
    ):
        """Hàm lấy object track truyền thống (tương thích ngược với Tầng 1 & Tầng 2)."""
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        tracks = {"players": [], "referees": [], "ball": []}
        ids = None

        ball_state = {
            "last_center": None,
            "last_frame": None,
            "velocity": np.zeros(2, dtype=float),
        }

        self.track_role_history.clear()
        self.reid_tracker.reset()

        for frame_index, d_frame in enumerate(self.detect_frame(list_frame)):
            if ids is None:
                ids = {v: k for k, v in d_frame.names.items()}

            det = sv.Detections.from_ultralytics(d_frame)

            is_ball = det.class_id == ids["ball"]
            ball_det = det[is_ball]

            if choose_Al == 0:
                people = self._suppress_cross_class_overlaps(det, ids, iou_thresh=0.45)
                tracked = self.tracker.update(people)
                tracked = tracked[tracked.tracker_id != -1]
            elif choose_Al == 1:
                people = self._suppress_cross_class_overlaps(det, ids, iou_thresh=0.45)
                tracked = self.BoTTracker.update(people, frame=d_frame.orig_img)
                tracked = tracked[tracked.tracker_id != -1]
            else:
                low_c = getattr(self.reid_tracker.config, "low_conf_thresh", 0.20) if hasattr(self, "reid_tracker") and hasattr(self.reid_tracker, "config") else 0.20
                p_c = min(self.player_conf, low_c)
                gk_c = min(self.goalkeeper_conf, low_c)
                ref_c = min(self.referee_conf, low_c)
                is_person = (
                    ((det.class_id == ids["player"]) & (det.confidence >= p_c)) |
                    ((det.class_id == ids["goalkeeper"]) & (det.confidence >= gk_c)) |
                    ((det.class_id == ids["referee"]) & (det.confidence >= ref_c))
                )
                person_det = det[is_person]
                people = self._suppress_cross_class_overlaps(person_det, ids, iou_thresh=0.45,
                                                             min_player_conf=p_c,
                                                             min_gk_conf=gk_c,
                                                             min_ref_conf=ref_c)
                tracked = self.reid_tracker.update(people, frame=d_frame.orig_img)

            players, referees, ball = {}, {}, {}

            if choose_Al == 2:
                players = tracked["players"]
                referees = tracked["referees"]

                # [Cross-Class Deduplication]: Khử trùng lặp triệt để giữa Player và Referee (IoU >= 0.35)
                # Đảm bảo 1 cơ thể người tuyệt đối không bao giờ bị gắn song song 2 ID
                for rid, rdata in list(referees.items()):
                    r_box = rdata.get("bbox")
                    if r_box is None:
                        continue
                    for pid, pdata in list(players.items()):
                        p_box = pdata.get("bbox")
                        if p_box is not None:
                            x1, y1 = max(r_box[0], p_box[0]), max(r_box[1], p_box[1])
                            x2, y2 = min(r_box[2], p_box[2]), min(r_box[3], p_box[3])
                            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                            a1 = (r_box[2] - r_box[0]) * (r_box[3] - r_box[1])
                            a2 = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                            iou = inter / (a1 + a2 - inter + 1e-6)
                            if iou >= 0.35:
                                # Trùng lặp: Ưu tiên Referee nếu tự tin hơn hoặc giữ 1 ID duy nhất
                                if rdata.get("confidence", 0) >= pdata.get("confidence", 0):
                                    players.pop(pid, None)
                                else:
                                    referees.pop(rid, None)
                                break
            else:
                for xyxy, cls_id, tid in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
                    entry = {"bbox": xyxy.tolist()}
                    int_tid = int(tid)

                    self.track_role_history[int_tid].append(cls_id)
                    history = self.track_role_history[int_tid]
                    total_obs = len(history)

                    ref_count = history.count(ids["referee"])
                    gk_count = history.count(ids["goalkeeper"])

                    if total_obs >= 3:
                        if ref_count / total_obs >= 0.25:
                            final_role = ids["referee"]
                        elif gk_count / total_obs >= 0.30:
                            final_role = ids["goalkeeper"]
                        else:
                            final_role = ids["player"]
                    else:
                        if ids["referee"] in history:
                            final_role = ids["referee"]
                        elif ids["goalkeeper"] in history:
                            final_role = ids["goalkeeper"]
                        else:
                            final_role = cls_id

                    if final_role == ids["referee"]:
                        referees[int_tid] = entry
                    elif final_role == ids["goalkeeper"]:
                        entry["is_goalkeeper"] = True
                        players[int_tid] = entry
                    else:
                        players[int_tid] = entry

            selected_ball = self._select_ball_detection(
                ball_det=ball_det,
                frame_index=frame_index,
                frame_shape=d_frame.orig_img.shape,
                state=ball_state,
            )

            if selected_ball is not None:
                ball[1] = selected_ball

            tracks["players"].append(players)
            tracks["referees"].append(referees)
            tracks["ball"].append(ball)

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None, is_gk=False):
        if isinstance(color, np.ndarray):
            color = tuple(map(int, color))
        elif isinstance(color, (list, tuple)):
            color = tuple(int(x) for x in color)
        y2 = int(bbox[3])
        x_center, _ = get_center_of_box(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        rectangle_width = 46 if is_gk else 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )
            label = "GK" if is_gk else f"{track_id}"
            x1_text = x1_rect + (11 if is_gk else 12)
            if not is_gk and track_id > 99:
                x1_text -= 10
            cv2.putText(
                frame,
                label,
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5 if is_gk else 0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_box(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2) 
        return frame

    draw_traingle = draw_triangle

    def draw_team_ball_control(self, frame, frame_num, team_ball_control, team_1_cumsum=None, team_2_cumsum=None):
        # Tối ưu bộ nhớ: Alpha blend trực tiếp trên vùng ROI dashboard thay vì copy toàn bộ frame 1080p
        sub_roi = frame[850:970, 1350:1900]
        white_box = np.full_like(sub_roi, 255)
        alpha = 0.4
        cv2.addWeighted(white_box, alpha, sub_roi, 1 - alpha, 0, sub_roi)
        frame[850:970, 1350:1900] = sub_roi

        if team_1_cumsum is not None and team_2_cumsum is not None:
            team_1_num_frames = team_1_cumsum[frame_num]
            team_2_num_frames = team_2_cumsum[frame_num]
        else:
            team_ball_control_till_frame = team_ball_control[: frame_num + 1]
            team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
            team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        total_control = team_1_num_frames + team_2_num_frames
        team_1 = team_1_num_frames / total_control if total_control > 0 else 0.0
        team_2 = team_2_num_frames / total_control if total_control > 0 else 0.0

        cv2.putText(frame, f"Team 1 Ball Control:{team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control:{team_2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_tracked_frames(self, video_frames, tracks, team_ball_control):
        team_1_cumsum = np.cumsum(team_ball_control == 1) if team_ball_control is not None else None
        team_2_cumsum = np.cumsum(team_ball_control == 2) if team_ball_control is not None else None

        for frame_index, frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict = tracks["players"][frame_index]
            ball_dict = tracks["ball"][frame_index]
            referee_dict = tracks["referees"][frame_index]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                is_gk = player.get("is_goalkeeper", False)
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id, is_gk=is_gk)
                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 234, 123), track_id=None)

            previous_ball_dict = (
                tracks["ball"][frame_index - 1]
                if frame_index > 0
                else {}
            )

            for _, ball in ball_dict.items():
                is_interp = ball.get("is_interpolated", False)
                if is_interp and not self.draw_interpolated_ball:
                    continue

                display_bbox = np.asarray(ball["bbox"], dtype=float).copy()

                if is_interp:
                    # Vẽ bóng nội suy: vòng rỗng có viền (hollow ring) khác biệt rõ với tam giác đặc của bóng thật
                    cx = int((display_bbox[0] + display_bbox[2]) / 2)
                    cy = int((display_bbox[1] + display_bbox[3]) / 2)
                    cv2.circle(frame, (cx, cy), 9, (0, 215, 255), 2, cv2.LINE_AA)
                    cv2.circle(frame, (cx, cy), 2, (0, 215, 255), -1, cv2.LINE_AA)
                    triangle_points = np.array([
                        [cx, cy - 10],
                        [cx - 8, cy - 22],
                        [cx + 8, cy - 22]
                    ])
                    cv2.drawContours(frame, [triangle_points], 0, (0, 215, 255), cv2.FILLED)
                    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 1, cv2.LINE_AA)
                    continue

                prev_fi = frame_index - 1
                previous_ball = previous_ball_dict.get(1) if prev_fi >= 0 else None
                prev_is_consecutive = (
                    previous_ball is not None
                    and not previous_ball.get("is_interpolated", False)
                    and tracks["ball"][prev_fi].get(1) is not None
                )

                if prev_is_consecutive:
                    prev_bbox = np.asarray(previous_ball["bbox"], dtype=float)
                    prev_center = np.array([
                        (prev_bbox[0] + prev_bbox[2]) / 2,
                        (prev_bbox[1] + prev_bbox[3]) / 2,
                    ])
                    curr_center = np.array([
                        (display_bbox[0] + display_bbox[2]) / 2,
                        (display_bbox[1] + display_bbox[3]) / 2,
                    ])
                    motion = curr_center - prev_center
                    speed = float(np.linalg.norm(motion))

                    if speed >= 12.0:
                        lead_frames = 0.35
                        offset = motion * lead_frames
                        height, width = frame.shape[:2]
                        max_offset = min(35.0, 0.02 * np.hypot(width, height))
                        offset_length = float(np.linalg.norm(offset))
                        if offset_length > max_offset:
                            offset *= max_offset / offset_length
                        display_bbox[[0, 2]] += offset[0]
                        display_bbox[[1, 3]] += offset[1]

                frame = self.draw_triangle(frame, display_bbox.tolist(), (255, 0, 0))

            if team_ball_control is not None:
                frame = self.draw_team_ball_control(frame, frame_index, team_ball_control, team_1_cumsum, team_2_cumsum)

            if (frame_index + 1) % 100 == 0:
                print(f"   -> [Pass 2/2 Annotation] Đã render {frame_index + 1} frames...")
            yield frame

    def _select_ball_detection(
        self,
        ball_det,
        frame_index,
        frame_shape,
        state,
    ):
        """
        [Approach 1 - Simplified]
        Không dùng pending confirmation 2-frame.
        Chấp nhận detection ngay nếu conf >= ball_conf hoặc nằm trong gate dự đoán.
        Gate rộng hơn khi bóng đã mất nhiều frame.
        Bỏ stale_at_old_position filter để không miss bóng chậm sau cú sút.
        """
        h, w = frame_shape[:2]
        diagonal = float(np.hypot(w, h))

        state.setdefault("last_center", None)
        state.setdefault("last_frame", None)
        state.setdefault("velocity", np.zeros(2, dtype=float))

        if len(ball_det) == 0:
            return None

        confs = np.asarray(ball_det.confidence, dtype=float)
        keep = confs >= max(0.10, self.ball_conf - 0.05)
        ball_det = ball_det[keep]
        confs = confs[keep]
        if len(ball_det) == 0:
            return None

        boxes = np.asarray(ball_det.xyxy, dtype=float)
        centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0

        last_center = state["last_center"]
        last_frame = state["last_frame"]
        velocity = state["velocity"]

        # --- Trường hợp chưa có lịch sử ---
        if last_center is None:
            idx = int(np.argmax(confs))
            if float(confs[idx]) >= max(0.20, self.ball_conf):
                state["last_center"] = centers[idx].copy()
                state["last_frame"] = frame_index
                state["velocity"] = np.zeros(2, dtype=float)
                return {"bbox": boxes[idx].tolist(), "confidence": float(confs[idx]), "is_interpolated": False}
            return None

        dt = max(1, frame_index - last_frame)
        speed = float(np.linalg.norm(velocity))
        predicted_center = last_center + velocity * dt

        gate = min(
            max(0.12 * diagonal, speed * 2.5 * dt + 0.04 * diagonal * dt),
            0.40 * diagonal,
        )

        distances = np.linalg.norm(centers - predicted_center, axis=1)
        in_gate = distances <= gate

        if np.any(in_gate):
            scores = confs.copy()
            scores[in_gate] += 0.3 * (1.0 - distances[in_gate] / gate)
            scores[~in_gate] = -1.0
            idx = int(np.argmax(scores))
        else:
            idx = int(np.argmax(confs))
            if float(confs[idx]) < max(0.35, self.ball_conf + 0.15):
                return None

        center = centers[idx].copy()
        measured_v = (center - last_center) / dt
        max_speed = min(50.0 * dt, 0.40 * diagonal)
        mv_speed = float(np.linalg.norm(measured_v))
        if mv_speed > max_speed:
            measured_v *= max_speed / mv_speed

        state["velocity"] = 0.20 * velocity + 0.80 * measured_v
        state["last_center"] = center
        state["last_frame"] = frame_index

        return {"bbox": boxes[idx].tolist(), "confidence": float(confs[idx]), "is_interpolated": False}
