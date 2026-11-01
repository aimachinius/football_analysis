"""
scripts/measure_baseline.py - Script do va kiem chung baseline cho du an CV_Football_Analys.

Ho tro:
  - Do 1 cau hinh don le hoac do toan bo cac cau hinh (--compare-all):
    + Nguoi: baseline (reid_min_conf=None), reid_min_conf=0.20, reid_min_conf=0.15.
    + Possession: Hysteresis N = 0, 3, 5, 8.
  - Thu thap cac chi so bo sung:
    + Proxy ID switch (mot ID LOST roi sinh ID moi trong <=30f gan vi tri cu).
    + So track "ma" (<10 frame ma toan bo detection conf < 0.40).
    + So lan doi chu bong theo ID VA theo doi, do dai trung binh mot luot giu bong.
    + Dem chinh xac so detection dai [0.15, 0.20) vs [0.20, 0.35) va ghi ro Stage 2 bo qua.
  - Xuat contact sheet (moi 5 frame) cho cac doan 165-185, 300-350, 510-545.
"""

import os
import sys
import time
import json
import threading
import argparse
import pickle
from collections import defaultdict
import numpy as np
import psutil
import cv2
import torch

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import read_video
from tracker import Tracker
from camera_motion import CameraMotion
from team_assignment import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from speed_distance import SpeedAndDistance_Calc
from pitch_homo import PitchCalibration, PitchMapper, load_points


class ResourceMonitor:
    """Non-invasive background resource monitor for peak RAM and VRAM."""
    def __init__(self, interval=0.1):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread = None
        self.peak_rss_bytes = 0
        self.peak_vram_alloc_bytes = 0
        self.peak_vram_res_bytes = 0

    def _run(self):
        while not self.stop_event.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss_bytes:
                    self.peak_rss_bytes = rss
            except Exception:
                pass

            if torch.cuda.is_available():
                try:
                    alloc = torch.cuda.memory_allocated()
                    res = torch.cuda.memory_reserved()
                    if alloc > self.peak_vram_alloc_bytes:
                        self.peak_vram_alloc_bytes = alloc
                    if res > self.peak_vram_res_bytes:
                        self.peak_vram_res_bytes = res
                except Exception:
                    pass

            time.sleep(self.interval)

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)


def compute_possession_stats(tracks_players, tracks_ball, hysteresis=0, fps=25.0):
    """
    Tinh toan chi so kiem soat bong voi nguong hysteresis N.
    - Doi chu theo ID
    - Doi chu theo Doi
    - Do dai trung binh 1 luot giu bong (theo ID va theo Doi)
    """
    assigner = PlayerBallAssigner(hysteresis=hysteresis)
    team_control = []
    player_control = []
    id_turnovers = 0
    team_turnovers = 0
    last_player = -1
    last_team = 0

    player_possession_durations = []
    team_possession_durations = []

    curr_p_len = 0
    curr_t_len = 0

    for f_idx, player_track in enumerate(tracks_players):
        ball_bbox = tracks_ball[f_idx][1]["bbox"] if 1 in tracks_ball[f_idx] else None
        p_id = assigner.assign_ball_to_player(player_track, ball_bbox)
        player_control.append(p_id)

        if p_id != -1 and p_id in player_track:
            team_id = player_track[p_id].get("team", 0)
        else:
            team_id = team_control[-1] if team_control else 0
        team_control.append(team_id)

        # ID changes
        if p_id != -1:
            if last_player != -1 and p_id != last_player:
                id_turnovers += 1
                if curr_p_len > 0:
                    player_possession_durations.append(curr_p_len)
                curr_p_len = 1
            else:
                curr_p_len += 1
            last_player = p_id
        else:
            if curr_p_len > 0:
                player_possession_durations.append(curr_p_len)
                curr_p_len = 0

        # Team changes
        if team_id in (1, 2):
            if last_team in (1, 2) and team_id != last_team:
                team_turnovers += 1
                if curr_t_len > 0:
                    team_possession_durations.append(curr_t_len)
                curr_t_len = 1
            else:
                curr_t_len += 1
            last_team = team_id

    if curr_p_len > 0:
        player_possession_durations.append(curr_p_len)
    if curr_t_len > 0:
        team_possession_durations.append(curr_t_len)

    total_time_min = len(tracks_players) / fps / 60.0
    mean_p_dur = float(np.mean(player_possession_durations)) if player_possession_durations else 0.0
    mean_t_dur = float(np.mean(team_possession_durations)) if team_possession_durations else 0.0

    return {
        "hysteresis": hysteresis,
        "id_turnovers": id_turnovers,
        "id_turnovers_per_min": round(id_turnovers / max(0.01, total_time_min), 2),
        "team_turnovers": team_turnovers,
        "team_turnovers_per_min": round(team_turnovers / max(0.01, total_time_min), 2),
        "mean_player_possession_frames": round(mean_p_dur, 2),
        "mean_player_possession_sec": round(mean_p_dur / fps, 2),
        "mean_team_possession_frames": round(mean_t_dur, 2),
        "mean_team_possession_sec": round(mean_t_dur / fps, 2),
        "team_ball_control": team_control,
        "player_control": player_control,
    }


def compute_proxy_id_switches(tracks_players, fps=25.0, max_gap_frames=30, max_dist_px=200.0, max_dist_m=6.0):
    """
    Dem so lan proxy ID switch: mot ID LOST roi sinh ID moi trong <=30f gan vi tri cu.
    Kiem tra tren ca toa do pixel va toa do met tren san (neu co).
    """
    track_info = {}
    for f_idx, frame_players in enumerate(tracks_players):
        for pid, p in frame_players.items():
            bbox = p["bbox"]
            foot_px = ((bbox[0] + bbox[2]) / 2.0, bbox[3])
            pos_m = p.get("position_transformed")
            team = p.get("team", 0)
            conf = p.get("confidence", 1.0)

            if pid not in track_info:
                track_info[pid] = {
                    "start_frame": f_idx,
                    "end_frame": f_idx,
                    "start_foot_px": foot_px,
                    "end_foot_px": foot_px,
                    "start_pos_m": pos_m,
                    "end_pos_m": pos_m,
                    "team": team,
                    "confs": [conf],
                    "count": 1,
                }
            else:
                track_info[pid]["end_frame"] = f_idx
                track_info[pid]["end_foot_px"] = foot_px
                if pos_m is not None:
                    track_info[pid]["end_pos_m"] = pos_m
                if track_info[pid]["start_pos_m"] is None and pos_m is not None:
                    track_info[pid]["start_pos_m"] = pos_m
                track_info[pid]["confs"].append(conf)
                track_info[pid]["count"] += 1

    total_frames = len(tracks_players)
    switches = []
    # Kiem tra cac track bi LOST truoc cuoi video
    for id_a, info_a in track_info.items():
        if info_a["end_frame"] >= total_frames - 5:
            continue  # Con song den cuoi video
        candidates = []
        for id_b, info_b in track_info.items():
            if id_b == id_a:
                continue
            dt = info_b["start_frame"] - info_a["end_frame"]
            if 0 <= dt <= max_gap_frames:
                # Kiem tra khoang cach met neu co, hoac pixel
                matched = False
                dist_val = 0.0
                if info_a["end_pos_m"] is not None and info_b["start_pos_m"] is not None:
                    dm = np.hypot(
                        info_b["start_pos_m"][0] - info_a["end_pos_m"][0],
                        info_b["start_pos_m"][1] - info_a["end_pos_m"][1]
                    )
                    if dm <= max_dist_m:
                        matched = True
                        dist_val = dm
                else:
                    dpx = np.hypot(
                        info_b["start_foot_px"][0] - info_a["end_foot_px"][0],
                        info_b["start_foot_px"][1] - info_a["end_foot_px"][1]
                    )
                    if dpx <= max_dist_px:
                        matched = True
                        dist_val = dpx

                if matched:
                    candidates.append((dist_val, dt, id_b))

        if candidates:
            candidates.sort()
            best_dist, best_dt, best_id_b = candidates[0]
            switches.append({
                "lost_id": id_a,
                "new_id": best_id_b,
                "lost_frame": info_a["end_frame"],
                "new_frame": track_info[best_id_b]["start_frame"],
                "frame_gap": best_dt,
                "distance": round(best_dist, 2),
            })

    # Track "ma": <10 frame ma toan bo detection conf < 0.40
    ghost_tracks = [
        pid for pid, inf in track_info.items()
        if inf["count"] < 10 and all(c < 0.40 for c in inf["confs"])
    ]

    return len(switches), switches, len(ghost_tracks), ghost_tracks, track_info


def generate_contact_sheets(video_path, tracks_dict, output_dir="reports/contact_sheets", step=5):
    """
    Xuat contact sheet (moi 5 frame) cho cac doan 165-185, 300-350, 510-545.
    """
    os.makedirs(output_dir, exist_ok=True)
    ranges = {
        "165_185": (165, 185),
        "300_350": (300, 350),
        "510_545": (510, 545),
    }

    cap = cv2.VideoCapture(video_path)
    thumb_w, thumb_h = 480, 270

    generated_paths = []
    for r_name, (f_start, f_end) in ranges.items():
        frame_list = list(range(f_start, f_end + 1, step))
        thumbnails = []

        for f_idx in frame_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            vis = frame.copy()
            # Draw players
            if f_idx < len(tracks_dict.get("players", [])):
                for pid, p in tracks_dict["players"][f_idx].items():
                    bbox = [int(v) for v in p["bbox"]]
                    color = p.get("team_color", (0, 0, 255))
                    cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    conf = p.get("confidence", 1.0)
                    cv2.putText(vis, f"P{pid} {conf:.2f}", (bbox[0], max(18, bbox[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Draw ball
            if f_idx < len(tracks_dict.get("ball", [])):
                if 1 in tracks_dict["ball"][f_idx]:
                    b = tracks_dict["ball"][f_idx][1]
                    bbox = [int(v) for v in b["bbox"]]
                    is_interp = b.get("is_interpolated", False)
                    cx = (bbox[0] + bbox[2]) // 2
                    cy = (bbox[1] + bbox[3]) // 2
                    if is_interp:
                        cv2.circle(vis, (cx, cy), 10, (0, 215, 255), 2, cv2.LINE_AA)
                        cv2.putText(vis, "BALL(interp)", (bbox[0] - 20, bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
                    else:
                        cv2.circle(vis, (cx, cy), 8, (255, 0, 0), -1, cv2.LINE_AA)
                        cv2.putText(vis, "BALL", (bbox[0] - 10, bbox[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

            # Header box
            cv2.rectangle(vis, (0, 0), (280, 42), (0, 0, 0), -1)
            cv2.putText(vis, f"Frame {f_idx}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            th = cv2.resize(vis, (thumb_w, thumb_h))
            thumbnails.append(th)

        if thumbnails:
            n = len(thumbnails)
            cols = min(4, n)
            rows = (n + cols - 1) // cols
            grid = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
            for i, th in enumerate(thumbnails):
                r = i // cols
                c = i % cols
                grid[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = th

            out_file = os.path.join(output_dir, f"contact_sheet_{r_name}.jpg")
            cv2.imwrite(out_file, grid)
            print(f"-> Da xuat contact sheet: {out_file} ({n} thumbnails)")
            generated_paths.append(out_file)

    cap.release()
    return generated_paths


def measure_single_config(
    video_path="video/sample.mp4",
    model_path=None,
    calib_path="calib/points.json",
    batch_size=8,
    reid_min_conf=None,
    draw_interpolated_ball=False,
    possession_hysteresis=0,
    stub_path=None,
    load_stub=False,
):
    """
    Chay pipeline cho mot cau hinh nhat dinh va thu thap so lieu day du.
    """
    if model_path is None:
        model_path = "best_V26_over_6000.pt" if os.path.exists("best_V26_over_6000.pt") else "best_YOLOv26s_FB_OD.pt"

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # Track instrumentation counters
    raw_person_detections_total = 0
    dropped_person_015_035_count = 0
    dropped_person_015_020_count = 0
    dropped_person_020_035_count = 0
    dropped_person_by_class = defaultdict(int)

    raw_ball_detections_total = 0
    frames_with_raw_ball = set()
    ball_gating_rejections_count = 0
    ball_in_gate_selections_count = 0
    ball_out_gate_accepted_count = 0

    new_track_allocations = []

    # Monitor resources
    monitor = ResourceMonitor(interval=0.05)
    monitor.start()

    target_device = 0 if torch.cuda.is_available() else "cpu"
    tracker = Tracker(
        model_path=model_path,
        fps=fps,
        player_conf=0.35,
        goalkeeper_conf=0.30,
        referee_conf=0.30,
        ball_conf=0.20,
        batch_size=batch_size,
        device=target_device,
        reid_min_conf=reid_min_conf,
        draw_interpolated_ball=draw_interpolated_ball,
    )
    camera_motion = CameraMotion()
    team_assigner = TeamAssigner()

    if os.path.exists(calib_path):
        pts_init = load_points(calib_path)
        if pts_init:
            first_frame_idx = min(pts_init.keys())
            calib_init = PitchCalibration.fit({first_frame_idx: pts_init[first_frame_idx]}, motion=None)
            H_init = calib_init.H_by_seg.get(0)
            if H_init is not None:
                cap_temp = cv2.VideoCapture(video_path)
                f_idx = 0
                target_frame = None
                while cap_temp.isOpened():
                    ret_tmp, f_tmp = cap_temp.read()
                    if not ret_tmp:
                        break
                    if f_idx == first_frame_idx:
                        target_frame = f_tmp
                    f_idx += 1
                cap_temp.release()
                if target_frame is not None:
                    gray_init = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY) if len(target_frame.shape) == 3 else target_frame
                    camera_motion.add_global_anchor(first_frame_idx, gray_init, H_init, seg=0)

    # Monitored ball selection
    orig_select_ball = tracker._select_ball_detection

    def monitored_select_ball(ball_det, frame_index, frame_shape, state):
        nonlocal ball_gating_rejections_count, ball_in_gate_selections_count, ball_out_gate_accepted_count
        h_f, w_f = frame_shape[:2]
        diag = float(np.hypot(w_f, h_f))
        if len(ball_det) > 0:
            c_confs = np.asarray(ball_det.confidence, dtype=float)
            c_keep = c_confs >= max(0.10, tracker.ball_conf - 0.05)
            filtered_det = ball_det[c_keep]
            if len(filtered_det) > 0:
                l_center = state.get("last_center")
                l_frame = state.get("last_frame")
                vel = state.get("velocity", np.zeros(2, dtype=float))
                if l_center is not None and l_frame is not None:
                    dt = max(1, frame_index - l_frame)
                    sp = float(np.linalg.norm(vel))
                    pred_c = l_center + vel * dt
                    g_rad = min(max(0.12 * diag, sp * 2.5 * dt + 0.04 * diag * dt), 0.40 * diag)
                    b_boxes = np.asarray(filtered_det.xyxy, dtype=float)
                    b_centers = (b_boxes[:, :2] + b_boxes[:, 2:4]) / 2.0
                    dists = np.linalg.norm(b_centers - pred_c, axis=1)
                    in_g = dists <= g_rad
                    if np.any(in_g):
                        ball_in_gate_selections_count += 1
                    else:
                        top_conf = float(np.max(filtered_det.confidence))
                        if top_conf < max(0.35, tracker.ball_conf + 0.15):
                            ball_gating_rejections_count += 1
                        else:
                            ball_out_gate_accepted_count += 1
        return orig_select_ball(ball_det, frame_index, frame_shape, state)

    tracker._select_ball_detection = monitored_select_ball

    # Monitored detect_frame
    orig_detect_frame = tracker.detect_frame

    def monitored_detect_frame(list_frame):
        nonlocal raw_person_detections_total, dropped_person_015_035_count
        nonlocal dropped_person_015_020_count, dropped_person_020_035_count, raw_ball_detections_total
        for f_idx, d_frame in enumerate(orig_detect_frame(list_frame)):
            names_map = {v: k for k, v in d_frame.names.items()}
            b_cls = names_map.get("ball", 0)
            p_cls = names_map.get("player", 2)
            gk_cls = names_map.get("goalkeeper", 1)
            ref_cls = names_map.get("referee", 3)

            boxes = d_frame.boxes
            if boxes is not None and len(boxes) > 0:
                classes = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for c_id, conf in zip(classes, confs):
                    if c_id == b_cls:
                        raw_ball_detections_total += 1
                        frames_with_raw_ball.add(f_idx)
                    elif c_id in (p_cls, gk_cls, ref_cls):
                        raw_person_detections_total += 1
                        # Determine threshold applied before Re-ID
                        if reid_min_conf is not None:
                            p_thresh = reid_min_conf
                            gk_thresh = min(reid_min_conf, tracker.goalkeeper_conf)
                            ref_thresh = min(reid_min_conf, tracker.referee_conf)
                        else:
                            p_thresh = tracker.player_conf
                            gk_thresh = tracker.goalkeeper_conf
                            ref_thresh = tracker.referee_conf

                        thresh = p_thresh if c_id == p_cls else (gk_thresh if c_id == gk_cls else ref_thresh)
                        c_name = "player" if c_id == p_cls else ("goalkeeper" if c_id == gk_cls else "referee")

                        if conf < thresh:
                            if conf >= 0.15:
                                dropped_person_015_035_count += 1
                                dropped_person_by_class[c_name] += 1
                                if conf < 0.20:
                                    dropped_person_015_020_count += 1
                                else:
                                    dropped_person_020_035_count += 1

            yield d_frame

    tracker.detect_frame = monitored_detect_frame

    # Monitored allocate_id
    orig_allocate_id = tracker.reid_tracker.manager._allocate_id

    def monitored_allocate_id():
        tid = orig_allocate_id()
        f_cnt = tracker.reid_tracker.manager.frame_count
        new_track_allocations.append((f_cnt, tid))
        return tid

    tracker.reid_tracker.manager._allocate_id = monitored_allocate_id

    # Run or load stub
    video_frames, _ = read_video(video_path)
    t0_pass1 = time.perf_counter()

    if load_stub and stub_path and os.path.exists(stub_path):
        print(f"Loading Pass 1 tracks from stub: {stub_path}")
        with open(stub_path, "rb") as f:
            tracks = pickle.load(f)
        motion = CameraMotion()  # dummy
    else:
        tracks, motion = tracker.process_pass1(
            video_frames,
            camera_motion=camera_motion,
            team_assigner=team_assigner,
            read_from_stub=False,
            choose_Al=2,
        )
        if stub_path:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

    t_pass1 = time.perf_counter() - t0_pass1
    fps_pass1 = total_frames / max(0.001, t_pass1)

    # Ball stabilization
    t0_ball = time.perf_counter()
    stabilized_ball = tracker.interpolate_ball_position(
        tracks["ball"],
        camera_motion=motion,
        player_tracks=tracks.get("players"),
    )
    t_ball = time.perf_counter() - t0_ball

    # Pitch mapping
    t0_pitch = time.perf_counter()
    tracker.add_position_to_track(tracks)
    calib = PitchCalibration.fit(load_points(calib_path), motion)
    mapper = PitchMapper(calib, motion)
    mapper.add_positions_to_tracks(tracks)
    t_pitch = time.perf_counter() - t0_pitch

    # Speed & Distance
    t0_speed = time.perf_counter()
    s_d_calculate = SpeedAndDistance_Calc(fps)
    s_d_calculate.add_speed_and_distance_to_tracks(tracks)
    t_speed = time.perf_counter() - t0_speed

    monitor.stop()

    # Calculate person tracking metrics
    players_data = tracks["players"]
    referees_data = tracks["referees"]

    player_appearances = defaultdict(list)
    for f_idx, frame_players in enumerate(players_data):
        for pid, pinfo in frame_players.items():
            player_appearances[pid].append(f_idx)

    ref_appearances = defaultdict(list)
    for f_idx, frame_refs in enumerate(referees_data):
        for rid, rinfo in frame_refs.items():
            ref_appearances[rid].append(f_idx)

    all_player_ids = set(player_appearances.keys())
    all_ref_ids = set(ref_appearances.keys())
    all_person_ids = all_player_ids | all_ref_ids

    all_durations = [len(app) for app in player_appearances.values()]
    all_durations.extend([len(app) for app in ref_appearances.values()])
    mean_dur = float(np.mean(all_durations)) if all_durations else 0.0
    median_dur = float(np.median(all_durations)) if all_durations else 0.0
    short_tracks = sum(1 for d in all_durations if d < 30)

    # Gaps within active tracks
    total_gaps_count = 0
    fragmented_tracks_count = 0
    for pid, app in player_appearances.items():
        if len(app) >= 2:
            diffs = np.diff(app)
            gaps = np.sum(diffs > 1)
            if gaps > 0:
                fragmented_tracks_count += 1
                total_gaps_count += int(gaps)

    # Proxy ID switches and ghost tracks
    n_proxy_switches, proxy_details, n_ghost_tracks, ghost_pids, track_info = compute_proxy_id_switches(
        players_data, fps=fps
    )

    # Multi-hysteresis possession stats
    possession_res = compute_possession_stats(
        players_data, tracks["ball"], hysteresis=possession_hysteresis, fps=fps
    )

    # Ball stats
    stab_detected = 0
    stab_interp = 0
    stab_lost = 0
    ball_presence = []
    for f in stabilized_ball:
        if 1 in f:
            if f[1].get("is_interpolated", False):
                stab_interp += 1
                ball_presence.append("INTERPOLATED")
            else:
                stab_detected += 1
                ball_presence.append("DETECTED")
        else:
            stab_lost += 1
            ball_presence.append("LOST")

    max_gap = 0
    cur_gap = 0
    for p in ball_presence:
        if p == "LOST":
            cur_gap += 1
            if cur_gap > max_gap:
                max_gap = cur_gap
        else:
            cur_gap = 0

    return {
        "config": {
            "reid_min_conf": reid_min_conf,
            "draw_interpolated_ball": draw_interpolated_ball,
            "possession_hysteresis": possession_hysteresis,
        },
        "person_tracking": {
            "unique_player_ids": len(all_player_ids),
            "unique_referee_ids": len(all_ref_ids),
            "total_unique_person_ids": len(all_person_ids),
            "total_tracklets_allocated": len(new_track_allocations) if new_track_allocations else len(track_info),
            "mean_track_duration_frames": round(mean_dur, 2),
            "median_track_duration_frames": round(median_dur, 2),
            "short_tracks_less_30f": short_tracks,
            "total_internal_gaps": total_gaps_count,
            "fragmented_tracks_count": fragmented_tracks_count,
            "proxy_id_switches": n_proxy_switches,
            "proxy_switch_details": proxy_details[:5],
            "ghost_tracks_count": n_ghost_tracks,
            "ghost_track_ids": ghost_pids,
            "dropped_person_015_035": dropped_person_015_035_count,
            "dropped_person_015_020": dropped_person_015_020_count,
            "dropped_person_020_035": dropped_person_020_035_count,
            "stage2_ignored_015_020_note": "Stage 2 has low_conf_thresh=0.20, so detections in [0.15, 0.20) are ignored by Stage 2." if reid_min_conf == 0.15 else "N/A",
        },
        "possession": possession_res,
        "ball_tracking": {
            "frames_with_raw_ball": len(frames_with_raw_ball),
            "frames_after_select": sum(1 for f in tracks["ball"] if 1 in f),
            "stabilizer_detected": stab_detected,
            "stabilizer_interpolated": stab_interp,
            "stabilizer_lost": stab_lost,
            "longest_ball_gap_frames": max_gap,
            "longest_ball_gap_sec": round(max_gap / fps, 2),
        },
        "system": {
            "pass1_time_sec": round(t_pass1, 2),
            "pass1_fps": round(fps_pass1, 2),
            "ball_stabilizer_time_ms": round(t_ball * 1000, 2),
            "pitch_mapping_time_ms": round(t_pitch * 1000, 2),
            "peak_ram_mb": round(monitor.peak_rss_bytes / (1024 * 1024), 2),
            "peak_vram_alloc_mb": round(monitor.peak_vram_alloc_bytes / (1024 * 1024), 2),
        },
        "tracks": tracks,
    }


def run_comprehensive_experiment(
    video_path="video/sample.mp4",
    model_path=None,
    calib_path="calib/points.json",
    batch_size=8,
    output_report="reports/step1_comparison.json",
    contact_sheet_dir="reports/contact_sheets",
):
    print("=" * 80)
    print("KHOI CHAY THUC NGHIEM TOAN DIEN 3 THAY DOI (STEP 1 BENCHMARK)")
    print("=" * 80)

    # 1. PERSON TRACKING: 3 CONFIGS (Baseline None, 0.20, 0.15)
    person_configs = [
        ("Baseline (cu)", None),
        ("reid_min_conf=0.20", 0.20),
        ("reid_min_conf=0.15", 0.15),
    ]

    results_by_conf = {}
    tracks_for_contact_sheet = None

    for label, min_conf in person_configs:
        stub_name = f"stubs/track_stubs_reid_{min_conf if min_conf is not None else 'baseline'}.pkl"
        print(f"\n>>> Dang chay cau hinh: {label} (stub={stub_name})...")
        res = measure_single_config(
            video_path=video_path,
            model_path=model_path,
            calib_path=calib_path,
            batch_size=batch_size,
            reid_min_conf=min_conf,
            draw_interpolated_ball=True,
            possession_hysteresis=0,
            stub_path=stub_name,
            load_stub=False,
        )
        tracks_for_contact_sheet = res["tracks"]
        # Do not store raw tracks in JSON
        clean_res = {k: v for k, v in res.items() if k != "tracks"}
        results_by_conf[label] = clean_res

    # 2. POSSESSION HYSTERESIS: N = 0, 3, 5, 8
    # Dung luon tracks cua baseline de so sanh cong bang su on dinh cua possession
    print("\n>>> Dang do cac cau hinh Possession Hysteresis (N = 0, 3, 5, 8)...")
    baseline_tracks = results_by_conf["Baseline (cu)"]
    # Lay tracks tu res vua chay
    poss_tracks_players = tracks_for_contact_sheet["players"]
    poss_tracks_ball = tracks_for_contact_sheet["ball"]

    possession_experiments = {}
    for n in [0, 3, 5, 8]:
        p_stat = compute_possession_stats(poss_tracks_players, poss_tracks_ball, hysteresis=n, fps=25.0)
        # remove raw lists from summary
        p_clean = {k: v for k, v in p_stat.items() if k not in ("team_ball_control", "player_control")}
        possession_experiments[f"Hysteresis N={n}"] = p_clean

    # 3. CONTACT SHEETS
    print("\n>>> Xuat contact sheets cho cac doan 165-185, 300-350, 510-545...")
    cs_paths = generate_contact_sheets(
        video_path=video_path,
        tracks_dict=tracks_for_contact_sheet,
        output_dir=contact_sheet_dir,
        step=5,
    )

    # 4. GHI FILE JSON
    all_data = {
        "person_tracking_comparison": results_by_conf,
        "possession_hysteresis_comparison": possession_experiments,
        "contact_sheets": cs_paths,
    }
    os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"\nDa luu toan bo du lieu thuc nghiem vao: {output_report}")

    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure baseline tracking metrics")
    parser.add_argument("--video", default="video/sample.mp4", help="Path to input video")
    parser.add_argument("--model", default=None, help="Path to YOLO model checkpoint")
    parser.add_argument("--calib", default="calib/points.json", help="Path to calibration points JSON")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--output", default="reports/step1_comparison.json", help="Output JSON path")
    parser.add_argument("--compare-all", action="store_true", default=True, help="Run all 3 changes comparison")
    args = parser.parse_args()

    run_comprehensive_experiment(
        video_path=args.video,
        model_path=args.model,
        calib_path=args.calib,
        batch_size=args.batch_size,
        output_report=args.output,
    )
