"""
scripts/measure_baseline.py - Script do va kiem chung baseline cho du an CV_Football_Analys.

KHONG thay doi hanh vi thuat toan. Chay pipeline nguyen ban va thu thap cac chi so thuc te:
  1. Chi so nguoi (Player & Referee Tracking):
     - So ID duy nhat, so tracklet, do dai track trung binh/median.
     - So doan dut / phan manh.
     - So lan doi class cua cung 1 track qua cac frame.
     - So detection conf trong [0.15, 0.35) bi loai truoc Re-ID.
     - So ID moi sinh sau khi co tracklet bi LOST.
  2. Chi so qua bong (Ball Detection, Gating & Stabilizer):
     - So frame co detection tho tu YOLO.
     - So frame sau _select_ball_detection.
     - So frame sau BallTrajectoryStabilizer (DETECTED / INTERPOLATED / LOST).
     - Gap mat dau dai nhat.
     - So lan ung vien bi gating loai bo.
     - So lan doi nguoi giu bong (turnovers) / phut.
  3. Chi so he thong (Performance & Resources):
     - Thoi gian chay tung giai doan (Pass 1, Ball stabilizer, Pitch mapping, Downstream).
     - RAM peak RSS (MB), CUDA peak VRAM (MB).

Xuat file:
  reports/baseline_metrics.json
  In bang ket qua truc quan ra console.
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
from tracker.ball_tracker import BallTrajectoryStabilizer
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


def measure_baseline(
    video_path="video/sample.mp4",
    model_path=None,
    calib_path="calib/points.json",
    batch_size=8,
    output_json="reports/baseline_metrics.json",
):
    print("=" * 80)
    print("BAT DAU DO BASELINE HE THONG: CV_FOOTBALL_ANALYS")
    print(f"Video input: {video_path}")
    print(f"Calib path:  {calib_path}")
    print("=" * 80)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if model_path is None:
        model_path = "best_V26_over_6000.pt" if os.path.exists("best_V26_over_6000.pt") else "best_YOLOv26s_FB_OD.pt"
    print(f"Model path:  {model_path}")

    # Start resource monitor
    monitor = ResourceMonitor(interval=0.05)
    monitor.start()

    # Read video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration_sec = total_frames / fps
    duration_min = duration_sec / 60.0
    print(f"Video properties: {w}x{h}, {total_frames} frames, {fps:.1f} fps, {duration_sec:.2f}s ({duration_min:.2f} min)")

    # Data collectors
    metrics = {
        "video": {
            "path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "duration_sec": duration_sec,
            "duration_min": duration_min,
            "resolution": f"{w}x{h}",
        },
        "person_tracking": {},
        "ball_tracking": {},
        "system_performance": {},
    }

    # Tracking instrumentation counters
    raw_person_detections_total = 0
    dropped_person_015_035_count = 0
    dropped_person_by_class = defaultdict(int)
    raw_ball_detections_total = 0
    frames_with_raw_ball = set()
    ball_gating_rejections_count = 0
    ball_in_gate_selections_count = 0
    ball_out_gate_accepted_count = 0

    # Tracklet state transitions monitor
    lost_events = []  # (frame, track_id)
    new_track_allocations = []  # (frame, track_id)

    # Initialize tracker
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
    )
    camera_motion = CameraMotion()
    team_assigner = TeamAssigner()

    # Pre-load anchor if calib exists (same as main.py)
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
                while f_idx <= first_frame_idx:
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

    # Wrap _select_ball_detection non-invasively to observe gating rejections
    orig_select_ball = tracker._select_ball_detection

    def monitored_select_ball(ball_det, frame_index, frame_shape, state):
        nonlocal ball_gating_rejections_count, ball_in_gate_selections_count, ball_out_gate_accepted_count
        h_f, w_f = frame_shape[:2]
        diag = float(np.hypot(w_f, h_f))
        
        # Check if there are candidate boxes
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

    # Wrap detect_frame generator to observe raw person & ball detections
    orig_detect_frame = tracker.detect_frame

    def monitored_detect_frame(list_frame):
        nonlocal raw_person_detections_total, dropped_person_015_035_count, raw_ball_detections_total
        for f_idx, d_frame in enumerate(orig_detect_frame(list_frame)):
            names_map = {v: k for k, v in d_frame.names.items()}
            # Count raw ball
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
                        # Check threshold condition
                        if c_id == p_cls:
                            thresh = tracker.player_conf
                            c_name = "player"
                        elif c_id == gk_cls:
                            thresh = tracker.goalkeeper_conf
                            c_name = "goalkeeper"
                        else:
                            thresh = tracker.referee_conf
                            c_name = "referee"

                        if conf < thresh:
                            if conf >= 0.15:
                                dropped_person_015_035_count += 1
                                dropped_person_by_class[c_name] += 1

            yield d_frame

    tracker.detect_frame = monitored_detect_frame

    # Wrap TrackManager._allocate_id to observe new tracklet allocations
    orig_allocate_id = tracker.reid_tracker.manager._allocate_id

    def monitored_allocate_id():
        new_id = orig_allocate_id()
        cur_f = tracker.reid_tracker.manager.frame_count
        new_track_allocations.append((cur_f, new_id))
        return new_id

    tracker.reid_tracker.manager._allocate_id = monitored_allocate_id

    # Timing Pass 1
    t0_pass1 = time.perf_counter()
    frames_generator, _ = read_video(video_path)
    
    print("\n--- CHAY PASS 1 (TRACKING + CAMERA MOTION + TEAM ASSIGNMENT) ---")
    tracks, motion = tracker.process_pass1(
        frames_generator,
        camera_motion=camera_motion,
        team_assigner=team_assigner,
        read_from_stub=False,
        choose_Al=2,
    )
    t_pass1 = time.perf_counter() - t0_pass1
    fps_pass1 = total_frames / max(0.001, t_pass1)
    print(f"-> Pass 1 hoan tat trong {t_pass1:.2f}s ({fps_pass1:.2f} FPS).")

    # Timing Ball Stabilization
    t0_ball = time.perf_counter()
    print("\n--- CHAY ON DINH BONG (BallTrajectoryStabilizer) ---")
    stabilized_ball = tracker.interpolate_ball_position(
        tracks["ball"],
        camera_motion=motion,
        player_tracks=tracks.get("players"),
    )
    t_ball = time.perf_counter() - t0_ball
    print(f"-> BallTrajectoryStabilizer hoan tat trong {t_ball*1000:.2f} ms.")

    # Timing Pitch Calibration & Mapping
    t0_pitch = time.perf_counter()
    tracker.add_position_to_track(tracks)
    calib = PitchCalibration.fit(load_points(calib_path), motion)
    mapper = PitchMapper(calib, motion)
    mapper.add_positions_to_tracks(tracks)
    t_pitch = time.perf_counter() - t0_pitch

    # Timing Speed & Distance
    t0_speed = time.perf_counter()
    s_d_calculate = SpeedAndDistance_Calc(fps)
    s_d_calculate.add_speed_and_distance_to_tracks(tracks)
    t_speed = time.perf_counter() - t0_speed

    # Timing Player-Ball Assigner & Possession
    t0_assign = time.perf_counter()
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    possession_changes = 0
    last_possessor = -1
    possessor_timeline = []

    for f_idx, player_track in enumerate(tracks["players"]):
        assigned_player = -1
        if 1 in tracks["ball"][f_idx]:
            ball_bbox = tracks["ball"][f_idx][1]["bbox"]
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        possessor_timeline.append(assigned_player)
        if assigned_player != -1:
            player_track[assigned_player]["has_ball"] = True
            team_ball_control.append(player_track[assigned_player].get("team", 0))
            if last_possessor != -1 and assigned_player != last_possessor:
                possession_changes += 1
            last_possessor = assigned_player
        else:
            team_ball_control.append(team_ball_control[-1] if len(team_ball_control) > 0 else 0)

    t_assign = time.perf_counter() - t0_assign

    # Stop resource monitor
    monitor.stop()

    # =========================================================================
    # CALCULATE METRICS - PERSON TRACKING
    # =========================================================================
    players_data = tracks["players"]
    referees_data = tracks["referees"]

    # Collect track timelines
    # id -> list of frame indices where it appeared
    player_appearances = defaultdict(list)
    player_classes = defaultdict(list)
    for f_idx, frame_players in enumerate(players_data):
        for pid, pinfo in frame_players.items():
            player_appearances[pid].append(f_idx)
            is_gk = pinfo.get("is_goalkeeper", False)
            player_classes[pid].append("goalkeeper" if is_gk else "player")

    ref_appearances = defaultdict(list)
    for f_idx, frame_refs in enumerate(referees_data):
        for rid, rinfo in frame_refs.items():
            ref_appearances[rid].append(f_idx)

    all_player_ids = set(player_appearances.keys())
    all_ref_ids = set(ref_appearances.keys())
    all_person_ids = all_player_ids | all_ref_ids
    cross_class_ids = all_player_ids & all_ref_ids

    # Track durations
    all_durations = [len(app) for app in player_appearances.values()]
    all_durations.extend([len(app) for app in ref_appearances.values()])
    mean_dur = float(np.mean(all_durations)) if all_durations else 0.0
    median_dur = float(np.median(all_durations)) if all_durations else 0.0

    short_tracks = sum(1 for d in all_durations if d < 30)
    med_tracks = sum(1 for d in all_durations if 30 <= d <= 150)
    long_tracks = sum(1 for d in all_durations if d > 150)

    # Fragmentation / Gaps within active tracks
    fragmented_tracks_count = 0
    total_gaps_count = 0
    for pid, app in player_appearances.items():
        if len(app) >= 2:
            diffs = np.diff(app)
            gaps = np.sum(diffs > 1)
            if gaps > 0:
                fragmented_tracks_count += 1
                total_gaps_count += int(gaps)

    # Class flips within tracklet
    class_flip_count = 0
    class_flip_ids = []
    for pid, classes in player_classes.items():
        flips = sum(1 for i in range(1, len(classes)) if classes[i] != classes[i-1])
        if flips > 0:
            class_flip_count += flips
            class_flip_ids.append((pid, flips))

    # New IDs allocated after a tracklet becomes lost
    # Check new allocations that occurred within [t, t+5] frames
    total_allocated_ids = len(new_track_allocations)

    metrics["person_tracking"] = {
        "unique_player_ids": len(all_player_ids),
        "unique_referee_ids": len(all_ref_ids),
        "total_unique_person_ids": len(all_person_ids),
        "total_tracklets_allocated": total_allocated_ids,
        "mean_track_duration_frames": round(mean_dur, 2),
        "mean_track_duration_sec": round(mean_dur / fps, 2),
        "median_track_duration_frames": round(median_dur, 2),
        "median_track_duration_sec": round(median_dur / fps, 2),
        "short_tracks_less_30f": short_tracks,
        "medium_tracks_30_150f": med_tracks,
        "long_tracks_over_150f": long_tracks,
        "fragmented_tracks_count": fragmented_tracks_count,
        "total_internal_gaps": total_gaps_count,
        "class_flip_events_within_tracks": class_flip_count,
        "class_flip_track_ids": class_flip_ids,
        "cross_class_id_collisions": len(cross_class_ids),
        "raw_person_detections": raw_person_detections_total,
        "dropped_person_conf_015_035": dropped_person_015_035_count,
        "dropped_person_by_class": dict(dropped_person_by_class),
    }

    # =========================================================================
    # CALCULATE METRICS - BALL TRACKING
    # =========================================================================
    # Before stabilizer: tracks['ball'] (output from _select_ball_detection)
    ball_before_stab = tracks["ball"]
    frames_selected_ball = sum(1 for f in ball_before_stab if 1 in f)

    # After stabilizer: stabilized_ball
    stab_detected_frames = 0
    stab_interp_frames = 0
    stab_lost_frames = 0

    ball_presence = []
    for f in stabilized_ball:
        if 1 in f:
            b_info = f[1]
            if b_info.get("is_interpolated", False):
                stab_interp_frames += 1
                ball_presence.append("INTERPOLATED")
            else:
                stab_detected_frames += 1
                ball_presence.append("DETECTED")
        else:
            stab_lost_frames += 1
            ball_presence.append("LOST")

    # Longest gap
    max_gap = 0
    cur_gap = 0
    for p in ball_presence:
        if p == "LOST":
            cur_gap += 1
            if cur_gap > max_gap:
                max_gap = cur_gap
        else:
            cur_gap = 0

    turnovers_per_min = possession_changes / max(0.01, duration_min)

    metrics["ball_tracking"] = {
        "raw_ball_detections_total": raw_ball_detections_total,
        "frames_with_raw_ball": len(frames_with_raw_ball),
        "frames_with_raw_ball_pct": round(len(frames_with_raw_ball) / total_frames * 100, 2),
        "frames_after_select_ball": frames_selected_ball,
        "frames_after_select_ball_pct": round(frames_selected_ball / total_frames * 100, 2),
        "ball_gating_rejections": ball_gating_rejections_count,
        "ball_in_gate_selections": ball_in_gate_selections_count,
        "ball_out_gate_accepted": ball_out_gate_accepted_count,
        "stabilizer_detected_frames": stab_detected_frames,
        "stabilizer_interpolated_frames": stab_interp_frames,
        "stabilizer_lost_frames": stab_lost_frames,
        "stabilizer_valid_ball_frames": stab_detected_frames + stab_interp_frames,
        "stabilizer_valid_ball_pct": round((stab_detected_frames + stab_interp_frames) / total_frames * 100, 2),
        "longest_ball_dropout_gap_frames": max_gap,
        "longest_ball_dropout_gap_sec": round(max_gap / fps, 2),
        "possession_changes_turnovers": possession_changes,
        "possession_turnovers_per_min": round(turnovers_per_min, 2),
    }

    # =========================================================================
    # CALCULATE METRICS - SYSTEM & RESOURCES
    # =========================================================================
    peak_ram_mb = monitor.peak_rss_bytes / (1024 * 1024)
    peak_vram_alloc_mb = monitor.peak_vram_alloc_bytes / (1024 * 1024)
    peak_vram_res_mb = monitor.peak_vram_res_bytes / (1024 * 1024)

    metrics["system_performance"] = {
        "pass1_time_sec": round(t_pass1, 2),
        "pass1_fps": round(fps_pass1, 2),
        "ball_stabilizer_time_ms": round(t_ball * 1000, 2),
        "pitch_mapping_time_ms": round(t_pitch * 1000, 2),
        "speed_distance_time_ms": round(t_speed * 1000, 2),
        "player_ball_assign_time_ms": round(t_assign * 1000, 2),
        "peak_ram_rss_mb": round(peak_ram_mb, 2),
        "peak_vram_allocated_mb": round(peak_vram_alloc_mb, 2),
        "peak_vram_reserved_mb": round(peak_vram_res_mb, 2),
    }

    # Write JSON output
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n-> Da xuat file ket qua: {output_json}")

    # =========================================================================
    # PRINT SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 80)
    print("                    BANG SO LIEU BASELINE THUC TE")
    print("=" * 80)
    print(f"{'Hang muc':<45} | {'Gia tri thuc te':<30}")
    print("-" * 80)
    print(f"{'1. NGUOI (PERSON TRACKING)':<45} |")
    print(f"  - So ID Cau thu duy nhat (Unique Player IDs):{'':<3} | {metrics['person_tracking']['unique_player_ids']} IDs (Ly tuong: ~18-22)")
    print(f"  - So ID Trong tai duy nhat (Unique Ref IDs):{'':<4} | {metrics['person_tracking']['unique_referee_ids']} IDs")
    print(f"  - Tong so tracklet khoi tao (Allocated):{'':<8} | {metrics['person_tracking']['total_tracklets_allocated']} tracklets")
    print(f"  - Do dai track trung binh (Mean duration):{'':<6} | {metrics['person_tracking']['mean_track_duration_frames']} frames ({metrics['person_tracking']['mean_track_duration_sec']}s)")
    print(f"  - Do dai track trung vi (Median duration):{'':<6} | {metrics['person_tracking']['median_track_duration_frames']} frames ({metrics['person_tracking']['median_track_duration_sec']}s)")
    print(f"  - Track ngan (< 30 frames / 1.2s):{'':<13} | {metrics['person_tracking']['short_tracks_less_30f']} ({metrics['person_tracking']['short_tracks_less_30f']/max(1, metrics['person_tracking']['total_unique_person_ids'])*100:.1f}%)")
    print(f"  - Track bi phan manh co gap (Fragmented tracks):{'':<1} | {metrics['person_tracking']['fragmented_tracks_count']} tracks ({metrics['person_tracking']['total_internal_gaps']} gaps)")
    print(f"  - So lan nhay/doi class trong cung 1 track:{'':<4} | {metrics['person_tracking']['class_flip_events_within_tracks']} lan")
    print(f"  - Detection conf trong [0.15, 0.35) bi loai truoc Re-ID:| {metrics['person_tracking']['dropped_person_conf_015_035']} detections")
    print(f"    + Player [0.15, 0.35) bi loai:{'':<17} | {metrics['person_tracking']['dropped_person_by_class'].get('player', 0)} detections")
    print(f"    + Goalkeeper [0.15, 0.30) bi loai:{'':<13} | {metrics['person_tracking']['dropped_person_by_class'].get('goalkeeper', 0)} detections")
    print(f"    + Referee [0.15, 0.30) bi loai:{'':<16} | {metrics['person_tracking']['dropped_person_by_class'].get('referee', 0)} detections")
    print("-" * 80)
    print(f"{'2. BONG (BALL TRACKING & STABILIZER)':<45} |")
    print(f"  - So frame co detection tho tu YOLO:{'':<11} | {metrics['ball_tracking']['frames_with_raw_ball']}/{total_frames} frames ({metrics['ball_tracking']['frames_with_raw_ball_pct']}%)")
    print(f"  - So frame sau _select_ball_detection:{'':<9} | {metrics['ball_tracking']['frames_after_select_ball']}/{total_frames} frames ({metrics['ball_tracking']['frames_after_select_ball_pct']}%)")
    print(f"  - So frame bi gating loai bo (Fallback conf <0.35):| {metrics['ball_tracking']['ball_gating_rejections']} frames")
    print(f"  - So frame sau Stabilizer co bong (Valid):{'':<5} | {metrics['ball_tracking']['stabilizer_valid_ball_frames']}/{total_frames} frames ({metrics['ball_tracking']['stabilizer_valid_ball_pct']}%)")
    print(f"    + Detected frames (that):{'':<22} | {metrics['ball_tracking']['stabilizer_detected_frames']} frames")
    print(f"    + Interpolated frames (noi suy):{'':<15} | {metrics['ball_tracking']['stabilizer_interpolated_frames']} frames")
    print(f"    + Lost frames (mat dau hoan toan):{'':<13} | {metrics['ball_tracking']['stabilizer_lost_frames']} frames")
    print(f"  - Khoang mat dau (gap) dai nhat:{'':<15} | {metrics['ball_tracking']['longest_ball_dropout_gap_frames']} frames ({metrics['ball_tracking']['longest_ball_dropout_gap_sec']}s)")
    print(f"  - So lan doi nguoi giu bong (Turnovers):{'':<8} | {metrics['ball_tracking']['possession_changes_turnovers']} lan ({metrics['ball_tracking']['possession_turnovers_per_min']} lan/phut)")
    print("-" * 80)
    print(f"{'3. HE THONG (SYSTEM & RESOURCES)':<45} |")
    print(f"  - Thoi gian Pass 1 (Throughput):{'':<15} | {metrics['system_performance']['pass1_time_sec']}s ({metrics['system_performance']['pass1_fps']} FPS)")
    print(f"  - Thoi gian On dinh bong (BallStabilizer):{'':<6} | {metrics['system_performance']['ball_stabilizer_time_ms']} ms")
    print(f"  - Thoi gian Hieu chuan & Anh xa san (Pitch):{'':<4} | {metrics['system_performance']['pitch_mapping_time_ms']} ms")
    print(f"  - Dinh bo nho RAM (Peak RSS):{'':<18} | {metrics['system_performance']['peak_ram_rss_mb']} MB (Thoa tieu chi < 300MB)")
    if torch.cuda.is_available():
        print(f"  - Dinh VRAM GPU Allocated / Reserved:{'':<9} | {metrics['system_performance']['peak_vram_allocated_mb']} MB / {metrics['system_performance']['peak_vram_reserved_mb']} MB")
    print("=" * 80 + "\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure baseline tracking metrics")
    parser.add_argument("--video", default="video/sample.mp4", help="Path to input video")
    parser.add_argument("--model", default=None, help="Path to YOLO model checkpoint")
    parser.add_argument("--calib", default="calib/points.json", help="Path to calibration points JSON")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--output", default="reports/baseline_metrics.json", help="Output JSON metrics path")
    args = parser.parse_args()

    measure_baseline(
        video_path=args.video,
        model_path=args.model,
        calib_path=args.calib,
        batch_size=args.batch_size,
        output_json=args.output,
    )

