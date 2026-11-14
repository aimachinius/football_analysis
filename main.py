import argparse
import json
import os
import time
import cv2
import numpy as np
from collections import defaultdict

from utils.video_utils import read_video, save_video
from tracker import Tracker
from camera_motion import CameraMotion
from team_assignment import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from speed_distance import SpeedAndDistance_Calc
from pitch_homo import PitchCalibration, PitchMapper, load_points
from minimap import MinimapRenderer

PATH_VID = 'video/sample.mp4'
DEFAULT_MODEL = 'best_V26_over_6000.pt' if os.path.exists('best_V26_over_6000.pt') else 'best_YOLOv26s_FB_OD.pt'

def validate_and_lock_special_roles(tracks, team_assigner, min_ref_persistence=10, video_path=None):
    """
    [TẦNG 3 NÂNG CAO]: Thẩm định Toàn diện & Khóa Cứng Vai Trò Đặc Biệt (Trọng tài vs Thủ môn vs Cầu thủ).
    Đặc trị triệt để lỗi: Trọng tài bị nhầm thành Cầu thủ mang số, nhấp nháy đổi vai trò giữa các frame.

    Cơ chế 4 Lớp:
      1. Khử Trọng tài 'Ma' (Eliminate Phantom Referees):
         - Bất kỳ Track ID nào chỉ xuất hiện làm Trọng tài < min_ref_persistence frames (ví dụ 2-5 frames)
           đều là nhiễu YOLO thoáng qua -> Loại bỏ hoặc chuyển trả về Cầu thủ.
      2. Phân tích Dị biệt Màu áo (Color Outlier Test):
         - Lấy màu áo 2 đội chính (C1, C2). Người có màu áo dị biệt với cả 2 đội (d > 45.0)
           chắc chắn là Trọng tài (áo Đen/Vàng chanh) hoặc Thủ môn (áo Neon khác biệt).
      3. Ràng buộc Không gian Sân bóng chuẩn FIFA (Spatial Geometry):
         - Trong vòng cấm (x <= 20m hoặc x >= 85m) & y ở giữa sân: KHẲNG ĐỊNH THỦ MÔN.
         - Ở giữa sân (22m <= x <= 83m): KHẲNG ĐỊNH TRỌNG TÀI.
      4. KHÓA CỨNG VAI TRÒ VĨNH VIỄN (Permanent Role Locking):
         - Đã là Trọng tài -> Xuyên suốt video nằm trong tracks['referees'], xóa sạch khỏi tracks['players'].
           TUYỆT ĐỐI KHÔNG BAO GIỜ mang số hay nhấp nháy thành Cầu thủ!
         - Đã là Thủ môn -> Xuyên suốt video nằm trong tracks['players'] với is_goalkeeper=True.
           TUYỆT ĐỐI KHÔNG BAO GIỜ bị nhầm thành Trọng tài hay Cầu thủ thường!
         - Đã là Cầu thủ -> Xuyên suốt video là Cầu thủ, xóa khỏi tracks['referees'].
    """
    player_positions = defaultdict(list)
    referee_positions = defaultdict(list)
    track_color_samples = defaultdict(list)
    total_frames = len(tracks["players"])

    # 1. Thu thập dữ liệu vị trí và màu áo của tất cả ID
    for frame in tracks["players"]:
        for pid, pinfo in frame.items():
            pos = pinfo.get("position_transformed")
            if pos is not None and np.all(np.isfinite(pos)):
                player_positions[pid].append(pos)
            tc = pinfo.get("team_color")
            if tc is not None:
                track_color_samples[pid].append(tc)

    for frame in tracks["referees"]:
        for rid, rinfo in frame.items():
            pos = rinfo.get("position_transformed")
            if pos is not None and np.all(np.isfinite(pos)):
                referee_positions[rid].append(pos)

    # Lấy màu sắc 2 đội từ team_assigner hoặc trích xuất từ tracks (khi chạy từ stub cache)
    tc1, tc2 = None, None
    if team_assigner and 1 in team_assigner.team_colors and 2 in team_assigner.team_colors:
        tc1 = np.array(team_assigner.team_colors[1], dtype=float)
        tc2 = np.array(team_assigner.team_colors[2], dtype=float)
    else:
        tc1_samples = [p["team_color"] for f in tracks["players"] for p in f.values() if p.get("team") == 1 and "team_color" in p]
        tc2_samples = [p["team_color"] for f in tracks["players"] for p in f.values() if p.get("team") == 2 and "team_color" in p]
        if tc1_samples:
            tc1 = np.mean(tc1_samples, axis=0)
        if tc2_samples:
            tc2 = np.mean(tc2_samples, axis=0)

    # Nếu chạy từ cache stub khiến team_assigner chưa có mẫu màu, lấy nhanh từ 60 frame video
    if team_assigner and (not team_assigner.player_raw_colors or not team_assigner.referee_raw_colors) and video_path and os.path.exists(video_path):
        cap_sample = cv2.VideoCapture(video_path)
        total_sample_f = min(60, total_frames)
        for f in range(total_sample_f):
            ret, frame = cap_sample.read()
            if not ret:
                break
            for pid, p in tracks["players"][f].items():
                if p.get("confidence", 0) >= 0.35 and not p.get("is_goalkeeper"):
                    c = team_assigner.get_player_color(frame, p["bbox"])
                    if c is not None:
                        team_assigner.player_raw_colors[pid].append(c)
            if "referees" in tracks and f < len(tracks["referees"]):
                for rid, r in tracks["referees"][f].items():
                    if r.get("confidence", 0) >= 0.30:
                        team_assigner.record_referee_color(frame, r["bbox"])
        cap_sample.release()

    # Thống kê toàn bộ ID xuất hiện trong video
    all_tids = set()
    for f in range(total_frames):
        all_tids.update(tracks["players"][f].keys())
        all_tids.update(tracks["referees"][f].keys())

    ref_frame_counts = {tid: sum(1 for f in tracks["referees"] if tid in f) for tid in all_tids}
    player_frame_counts = {tid: sum(1 for f in tracks["players"] if tid in f) for tid in all_tids}

    # 2. Thu thập màu áo chuẩn của Trọng tài biên (Assistant Referees)
    ref_color_samples = []
    if team_assigner and hasattr(team_assigner, "referee_raw_colors") and team_assigner.referee_raw_colors:
        ref_color_samples = team_assigner.referee_raw_colors
    has_real_ref_samples = bool(ref_color_samples)
    mean_ref_color = np.mean(ref_color_samples, axis=0) if ref_color_samples else np.array([57.4, 65.1, 87.0])

    final_roles = {}  # tid -> 'referee' | 'goalkeeper' | 'player'

    for tid in all_tids:
        r_cnt = ref_frame_counts.get(tid, 0)
        p_cnt = player_frame_counts.get(tid, 0)
        total_cnt = r_cnt + p_cnt
        if total_cnt == 0:
            continue

        # Tọa độ metric FIFA trung bình
        coords = player_positions.get(tid, []) + referee_positions.get(tid, [])
        mean_x = float(np.mean([c[0] for c in coords])) if coords else None
        mean_y = float(np.mean([c[1] for c in coords])) if coords else None

        # Tọa độ pixel ảnh trung bình (fallback khi homography chưa calib)
        pixel_xs = []
        for f in range(total_frames):
            if tid in tracks["players"][f]:
                b = tracks["players"][f][tid].get("bbox")
                if b: pixel_xs.append((b[0] + b[2]) / 2.0)
            elif tid in tracks["referees"][f]:
                b = tracks["referees"][f][tid].get("bbox")
                if b: pixel_xs.append((b[0] + b[2]) / 2.0)
        mean_pixel_x = float(np.mean(pixel_xs)) if pixel_xs else 960.0

        # Kiểm tra màu áo thực tế (RGB raw color profile)
        raw_cols = team_assigner.player_raw_colors.get(tid, []) if (team_assigner and hasattr(team_assigner, "player_raw_colors")) else []
        is_ref_by_color = False
        sim_ref = 0.0
        d_team_min = 0.0
        d_ref = 999.0

        if raw_cols and tc1 is not None and tc2 is not None and has_real_ref_samples and mean_ref_color is not None:
            mean_col = np.mean(raw_cols, axis=0)
            d1 = np.linalg.norm(mean_col - tc1)
            d2 = np.linalg.norm(mean_col - tc2)
            d_team_min = min(d1, d2)
            d_ref = np.linalg.norm(mean_col - mean_ref_color)
            sim_ref = max(0.0, 1.0 - d_ref / 441.67) * 100.0

            # Tiêu chí: Dị biệt rõ ràng với cả 2 đội (d_team_min > 50),
            # gần màu áo trọng tài (d_ref <= 60 hoặc sim_ref >= 78%),
            # và phải gần màu trọng tài hơn bất kỳ đội nào (d_ref < d_team_min)
            if d_team_min > 50.0 and (sim_ref >= 78.0 or d_ref <= 60.0) and d_ref < d_team_min:
                is_ref_by_color = True

        # Quyết định vai trò:
        # A. THỦ MÔN:
        # - Trên tọa độ metric FIFA: mean_x <= 20.0 hoặc mean_x >= 85.0 và y ở khoảng giữa sân
        # - HOẶC trên tọa độ pixel ảnh: luôn đứng sát mép khung thành (mean_pixel_x < 350px hoặc > 1600px)
        # - HOẶC có frame được YOLO gắn cờ is_goalkeeper
        is_gk_detected = any(tracks["players"][f][tid].get("is_goalkeeper", False)
                             for f in range(total_frames) if tid in tracks["players"][f])

        in_penalty_metric = (mean_x is not None) and (mean_x <= 20.0 or mean_x >= 85.0) and (14.0 <= mean_y <= 54.0)
        in_penalty_pixel = (mean_pixel_x < 350.0 or mean_pixel_x > 1600.0)

        if in_penalty_metric or (mean_x is None and in_penalty_pixel) or (is_gk_detected and (in_penalty_metric or in_penalty_pixel)):
            final_roles[tid] = 'goalkeeper'
        # B. TRỌNG TÀI (Được ưu tiên cao hơn quyền detect của YOLO):
        # - Khác màu 2 đội và trùng màu >= 78% với tài biên thực tế trận này -> KHẲNG ĐỊNH TRỌNG TÀI NGAY!
        elif is_ref_by_color and not in_penalty_metric:
            final_roles[tid] = 'referee'
            print(f"-> [Tầng 3 Referee Lock] Khẳng định Track ID {tid} là TRỌNG TÀI (Màu áo giống {sim_ref:.1f}% tài biên, d_ref={d_ref:.1f}, khác biệt 2 đội d={d_team_min:.1f})")
        # - Hoặc có tín hiệu referee bền vững từ YOLO (phải xuất hiện tối thiểu min_ref_persistence frames và chiếm tỷ lệ cao)
        elif ((r_cnt >= min_ref_persistence and r_cnt / total_cnt >= 0.35) or (r_cnt > p_cnt and r_cnt >= min_ref_persistence)) and not in_penalty_metric:
            final_roles[tid] = 'referee'
        else:
            final_roles[tid] = 'player'

    # Áp dụng KHÓA CỨNG VAI TRÒ VĨNH VIỄN vào tracks
    team1_x, team2_x = [], []
    for frame in tracks["players"]:
        for pid, p in frame.items():
            if final_roles.get(pid) == 'player':
                pos = p.get("position_transformed")
                if pos is not None and np.all(np.isfinite(pos)):
                    t = p.get("team")
                    if t == 1:
                        team1_x.append(pos[0])
                    elif t == 2:
                        team2_x.append(pos[0])

    team_defending_left = 1 if (team1_x and team2_x and np.median(team1_x) < np.median(team2_x)) else 1
    team_defending_right = 2 if team_defending_left == 1 else 1

    for tid, role in final_roles.items():
        if role == 'referee':
            # Chuyển toàn bộ vào tracks['referees'], xóa khỏi tracks['players']
            for f in range(total_frames):
                if tid in tracks["players"][f]:
                    pdata = tracks["players"][f].pop(tid)
                    pdata.pop("is_goalkeeper", None)
                    pdata.pop("team", None)
                    pdata.pop("team_color", None)
                    tracks["referees"][f][tid] = pdata
        elif role == 'goalkeeper':
            # Khóa cứng is_goalkeeper = True trong tracks['players'], xóa khỏi tracks['referees']
            coords = player_positions.get(tid, []) + referee_positions.get(tid, [])
            mean_x = float(np.mean([c[0] for c in coords])) if coords else (0.0 if mean_pixel_x < 960 else 105.0)
            gk_team = team_defending_left if mean_x < 52.5 else team_defending_right
            gk_color = team_assigner.team_colors.get(gk_team, (0, 0, 255)) if team_assigner else (0, 0, 255)

            for f in range(total_frames):
                if tid in tracks["referees"][f]:
                    rdata = tracks["referees"][f].pop(tid)
                    rdata["is_goalkeeper"] = True
                    rdata["team"] = gk_team
                    rdata["team_color"] = gk_color
                    tracks["players"][f][tid] = rdata
                elif tid in tracks["players"][f]:
                    pdata = tracks["players"][f][tid]
                    pdata["is_goalkeeper"] = True
                    pdata["team"] = gk_team
                    pdata["team_color"] = gk_color
        else:  # player
            # Khóa cứng là cầu thủ, xóa khỏi tracks['referees'] nếu có lọt vào
            for f in range(total_frames):
                if tid in tracks["referees"][f]:
                    rdata = tracks["referees"][f].pop(tid)
                    rdata["is_goalkeeper"] = False
                    tracks["players"][f][tid] = rdata
                if tid in tracks["players"][f]:
                    tracks["players"][f][tid]["is_goalkeeper"] = False

    # Loại bỏ các frame phát hiện thủ môn bị trôi dạt ra giữa sân (450 < cx < 1450)
    for f in range(total_frames):
        for tid in list(tracks["players"][f].keys()):
            p = tracks["players"][f][tid]
            if p.get("is_goalkeeper", False):
                b = p.get("bbox")
                if b:
                    cx = (b[0] + b[2]) / 2.0
                    if 450.0 < cx < 1450.0:
                        p["is_goalkeeper"] = False

    # Khâu nối (Stitching) tất cả các Tracklet thủ môn về duy nhất 1 ID chuẩn cho mỗi khung thành
    left_gk_ids = []
    right_gk_ids = []
    for tid, role in final_roles.items():
        if role == 'goalkeeper':
            coords = player_positions.get(tid, []) + referee_positions.get(tid, [])
            mx = float(np.mean([c[0] for c in coords])) if coords else (0.0 if mean_pixel_x < 960 else 105.0)
            if mx < 52.5:
                left_gk_ids.append(tid)
            else:
                right_gk_ids.append(tid)

    if left_gk_ids:
        canon_left = min(left_gk_ids)
        for tid in left_gk_ids:
            if tid != canon_left:
                for f in range(total_frames):
                    if tid in tracks["players"][f]:
                        pdata = tracks["players"][f].pop(tid)
                        b = pdata.get("bbox")
                        cx = (b[0] + b[2]) / 2.0 if b else 0.0
                        if cx < 500.0 and canon_left not in tracks["players"][f]:
                            tracks["players"][f][canon_left] = pdata
        print(f"-> [Tầng 3 GK Stitching] Đã hợp nhất các Tracklet thủ môn trái {left_gk_ids} vào ID chuẩn: {canon_left}")

    if right_gk_ids:
        canon_right = min(right_gk_ids)
        for tid in right_gk_ids:
            if tid != canon_right:
                for f in range(total_frames):
                    if tid in tracks["players"][f]:
                        pdata = tracks["players"][f].pop(tid)
                        b = pdata.get("bbox")
                        cx = (b[0] + b[2]) / 2.0 if b else 1920.0
                        if cx > 1400.0 and canon_right not in tracks["players"][f]:
                            tracks["players"][f][canon_right] = pdata
        print(f"-> [Tầng 3 GK Stitching] Đã hợp nhất các Tracklet thủ môn phải {right_gk_ids} vào ID chuẩn: {canon_right}")

    print(f"-> [Tầng 3 Nâng Cao] Đã khóa cứng vai trò vĩnh viễn cho {len(final_roles)} Track IDs:")
    ref_list = [tid for tid, r in final_roles.items() if r == 'referee']
    gk_list = [tid for tid, r in final_roles.items() if r == 'goalkeeper']
    print(f"   • Trọng tài (Referees - Khóa vĩnh viễn): {ref_list}")
    print(f"   • Thủ môn (Goalkeepers - Khóa vĩnh viễn): {gk_list}")


def main(
    output_name=None,
    video_path=PATH_VID,
    model_path=DEFAULT_MODEL,
    player_conf=0.35,
    goalkeeper_conf=0.30,
    referee_conf=0.30,
    ball_conf=0.20,
    batch_size=8,
    calib_path="calib/points.json",
    generate_minimap=True,
    device=None,
    reid_min_conf=None,
    draw_interpolated_ball=True,
    possession_hysteresis=3,
    ball_selector="viterbi",
    ball_imgsz=1920,
    use_sam2_backward=True,
    sam2_frames_dir="data/sample_frames",
    export_tracks_path=None,
    no_render=False,
    read_from_stub=False,
    stub_path=None,
    camera_stub_path=None,
):
    t_start = time.time()
    # Xác định tên và đường dẫn lưu video
    if not output_name:
        output_name = "output.mp4"
    elif not output_name.lower().endswith(".mp4"):
        output_name += ".mp4"

    if not os.path.dirname(output_name):
        output_path = os.path.join("output", output_name)
    else:
        output_path = output_name

    # =========================================================================
    # PASS 1: Phân tích toàn bộ video trong 1 lần đọc duy nhất (Streaming)
    #   - [TẦNG 1] Khử trùng lặp đa nhãn: Referee > GK > Player
    #   - [TẦNG 2] Bỏ phiếu chuỗi thời gian khóa cố định vai trò cho từng ID
    #   - BoT-SORT Tracking người & Gating tracking bóng
    #   - Camera Motion Homography
    #   - Team Assigner trực tiếp từ frame
    #   => Bộ nhớ RAM luôn duy trì an toàn < 300MB
    # =========================================================================
    print(f"=== [PASS 1/2] BẮT ĐẦU PHÂN TÍCH VIDEO (Hệ thống 3 Tầng: Tracking + Camera + Team) ===")
    print(f"Model: {model_path}")
    print(f"Conf thresholds: player={player_conf}, gk={goalkeeper_conf}, ref={referee_conf}, ball={ball_conf}")
    print(f"Input video: {video_path}")

    import torch
    target_device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    if target_device == "cpu":
        print("\n" + "!" * 76)
        print("[CẢNH BÁO HIỆU NĂNG] Hệ thống đang chạy trên CPU (không tìm thấy CUDA GPU)!")
        print("Inference YOLO trên CPU với imgsz=1536 có thể mất 15-20 phút cho video 30s!")
        print("Để đạt tốc độ tối đa ~2 phút, vui lòng sử dụng môi trường GPU NVIDIA có CUDA.")
        print("!" * 76 + "\n")
    else:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
        print(f"-> Thiết bị tăng tốc: CUDA GPU ({gpu_name})")

    video_frames, fps = read_video(video_path)
    tracker = Tracker(
        model_path=model_path,
        fps=fps,
        player_conf=player_conf,
        goalkeeper_conf=goalkeeper_conf,
        referee_conf=referee_conf,
        ball_conf=ball_conf,
        batch_size=batch_size,
        device=target_device,
        reid_min_conf=reid_min_conf,
        draw_interpolated_ball=draw_interpolated_ball,
        ball_selector=ball_selector,
        ball_imgsz=ball_imgsz,
        calib_path=calib_path,
    )
    camera_motion = CameraMotion()
    team_assigner = TeamAssigner()

    # [GIAI ĐOẠN 3]: Nạp Global Calibration Anchor ban đầu vào CameraMotion để hỗ trợ Re-localization sau camera cut
    # Đọc frame độc lập bằng VideoCapture tạm thời để bảo toàn generator video_frames cho Pass 1
    if os.path.exists(calib_path):
        pts_init = load_points(calib_path)
        if pts_init:
            valid_frames = [f for f, p in pts_init.items() if len(p) >= 4]
            if valid_frames:
                first_frame_idx = min(valid_frames)
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
                    print(f"-> [Stage 3] Đã nạp Global Calibration Anchor frame {first_frame_idx} vào CameraMotion.")

    base_vid_name = os.path.splitext(os.path.basename(video_path))[0]
    if stub_path is None:
        cand_stub = f"stubs/track_stubs_{base_vid_name}.pkl"
        if not os.path.exists(cand_stub) and base_vid_name == "sample" and os.path.exists("stubs/track_stubs_reid.pkl"):
            stub_path = "stubs/track_stubs_reid.pkl"
        else:
            stub_path = cand_stub
    if camera_stub_path is None:
        cand_cam = f"stubs/camera_motion_{base_vid_name}.pkl"
        if not os.path.exists(cand_cam) and base_vid_name == "sample" and os.path.exists("stubs/camera_motion.pkl"):
            camera_stub_path = "stubs/camera_motion.pkl"
        else:
            camera_stub_path = cand_cam

    t_p1_start = time.time()
    tracks, motion = tracker.process_pass1(
        video_frames,
        camera_motion=camera_motion,
        team_assigner=team_assigner,
        read_from_stub=read_from_stub,
        stub_path=stub_path,
        camera_stub_path=camera_stub_path,
        choose_Al=2,
        video_path=video_path,
    )
    t_pass1 = time.time() - t_p1_start
    print(f"-> Pass 1 hoàn tất ({t_pass1:.2f}s): đã phân tích {len(tracks['players'])} frames.")
    print(f"   + Số frame có cầu thủ/thủ môn: {sum(len(f) > 0 for f in tracks['players'])}")
    print(f"   + Số frame có trọng tài: {sum(len(f) > 0 for f in tracks['referees'])}")
    print(f"   + Số frame có bóng: {sum(len(f) > 0 for f in tracks['ball'])}")

    # =========================================================================
    # GIAI ĐOẠN TRUNG GIAN (PHẦN 1): Hiệu chuẩn sân & Khóa Cứng Vai Trò Tầng 3
    # =========================================================================
    t_inter_start = time.time()
    print("=== [XỬ LÝ DỮ LIỆU] Ổn định quỹ đạo bóng, Hiệu chuẩn sân & Khóa Cứng Vai Trò Tầng 3 ===")

    # [Milestone P1] Ổn định quỹ đạo bóng, loại bỏ ngoại lai & nội suy thích nghi 3 tầng TRƯỚC ánh xạ Homography
    tracks["ball"] = tracker.interpolate_ball_position(
        tracks["ball"],
        camera_motion=motion,
        player_tracks=tracks.get("players"),
    )

    tracker.add_position_to_track(tracks)

    # Hiệu chuẩn sân và ánh xạ tọa độ sang mét
    calib = PitchCalibration.fit(load_points(calib_path), motion)
    print("Pitch Calibration Report:\n", calib.report)

    # Chẩn đoán trạng thái phân đoạn và hiệu chuẩn toàn cục
    total_segs = set(int(s) for s in motion.seg)
    calib_segs = set(calib.H_by_seg.keys())
    missing_segs = total_segs - calib_segs
    if missing_segs:
        uncalibrated_frames = sum(1 for s in motion.seg if int(s) in missing_segs)
        print(f"[PitchCalibration CẢNH BÁO] Phát hiện {len(missing_segs)} segment chưa có ma trận hiệu chuẩn toàn cục: {sorted(missing_segs)}.")
        print(f"   -> Có {uncalibrated_frames}/{len(motion.seg)} frames (chiếm {uncalibrated_frames/len(motion.seg)*100:.1f}%) thuộc trạng thái UNREGISTERED.")

    mapper = PitchMapper(calib, motion)
    mapper.add_positions_to_tracks(tracks)

    # [TẦNG 3 NÂNG CAO] Thẩm định không gian sân bóng chuẩn FIFA, Dị biệt Màu áo & Khóa Cứng Vai Trò Vĩnh Viễn
    validate_and_lock_special_roles(tracks, team_assigner, video_path=video_path)

    # =========================================================================
    # GIAI ĐOẠN REFINEMENT: SAM 2 BACKWARD TRACKING (Xử lý cụm va chạm cầu thủ thực)
    # =========================================================================
    t_sam2 = 0.0
    if use_sam2_backward:
        t_sam2_start = time.time()
        from sam2_tracker import refine_tracks_with_sam2_backward
        tracks = refine_tracks_with_sam2_backward(
            tracks,
            video_path=video_path,
            frames_dir=sam2_frames_dir,
            device=target_device,
            verbose=True,
        )
        t_sam2 = time.time() - t_sam2_start

    # [Universal Trajectory Smoothing]: Làm mượt quỹ đạo và bù đắp khoảng mất dấu ngắn (1..16 frames)
    # Áp dụng đồng bộ cho toàn bộ Cầu thủ, Trọng tài và Thủ môn sau khi SAM 2 hoàn tất
    tracker.interpolate_player_trajectories(tracks, max_gap=16)
    mapper.add_positions_to_tracks(tracks)

    # [Team Consistency Lock]: Khóa cố định màu áo và đội bóng cho từng Track ID
    # Tránh tình trạng cầu thủ bị nhấp nháy màu áo hoặc nhảy đội giữa các frame
    track_team_votes = defaultdict(list)
    track_color_samples = defaultdict(list)
    for frame in tracks["players"]:
        for pid, p in frame.items():
            if "team" in p and p["team"] is not None:
                track_team_votes[pid].append(p["team"])
            if "team_color" in p and p["team_color"] is not None:
                track_color_samples[pid].append(p["team_color"])

    locked_teams = {pid: max(set(v), key=v.count) for pid, v in track_team_votes.items() if v}
    locked_colors = {pid: c[len(c) // 2] for pid, c in track_color_samples.items() if c}

    for frame in tracks["players"]:
        for pid, p in frame.items():
            if pid in locked_teams:
                p["team"] = locked_teams[pid]
            if pid in locked_colors:
                p["team_color"] = locked_colors[pid]

    # Tính toán vận tốc (km/h) và quãng đường (m)
    s_d_calculate = SpeedAndDistance_Calc(fps)
    s_d_calculate.add_speed_and_distance_to_tracks(tracks)

    # Xác định quyền kiểm soát bóng của từng đội
    team_ball_control = []
    player_assigner = PlayerBallAssigner(hysteresis=possession_hysteresis)
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox'] if 1 in tracks['ball'][frame_num] else None
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player].get('team', 0))
        else:
            team_ball_control.append(team_ball_control[-1] if len(team_ball_control) > 0 else 0)
    team_ball_control = np.array(team_ball_control)
    t_intermediate = time.time() - t_inter_start

    # =========================================================================
    # PASS 2: Vẽ Annotation và Lưu Video ra Đĩa (Streaming 1 frame tại một thời điểm)
    # =========================================================================
    # Khử triệt để shadow / duplicate tracks trong cùng frame (IoU >= 0.60)
    from sam2_tracker.sam2_refiner import calculate_box_iou
    for f in range(len(tracks["players"])):
        pids = list(tracks["players"][f].keys())
        to_remove = set()
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                id1, id2 = pids[i], pids[j]
                if id1 in to_remove or id2 in to_remove:
                    continue
                b1 = tracks["players"][f][id1].get("bbox")
                b2 = tracks["players"][f][id2].get("bbox")
                if b1 and b2 and calculate_box_iou(b1, b2) >= 0.60:
                    int1 = tracks["players"][f][id1].get("is_interpolated", False)
                    int2 = tracks["players"][f][id2].get("is_interpolated", False)
                    if int1 and not int2:
                        to_remove.add(id1)
                    elif int2 and not int1:
                        to_remove.add(id2)
                    elif tracks["players"][f][id1].get("confidence", 0) < tracks["players"][f][id2].get("confidence", 0):
                        to_remove.add(id1)
                    else:
                        to_remove.add(id2)
        for tid in to_remove:
            tracks["players"][f].pop(tid, None)

    # [Pruning Spurious Blips]: Loại bỏ triệt để các tracklet rác thoáng qua (tổng vòng đời <= 2 frames)
    for cat in ("players", "referees"):
        id_counts = defaultdict(int)
        for frame in tracks[cat]:
            for tid in frame:
                id_counts[tid] += 1
        spurious_ids = {tid for tid, cnt in id_counts.items() if cnt <= 2}
        if spurious_ids:
            for frame in tracks[cat]:
                for tid in spurious_ids:
                    frame.pop(tid, None)
            print(f"-> [Pruning Spurious Blips] Đã loại bỏ các tracklet rác <= 2 frames: {spurious_ids} khỏi {cat}")

    if export_tracks_path:
        os.makedirs(os.path.dirname(export_tracks_path) or ".", exist_ok=True)
        cap_dim = cv2.VideoCapture(video_path)
        vid_w = int(cap_dim.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        vid_h = int(cap_dim.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        cap_dim.release()
        export_data = {
            "video": video_path,
            "fps": float(fps),
            "width": int(vid_w),
            "height": int(vid_h),
            "total_frames": int(len(tracks["players"])),
            "frames": {}
        }
        for f in range(len(tracks["players"])):
            frame_boxes = []
            for pid, pdata in tracks["players"][f].items():
                b = pdata.get("bbox")
                if b:
                    frame_boxes.append({
                        "id": int(pid),
                        "bbox": [round(float(x), 1) for x in b],
                        "role": "goalkeeper" if pdata.get("is_goalkeeper", False) else "player",
                        "team": int(pdata.get("team", 0)) if pdata.get("team") is not None else None,
                        "conf": round(float(pdata.get("confidence", 0.9)), 2),
                        "speed": round(float(pdata.get("speed", 0.0)), 1) if pdata.get("speed") is not None else 0.0
                    })
            if "referees" in tracks and f < len(tracks["referees"]):
                for rid, rdata in tracks["referees"][f].items():
                    b = rdata.get("bbox")
                    if b:
                        frame_boxes.append({
                            "id": int(rid),
                            "bbox": [round(float(x), 1) for x in b],
                            "role": "referee",
                            "team": None,
                            "conf": round(float(rdata.get("confidence", 0.9)), 2),
                            "speed": round(float(rdata.get("speed", 0.0)), 1) if rdata.get("speed") is not None else 0.0
                        })
            export_data["frames"][str(f)] = frame_boxes
        with open(export_tracks_path, "w", encoding="utf-8") as fp:
            json.dump(export_data, fp, indent=2, ensure_ascii=False)
        print(f"-> [Export Tracks] Đã lưu Pre-labeled Tracks đầy đủ vào: {export_tracks_path}")

    if no_render:
        t_total = time.time() - t_start
        mins, secs = divmod(t_total, 60)
        print("\n" + "=" * 74)
        print(f"✅ HOÀN TẤT TRÍCH XUẤT TRACKS (--no-render) TRONG {int(mins)}m {secs:04.1f}s")
        print("=" * 74 + "\n")
        return tracks

    t_p2_start = time.time()
    print(f"=== [PASS 2/2] VẼ ANNOTATION VÀ LƯU VIDEO CHÍNH ===")
    video_frame_to_save, _ = read_video(video_path)

    output_frames = tracker.draw_tracked_frames(video_frame_to_save, tracks, team_ball_control)
    output_frames = CameraMotion.draw_camera_movement(output_frames, motion)
    output_frames = s_d_calculate.draw_speed_and_distance(output_frames, tracks)

    save_video(output_frames, output_path, fps)
    t_pass2 = time.time() - t_p2_start
    print(f"XONG! Video phân tích chính đã lưu tại: {output_path} ({t_pass2:.2f}s)")

    # =========================================================================
    # MINIMAP VIDEO: Tạo video 2D Sa bàn chiến thuật (Tactical Radar) độc lập
    # =========================================================================
    t_minimap = 0.0
    if generate_minimap:
        t_mini_start = time.time()
        if output_path.lower().endswith(".mp4"):
            minimap_output_path = output_path[:-4] + "_minimap.mp4"
        else:
            minimap_output_path = output_path + "_minimap.mp4"

        print(f"\n=== [MINIMAP] BẮT ĐẦU TẠO VIDEO SA BÀN CHIẾN THUẬT 2D ===")
        minimap_renderer = MinimapRenderer(fps=fps, canvas_width=1280, canvas_height=720)
        minimap_renderer.render_video(
            tracks,
            team_ball_control,
            minimap_output_path,
            total_frames=len(tracks["players"]),
        )
        t_minimap = time.time() - t_mini_start
        print(f"XONG! Video Minimap chiến thuật đã lưu tại: {minimap_output_path} ({t_minimap:.2f}s)")

    # =========================================================================
    # BẢNG TỔNG KẾT THỜI GIAN THỰC THI (PROFILER)
    # =========================================================================
    t_total = time.time() - t_start
    mins, secs = divmod(t_total, 60)

    print("\n" + "=" * 74)
    print("⏱️  BẢNG TỔNG KẾT THỜI GIAN THỰC THI TOÀN BỘ PIPELINE")
    print("=" * 74)
    print(f"  • Pass 1 (YOLO Inference + ReID Tracking + Camera): {t_pass1:6.2f} s ({t_pass1/t_total*100:5.1f}%)")
    if use_sam2_backward:
        print(f"  • SAM 2 Backward Refinement (Phân rã Cluster):      {t_sam2:6.2f} s ({t_sam2/t_total*100:5.1f}%)")
    print(f"  • Xử lý Trung gian (Viterbi + Sân + Vận tốc/Cự ly): {t_intermediate:6.2f} s ({t_intermediate/t_total*100:5.1f}%)")
    print(f"  • Pass 2 (Vẽ Broadcast Annotations & Encode MP4):   {t_pass2:6.2f} s ({t_pass2/t_total*100:5.1f}%)")
    if generate_minimap:
        print(f"  • Tạo Video Sa bàn chiến thuật (Tactical Minimap):  {t_minimap:6.2f} s ({t_minimap/t_total*100:5.1f}%)")
    print("-" * 74)
    print(f"  👉 TỔNG THỜI GIAN THỰC THI TOÀN BỘ:                 {int(mins)} phút {secs:04.1f} giây ({t_total:.2f} s)")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Analysis Video Processing (3-Tier & Minimap Radar)")
    parser.add_argument(
        "output_name",
        nargs="?",
        default=None,
        help="Tên video đầu ra (ví dụ: output_optimized.mp4)",
    )
    parser.add_argument(
        "--video",
        default=PATH_VID,
        help="Đường dẫn video đầu vào (mặc định: video/sample.mp4)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Đường dẫn trọng số model YOLO (mặc định: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--player-conf",
        type=float,
        default=0.35,
        help="Ngưỡng confidence cho Cầu thủ (mặc định: 0.35)",
    )
    parser.add_argument(
        "--gk-conf",
        type=float,
        default=0.30,
        help="Ngưỡng confidence cho Thủ môn (mặc định: 0.30)",
    )
    parser.add_argument(
        "--ref-conf",
        type=float,
        default=0.30,
        help="Ngưỡng confidence cho Trọng tài (mặc định: 0.30)",
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.20,
        help="Ngưỡng confidence cho Quả bóng (mặc định: 0.20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Kích thước batch cho YOLO inference (mặc định: 8)",
    )
    parser.add_argument(
        "--calib",
        default="calib/points.json",
        help="Đường dẫn file JSON điểm hiệu chuẩn sân (mặc định: calib/points.json)",
    )
    parser.add_argument(
        "--no-minimap",
        action="store_true",
        help="Bật cờ này nếu không muốn xuất video Minimap radar 2D",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Thiết bị xử lý (ví dụ: 0, 'cpu'; mặc định tự động chọn CUDA nếu có)",
    )
    parser.add_argument(
        "--reid-min-conf",
        type=float,
        default=None,
        help="Ngưỡng lọc confidence riêng trước ReIDTrackerFacade.update (mặc định: None = dùng player_conf/gk_conf/ref_conf)",
    )
    parser.add_argument(
        "--draw-interpolated-ball",
        action="store_true",
        default=True,
        help="Vẽ bóng nội suy với ký hiệu riêng (vòng tròn rỗng) thay vì ẩn đi (mặc định: True)",
    )
    parser.add_argument(
        "--no-draw-interpolated-ball",
        action="store_false",
        dest="draw_interpolated_ball",
        help="Ẩn bóng nội suy, chỉ vẽ bóng raw detection",
    )
    parser.add_argument(
        "--possession-hysteresis",
        type=int,
        default=3,
        help="Số frame thắng liên tục yêu cầu để đổi chủ bóng (mặc định: 3 = ổn định chống nhấp nháy)",
    )
    parser.add_argument(
        "--ball-selector",
        choices=["greedy", "viterbi"],
        default="viterbi",
        help="Bộ chọn bóng: 'viterbi' (mặc định tối ưu toàn cục) hoặc 'greedy' (bộ chọn đơn frame cũ)",
    )
    parser.add_argument(
        "--ball-imgsz",
        type=int,
        default=1920,
        help="Độ phân giải inference YOLO riêng cho lớp bóng (mặc định: 1920).",
    )
    parser.add_argument(
        "--use-sam2-backward",
        action="store_true",
        default=True,
        help="Kích hoạt SAM 2 Backward Tracking để phân tách các cụm va chạm/che khuất (mặc định: True)",
    )
    parser.add_argument(
        "--no-sam2-backward",
        action="store_false",
        dest="use_sam2_backward",
        help="Tắt SAM 2 Backward Tracking",
    )
    parser.add_argument(
        "--export-tracks",
        default=None,
        help="Đường dẫn file JSON xuất kết quả Pre-labeled Tracks đầy đủ (ví dụ: data/prelabeled_tracks/sample.json)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Bỏ qua bước vẽ và encode video MP4 (chỉ xuất tracks & homography)",
    )
    parser.add_argument(
        "--read-from-stub",
        action="store_true",
        help="Đọc dữ liệu Pass 1 từ cache stub nếu có (tiết kiệm thời gian)",
    )
    parser.add_argument(
        "--stub-path",
        default=None,
        help="Đường dẫn file stub cache tracks (mặc định: tự động theo tên video)",
    )
    parser.add_argument(
        "--camera-stub-path",
        default=None,
        help="Đường dẫn file stub camera motion (mặc định: tự động theo tên video)",
    )
    args = parser.parse_args()

    main(
        output_name=args.output_name,
        video_path=args.video,
        model_path=args.model,
        player_conf=args.player_conf,
        goalkeeper_conf=args.gk_conf,
        referee_conf=args.ref_conf,
        ball_conf=args.ball_conf,
        batch_size=args.batch_size,
        calib_path=args.calib,
        generate_minimap=not args.no_minimap,
        device=args.device,
        reid_min_conf=args.reid_min_conf,
        draw_interpolated_ball=args.draw_interpolated_ball,
        possession_hysteresis=args.possession_hysteresis,
        ball_selector=args.ball_selector,
        ball_imgsz=args.ball_imgsz,
        use_sam2_backward=args.use_sam2_backward,
        export_tracks_path=args.export_tracks,
        no_render=args.no_render,
        read_from_stub=args.read_from_stub,
        stub_path=args.stub_path,
        camera_stub_path=args.camera_stub_path,
    )
