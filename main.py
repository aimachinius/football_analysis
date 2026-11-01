import argparse
import os
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


def validate_roles_and_assign_goalkeepers(tracks, team_assigner):
    """
    [TẦNG 3]: Thẩm định không gian sân bóng chuẩn FIFA (105m x 68m) & Gán đội cho Thủ môn.
      1. Phân định Thủ môn vs Trọng tài dựa vào vị trí thực tế trên sân:
         - Vị trí trung bình trong vòng cấm (x <= 18m hoặc x >= 87m): Khẳng định là THỦ MÔN.
         - Vị trí trung bình ở giữa sân (25m <= x <= 80m): Khẳng định là TRỌNG TÀI.
      2. Gán Thủ môn về đúng Đội bóng theo khung thành bảo vệ:
         - Tính nửa sân mà Đội 1 và Đội 2 đang phòng thủ.
         - Thủ môn bên nào thì thuộc về Đội đang phòng thủ bên đó.
    """
    player_positions = defaultdict(list)
    referee_positions = defaultdict(list)

    for frame in tracks["players"]:
        for pid, pinfo in frame.items():
            pos = pinfo.get("position_transformed")
            if pos is not None and np.all(np.isfinite(pos)):
                player_positions[pid].append(pos)

    for frame in tracks["referees"]:
        for rid, rinfo in frame.items():
            pos = rinfo.get("position_transformed")
            if pos is not None and np.all(np.isfinite(pos)):
                referee_positions[rid].append(pos)

    # 1. Thẩm định không gian: Trọng tài bị nhầm khi đứng cố định trong vòng cấm -> Chuyển thành Thủ môn
    to_move_from_ref_to_gk = []
    for rid, coords in referee_positions.items():
        if len(coords) >= 3:
            mean_x = float(np.mean([c[0] for c in coords]))
            mean_y = float(np.mean([c[1] for c in coords]))
            if (mean_x <= 18.0 or mean_x >= 87.0) and (15.0 <= mean_y <= 53.0):
                to_move_from_ref_to_gk.append(rid)

    for rid in to_move_from_ref_to_gk:
        print(f"-> [Tầng 3 Spatial Check] Chuyển Track ID {rid} từ Referee sang Thủ môn (GK) do đứng trong vòng cấm.")
        for frame_idx, frame in enumerate(tracks["referees"]):
            if rid in frame:
                rinfo = frame.pop(rid)
                rinfo["is_goalkeeper"] = True
                tracks["players"][frame_idx][rid] = rinfo

    # 2. Thẩm định không gian: Người bị đánh nhãn là GK nhưng lại chạy giữa sân -> Chuyển thành Trọng tài
    to_move_from_gk_to_ref = []
    for pid, coords in player_positions.items():
        if len(coords) >= 3:
            mean_x = float(np.mean([c[0] for c in coords]))
            is_gk_flag = any(frame.get(pid, {}).get("is_goalkeeper", False) for frame in tracks["players"])
            if is_gk_flag and (25.0 <= mean_x <= 80.0):
                to_move_from_gk_to_ref.append(pid)

    for pid in to_move_from_gk_to_ref:
        print(f"-> [Tầng 3 Spatial Check] Chuyển Track ID {pid} từ Goalkeeper sang Referee do chạy ở giữa sân.")
        for frame_idx, frame in enumerate(tracks["players"]):
            if pid in frame:
                pinfo = frame.pop(pid)
                pinfo.pop("is_goalkeeper", None)
                pinfo.pop("team", None)
                pinfo.pop("team_color", None)
                tracks["referees"][frame_idx][pid] = pinfo

    # 3. Gán Thủ môn về đúng Đội theo khung thành bảo vệ
    team1_x = []
    team2_x = []
    for frame in tracks["players"]:
        for p in frame.values():
            if p.get("is_goalkeeper", False):
                continue
            pos = p.get("position_transformed")
            if pos is not None and np.all(np.isfinite(pos)):
                t = p.get("team")
                if t == 1:
                    team1_x.append(pos[0])
                elif t == 2:
                    team2_x.append(pos[0])

    if team1_x and team2_x:
        med1 = float(np.median(team1_x))
        med2 = float(np.median(team2_x))
        team_defending_left = 1 if med1 < med2 else 2
        team_defending_right = 2 if team_defending_left == 1 else 1
    else:
        team_defending_left = 1
        team_defending_right = 2

    # Gán team và màu đội cho tất cả thủ môn
    for frame in tracks["players"]:
        for pid, p in frame.items():
            if p.get("is_goalkeeper", False):
                pos = p.get("position_transformed")
                if pos is not None and np.all(np.isfinite(pos)):
                    gk_team = team_defending_left if pos[0] < 52.5 else team_defending_right
                else:
                    mean_x = float(np.mean([c[0] for c in player_positions[pid]])) if player_positions[pid] else 0.0
                    gk_team = team_defending_left if mean_x < 52.5 else team_defending_right

                p["team"] = gk_team
                if team_assigner and hasattr(team_assigner, "team_colors"):
                    p["team_color"] = team_assigner.team_colors.get(gk_team, (0, 0, 255))


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
    draw_interpolated_ball=False,
    possession_hysteresis=0,
):
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
    )
    camera_motion = CameraMotion()
    team_assigner = TeamAssigner()

    # [GIAI ĐOẠN 3]: Nạp Global Calibration Anchor ban đầu vào CameraMotion để hỗ trợ Re-localization sau camera cut
    # Đọc frame độc lập bằng VideoCapture tạm thời để bảo toàn generator video_frames cho Pass 1
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
                    print(f"-> [Stage 3] Đã nạp Global Calibration Anchor frame {first_frame_idx} vào CameraMotion.")

    tracks, motion = tracker.process_pass1(
        video_frames,
        camera_motion=camera_motion,
        team_assigner=team_assigner,
        read_from_stub=False,
        stub_path="stubs/track_stubs_reid.pkl",
        camera_stub_path="stubs/camera_motion.pkl",
        choose_Al=2,
    )
    print(f"-> Pass 1 hoàn tất: đã phân tích {len(tracks['players'])} frames.")
    print(f"   + Số frame có cầu thủ/thủ môn: {sum(len(f) > 0 for f in tracks['players'])}")
    print(f"   + Số frame có trọng tài: {sum(len(f) > 0 for f in tracks['referees'])}")
    print(f"   + Số frame có bóng: {sum(len(f) > 0 for f in tracks['ball'])}")

    # =========================================================================
    # GIAI ĐOẠN TRUNG GIAN: Xử lý Toán học trên Metadata trong RAM (< 1 giây)
    # =========================================================================
    print("=== [XỬ LÝ DỮ LIỆU] Ổn định quỹ đạo bóng, Hiệu chuẩn sân, Thẩm định Không gian Tầng 3 & Tính tốc độ ===")

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

    # [TẦNG 3] Thẩm định không gian sân bóng & Gán đội cho Thủ môn
    validate_roles_and_assign_goalkeepers(tracks, team_assigner)

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

    # =========================================================================
    # PASS 2: Vẽ Annotation và Lưu Video ra Đĩa (Streaming 1 frame tại một thời điểm)
    # =========================================================================
    print(f"=== [PASS 2/2] VẼ ANNOTATION VÀ LƯU VIDEO CHÍNH ===")
    video_frame_to_save, _ = read_video(video_path)

    output_frames = tracker.draw_tracked_frames(video_frame_to_save, tracks, team_ball_control)
    output_frames = CameraMotion.draw_camera_movement(output_frames, motion)
    output_frames = s_d_calculate.draw_speed_and_distance(output_frames, tracks)

    save_video(output_frames, output_path, fps)
    print(f"XONG! Video phân tích chính đã lưu tại: {output_path}")

    # =========================================================================
    # MINIMAP VIDEO: Tạo video 2D Sa bàn chiến thuật (Tactical Radar) độc lập
    # =========================================================================
    if generate_minimap:
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
        print(f"XONG! Video Minimap chiến thuật đã lưu tại: {minimap_output_path}")


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
        help="Vẽ bóng nội suy với ký hiệu riêng (vòng tròn rỗng) thay vì ẩn đi (mặc định: False)",
    )
    parser.add_argument(
        "--possession-hysteresis",
        type=int,
        default=0,
        help="Số frame thắng liên tục yêu cầu để đổi chủ bóng (mặc định: 0 = tắt)",
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
    )
