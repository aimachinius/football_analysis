# BÁO CÁO KIỂM TOÁN MÃ NGUỒN: HỆ THỐNG THEO DÕI NGƯỜI VÀ BÓNG (TRACKING AUDIT)
**Dự án**: Football CV Analysis (`e:\CV_Football_Analys`)
**Thời điểm kiểm toán**: 2026-10-30
**Chế độ**: CHỈ ĐỌC (Code Audit - Read-Only Verification)
**Mục tiêu**: Trích xuất toàn diện, chuẩn xác và có dẫn chứng code thật về cơ chế theo dõi người (player, goalkeeper, referee) và bóng (ball).

---


## MỤC LỤC BÁO CÁO TOÀN DIỆN (CHIA THEO NHÓM MỤC)
Do tổng dung lượng báo cáo đạt ~186 KB (> 150 KB), báo cáo được chia thành 4 phần tự đủ ngữ cảnh theo quy định:
- [**Phần 1: Mục 1 – 3** (Cây thư mục, Luồng dữ liệu, Bước phát hiện YOLO)](CODE_AUDIT_TRACKING_part1.md)
- [**Phần 2: Mục 4** (Tracking người & Chi tiết Package Algorithmic Re-ID)](CODE_AUDIT_TRACKING_part2.md)
- [**Phần 3: Mục 5 – 7** (Tracking bóng, Hợp đồng dữ liệu, Bảng tham số tổng hợp)](CODE_AUDIT_TRACKING_part3.md)
- [**Phần 4: Mục 8 – 11 & Bảng Tự Kiểm** (Kiểm thử, Dữ liệu đánh giá, Bất thường code, Câu hỏi còn lại)](CODE_AUDIT_TRACKING_part4.md)

---

### MỤC 1. Cây thư mục và vai trò từng file

Dưới đây là danh mục toàn bộ các file Python trong dự án cùng 1 dòng mô tả vai trò thực tế sau khi đã mở và đọc nội dung từng file:

| STT | Đường dẫn file | Vai trò trong hệ thống |
|---|---|---|
| 1 | [`main.py`](main.py) | Điểm khởi nhập (entrypoint) chính điều phối toàn bộ pipeline 2 pass: đọc video, gọi Pass 1 tracking, xử lý metadata trung gian (ổn định bóng, homography sân, thẩm định vai trò tầng 3, tính tốc độ/quãng đường, gán bóng), Pass 2 render annotation và xuất video minimap chiến thuật 2D. |
| 2 | [`tracker/tracker.py`](tracker/tracker.py) | Module quản lý phát hiện và bám vết đối tượng; thực hiện batch inference YOLOv8/YOLOv26 (imgsz=1536), khử trùng lặp đa nhãn (Tầng 1), điều phối tracker (ByteTrack/BoT-SORT/ReID facade), duy trì lịch sử vai trò (Tầng 2), chọn bóng thô và render annotation hình elip/tam giác. |
| 3 | [`tracker/ball_tracker.py`](tracker/ball_tracker.py) | Module độc lập `BallTrajectoryStabilizer` (Milestone P1) ổn định quỹ đạo bóng; lọc ngoại lai động học (v_max=50 px/f), lọc chuỗi ngoại lai liên tiếp (lookahead chord burst k_max=4), bù chuyển động camera, nội suy thích ứng 3 tầng (PCHIP cho 1-4f, kiểm tra góc hướng cho 5-10f, strict no-op cho >10f) và làm mượt hồi quy ballistic <= 2.5 px. |
| 4 | [`tracker/__init__.py`](tracker/__init__.py) | File khởi tạo package `tracker`, export class `Tracker`. |
| 5 | [`algorithmic_reid/config.py`](algorithmic_reid/config.py) | File định nghĩa dataclass `ReIDConfig`, lưu trữ tập trung toàn bộ siêu tham số, trọng số ma trận chi phí, ngưỡng gating, ngưỡng confidence và cấu hình Kalman cho Algorithmic Re-ID. |
| 6 | [`algorithmic_reid/reid_tracker_facade.py`](algorithmic_reid/reid_tracker_facade.py) | Lớp Facade trung gian kết nối `TrackManager` với `tracker.py` và `main.py`; tiếp nhận `sv.Detections` cùng ma trận homography camera G_t, trả về container tương thích kép hỗ trợ truy xuất theo dict và supervision. |
| 7 | [`algorithmic_reid/__init__.py`](algorithmic_reid/__init__.py) | File khởi tạo package `algorithmic_reid`, export `ReIDTrackerFacade` và `ReIDConfig`. |
| 8 | [`algorithmic_reid/appearance/color_descriptor.py`](algorithmic_reid/appearance/color_descriptor.py) | Trích xuất biểu đồ màu đa vùng HSV (thân áo trên và quần/tất dưới) từ bbox cầu thủ và tính ma trận khoảng cách Bhattacharyya. |
| 9 | [`algorithmic_reid/appearance/temporal_aggregator.py`](algorithmic_reid/appearance/temporal_aggregator.py) | Quản lý cập nhật trung bình động lũy thừa (EMA) cho template ngoại quan theo thời gian, tích hợp cơ chế đóng băng (freeze) khi bị che khuất hoặc confidence thấp. |
| 10 | [`algorithmic_reid/appearance/__init__.py`](algorithmic_reid/appearance/__init__.py) | File khởi tạo package `appearance`, export các hàm trích xuất histogram và khoảng cách Bhattacharyya. |
| 11 | [`algorithmic_reid/association/cost_matrix.py`](algorithmic_reid/association/cost_matrix.py) | Tính toán ma trận chi phí chuẩn hóa hợp nhất đa đặc trưng: Distance-IoU (DIoU), khoảng cách Mahalanobis, ngoại quan Bhattacharyya và độ lệch góc hướng di chuyển. |
| 12 | [`algorithmic_reid/association/gating.py`](algorithmic_reid/association/gating.py) | Áp dụng các rào cản loại trừ cứng (hard gating): cách ly loài cứng (Species Isolation giữa player/gk/referee), rào cản ngoại quan (Bhattacharyya gate) và rào cản không gian Mahalanobis kết hợp khoảng cách Euclid. |
| 13 | [`algorithmic_reid/association/hungarian.py`](algorithmic_reid/association/hungarian.py) | Giải bài toán gán cặp tối ưu hai phía (Bipartite Matching) thông qua `scipy.optimize.linear_sum_assignment` và lọc bỏ các cặp vượt ngưỡng chi phí tối đa. |
| 14 | [`algorithmic_reid/association/__init__.py`](algorithmic_reid/association/__init__.py) | File khởi tạo package `association`, export hàm chi phí, gating và matching. |
| 15 | [`algorithmic_reid/lifecycle/tracklet.py`](algorithmic_reid/lifecycle/tracklet.py) | Cấu trúc dữ liệu `Tracklet` và máy trạng thái vòng đời (TENTATIVE, CONFIRMED, LOST, DELETED), quản lý trạng thái Kalman (mean, covariance) và lịch sử tọa độ. |
| 16 | [`algorithmic_reid/lifecycle/track_manager.py`](algorithmic_reid/lifecycle/track_manager.py) | Bộ điều phối liên kết 4 tầng (Stage 1: active confirmed <-> high-conf; Stage 2: remaining active <-> low-conf; Stage 3: lost confirmed <-> remaining high-conf; Stage 4: tentative <-> remaining dets), sinh track mới và khâu nối tracklet. |
| 17 | [`algorithmic_reid/lifecycle/stitcher.py`](algorithmic_reid/lifecycle/stitcher.py) | Khâu nối các phân mảnh tracklet bị đứt quãng từ 1 đến 30 frame dựa trên không chồng lấn thời gian, cùng class, vận tốc khả thi và độ tương đồng ngoại quan. |
| 18 | [`algorithmic_reid/lifecycle/__init__.py`](algorithmic_reid/lifecycle/__init__.py) | File khởi tạo package `lifecycle`, export `Tracklet`, `TrackManager`, `TrackletStitcher`. |
| 19 | [`algorithmic_reid/motion/kalman_filter.py`](algorithmic_reid/motion/kalman_filter.py) | Bộ lọc Kalman 8 trạng thái có hệ số cản vận tốc gamma=0.90 khi mất quan sát và cố định tỷ lệ khung hình (dot_a = 0) để chống vọt lố tọa độ. |
| 20 | [`algorithmic_reid/motion/mahalanobis.py`](algorithmic_reid/motion/mahalanobis.py) | Tính toán ma trận khoảng cách Mahalanobis từ sai số đổi mới (innovation) và ma trận hiệp phương sai chiếu S thông qua phân rã Cholesky. |
| 21 | [`algorithmic_reid/motion/motion_predictor.py`](algorithmic_reid/motion/motion_predictor.py) | Bù trừ chuyển động camera (CMC) cho các tracklet bằng cách biến đổi 4 góc bounding box dự đoán qua ma trận homography liên frame G_t. |
| 22 | [`algorithmic_reid/motion/__init__.py`](algorithmic_reid/motion/__init__.py) | File khởi tạo package `motion`, export `DampedKalmanFilter`, hàm Mahalanobis và CMC. |
| 23 | [`camera_motion.py`](camera_motion.py) | Ước lượng chuyển động camera bằng Anchor-Grounded Tracking; phát hiện camera cut, quản lý anchor toàn cục/cục bộ, tính homography liên frame G_t và tái định vị (re-localization). |
| 24 | [`camera_movement_estimate/camera_movement_estimate.py`](camera_movement_estimate/camera_movement_estimate.py) | Module ước lượng chuyển động camera cũ (legacy) dựa trên luồng quang học Lucas-Kanade điểm mốc thô không có mô hình anchor. |
| 25 | [`camera_movement_estimate/__init__.py`](camera_movement_estimate/__init__.py) | File khởi tạo package `camera_movement_estimate`. |
| 26 | [`pitch_homo.py`](pitch_homo.py) | Module chuyển đổi tọa độ sân bóng FIFA (105x68m) qua Homography MAGSAC; tính toán safe_project, extrapolation_risk, quản lý Model Hull và Observed Hull, tích hợp bộ giải conic Giai đoạn 4. |
| 27 | [`pitch_homo_v2.py`](pitch_homo_v2.py) | Bản sao lưu thử nghiệm trước đây của module homography sân cỏ với tiện ích CLI pick điểm mốc từ video. |
| 28 | [`central_view_geometry.py`](central_view_geometry.py) | Giải thuật hình học xạ ảnh cho khung nhìn trung tâm (Central View) khi các điểm mốc bị thẳng hàng; khớp conic elip vòng tròn giữa sân và đường đối cực/hàng điểm điều hòa. |
| 29 | [`minimap/minimap_renderer.py`](minimap/minimap_renderer.py) | Bộ tạo video sa bàn chiến thuật 2D độc lập (1280x720); vẽ sân cỏ FIFA, hiển thị vị trí cầu thủ theo màu đội, số ID, thủ môn (GK), trọng tài (R), quả bóng và vệt chuyển động. |
| 30 | [`minimap/__init__.py`](minimap/__init__.py) | File khởi tạo package `minimap`, export `MinimapRenderer`. |
| 31 | [`player_ball_assigner/player_ball_assigner.py`](player_ball_assigner/player_ball_assigner.py) | Gán quyền kiểm soát bóng cho cầu thủ gần nhất dựa trên khoảng cách Euclid 2D giữa tâm bóng và 2 điểm góc chân cầu thủ trong không gian pixel ảnh (ngưỡng 70 px). |
| 32 | [`player_ball_assigner/__init__.py`](player_ball_assigner/__init__.py) | File khởi tạo package `player_ball_assigner`. |
| 33 | [`speed_distance/speed_distance_calc.py`](speed_distance/speed_distance_calc.py) | Tính toán vận tốc (km/h) và quãng đường tích lũy (m) của cầu thủ theo cửa sổ trượt 5 frame trong hệ tọa độ mét mặt sân, có kiểm tra cờ trạng thái hiệu chuẩn. |
| 34 | [`speed_distance/__init__.py`](speed_distance/__init__.py) | File khởi tạo package `speed_distance`. |
| 35 | [`team_assignment/team_assigner.py`](team_assignment/team_assigner.py) | Phân chia đội bóng dựa trên K-Means clustering (2 hoặc 3 cụm) màu áo nửa thân trên cầu thủ trong không gian BGR và bỏ phiếu đa số qua cửa sổ trượt 15 frame. |
| 36 | [`team_assignment/__init__.py`](team_assignment/__init__.py) | File khởi tạo package `team_assignment`. |
| 37 | [`utils/bbox_utils.py`](utils/bbox_utils.py) | Chứa các hàm tiện ích hình học bounding box: `get_center_of_box`, `get_bbox_width`, `measure_distance`, `measure_xy_distance`, `get_foot_position`. |
| 38 | [`utils/video_utils.py`](utils/video_utils.py) | Tiện ích đọc video dạng generator (`read_video`) và ghi video (`save_video`) với cơ chế giải phóng capture/writer an toàn. |
| 39 | [`utils/__init__.py`](utils/__init__.py) | File khởi tạo package `utils`, export các tiện ích video và bbox. |
| 40 | [`view_transformer/view_transformer.py`](view_transformer/view_transformer.py) | Module biến đổi tọa độ 4 điểm góc sân cũ (legacy), đã được thay thế hoàn toàn bởi `pitch_homo.py`. |
| 41 | [`view_transformer/__init__.py`](view_transformer/__init__.py) | File khởi tạo package `view_transformer`. |
| 42 | [`detect/yolo_detect_test.py`](detect/yolo_detect_test.py) | Script kiểm thử nhanh khả năng suy luận phát hiện của model YOLO trên video mẫu cũ. |
| 43 | [`checkupdate.py`](checkupdate.py) | Script kiểm tra các thuộc tính có sẵn trong thư viện `supervision`. |
| 44 | [`extract_frame.py`](extract_frame.py) | Script trích xuất frame định kỳ từ video sang file ảnh (lọc theo độ nét Laplacian) để gán nhãn dữ liệu. |
| 45 | [`high_frame.py`](high_frame.py) | Script tải và xử lý video YouTube chất lượng cao bằng `yt_dlp` và `imageio_ffmpeg`. |
| 46 | [`test.py`](test.py) | Script kiểm thử nạp model `best_YOLOV11s_FB_detect.pt` và chạy predict trên ảnh tĩnh đơn lẻ `image.png` với ngưỡng conf=0.001. |
| 47 | [`test_reid_tracking_suite.py`](test_reid_tracking_suite.py) | Bộ kiểm thử chuẩn hóa 8 kịch bản cho Player Tracking & Re-ID (cầu thủ giao nhau, dropout 1-10 frame, dao động conf, đồng đội sát nhau, camera pan, false det, species isolation, downstream contract). |
| 48 | [`test_reid_adversarial_stress.py`](test_reid_adversarial_stress.py) | Bộ kiểm thử ứng suất đối kháng 16 kịch bản cho Algorithmic Re-ID (4 người giao nhau, mất dấu 15-20 frame đường cong/đổi hướng, camera pan 25-75 px/f, penalty box cluster). |
| 49 | [`test_challenger_p0_stress.py`](test_challenger_p0_stress.py) | Bộ kiểm thử ứng suất đối kháng P0 của Challenger (rung lắc bbox +-15px, dao động conf 0.20-0.25, đổi hướng zigzag 10-32 px/f). |
| 50 | [`test_ball_trajectory_suite.py`](test_ball_trajectory_suite.py) | Bộ kiểm thử chuẩn hóa 10 kịch bản cho Ball Trajectory (làm mượt RMS, loại ngoại lai 300px, dropout 1-3f, dropout đổi hướng, strict no-op dropout 60+f, sút bóng gia tốc, bóng đứng yên, camera pan, bóng ra biên, hợp đồng minimap). |
| 51 | [`test_challenger_p1_stress.py`](test_challenger_p1_stress.py) | Bộ kiểm thử ứng suất đối kháng P1 của Challenger (chuỗi ngoại lai liên tiếp ADV-B01 2-3 frame, ngoại lai biên, camera rung giật). |
| 52 | [`test_stage2_suite.py`](test_stage2_suite.py) | Bộ kiểm thử hồi quy Giai đoạn 2 (Homography vững chắc, MAGSAC vs DLT, safe_project, extrapolation_risk, invariant data contract). |
| 53 | [`test_stage3_suite.py`](test_stage3_suite.py) | Bộ kiểm thử hồi quy Giai đoạn 3 (Anchor-grounded camera tracking, kiểm soát drift < 5px, tái định vị sau camera cut). |
| 54 | [`test_stage4_suite.py`](test_stage4_suite.py) | Bộ kiểm thử hồi quy Giai đoạn 4 (Central View Conic Geometry, Projective Ellipse Fitting, Harmonic Tetrad, Halfway Line Recovery). |
| 55 | [`verify_central_view_real_frame.py`](verify_central_view_real_frame.py) | Script kiểm chứng thực nghiệm độc lập hình học Conic và Support Hull trên frame 1 của video thực tế `video/sample.mp4`. |
| 56 | [`benchmark_pipeline.py`](benchmark_pipeline.py) | Script đo đạc hiệu năng toàn diện của pipeline (thời gian, FPS, RAM RSS, VRAM GPU) trên video `video/sample.mp4`. |
| 57 | [`benchmark_reid_tracking.py`](benchmark_reid_tracking.py) | Script đánh giá định lượng độ ổn định tracking trên video thực tế (số lượng unique IDs, track fragmentation, team swapping, cross-class contamination). |
| 58 | [`run_benchmark_m6.py`](run_benchmark_m6.py) | Script giám sát tài nguyên đa luồng và benchmark pipeline cho Milestone M6. |
| 59 | [`evaluate_video_stage3.py`](evaluate_video_stage3.py) | Script đánh giá thực nghiệm tracking chuyển động camera và phát hiện cắt cảnh Giai đoạn 3 trên `video/sample.mp4`. |

---

### MỤC 2. Luồng dữ liệu tổng thể

#### 2.1 Trích NGUYÊN VĂN phần thân hàm điều phối chính trong `main.py`

Dưới đây là nguyên văn phần thân hàm điều phối chính `main` trong `main.py` từ khai báo tham số, đọc video, Pass 1, xử lý metadata trung gian, Pass 2 đến kết xuất video minimap:

```python
# main.py:121-308
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
    player_assigner = PlayerBallAssigner()
    for frame_num, player_track in enumerate(tracks['players']):
        assigned_player = -1
        if 1 in tracks['ball'][frame_num]:
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
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

```

#### 2.2 Danh sách các bước theo ĐÚNG thứ tự thực thi thật

| Bước | Tên bước / Thao tác | File:Dòng | Dữ liệu đầu vào | Dữ liệu đầu ra |
|---|---|---|---|---|
| 1. Khởi tạo cấu hình và tham số đầu vào | main.py:134-144 | `output_name, video_path, model_path, calib_path` | `output_path` |
| 2. Tự động phát hiện và lựa chọn thiết bị phần cứng | main.py:159-170 | `device (hoặc None)` | `target_device ('cuda:0' hoặc 'cpu')` |
| 3. Mở luồng đọc video đầu vào (Generator) | main.py:171 | `video_path` | `video_frames (generator), fps (float)` |
| 4. Khởi tạo các đối tượng điều phối | main.py:172-184 | `model_path, fps, confs, batch_size, device` | `tracker (Tracker), camera_motion (CameraMotion), team_assigner (TeamAssigner)` |
| 5. Nạp Global Calibration Anchor ban đầu | main.py:187-208 | `calib_path, video_path` | `camera_motion.global_anchors (frame 0)` |
| 6. Thực thi [PASS 1] Tracking, Camera Motion & Team Assigner | main.py:210-218 | `video_frames, camera_motion, team_assigner, choose_Al=2` | `tracks (dict gồm 'players', 'referees', 'ball'), motion (CameraMotionResult)` |
| 7. Ổn định quỹ đạo bóng và nội suy thích ứng 3 tầng [Milestone P1] | main.py:230-234 | `tracks['ball'], motion, tracks['players']` | `tracks['ball'] (đã ổn định, lọc ngoại lai, nội suy PCHIP)` |
| 8. Gán tọa độ điểm chân và tâm bóng vào tracks | main.py:236 | `tracks` | `tracks[...][...]['position'] (tọa độ pixel ảnh thô)` |
| 9. Khớp mô hình hiệu chuẩn sân cỏ (Pitch Calibration) | main.py:239 | `load_points(calib_path), motion` | `calib (PitchCalibration)` |
| 10. Chẩn đoán trạng thái phân đoạn và cảnh báo UNREGISTERED | main.py:243-249 | `motion.seg, calib.H_by_seg` | `Log thống kê phân đoạn` |
| 11. Khởi tạo PitchMapper và chiếu tọa độ toàn bộ tracks sang mét | main.py:251-252 | `calib, motion, tracks` | `tracks[...][...]['position_local_raw', 'position_adjusted', 'position_transformed', 'projection_valid', 'calibration_status', 'extrapolation_risk', 'inside_pitch']` |
| 12. [TẦNG 3] Thẩm định không gian sân bóng & Gán đội cho Thủ môn | main.py:255 | `tracks, team_assigner` | `Chuyển role referee <-> goalkeeper theo vị trí mét, gán team/màu cho thủ môn` |
| 13. Tính toán vận tốc (km/h) và quãng đường (m) | main.py:258-259 | `fps, tracks` | `tracks['players'][...]['speed', 'distance', 'is_predicted', 'speed_is_predicted']` |
| 14. Gán quyền kiểm soát bóng cho cầu thủ và thống kê tỷ lệ đội | main.py:262-275 | `tracks['players'], tracks['ball']` | `tracks['players'][...]['has_ball'], team_ball_control (np.ndarray)` |
| 15. Mở lại luồng đọc video lần 2 cho [PASS 2] Render Annotation | main.py:281 | `video_path` | `video_frame_to_save (generator)` |
| 16. Vẽ Annotation cầu thủ, trọng tài, bóng và dashboard | main.py:283 | `video_frame_to_save, tracks, team_ball_control` | `output_frames (generator)` |
| 17. Vẽ thông tin chuyển động camera (Pan/Zoom) | main.py:284 | `output_frames, motion` | `output_frames (generator)` |
| 18. Vẽ vận tốc và quãng đường lên video | main.py:285 | `output_frames, tracks` | `output_frames (generator)` |
| 19. Ghi video phân tích chính ra file MP4 | main.py:287 | `output_frames, output_path, fps` | `File video output_path hoàn tất` |
| 20. Kết xuất video sa bàn chiến thuật 2D độc lập (Minimap Radar) | main.py:300-306 | `tracks, team_ball_control, minimap_output_path` | `File video minimap_output_path hoàn tất` |

#### 2.3 Cờ, tham số dòng lệnh và biến cấu hình chọn thuật toán

Trong hệ thống, biến `choose_Al` quyết định bộ điều khiển tracking người được sử dụng:

- **Vị trí định nghĩa và mặc định thực tế**:

  - Trong `main.py:217`: `choose_Al=2` được truyền trực tiếp khi gọi `tracker.process_pass1(...)`. **Đây là giá trị mặc định thực tế khi chạy pipeline.**

  - Trong `tracker/tracker.py:227`: `choose_Al=2` là giá trị mặc định của tham số hàm `process_pass1`.

  - Trong `tracker/tracker.py:412`: `choose_Al=0` là giá trị mặc định trong hàm cũ `get_object_track` (phục vụ tương thích ngược).

- **Các giá trị có thể của `choose_Al` và luồng mã được kích hoạt**:

  1. `choose_Al == 0`:

     - Kích hoạt `self.tracker = ByteTrackTracker(...)` (`tracker/tracker.py:36-43`).

     - Lọc phát hiện người qua hàm `_suppress_cross_class_overlaps(det, ids, iou_thresh=0.45)` (`tracker/tracker.py:281`).

     - Hậu xử lý vai trò bằng cơ chế bỏ phiếu chuỗi thời gian Tầng 2 `track_role_history` (`tracker/tracker.py:305-337`).

  2. `choose_Al == 1`:

     - Kích hoạt `self.BoTTracker = BoTSORTTracker(...)` có bật `enable_cmc=True, cmc_method='sparseOptFlow'` (`tracker/tracker.py:44-56`).

     - Lọc phát hiện người qua hàm `_suppress_cross_class_overlaps(det, ids, iou_thresh=0.45)` (`tracker/tracker.py:285`).

     - Hậu xử lý vai trò bằng cơ chế bỏ phiếu chuỗi thời gian Tầng 2 `track_role_history` (`tracker/tracker.py:305-337`).

  3. `choose_Al == 2`:

     - Kích hoạt `self.reid_tracker = ReIDTrackerFacade(...)` từ package `algorithmic_reid` (`tracker/tracker.py:58-66`).

     - Lọc phát hiện người bằng mask boolean trực tiếp `is_person` (`tracker/tracker.py:290-295`).

     - Bù trừ chuyển động camera qua homography $G_t$ (`tracker/tracker.py:296`).

     - Áp dụng 4 tầng liên kết phân cấp, lọc Kalman có cản vận tốc $\gamma=0.90$, gating Mahalanobis, rào cản loài cứng (Species Isolation) và khâu nối tracklet (`TrackletStitcher`).

     - Phân định role trực tiếp từ kết quả của Facade (`tracked['players']`, `tracked['referees']`) (`tracker/tracker.py:300-302`).


---
### MỤC 3. Bước phát hiện (YOLO) và hậu xử lý detection

#### 3.1 Chỗ nạp model và gọi `predict` / `track` / `__call__`

Trong dự án, việc nạp model và thực thi suy luận diễn ra tập trung tại `tracker/tracker.py` thông qua thư viện `ultralytics.YOLO`:

- **Khai báo model mặc định** (`main.py:17-18`):

```python
# main.py:16-18
PATH_VID = 'video/sample.mp4'
DEFAULT_MODEL = 'best_V26_over_6000.pt' if os.path.exists('best_V26_over_6000.pt') else 'best_YOLOv26s_FB_OD.pt'

```

- **Nạp model trong hàm khởi tạo `Tracker.__init__`** (`tracker/tracker.py:35`):

```python
# tracker/tracker.py:35-35
        self.model = YOLO(model_path)   
```

- **Phương thức thực thi suy luận batch `Tracker.detect_frame`** (`tracker/tracker.py:185-218`):

```python
# tracker/tracker.py:185-218
    def detect_frame(self, list_frame):
        batch_size = self.batch_size
        device = self.device
        batch = []

        for frame in list_frame:
            batch.append(frame)

            if len(batch) == batch_size:
                results = self.model.predict(
                    source=batch,
                    stream=True,
                    verbose=False,
                    conf=self.min_predict_conf,
                    device=device,
                    imgsz=1536,
                    agnostic_nms=False
                )
                batch = []
                yield from results

        if batch:
            results = self.model.predict(
                source=batch,
                stream=True,
                verbose=False,
                conf=self.min_predict_conf,
                device=device,
                imgsz=1536,
                agnostic_nms=False
            )
            batch = []
            yield from results

```


**Bảng chi tiết toàn bộ tham số truyền vào `model.predict(...)`:**

| Tên tham số | Giá trị cụ thể | Nơi định nghĩa / truyền vào | Tác dụng trong mã nguồn |
|---|---|---|---|
| `source` | `batch` | `tracker/tracker.py:195, 208` | Danh sách frame ảnh BGR (độ dài = batch_size = 8). |
| `stream` | `True` | `tracker/tracker.py:196, 209` | Chế độ streaming generator, giải phóng bộ nhớ từng frame sau khi yield. |
| `verbose` | `False` | `tracker/tracker.py:197, 210` | Tắt log in chi tiết từng batch của Ultralytics. |
| `conf` | `self.min_predict_conf` | `tracker/tracker.py:198, 211` | Giá trị tính toán động: max(0.10, min(player_conf, gk_conf, ref_conf, ball_conf) - 0.05). Với mặc định (0.35, 0.30, 0.30, 0.20), giá trị là 0.15. Định nghĩa tại tracker/tracker.py:86-88. |
| `device` | `self.device` | `tracker/tracker.py:199, 212` | 0 (CUDA GPU) nếu torch.cuda.is_available(), ngược lại 'cpu' hoặc giá trị do người dùng truyền qua CLI --device. Định nghĩa tại main.py:160 và tracker/tracker.py:31. |
| `imgsz` | `1536` | `tracker/tracker.py:200, 213` | Kích thước ảnh đầu vào khi suy luận YOLO (hardcode 1536x1536). |
| `agnostic_nms` | `False` | `tracker/tracker.py:201, 214` | NMS theo từng class độc lập của Ultralytics, không gộp class. |
| `iou` | `KHÔNG TRUYỀN (mặc định 0.7)` | `Mặc định Ultralytics` | Sử dụng giá trị mặc định của Ultralytics cho NMS IoU. |
| `max_det` | `KHÔNG TRUYỀN (mặc định 300)` | `Mặc định Ultralytics` | Sử dụng giá trị mặc định của Ultralytics cho số lượng detection tối đa mỗi ảnh. |
| `classes` | `KHÔNG TRUYỀN (None)` | `Mặc định Ultralytics` | Nhận diện toàn bộ các class mà model đã học. |
| `half` | `KHÔNG TRUYỀN (False / FP32)` | `Mặc định Ultralytics` | Chạy ở độ chính xác dấu phẩy động 32-bit (FP32) đầy đủ. |
| `augment` | `KHÔNG TRUYỀN (False)` | `Mặc định Ultralytics` | Không sử dụng Test-Time Augmentation (TTA). |

#### 3.2 Tên class và ánh xạ id -> tên (names)

Tên class và ánh xạ ID được xác định trực tiếp từ weights model thông qua `d_frame.names` tại runtime:

```python
# tracker/tracker.py:264-265
            if ids is None:
                ids = {v: k for k, v in d_frame.names.items()}
```

Đối chiếu với file nhãn [`data.yaml:16-21`](data.yaml:16-21):

```yaml
# data.yaml:16-21
nc: 4
names:
- ball
- goalkeeper
- player
- referee```

- **Bảng ánh xạ chuẩn**:

  - ID `0` -> `ball` (Quả bóng)

  - ID `1` -> `goalkeeper` (Thủ môn)

  - ID `2` -> `player` (Cầu thủ)

  - ID `3` -> `referee` (Trọng tài)

- **Xác nhận đổi tên / đổi ID class**: Trong toàn bộ codebase, **KHÔNG CÓ** thao tác đổi tên hoặc đổi ID của các class này. Trong `algorithmic_reid/config.py:53-57`, các hằng số class cũng được định nghĩa khớp 100%:

```python
# algorithmic_reid/config.py:53-57
    # Species class mapping
    class_ball: int = 0
    class_goalkeeper: int = 1
    class_player: int = 2
    class_referee: int = 3
```


#### 3.3 Bảng tổng hợp mọi ngưỡng confidence đang dùng trong hệ thống

| Tên tham số | Giá trị | File:Dòng | Áp dụng cho class nào | Tác dụng |
|---|---|---|---|---|
| `player_conf` | `0.35` | `main.py:125, 329-333` | `player` | Ngưỡng lọc confidence cho Cầu thủ, truyền từ CLI hoặc giá trị mặc định. |
| `goalkeeper_conf` | `0.30` | `main.py:126, 335-339` | `goalkeeper` | Ngưỡng lọc confidence cho Thủ môn, truyền từ CLI hoặc giá trị mặc định. |
| `referee_conf` | `0.30` | `main.py:127, 341-345` | `referee` | Ngưỡng lọc confidence cho Trọng tài, truyền từ CLI hoặc giá trị mặc định. |
| `ball_conf` | `0.20` | `main.py:128, 347-351` | `ball` | Ngưỡng lọc confidence cho Quả bóng, truyền từ CLI hoặc giá trị mặc định. |
| `min_predict_conf` | `0.15` | `tracker/tracker.py:86-88` | `Cả 4 class` | Ngưỡng tối thiểu truyền vào YOLO.predict để model không bỏ sót ứng viên tiềm năng. |
| `track_thresh` | `0.20` | `tracker/tracker.py:24` | `Người (player/gk/ref)` | Ngưỡng khởi tạo track_activation_threshold cho ByteTrackTracker và BoTSORTTracker. |
| `high_conf_thresh` | `0.35 (gán từ player_conf)` | `tracker/tracker.py:62` | `Người (ReID)` | Ngưỡng liên kết Stage 1 trong ReIDTrackerFacade (mặc định trong config là 0.40 nhưng bị ghi đè thành player_conf=0.35). |
| `low_conf_thresh` | `0.20 (gán từ track_thresh)` | `tracker/tracker.py:63` | `Người (ReID)` | Sàn phục hồi liên kết Stage 2 cho detection yếu trong ReIDTrackerFacade. |
| `init_conf_thresh` | `0.35 (gán từ player_conf)` | `tracker/tracker.py:64` | `Người (ReID)` | Ngưỡng tối thiểu để detection chưa gán được phép khởi tạo tentative track mới (mặc định trong config là 0.45 nhưng bị ghi đè thành 0.35). |
| `is_person filter` | `0.35 (player), 0.30 (gk, ref)` | `tracker/tracker.py:290-295` | `player, gk, ref` | Lọc cứng trước khi đưa vào ReID tracker update (lưu ý: loại bỏ các detection player có conf < 0.35). |
| `_suppress_cross_class_overlaps confs` | `0.35, 0.30, 0.30` | `tracker/tracker.py:100-102` | `player, gk, ref` | Lọc box hợp lệ trước khi tính IoU khử trùng lặp đa nhãn ở choose_Al=0,1. |
| `ball candidate floor` | `max(0.10, ball_conf - 0.05) = 0.15` | `tracker/tracker.py:705` | `ball` | Sàn lọc conf tối thiểu để xét ứng viên bóng trong _select_ball_detection. |
| `ball initial conf` | `max(0.20, ball_conf) = 0.20` | `tracker/tracker.py:721` | `ball` | Ngưỡng chấp nhận bóng frame đầu tiên khi chưa có lịch sử vị trí. |
| `ball out-of-gate conf` | `max(0.35, ball_conf + 0.15) = 0.35` | `tracker/tracker.py:747` | `ball` | Ngưỡng chấp nhận bóng khi detection nằm ngoài elip dự đoán gating. |
| `interp_confidence` | `0.30` | `tracker/ball_tracker.py:35, 410` | `ball` | Confidence giả định gán cho các frame bóng được nội suy bởi BallTrajectoryStabilizer. |
| `default raw conf` | `0.80 / 0.85` | `tracker/ball_tracker.py:83, 411` | `ball` | Giá trị fallback cho confidence bóng nếu trường confidence bị thiếu. |
| `appearance freeze min_conf` | `0.50` | `algorithmic_reid/appearance/temporal_aggregator.py:17, 29` | `Người (ReID)` | Đóng băng cập nhật template ngoại quan EMA nếu detection có confidence < 0.50. |

#### 3.4 Toàn bộ bước hậu xử lý detection trước khi vào tracker

Hệ thống có hai nhánh hậu xử lý detection tùy thuộc vào giá trị của `choose_Al`:

##### Nhánh 1: Với `choose_Al == 0` (ByteTrack) và `choose_Al == 1` (BoT-SORT)

Kích hoạt hàm khử trùng lặp đa nhãn `_suppress_cross_class_overlaps` (`tracker/tracker.py:90-153`):

```python
# tracker/tracker.py:90-153
    def _suppress_cross_class_overlaps(self, det, ids, iou_thresh=0.45):
        """
        [TẦNG 1]: Khử trùng lặp đa nhãn giữa Cầu thủ - Thủ môn - Trọng tài.
        Quy tắc ưu tiên (Priority Suppression):
          Referee (3) > Goalkeeper (2) > Player (1)
        Khi 2 box đè nhau (IoU >= iou_thresh):
          - Nếu trùng Referee & Player -> Xóa Player, giữ Referee.
          - Nếu trùng Goalkeeper & Player -> Xóa Player, giữ Goalkeeper.
          - Nếu cùng loại -> Giữ box có confidence cao hơn.
        """
        is_player = (det.class_id == ids["player"]) & (det.confidence >= self.player_conf)
        is_gk = (det.class_id == ids["goalkeeper"]) & (det.confidence >= self.goalkeeper_conf)
        is_ref = (det.class_id == ids["referee"]) & (det.confidence >= self.referee_conf)

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
```

**Tiêu chí lọc và thứ tự thực hiện của `_suppress_cross_class_overlaps`**:

1. Lọc ngưỡng conf độc lập theo class: player >= 0.35, goalkeeper >= 0.30, referee >= 0.30.

2. Tính ma trận IoU giữa tất cả các bounding box hợp lệ.

3. Gán mức ưu tiên phân cấp: `Referee (3) > Goalkeeper (2) > Player (1)`.

4. Duyệt các cặp có `IoU >= 0.45`:

   - Nếu khác mức ưu tiên: Triệt tiêu (xóa) box có ưu tiên thấp hơn (ví dụ: Referee đè Player thì xóa Player; Goalkeeper đè Player thì xóa Player).

   - Nếu cùng mức ưu tiên: Giữ box có confidence cao hơn, xóa box còn lại.


##### Nhánh 2: Với `choose_Al == 2` (Algorithmic Re-ID - Mặc định thực tế)

Tại `tracker/tracker.py:290-295`, detection được lọc qua mask boolean `is_person` trước khi gọi `self.reid_tracker.update`:

```python
# tracker/tracker.py:290-295
                is_person = (
                    ((det.class_id == ids["player"]) & (det.confidence >= self.player_conf)) |
                    ((det.class_id == ids["goalkeeper"]) & (det.confidence >= self.goalkeeper_conf)) |
                    ((det.class_id == ids["referee"]) & (det.confidence >= self.referee_conf))
                )
                person_det = det[is_person]
```

**Đặc điểm hậu xử lý của nhánh này**:

- Không gọi hàm `_suppress_cross_class_overlaps`.

- Không có NMS tự cài bổ sung (dựa hoàn toàn vào NMS nội bộ của Ultralytics với `agnostic_nms=False`).

- Không lọc diện tích hoặc tỷ lệ bbox tại tầng này (việc lọc tỷ lệ khía cạnh aspect ratio [1.3, 4.2] được dời vào hàm `AppearanceAggregator.should_freeze_update` trong ReID).

- Rào cản tách biệt nhãn được xử lý bằng **Hard Species Isolation** trong ma trận gating của ReID (`algorithmic_reid/association/gating.py:44-50`), cấm hoàn toàn việc gán detection khác class vào tracklet.


#### 3.5 Kiểm tra thuộc tính liên quan NMS hoặc End-to-End đối với YOLO26

Đã quét (grep) toàn bộ codebase với các từ khóa: `nms`, `end2end`, `agnostic`, `iou`, `max_det`. Kết quả thực tế như sau:

- `end2end`: **KHÔNG CÓ TRONG CODE** (không tìm thấy bất kỳ dòng nào chứa `end2end`).

- `agnostic`: Chỉ xuất hiện ở tham số `agnostic_nms=False` truyền vào `predict()`:

  - `tracker/tracker.py:201`: `agnostic_nms=False`

  - `tracker/tracker.py:214`: `agnostic_nms=False`

  - `test.py:18`: `agnostic_nms=False,`

- `max_det`: Chỉ xuất hiện duy nhất tại file kiểm thử tĩnh `test.py:19` (`max_det=1000,`). Trong mã nguồn chính (`main.py`, `tracker/tracker.py`), tham số này không được thiết lập.

- `nms`: Xuất hiện ở cờ `agnostic_nms=False` nói trên.

- `iou`: Xuất hiện tại:

  - `tracker/tracker.py:41`: `minimum_iou_threshold=0.2,` (ByteTrackTracker)

  - `tracker/tracker.py:49-51`: `minimum_iou_threshold_first_assoc=0.3, minimum_iou_threshold_second_assoc=0.5, minimum_iou_threshold_unconfirmed_assoc=0.3,` (BoTSORTTracker)

  - `tracker/tracker.py:90, 139`: `iou_thresh=0.45` trong `_suppress_cross_class_overlaps`

  - `algorithmic_reid/config.py:25, 40`: `w_diou=0.40, overlap_occlusion_iou=0.15`

  - `test.py:17`: `iou=0.99,`


---

### MỤC 4. Tracking người

#### 4.1 Tracker thư viện đang dùng thật sự và khởi tạo

Trong file `tracker/tracker.py`, cả 3 tracker đều được nạp và khởi tạo trong `Tracker.__init__`:

```python
# tracker/tracker.py:35-67
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
        reid_cfg = ReIDConfig(
            fps=fps,
            high_conf_thresh=player_conf,
            low_conf_thresh=track_thresh,
            init_conf_thresh=player_conf,
        )
        self.reid_tracker = ReIDTrackerFacade(config=reid_cfg, fps=fps)

```


**Vị trí gọi `update` thực tế trong mã nguồn:**

- Khi `choose_Al == 0` (ByteTrack):

```python
# tracker/tracker.py:280-283
            if choose_Al == 0:
                people = self._suppress_cross_class_overlaps(det, ids, iou_thresh=0.45)
                tracked = self.tracker.update(people)
                tracked = tracked[tracked.tracker_id != -1]
```

- Khi `choose_Al == 1` (BoT-SORT):

```python
# tracker/tracker.py:284-287
            elif choose_Al == 1:
                people = self._suppress_cross_class_overlaps(det, ids, iou_thresh=0.45)
                tracked = self.BoTTracker.update(people, frame=frame)
                tracked = tracked[tracked.tracker_id != -1]
```

- Khi `choose_Al == 2` (Algorithmic Re-ID Facade - Mặc định runtime):

```python
# tracker/tracker.py:288-297
            else:
                # choose_Al == 2: Algorithmic Re-ID Tracker (P0)
                is_person = (
                    ((det.class_id == ids["player"]) & (det.confidence >= self.player_conf)) |
                    ((det.class_id == ids["goalkeeper"]) & (det.confidence >= self.goalkeeper_conf)) |
                    ((det.class_id == ids["referee"]) & (det.confidence >= self.referee_conf))
                )
                person_det = det[is_person]
                tracked = self.reid_tracker.update(person_det, frame=frame, camera_homography=G_t)

```


#### 4.2 Đầu vào đưa vào tracker

Trong luồng mặc định (`choose_Al == 2`):

- **Các detection được đưa vào**: Lọc chung các detection thuộc 3 class `player`, `goalkeeper`, `referee` thỏa mãn ngưỡng confidence tương ứng:

  - `player`: `confidence >= self.player_conf` (0.35)

  - `goalkeeper`: `confidence >= self.goalkeeper_conf` (0.30)

  - `referee`: `confidence >= self.referee_conf` (0.30)

- **Định dạng dữ liệu**: Đối tượng `supervision.Detections` chứa `xyxy` (N, 4), `confidence` (N,), `class_id` (N,).

- **Gộp / Tách class**: Ba class người chơi/thủ môn/trọng tài được gộp chung vào một đối tượng `person_det` duy nhất nhưng bảo toàn nguyên vẹn trường `class_id` của từng detection.

- **Xử lý bên trong Facade** (`algorithmic_reid/reid_tracker_facade.py:105-162`):

```python
# algorithmic_reid/reid_tracker_facade.py:105-162
        if len(detections) == 0:
            active_tracks = self.manager.step(
                det_boxes=np.empty((0, 4), dtype=np.float32),
                det_confs=np.empty((0,), dtype=np.float32),
                det_classes=np.empty((0,), dtype=np.int32),
                frame=frame,
                camera_homography=camera_homography,
            )
            empty_det = sv.Detections.empty()
            empty_det.tracker_id = np.empty((0,), dtype=np.int32)
            return TrackedDetectionsContainer(players={}, referees={}, tracked_det=empty_det)

        det_boxes = detections.xyxy.astype(np.float32)
        det_confs = (
            detections.confidence.astype(np.float32)
            if detections.confidence is not None
            else np.ones(len(det_boxes), dtype=np.float32)
        )
        det_classes = (
            detections.class_id.astype(np.int32)
            if detections.class_id is not None
            else np.full(len(det_boxes), self.config.class_player, dtype=np.int32)
        )

        # On the very first frame, confirm high-confidence tracks immediately
        if self.manager.frame_count == 0:
            for k in range(len(det_boxes)):
                if det_confs[k] >= self.config.high_conf_thresh:
                    # Pre-initialize as confirmed
                    tid = self.manager._allocate_id()
                    feat = None
                    if frame is not None:
                        from .appearance.color_descriptor import extract_multizone_histogram
                        feat = extract_multizone_histogram(frame, det_boxes[k])
                    track = self.manager.tracks
                    from .lifecycle.tracklet import Tracklet
                    t = Tracklet(
                        track_id=tid,
                        bbox=det_boxes[k],
                        confidence=det_confs[k],
                        class_id=det_classes[k],
                        feature=feat,
                        kf=self.manager.kf,
                        aggregator=self.manager.aggregator,
                        n_init=self.config.n_init,
                        max_age=self.config.max_age,
                        start_frame=0,
                    )
                    t.state = TrackletState.CONFIRMED
                    self.manager.tracks.append(t)

        active_tracks = self.manager.step(
            det_boxes=det_boxes,
            det_confs=det_confs,
            det_classes=det_classes,
            frame=frame,
            camera_homography=camera_homography,
        )
```


#### 4.3 Đầu ra của tracker và xử lý SAU tracker

- **Đầu ra từ tracker**: Trả về `TrackedDetectionsContainer` chứa 2 dictionary `players` và `referees`:

```python
# algorithmic_reid/reid_tracker_facade.py:172-204
        for track in active_tracks:
            bbox = track.bbox.tolist()
            tid = int(track.track_id)
            cls_id = int(track.class_id)
            conf = float(track.confidence)

            entry = {"bbox": bbox, "confidence": conf}

            if cls_id == self.config.class_referee:
                referees[tid] = entry
            elif cls_id == self.config.class_goalkeeper:
                entry["is_goalkeeper"] = True
                players[tid] = entry
            else:
                players[tid] = entry

            out_boxes.append(bbox)
            out_confs.append(conf)
            out_classes.append(cls_id)
            out_tids.append(tid)

        if len(out_boxes) > 0:
            tracked_det = sv.Detections(
                xyxy=np.array(out_boxes, dtype=np.float32),
                confidence=np.array(out_confs, dtype=np.float32),
                class_id=np.array(out_classes, dtype=np.int32),
                tracker_id=np.array(out_tids, dtype=np.int32),
            )
        else:
            tracked_det = sv.Detections.empty()
            tracked_det.tracker_id = np.empty((0,), dtype=np.int32)

        return TrackedDetectionsContainer(players=players, referees=referees, tracked_det=tracked_det)
```

- **Bỏ phiếu chuỗi thời gian (Role Voting) ở chế độ cũ (`choose_Al == 0, 1`)** (`tracker/tracker.py:304-337`):

```python
# tracker/tracker.py:304-337
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
```

- **Thẩm định không gian Tầng 3 và phân định lại Goalkeeper vs Referee** (`main.py:20-120`):

```python
# main.py:20-120
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


```


#### 4.4 Khảo sát chi tiết Package `algorithmic_reid/`

Package `algorithmic_reid` được tổ chức thành 4 phân hệ chức năng độc lập, hoàn toàn không sử dụng neural network hay transformer:

##### 4.4.1 Phân hệ Chuyển động (Motion)

- `algorithmic_reid/motion/kalman_filter.py`: Lớp `DampedKalmanFilter`. State vector: $x = [x_c, y_c, a, h, v_x, v_y, 0, v_h]^T$. Khóa $\dot{a}=0$. Vận tốc suy giảm theo hệ số cản $\gamma = 0.90$ khi mất dấu. Đo đạc $z = [x_c, y_c, a, h]$.

```python
# algorithmic_reid/motion/kalman_filter.py:38-160
class DampedKalmanFilter:
    def __init__(
        self,
        std_weight_position: float = 1.0 / 20.0,
        std_weight_velocity: float = 1.0 / 160.0,
        velocity_damping_gamma: float = 0.90,
        max_velocity_px: float = 35.0,
    ):
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity
        self.gamma = velocity_damping_gamma
        self.max_velocity_px = max_velocity_px

        # Measurement matrix H (4 x 8)
        self._update_mat = np.eye(4, 8, dtype=np.float64)

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize state vector mean (8,) and covariance matrix (8, 8) from initial bbox measurement.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray, is_observed: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman prediction step.
        If is_observed is False, apply exponential velocity damping (gamma).
        """
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # State transition matrix F with aspect ratio damping (\dot{a} = 0)
        F = np.eye(8, dtype=np.float64)
        F[0, 4] = 1.0  # xc += vx
        F[1, 5] = 1.0  # yc += vy
        F[3, 7] = 1.0  # h += vh
        # Fixed aspect ratio: F[2, 6] = 0.0

        if not is_observed:
            # Apply velocity damping during unobserved dropout
            decay = self.gamma
            F[4, 4] = decay
            F[5, 5] = decay
            F[7, 7] = decay

        # Bound velocities to max sprint speed
        v_norm = np.hypot(mean[4], mean[5])
        if v_norm > self.max_velocity_px and v_norm > 1e-4:
            scale = self.max_velocity_px / v_norm
            mean[4] *= scale
            mean[5] *= scale

        mean = np.dot(F, mean)
        covariance = np.linalg.multi_dot((F, covariance, F.T)) + motion_cov
        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space [xc, yc, a, h].
        Returns: (projected_mean, innovation_covariance S)
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        R = np.diag(np.square(std))

        H = self._update_mat
        projected_mean = np.dot(H, mean)
        S = np.linalg.multi_dot((H, covariance, H.T)) + R
        return projected_mean, S

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman update step with new measurement [xc, yc, a, h].
        """
        projected_mean, S = self.project(mean, covariance)

        # Kalman gain K = P H^T S^{-1}
        H = self._update_mat
        chol_factor = np.linalg.cholesky(S)
        K = np.linalg.solve(chol_factor.T, np.linalg.solve(chol_factor, np.dot(H, covariance))).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, K.T)
        new_covariance = covariance - np.linalg.multi_dot((K, S, K.T))
        return new_mean, new_covariance
```

- `algorithmic_reid/motion/mahalanobis.py`: Hàm `compute_mahalanobis_distance_matrix`. Tính khoảng cách Mahalanobis qua phân rã Cholesky của ma trận hiệp phương sai đổi mới $S$:

```python
# algorithmic_reid/motion/mahalanobis.py:12-52
def compute_mahalanobis_distance_matrix(
    kf: DampedKalmanFilter,
    tracks: List[Any],
    measurements: np.ndarray,
    only_position: bool = False,
) -> np.ndarray:
    """
    Compute Mahalanobis distance matrix of shape (num_tracks, num_measurements).
    tracks: list of tracklets with (mean, covariance)
    measurements: array of shape (N, 4) in [xc, yc, a, h] format
    """
    if len(tracks) == 0 or len(measurements) == 0:
        return np.empty((len(tracks), len(measurements)), dtype=np.float64)

    dists = np.zeros((len(tracks), len(measurements)), dtype=np.float64)

    for i, track in enumerate(tracks):
        mean, cov = track.mean, track.covariance
        proj_mean, S = kf.project(mean, cov)

        if only_position:
            proj_mean = proj_mean[:2]
            S = S[:2, :2]
            diff = measurements[:, :2] - proj_mean
        else:
            diff = measurements - proj_mean

        try:
            chol = np.linalg.cholesky(S)
            # Solve L * y = diff.T -> y = L^{-1} diff.T
            y = np.linalg.solve(chol, diff.T)
            # d_M^2 = sum(y^2, axis=0)
            d2 = np.sum(np.square(y), axis=0)
            dists[i, :] = np.sqrt(np.maximum(0.0, d2))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            S_inv = np.linalg.pinv(S)
            d2 = np.sum(np.dot(diff, S_inv) * diff, axis=1)
            dists[i, :] = np.sqrt(np.maximum(0.0, d2))

    return dists
```

- `algorithmic_reid/motion/motion_predictor.py`: Hàm `apply_camera_motion_to_tracks`. Bù trừ camera bằng ma trận $G_t$ cho 4 đỉnh của bounding box dự đoán:

```python
# algorithmic_reid/motion/motion_predictor.py:11-50
def apply_camera_motion_to_tracks(tracks: List[Any], homography: np.ndarray) -> None:
    """
    Warp tracklet predicted positions and bboxes using 3x3 camera homography.
    homography: G_t mapping points from frame t-1 to frame t (or current to reference).
    """
    if homography is None or len(tracks) == 0:
        return

    # Check validity of homography
    if homography.shape != (3, 3) or not np.all(np.isfinite(homography)):
        return

    for track in tracks:
        # Extract bbox from mean [xc, yc, a, h]
        bbox = z_to_bbox(track.mean[:4])
        # 4 corners of bbox
        corners = np.array([
            [bbox[0], bbox[1], 1.0],
            [bbox[2], bbox[1], 1.0],
            [bbox[2], bbox[3], 1.0],
            [bbox[0], bbox[3], 1.0],
        ], dtype=np.float64).T

        warped = np.dot(homography, corners)
        w_norm = warped[2, :]
        valid_mask = np.abs(w_norm) > 1e-5
        if not np.all(valid_mask):
            continue

        warped_pts = warped[:2, :] / w_norm
        new_x1 = np.min(warped_pts[0, :])
        new_x2 = np.max(warped_pts[0, :])
        new_y1 = np.min(warped_pts[1, :])
        new_y2 = np.max(warped_pts[1, :])

        new_bbox = np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float64)
        new_z = bbox_to_z(new_bbox)

        # Update mean position components
        track.mean[:4] = new_z
```

##### 4.4.2 Phân hệ Ngoại quan (Appearance)

- `algorithmic_reid/appearance/color_descriptor.py`: Hàm `extract_multizone_histogram` và `bhattacharyya_distance`:

```python
# algorithmic_reid/appearance/color_descriptor.py:44-105
def extract_multizone_histogram(
    frame: np.ndarray,
    bbox: np.ndarray,
    bins_h: int = 16,
    bins_s: int = 8,
    bins_v: int = 8,
) -> Optional[np.ndarray]:
    """
    Extract multi-zone (upper + lower) color descriptor from person bbox in frame.
    Returns concatenated unit-normalized vector of shape (2 * (bins_h*bins_s + bins_v),).
    """
    if frame is None or frame.size == 0:
        return None

    h_frame, w_frame = frame.shape[:2]
    x1 = max(0, min(int(bbox[0]), w_frame - 1))
    y1 = max(0, min(int(bbox[1]), h_frame - 1))
    x2 = max(0, min(int(bbox[2]), w_frame))
    y2 = max(0, min(int(bbox[3]), h_frame))

    w = x2 - x1
    h = y2 - y1
    if w < 6 or h < 10:
        return None

    # Upper Torso: y in [0.15h, 0.50h], x in [0.20w, 0.80w]
    uy1 = y1 + int(0.15 * h)
    uy2 = y1 + int(0.50 * h)
    ux1 = x1 + int(0.20 * w)
    ux2 = x1 + int(0.80 * w)
    upper_roi = frame[uy1:uy2, ux1:ux2]

    # Lower Body: y in [0.55h, 0.85h], x in [0.20w, 0.80w]
    ly1 = y1 + int(0.55 * h)
    ly2 = y1 + int(0.85 * h)
    lx1 = x1 + int(0.20 * w)
    lx2 = x1 + int(0.80 * w)
    lower_roi = frame[ly1:ly2, lx1:lx2]

    h_upper = extract_zone_hist(upper_roi, bins_h, bins_s, bins_v)
    h_lower = extract_zone_hist(lower_roi, bins_h, bins_s, bins_v)

    multizone = np.concatenate([h_upper, h_lower]).astype(np.float32)
    s = np.sum(multizone)
    if s > 1e-6:
        multizone /= s
    return multizone


def bhattacharyya_distance(f1: Optional[np.ndarray], f2: Optional[np.ndarray]) -> float:
    """
    Compute Bhattacharyya distance between two normalized histograms.
      D_Bhatt = sqrt(1 - sum(sqrt(f1 * f2)))
    Range: [0.0, 1.0]. Returns 0.5 if either feature is None.
    """
    if f1 is None or f2 is None:
        return 0.5

    bc = np.sum(np.sqrt(np.maximum(0.0, f1 * f2)))
    bc = np.clip(bc, 0.0, 1.0)
    return float(np.sqrt(1.0 - bc))

```

- `algorithmic_reid/appearance/temporal_aggregator.py`: Lớp `AppearanceAggregator`. Cập nhật template qua EMA ($F_t = (1-\alpha)F_{t-1} + \alpha F_{det}$ với $\alpha=0.15$). Tích hợp rào chắn đóng băng (freeze update) khi confidence < 0.50, overlap IoU > 0.15 hoặc aspect ratio ngoài [1.3, 4.2]:

```python
# algorithmic_reid/appearance/temporal_aggregator.py:16-68
class AppearanceAggregator:
    def __init__(self, alpha: float = 0.15, min_conf: float = 0.50, max_overlap_iou: float = 0.15):
        self.alpha = alpha
        self.min_conf = min_conf
        self.max_overlap_iou = max_overlap_iou

    def should_freeze_update(
        self,
        confidence: float,
        bbox: np.ndarray,
        max_inter_iou: float = 0.0,
    ) -> bool:
        """Evaluate whether appearance update must be frozen."""
        if confidence < self.min_conf:
            return True
        if max_inter_iou > self.max_overlap_iou:
            return True

        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        aspect = h / w
        if aspect < 1.3 or aspect > 4.2:
            return True

        return False

    def update_template(
        self,
        current_template: Optional[np.ndarray],
        new_feature: Optional[np.ndarray],
        confidence: float,
        bbox: np.ndarray,
        max_inter_iou: float = 0.0,
    ) -> Optional[np.ndarray]:
        """
        Update appearance template via EMA if update freeze guards pass.
        """
        if new_feature is None:
            return current_template

        if current_template is None:
            # First initialization
            return new_feature.copy()

        if self.should_freeze_update(confidence, bbox, max_inter_iou):
            # Freeze template
            return current_template

        updated = (1.0 - self.alpha) * current_template + self.alpha * new_feature
        s = np.sum(updated)
        if s > 1e-6:
            updated /= s
        return updated
```

##### 4.4.3 Phân hệ Liên kết (Association)

- `algorithmic_reid/association/cost_matrix.py`: Hàm `compute_diou_matrix` và `compute_fused_cost_matrix`:

```python
# algorithmic_reid/association/cost_matrix.py:62-134
def compute_fused_cost_matrix(
    tracks: List[Any],
    det_boxes: np.ndarray,
    det_features: List[Optional[np.ndarray]],
    det_measurements: np.ndarray,
    kf: DampedKalmanFilter,
    w_diou: float = 0.40,
    w_mot: float = 0.20,
    w_app: float = 0.30,
    w_dir: float = 0.10,
    use_appearance: bool = True,
    chi2_gate: float = 13.28,
    det_inter_ious: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute normalized fused cost matrix between tracks and detections.
    Cost components:
      - C_diou = (1.0 - diou) / 2.0 in [0, 1]
      - C_mot  = min(1.0, d_M / sqrt(chi2_gate)) in [0, 1]
      - C_app  = D_Bhatt in [0, 1]
      - C_dir  = (1.0 - cos_theta) / 2.0 in [0, 1]
    """
    M = len(tracks)
    N = len(det_boxes)
    if M == 0 or N == 0:
        return np.empty((M, N), dtype=np.float64)

    # 1. DIoU Cost
    track_boxes = np.array([z_to_bbox(t.mean[:4]) for t in tracks], dtype=np.float64)
    diou_mat = compute_diou_matrix(track_boxes, det_boxes)
    c_diou = (1.0 - diou_mat) / 2.0
    c_diou = np.clip(c_diou, 0.0, 1.0)

    # 2. Mahalanobis Motion Cost
    d_m_mat = compute_mahalanobis_distance_matrix(kf, tracks, det_measurements)
    c_mot = np.clip(d_m_mat / np.sqrt(chi2_gate), 0.0, 1.0)

    # 3. Appearance Cost
    if use_appearance:
        track_templates = [t.appearance_template for t in tracks]
        c_app = batch_bhattacharyya_distance(track_templates, det_features)
        if det_inter_ious is not None and np.any(det_inter_ious > 0.15):
            occ_mask = det_inter_ious > 0.15
            c_app[:, occ_mask] = np.minimum(c_app[:, occ_mask], 0.60)
    else:
        c_app = np.zeros((M, N), dtype=np.float64)
        w_app = 0.0

    # 4. Heading / Motion Direction Cost
    c_dir = np.zeros((M, N), dtype=np.float64)
    for i, t in enumerate(tracks):
        vx, vy = float(t.mean[4]), float(t.mean[5])
        v_speed = np.hypot(vx, vy)
        if v_speed < 2.0:
            continue
        c_x, c_y = float(t.mean[0]), float(t.mean[1])
        cand_x = (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0
        cand_y = (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0
        dx = cand_x - c_x
        dy = cand_y - c_y
        dist = np.hypot(dx, dy)
        valid_dist = dist > 2.0
        if np.any(valid_dist):
            cos_theta = (vx * dx[valid_dist] + vy * dy[valid_dist]) / (v_speed * dist[valid_dist])
            c_dir[i, valid_dist] = (1.0 - np.clip(cos_theta, -1.0, 1.0)) / 2.0

    # Normalize active weights
    tot_weight = w_diou + w_mot + w_app + w_dir
    if tot_weight < 1e-4:
        tot_weight = 1.0

    fused_cost = (w_diou * c_diou + w_mot * c_mot + w_app * c_app + w_dir * c_dir) / tot_weight
    return fused_cost
```

- `algorithmic_reid/association/gating.py`: Hàm `apply_gating_barriers`. Áp dụng Hard Species Isolation ($10^5$), Appearance Barrier (Bhattacharyya gate 0.55 / nới 0.70 khi che khuất) và Spatial Mahalanobis Barrier (ngưỡng chi-squared $\chi^2=13.28$):

```python
# algorithmic_reid/association/gating.py:18-141
def apply_gating_barriers(
    cost_matrix: np.ndarray,
    tracks: List[Any],
    det_boxes: np.ndarray,
    det_classes: np.ndarray,
    det_features: List[Optional[np.ndarray]],
    det_measurements: np.ndarray,
    kf: DampedKalmanFilter,
    chi2_gate: float = 13.28,
    appearance_gate: float = 0.55,
    max_velocity_px: float = 35.0,
    check_appearance: bool = True,
    inf_cost: float = 1e5,
    det_inter_ious: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Applies hard gating barriers in-place (or returning copy) to cost_matrix.
    Any violation sets cost to inf_cost (1e5).
    """
    M = len(tracks)
    N = len(det_boxes)
    if M == 0 or N == 0:
        return cost_matrix

    gated_cost = cost_matrix.copy()

    # 1. Species Isolation Barrier (Player != Referee != Goalkeeper)
    for i, t in enumerate(tracks):
        t_cls = t.class_id
        mismatch_mask = (det_classes != t_cls)
        if np.any(mismatch_mask):
            gated_cost[i, mismatch_mask] = inf_cost

    # 2. Appearance Barrier with Occlusion Awareness
    if check_appearance:
        track_templates = [t.appearance_template for t in tracks]
        app_dist = batch_bhattacharyya_distance(track_templates, det_features)

        # Precompute inter-detection overlap if not provided
        if det_inter_ious is None:
            if N > 1:
                x1 = np.maximum(det_boxes[:, 0][:, None], det_boxes[:, 0][None, :])
                y1 = np.maximum(det_boxes[:, 1][:, None], det_boxes[:, 1][None, :])
                x2 = np.minimum(det_boxes[:, 2][:, None], det_boxes[:, 2][None, :])
                y2 = np.minimum(det_boxes[:, 3][:, None], det_boxes[:, 3][None, :])
                inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
                areas = np.maximum(1e-4, (det_boxes[:, 2] - det_boxes[:, 0]) * (det_boxes[:, 3] - det_boxes[:, 1]))
                union = areas[:, None] + areas[None, :] - inter
                iou_mat = np.where(union > 0, inter / union, 0.0)
                np.fill_diagonal(iou_mat, 0.0)
                inter_ious = np.max(iou_mat, axis=1)
            else:
                inter_ious = np.zeros(N, dtype=np.float64)
        else:
            inter_ious = det_inter_ious

        # Spatial pairwise DIoU
        track_boxes = np.array([z_to_bbox(t.mean[:4]) for t in tracks], dtype=np.float64)
        diou_mat = compute_diou_matrix(track_boxes, det_boxes)

        # Track-level occlusion overlap
        if M > 1:
            tr_x1 = np.maximum(track_boxes[:, 0][:, None], track_boxes[:, 0][None, :])
            tr_y1 = np.maximum(track_boxes[:, 1][:, None], track_boxes[:, 1][None, :])
            tr_x2 = np.minimum(track_boxes[:, 2][:, None], track_boxes[:, 2][None, :])
            tr_y2 = np.minimum(track_boxes[:, 3][:, None], track_boxes[:, 3][None, :])
            tr_inter = np.maximum(0.0, tr_x2 - tr_x1) * np.maximum(0.0, tr_y2 - tr_y1)
            tr_areas = np.maximum(1e-4, (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1]))
            tr_union = tr_areas[:, None] + tr_areas[None, :] - tr_inter
            tr_iou_mat = np.where(tr_union > 0, tr_inter / tr_union, 0.0)
            np.fill_diagonal(tr_iou_mat, 0.0)
            is_occluded_track = np.max(tr_iou_mat, axis=1) > 0.15
        else:
            is_occluded_track = np.zeros(M, dtype=bool)

        is_occluded_det = (inter_ious > 0.15)
        in_occlusion = is_occluded_det[None, :] | is_occluded_track[:, None]

        # Relax appearance gate to 0.70 during occlusion
        effective_gate = np.where(in_occlusion, max(appearance_gate, 0.70), appearance_gate)

        # Heading alignment check
        heading_align = np.ones((M, N), dtype=bool)
        for i, t in enumerate(tracks):
            vx, vy = float(t.mean[4]), float(t.mean[5])
            v_speed = np.hypot(vx, vy)
            if v_speed >= 2.0:
                c_x, c_y = float(t.mean[0]), float(t.mean[1])
                cand_x = (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0
                cand_y = (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0
                dx = cand_x - c_x
                dy = cand_y - c_y
                dist = np.hypot(dx, dy)
                non_zero = dist > 2.0
                if np.any(non_zero):
                    cos_theta = np.ones(N, dtype=np.float64)
                    cos_theta[non_zero] = (vx * dx[non_zero] + vy * dy[non_zero]) / (v_speed * dist[non_zero])
                    heading_align[i, :] = (cos_theta >= -0.2)

        # Bypass appearance gating if in occlusion context and spatial DIoU >= 0.10 and heading aligns
        bypass_app = in_occlusion & (diou_mat >= 0.10) & heading_align

        app_violation = (app_dist > effective_gate) & ~bypass_app
        gated_cost[app_violation] = inf_cost

    # 3. Spatial / Mahalanobis Barrier
    d_m_mat = compute_mahalanobis_distance_matrix(kf, tracks, det_measurements)
    d_m_sq = np.square(d_m_mat)

    for i, t in enumerate(tracks):
        time_lost = max(1, t.time_since_update)
        max_dist = max(2.0 * max_velocity_px * time_lost, 75.0)

        # Euclidean center distance
        c_x, c_y = float(t.mean[0]), float(t.mean[1])
        cand_x = (det_boxes[:, 0] + det_boxes[:, 2]) / 2.0
        cand_y = (det_boxes[:, 1] + det_boxes[:, 3]) / 2.0
        dist = np.hypot(cand_x - c_x, cand_y - c_y)

        # Gate if Mahalanobis exceeds chi2 AND spatial distance exceeds max_dist
        spatial_violation = (d_m_sq[i, :] > chi2_gate) & (dist > max_dist)
        gated_cost[i, spatial_violation] = inf_cost

    return gated_cost
```

- `algorithmic_reid/association/hungarian.py`: Hàm `min_cost_matching`. Giải gán cặp tối ưu có chặn trần `max_cost`:

```python
# algorithmic_reid/association/hungarian.py:13-45
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
```

##### 4.4.4 Phân hệ Vòng đời (Lifecycle) và Facade

- `algorithmic_reid/lifecycle/tracklet.py`: Lớp `Tracklet` và enum `TrackletState`:

```python
# algorithmic_reid/lifecycle/tracklet.py:12-129
class TrackletState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    LOST = 3
    DELETED = 4


class Tracklet:
    def __init__(
        self,
        track_id: int,
        bbox: np.ndarray,
        confidence: float,
        class_id: int,
        feature: Optional[np.ndarray],
        kf: DampedKalmanFilter,
        aggregator: AppearanceAggregator,
        n_init: int = 2,
        max_age: int = 70,
        start_frame: int = 0,
    ):
        self.track_id = track_id
        self.class_id = class_id
        self.confidence = float(confidence)
        self.kf = kf
        self.aggregator = aggregator
        self.n_init = n_init
        self.max_age = max_age
        self.start_frame = start_frame
        self.last_frame = start_frame

        # Initialize Kalman state
        z = bbox_to_z(bbox)
        self.mean, self.covariance = self.kf.initiate(z)

        # Initialize appearance template
        self.appearance_template = feature.copy() if feature is not None else None

        # Lifecycle state
        self.state = TrackletState.TENTATIVE
        self.hits = 1
        self.consecutive_hits = 1
        self.age = 1
        self.time_since_update = 0

        # Trajectory history: list of (frame_idx, bbox)
        self.history: List[Tuple[int, np.ndarray]] = [(start_frame, bbox.copy())]

    @property
    def bbox(self) -> np.ndarray:
        """Current bounding box [x1, y1, x2, y2]."""
        return z_to_bbox(self.mean[:4])

    def predict(self, homography: Optional[np.ndarray] = None) -> None:
        """
        Predict state for current frame. Apply velocity damping if unobserved.
        """
        is_observed = (self.time_since_update == 0)
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance, is_observed=is_observed)
        self.age += 1

        if homography is not None:
            from ..motion.motion_predictor import apply_camera_motion_to_tracks
            apply_camera_motion_to_tracks([self], homography)

    def update(
        self,
        bbox: np.ndarray,
        confidence: float,
        class_id: int,
        feature: Optional[np.ndarray] = None,
        max_inter_iou: float = 0.0,
        frame_idx: int = 0,
    ) -> None:
        """
        Update tracklet with matched detection.
        """
        z = bbox_to_z(bbox)
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, z)

        # Appearance update with occlusion freeze guard
        if feature is not None:
            self.appearance_template = self.aggregator.update_template(
                self.appearance_template,
                feature,
                confidence=confidence,
                bbox=bbox,
                max_inter_iou=max_inter_iou,
            )

        self.confidence = float(confidence)
        self.class_id = class_id
        self.hits += 1
        self.consecutive_hits += 1
        self.time_since_update = 0
        self.last_frame = frame_idx
        self.history.append((frame_idx, bbox.copy()))

        if self.state == TrackletState.TENTATIVE:
            if self.consecutive_hits >= self.n_init:
                self.state = TrackletState.CONFIRMED
        elif self.state == TrackletState.LOST:
            self.state = TrackletState.CONFIRMED

    def mark_missed(self) -> None:
        """Mark tracklet as missed in current frame."""
        self.time_since_update += 1
        self.consecutive_hits = 0

        if self.state == TrackletState.TENTATIVE:
            # Tentative track that misses an update is deleted
            self.state = TrackletState.DELETED
        elif self.state == TrackletState.CONFIRMED:
            if self.time_since_update > 0:
                self.state = TrackletState.LOST

        if self.time_since_update > self.max_age:
            self.state = TrackletState.DELETED
```

- `algorithmic_reid/lifecycle/stitcher.py`: Lớp `TrackletStitcher`. Khâu nối tracklet bị đứt quãng:

```python
# algorithmic_reid/lifecycle/stitcher.py:17-98
class TrackletStitcher:
    def __init__(
        self,
        max_gap_frames: int = 30,
        max_velocity_px: float = 35.0,
        appearance_thresh: float = 0.40,
    ):
        self.max_gap_frames = max_gap_frames
        self.max_velocity_px = max_velocity_px
        self.appearance_thresh = appearance_thresh

    def can_stitch(self, t_prev: Tracklet, t_next: Tracklet) -> bool:
        """Evaluate whether t_next can be stitched as a continuation of t_prev."""
        # 1. Species barrier
        if t_prev.class_id != t_next.class_id:
            return False

        # 2. Temporal non-overlap
        gap = t_next.start_frame - t_prev.last_frame
        if gap < 1 or gap > self.max_gap_frames:
            return False

        # 3. Velocity feasibility
        p_prev = ((t_prev.history[-1][1][0] + t_prev.history[-1][1][2]) / 2.0,
                  (t_prev.history[-1][1][1] + t_prev.history[-1][1][3]) / 2.0)
        p_next = ((t_next.history[0][1][0] + t_next.history[0][1][2]) / 2.0,
                  (t_next.history[0][1][1] + t_next.history[0][1][3]) / 2.0)

        dist = np.hypot(p_next[0] - p_prev[0], p_next[1] - p_prev[1])
        if (dist / float(gap)) > self.max_velocity_px:
            return False

        # 4. Appearance similarity
        if t_prev.appearance_template is not None and t_next.appearance_template is not None:
            app_dist = bhattacharyya_distance(t_prev.appearance_template, t_next.appearance_template)
            if app_dist > self.appearance_thresh:
                return False

        return True

    def stitch_tracklets(self, target_track: Tracklet, source_track: Tracklet) -> None:
        """Merge source_track into target_track and interpolate the gap."""
        # Linear interpolation over unobserved gap in history if gap > 1
        t_gap = source_track.history[0][0] - target_track.history[-1][0]
        if t_gap > 1:
            box_start = target_track.history[-1][1]
            box_end = source_track.history[0][1]
            f_start = target_track.history[-1][0]
            interpolated = []
            for step in range(1, t_gap):
                alpha = float(step) / float(t_gap)
                interp_box = (1.0 - alpha) * box_start + alpha * box_end
                interpolated.append((f_start + step, interp_box))
            target_track.history.extend(interpolated)

        # Extend history
        target_track.history.extend(source_track.history)
        target_track.last_frame = source_track.last_frame
        target_track.time_since_update = 0
        target_track.state = TrackletState.CONFIRMED
        target_track.hits += source_track.hits
        target_track.consecutive_hits = source_track.consecutive_hits
        target_track.age += source_track.age

        # Update Kalman state to source track's latest state
        target_track.mean = source_track.mean.copy()
        target_track.covariance = source_track.covariance.copy()

        # Update appearance template with source
        if source_track.appearance_template is not None:
            if target_track.appearance_template is None:
                target_track.appearance_template = source_track.appearance_template.copy()
            else:
                target_track.appearance_template = 0.5 * (
                    target_track.appearance_template + source_track.appearance_template
                )
                s = np.sum(target_track.appearance_template)
                if s > 1e-6:
                    target_track.appearance_template /= s

        # Mark source as deleted
        source_track.state = TrackletState.DELETED
```

- `algorithmic_reid/lifecycle/track_manager.py`: Lớp `TrackManager`. Điều phối 4 tầng liên kết phân cấp và khâu nối tracklet:

```python
# algorithmic_reid/lifecycle/track_manager.py:26-115
class TrackManager:
    def __init__(self, config: Optional[ReIDConfig] = None):
        self.config = config or ReIDConfig()

        self.kf = DampedKalmanFilter(
            std_weight_position=self.config.std_weight_position,
            std_weight_velocity=self.config.std_weight_velocity,
            velocity_damping_gamma=self.config.velocity_damping_gamma,
            max_velocity_px=self.config.max_velocity_px,
        )

        self.aggregator = AppearanceAggregator(
            alpha=self.config.ema_alpha,
            min_conf=0.50,
            max_overlap_iou=self.config.overlap_occlusion_iou,
        )

        self.stitcher = TrackletStitcher(
            max_gap_frames=self.config.reactivation_window,
            max_velocity_px=self.config.max_velocity_px,
            appearance_thresh=0.40,
        )

        self.tracks: List[Tracklet] = []
        self.next_id: int = 0
        self.frame_count: int = 0

    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0

    def _allocate_id(self) -> int:
        tid = self.next_id
        self.next_id += 1
        return tid

    def _stitch_candidates(self, matched_track_ids: set) -> None:
        """
        Evaluate and stitch newly confirmed active tracks into recently lost confirmed tracks.
        """
        lost_candidates = [
            t for t in self.tracks
            if t.state in (TrackletState.LOST, TrackletState.CONFIRMED)
            and t.time_since_update > 0
            and len(t.history) >= self.config.n_init
        ]
        if not lost_candidates:
            return

        active_candidates = [
            t for t in self.tracks
            if t.state == TrackletState.CONFIRMED
            and t.time_since_update == 0
            and t.start_frame > 0
        ]
        if not active_candidates:
            return

        for new_track in active_candidates:
            best_target = None
            best_cost = float("inf")

            for lost_track in lost_candidates:
                if self.stitcher.can_stitch(lost_track, new_track):
                    p_prev = ((lost_track.history[-1][1][0] + lost_track.history[-1][1][2]) / 2.0,
                              (lost_track.history[-1][1][1] + lost_track.history[-1][1][3]) / 2.0)
                    p_next = ((new_track.history[0][1][0] + new_track.history[0][1][2]) / 2.0,
                              (new_track.history[0][1][1] + new_track.history[0][1][3]) / 2.0)
                    dist = np.hypot(p_next[0] - p_prev[0], p_next[1] - p_prev[1])
                    gap = max(1, new_track.start_frame - lost_track.last_frame)
                    vel = dist / float(gap)

                    app_cost = 0.0
                    if lost_track.appearance_template is not None and new_track.appearance_template is not None:
                        from ..appearance.color_descriptor import bhattacharyya_distance
                        app_cost = bhattacharyya_distance(lost_track.appearance_template, new_track.appearance_template)

                    stitch_cost = app_cost + 0.01 * vel
                    if stitch_cost < best_cost:
                        best_cost = stitch_cost
                        best_target = lost_track

            if best_target is not None:
                self.stitcher.stitch_tracklets(best_target, new_track)
                lost_candidates.remove(best_target)
                matched_track_ids.discard(new_track.track_id)
                matched_track_ids.add(best_target.track_id)

```

- `algorithmic_reid/reid_tracker_facade.py`: Lớp `ReIDTrackerFacade` và `TrackedDetectionsContainer`:

```python
# algorithmic_reid/reid_tracker_facade.py:81-204
class ReIDTrackerFacade:
    def __init__(self, config: Optional[ReIDConfig] = None, fps: float = 25.0):
        if config is None:
            config = ReIDConfig(fps=fps)
        else:
            config.fps = fps
        self.config = config
        self.manager = TrackManager(config=self.config)

    def reset(self) -> None:
        self.manager.reset()

    def update(
        self,
        detections: sv.Detections,
        frame: Optional[np.ndarray] = None,
        camera_homography: Optional[np.ndarray] = None,
    ) -> TrackedDetectionsContainer:
        """
        Update tracker with current frame detections.
        detections: sv.Detections containing xyxy, confidence, class_id
        frame: optional BGR image for multi-zone appearance extraction
        camera_homography: optional 3x3 homography matrix (e.g. from Stage 3 CameraMotion)
        """
        if len(detections) == 0:
            active_tracks = self.manager.step(
                det_boxes=np.empty((0, 4), dtype=np.float32),
                det_confs=np.empty((0,), dtype=np.float32),
                det_classes=np.empty((0,), dtype=np.int32),
                frame=frame,
                camera_homography=camera_homography,
            )
            empty_det = sv.Detections.empty()
            empty_det.tracker_id = np.empty((0,), dtype=np.int32)
            return TrackedDetectionsContainer(players={}, referees={}, tracked_det=empty_det)

        det_boxes = detections.xyxy.astype(np.float32)
        det_confs = (
            detections.confidence.astype(np.float32)
            if detections.confidence is not None
            else np.ones(len(det_boxes), dtype=np.float32)
        )
        det_classes = (
            detections.class_id.astype(np.int32)
            if detections.class_id is not None
            else np.full(len(det_boxes), self.config.class_player, dtype=np.int32)
        )

        # On the very first frame, confirm high-confidence tracks immediately
        if self.manager.frame_count == 0:
            for k in range(len(det_boxes)):
                if det_confs[k] >= self.config.high_conf_thresh:
                    # Pre-initialize as confirmed
                    tid = self.manager._allocate_id()
                    feat = None
                    if frame is not None:
                        from .appearance.color_descriptor import extract_multizone_histogram
                        feat = extract_multizone_histogram(frame, det_boxes[k])
                    track = self.manager.tracks
                    from .lifecycle.tracklet import Tracklet
                    t = Tracklet(
                        track_id=tid,
                        bbox=det_boxes[k],
                        confidence=det_confs[k],
                        class_id=det_classes[k],
                        feature=feat,
                        kf=self.manager.kf,
                        aggregator=self.manager.aggregator,
                        n_init=self.config.n_init,
                        max_age=self.config.max_age,
                        start_frame=0,
                    )
                    t.state = TrackletState.CONFIRMED
                    self.manager.tracks.append(t)

        active_tracks = self.manager.step(
            det_boxes=det_boxes,
            det_confs=det_confs,
            det_classes=det_classes,
            frame=frame,
            camera_homography=camera_homography,
        )

        players = {}
        referees = {}

        out_boxes = []
        out_confs = []
        out_classes = []
        out_tids = []

        for track in active_tracks:
            bbox = track.bbox.tolist()
            tid = int(track.track_id)
            cls_id = int(track.class_id)
            conf = float(track.confidence)

            entry = {"bbox": bbox, "confidence": conf}

            if cls_id == self.config.class_referee:
                referees[tid] = entry
            elif cls_id == self.config.class_goalkeeper:
                entry["is_goalkeeper"] = True
                players[tid] = entry
            else:
                players[tid] = entry

            out_boxes.append(bbox)
            out_confs.append(conf)
            out_classes.append(cls_id)
            out_tids.append(tid)

        if len(out_boxes) > 0:
            tracked_det = sv.Detections(
                xyxy=np.array(out_boxes, dtype=np.float32),
                confidence=np.array(out_confs, dtype=np.float32),
                class_id=np.array(out_classes, dtype=np.int32),
                tracker_id=np.array(out_tids, dtype=np.int32),
            )
        else:
            tracked_det = sv.Detections.empty()
            tracked_det.tracker_id = np.empty((0,), dtype=np.int32)

        return TrackedDetectionsContainer(players=players, referees=referees, tracked_det=tracked_det)
```


#### 4.5 Camera motion trong tracking người

Qua khảo sát mã nguồn chi tiết:

1. **Biến tọa độ đưa vào Kalman filter**: Toàn bộ trạng thái của Kalman (`track.mean`) được lưu trong **hệ tọa độ PIXEL ẢNH THÔ** `[xc, yc, a, h]`.

2. **Cơ chế bù trừ camera trong tracking người**:

   - Tại `tracker/tracker.py:276-277`, sau khi gọi `camera_motion.step(frame, prev_boxes)`, ma trận biến đổi phối cảnh liên frame $G_t$ được lấy ra từ `camera_motion.G_prev`.

   - Tại `tracker/tracker.py:296`, ma trận $G_t$ được truyền vào `self.reid_tracker.update(..., camera_homography=G_t)`.

   - Trong `Tracklet.predict()`, hàm `apply_camera_motion_to_tracks([self], homography)` lấy 4 đỉnh bbox từ `track.mean[:4]`, nhân với ma trận $G_t$, và gán lại `track.mean[:4]`.

   - **ĐIỂM ĐẶC BIỆT**: Các thành phần vận tốc trong Kalman `track.mean[4:8]` ($v_x, v_y, 0, v_h$) **KHÔNG** được nhân với ma trận $G_t$, chỉ có tọa độ vị trí $[x_c, y_c, a, h]$ được biến đổi.

3. **Tọa độ bù camera chuẩn (`position_adjusted`) và tọa độ sân (`position_transformed`)**:

   - KHÔNG được sử dụng trong vòng lặp liên kết tracking người.

   - Chỉ được tính toán SAU KHI Pass 1 kết thúc, thông qua `PitchMapper.add_positions_to_tracks(tracks)` tại `main.py:252`.


#### 4.6 Cách xử lý class của từng track trong toàn pipeline

Class của từng tracklet được lưu trữ và biến đổi qua các giai đoạn như sau:

1. **Giai đoạn Detection**: `det.class_id` mang giá trị 0 (ball), 1 (gk), 2 (player), 3 (referee).

2. **Giai đoạn Tracking**: `tracklet.class_id` được gán khi khởi tạo (`algorithmic_reid/lifecycle/tracklet.py:34`). Trong mỗi lần update, `tracklet.class_id = class_id` được cập nhật lại theo detection gán (`tracklet.py:103`). Nhờ Hard Species Isolation (`gating.py:44-50`), detection khác class bị gán cost = $10^5$, ngăn chặn việc tráo đổi class giữa các frame.

3. **Giai đoạn Xuất từ Facade**: `reid_tracker_facade.py:180-186` phân tách track vào `referees` nếu `cls_id == 3`, hoặc vào `players` nếu `cls_id in (1, 2)`. Nếu `cls_id == 1`, đặt thêm cờ `entry['is_goalkeeper'] = True`.

4. **Giai đoạn Thẩm định Tầng 3** (`main.py:45-80`): Tọa độ trung bình sân bóng FIFA được dùng để can thiệp trực tiếp vào class của track:

   - Nếu một track ở `referees` nhưng có vị trí trung bình trong vòng cấm (`x <= 18` hoặc `x >= 87` và `15 <= y <= 53`), track đó bị xóa khỏi `referees` và chuyển sang `players` với `is_goalkeeper = True` (`main.py:54-60`).

   - Nếu một track ở `players` có cờ `is_goalkeeper == True` nhưng vị trí trung bình ở giữa sân (`25 <= x <= 80`), track đó bị xóa khỏi `players` và chuyển sang `referees`, đồng thời bị xóa cờ `is_goalkeeper`, `team`, `team_color` (`main.py:71-79`).

5. **Giai đoạn Vẽ và Hiển thị**:

   - Video chính (`tracker/tracker.py:617-626`): Duyệt `tracks['players']` vẽ hình elip mang màu đội và text `'GK{id}'` (nếu `is_goalkeeper`) hoặc `'{id}'`; duyệt `tracks['referees']` vẽ hình elip màu xanh neon cố định `(0, 234, 123)`.

   - Minimap radar (`minimap/minimap_renderer.py:204-257`): Duyệt `tracks['referees']` vẽ chấm tròn chữ 'R'; duyệt `tracks['players']` vẽ chấm tròn màu áo đội mang số ID hoặc chữ 'GK' có viền vàng rực.


---
### MỤC 5. Tracking bóng

#### 5.1 Cách chọn bóng từ detection mỗi frame

Trong quá trình đọc từng frame ở Pass 1 (`tracker/tracker.py:340-348`), hàm `_select_ball_detection` được gọi để chọn ứng viên bóng duy nhất (Top-1):

```python
# tracker/tracker.py:680-761
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
```

**Phân tích cơ chế chọn bóng trong `_select_ball_detection`**:

1. **Số lượng ứng viên**: Lấy duy nhất Top-1 ứng viên (`argmax`), lưu vào `tracks['ball'][frame_index][1]`.

2. **Lọc confidence**: Lọc thô ban đầu `confs >= max(0.10, ball_conf - 0.05)` (với mặc định là 0.15). Nếu không còn box nào thì trả về `None`.

3. **Lọc theo vùng / lịch sử**:

   - Khi chưa có lịch sử (`last_center is None`): Yêu cầu confidence cao `conf >= max(0.20, ball_conf) = 0.20`. Nếu đạt, khởi tạo `velocity = [0, 0]`.

   - Khi đã có lịch sử: Tính tâm dự đoán `predicted_center = last_center + velocity * dt` trong **hệ tọa độ PIXEL ẢNH THÔ**.

   - Bán kính elip gating: $gate = \min(\max(0.12 \cdot D, 2.5 \cdot speed \cdot dt + 0.04 \cdot D \cdot dt), 0.40 \cdot D)$ với $D$ là đường chéo khung hình.

   - Nếu có box nằm trong gate (`distances <= gate`): Cộng điểm ưu tiên theo khoảng cách $0.3 \cdot (1 - dist/gate)$, chọn box có điểm cao nhất.

   - Nếu không có box nào trong gate: Chỉ chấp nhận box có conf cao nhất nếu `conf >= max(0.35, ball_conf + 0.15) = 0.35` (chấp nhận pha sút đổi hướng mạnh khi model đủ tự tin).

4. **Cập nhật vận tốc**: Cập nhật EMA vận tốc $v_{t} = 0.20 v_{t-1} + 0.80 v_{meas}$ có chặn trần $v_{max} = \min(50 \cdot dt, 0.40 \cdot D)$.


#### 5.2 Khảo sát module `tracker/ball_tracker.py`

- **Class chính**: `BallTrajectoryStabilizer` (`tracker/ball_tracker.py:20-551`).

- **Danh mục phương thức trong `BallTrajectoryStabilizer`**:

| Tên phương thức | Vai trò chức năng |
|---|---|
| `__init__(self, v_max=50.0, max_acc=30.0, foot_proximity_thresh=60.0, tier1_max_gap=4, tier2_max_gap=10, tier2_max_angle_deg=60.0, max_smoothing_deviation=2.5, interp_confidence=0.30)` | Khởi tạo các ngưỡng siêu tham số vật lý và khoảng cách. |
| `process(self, ball_tracks, camera_motion=None, player_tracks=None) -> List[Dict]` | Hàm điều phối tổng thể 6 bước xử lý chuỗi quỹ đạo bóng. |
| `_to_reference(self, camera_motion, t, pt) -> np.ndarray` | Chuyển đổi điểm pixel từ frame t về hệ quy chiếu tham chiếu camera. |
| `_to_frame(self, camera_motion, t, pt_ref) -> np.ndarray` | Chuyển đổi điểm từ hệ quy chiếu tham chiếu ngược về pixel frame t. |
| `_is_near_player(self, player_tracks, t, ball_center) -> bool` | Kiểm tra xem bóng có nằm trong phạm vi 60px cạnh chân cầu thủ nào không. |
| `_get_velocity_at(self, positions, t, direction='in') -> Optional[np.ndarray]` | Ước lượng vector vận tốc đi vào (in) hoặc đi ra (out) tại frame t. |
| `_interpolate_segment(self, positions, is_interp_mask, track_states, t_a, t_b, confirmed_indices)` | Thực hiện nội suy đơn điệu PCHIP giữa 2 frame t_a và t_b. |

**Bảng toàn bộ hằng số và ngưỡng trong `tracker/ball_tracker.py`**:

| Tên hằng số / tham số | Giá trị | File:Dòng | Tác dụng |
|---|---|---|---|
| `v_max` | `50.0 px/frame` | `tracker/ball_tracker.py:28` | Trần vận tốc bóng tối đa theo quy chuẩn vật lý (~126 km/h ở 25fps). |
| `max_acc` | `30.0 px/frame^2` | `tracker/ball_tracker.py:29` | Trần gia tốc đột ngột tối đa nếu không có chân cầu thủ tác động. |
| `foot_proximity_thresh` | `60.0 px` | `tracker/ball_tracker.py:30` | Khoảng cách pixel tối đa tới chân cầu thủ để công nhận pha sút bóng. |
| `tier1_max_gap` | `4 frames` | `tracker/ball_tracker.py:31` | Khoảng mất dấu ngắn (1-4 frames) -> Áp dụng nội suy PCHIP. |
| `tier2_max_gap` | `10 frames` | `tracker/ball_tracker.py:32` | Khoảng mất dấu trung bình (5-10 frames) -> Kiểm tra tính nhất quán động học. |
| `tier2_max_angle_deg` | `60.0 degrees` | `tracker/ball_tracker.py:33` | Góc lệch hướng vận tốc tối đa cho phép ở Tier 2 để được nội suy. |
| `max_smoothing_deviation` | `2.5 px` | `tracker/ball_tracker.py:34` | Giới hạn dịch chuyển tối đa khi làm mượt đạn đạo so với detection gốc. |
| `interp_confidence` | `0.30` | `tracker/ball_tracker.py:35` | Confidence gán cho các frame bóng được nội suy. |
| `k_max` | `4 candidates` | `tracker/ball_tracker.py:170` | Số lượng detection ứng viên nhìn trước trong thuật toán Lookahead Chord Burst. |
| `default_size` | `(20.0, 20.0) px` | `tracker/ball_tracker.py:385` | Kích thước bbox mặc định (w, h) nếu không có detection hợp lệ. |

#### 5.3 Trích NGUYÊN VĂN các hàm cốt lõi của `ball_tracker.py`

Dưới đây là nguyên văn toàn bộ các khối xử lý cốt lõi trong `tracker/ball_tracker.py`:

##### 5.3.1 Hàm điều phối chính `BallTrajectoryStabilizer.process` (Trích từng khối logic đầy đủ)

- **Khối 1: Khởi tạo và trích xuất dữ liệu thô (Lines 46-90)**:

```python
# tracker/ball_tracker.py:46-90
    def process(
        self,
        ball_tracks: List[Dict[int, Dict[str, Any]]],
        camera_motion: Optional[Any] = None,
        player_tracks: Optional[List[Dict[int, Dict[str, Any]]]] = None,
    ) -> List[Dict[int, Dict[str, Any]]]:
        """
        Process and stabilize full ball trajectory across all video frames.

        Args:
            ball_tracks: List of per-frame ball dictionaries [{1: {"bbox": ...}}, ...].
            camera_motion: CameraMotion object providing `.to_reference(t, pts)` or `.Ms`.
            player_tracks: Optional list of player bounding boxes for kick context gating.

        Returns:
            List of stabilized ball dictionaries preserving data contracts.
        """
        num_frames = len(ball_tracks)
        if num_frames == 0:
            return []

        # ---------------------------------------------------------------------
        # Step 1: Extract raw centers, dimensions, and confidences
        # ---------------------------------------------------------------------
        raw_centers = [None] * num_frames
        raw_sizes = [None] * num_frames
        raw_confs = [None] * num_frames
        raw_interpolated = [False] * num_frames

        for t in range(num_frames):
            frame_data = ball_tracks[t]
            if 1 in frame_data:
                bbox = frame_data[1]["bbox"]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                w = max(1.0, float(bbox[2] - bbox[0]))
                h = max(1.0, float(bbox[3] - bbox[1]))
                conf = float(frame_data[1].get("confidence", 0.80))
                is_interp = bool(frame_data[1].get("is_interpolated", False))

                raw_centers[t] = np.array([cx, cy], dtype=np.float64)
                raw_sizes[t] = (w, h)
                raw_confs[t] = conf
                raw_interpolated[t] = is_interp

```

- **Khối 2: Chuyển sang không gian tham chiếu camera & Lọc ngoại lai biên (Lines 91-164)**:

```python
# tracker/ball_tracker.py:91-164
        # ---------------------------------------------------------------------
        # Step 2: Transform to Reference Coordinate Space (Camera Decoupling)
        # ---------------------------------------------------------------------
        ref_centers = [None] * num_frames
        for t in range(num_frames):
            if raw_centers[t] is not None:
                ref_centers[t] = self._to_reference(camera_motion, t, raw_centers[t])

        # ---------------------------------------------------------------------
        # Step 3: Outlier Rejection & Kinematic Gating
        # ---------------------------------------------------------------------
        valid_mask = [ref_centers[t] is not None for t in range(num_frames)]
        detected_indices = [t for t in range(num_frames) if valid_mask[t]]

        # 3.1 Boundary Outlier Gating
        if len(detected_indices) >= 2:
            # Check initial detection (i=0)
            t_0 = detected_indices[0]
            t_1 = detected_indices[1]
            dt_0 = t_1 - t_0
            v_01 = (ref_centers[t_1] - ref_centers[t_0]) / max(1, dt_0)
            speed_0 = np.linalg.norm(v_01)
            is_near_foot_0 = self._is_near_player(
                player_tracks, t_0, raw_centers[t_0]
            ) or self._is_near_player(
                player_tracks, t_1, raw_centers[t_1]
            )

            is_initial_outlier = False
            if speed_0 > self.v_max:
                if speed_0 > 2.0 * self.v_max:
                    is_initial_outlier = True
                elif len(detected_indices) >= 3:
                    t_2 = detected_indices[2]
                    dt_1 = t_2 - t_1
                    v_12 = (ref_centers[t_2] - ref_centers[t_1]) / max(1, dt_1)
                    acc_0 = np.linalg.norm(v_01 - v_12)
                    if acc_0 > self.max_acc and not is_near_foot_0:
                        is_initial_outlier = True
                elif not is_near_foot_0:
                    is_initial_outlier = True

            if is_initial_outlier:
                valid_mask[t_0] = False

            # Check trailing detection (i=len-1)
            t_n = detected_indices[-1]
            t_prev = detected_indices[-2]
            dt_n = t_n - t_prev
            v_prev_n = (ref_centers[t_n] - ref_centers[t_prev]) / max(1, dt_n)
            speed_n = np.linalg.norm(v_prev_n)
            is_near_foot_n = self._is_near_player(
                player_tracks, t_n, raw_centers[t_n]
            ) or self._is_near_player(
                player_tracks, t_prev, raw_centers[t_prev]
            )

            is_trailing_outlier = False
            if speed_n > self.v_max:
                if speed_n > 2.0 * self.v_max:
                    is_trailing_outlier = True
                elif len(detected_indices) >= 3:
                    t_prev2 = detected_indices[-3]
                    dt_prev = t_prev - t_prev2
                    v_prev2_prev = (ref_centers[t_prev] - ref_centers[t_prev2]) / max(1, dt_prev)
                    acc_n = np.linalg.norm(v_prev_n - v_prev2_prev)
                    if acc_n > self.max_acc and not is_near_foot_n:
                        is_trailing_outlier = True
                elif not is_near_foot_n:
                    is_trailing_outlier = True

            if is_trailing_outlier:
                valid_mask[t_n] = False

```

- **Khối 3: Gating chuỗi ngoại lai liên tiếp (Lookahead Chord Burst) (Lines 165-227)**:

```python
# tracker/ball_tracker.py:165-227
        # 3.2 Multi-Frame Outlier Burst Gating (Lookahead Chords)
        detected_indices = [t for t in range(num_frames) if valid_mask[t]]
        if len(detected_indices) > 2:
            i = 1
            prev_valid_idx = 0
            k_max = 4

            while i < len(detected_indices):
                prev_t = detected_indices[prev_valid_idx]
                curr_t = detected_indices[i]
                dt_prev = curr_t - prev_t
                pos_prev = ref_centers[prev_t]
                pos_curr = ref_centers[curr_t]
                speed_prev = np.linalg.norm(pos_curr - pos_prev) / max(1, dt_prev)

                if speed_prev > self.v_max:
                    # Candidate at index i deviates from previous valid detection by speed > v_max
                    # Search ahead up to k_max = 4 candidate detections
                    found_reconnection = False
                    reconnect_k = -1

                    for k in range(1, k_max + 1):
                        if i + k < len(detected_indices):
                            next_t = detected_indices[i + k]
                            pos_next = ref_centers[next_t]
                            chord_dt = next_t - prev_t
                            chord_speed = np.linalg.norm(pos_next - pos_prev) / max(1, chord_dt)

                            if chord_speed <= self.v_max:
                                # Check if intermediate burst frames are explained by foot proximity kicks
                                any_kick_explained = False
                                for m in range(i, i + k):
                                    cand_m_t = detected_indices[m]
                                    cand_m_prev_t = prev_t if m == i else detected_indices[m - 1]
                                    m_speed = np.linalg.norm(ref_centers[cand_m_t] - ref_centers[cand_m_prev_t]) / max(1, cand_m_t - cand_m_prev_t)
                                    m_near_foot = self._is_near_player(player_tracks, cand_m_t, raw_centers[cand_m_t])
                                    if m_near_foot and m_speed <= 2.0 * self.v_max:
                                        any_kick_explained = True
                                        break

                                if not any_kick_explained:
                                    found_reconnection = True
                                    reconnect_k = k
                                    break

                    if found_reconnection:
                        # Mark all intermediate burst frames {i, ..., i+k-1} as outliers
                        for m in range(i, i + reconnect_k):
                            burst_t = detected_indices[m]
                            valid_mask[burst_t] = False

                        # Advance past the burst: candidate i + reconnect_k is confirmed valid
                        prev_valid_idx = i + reconnect_k
                        i = i + reconnect_k + 1
                    else:
                        # No chord reconnection within k_max (legitimate forward transition)
                        prev_valid_idx = i
                        i += 1
                else:
                    # speed_prev <= v_max: candidate is kinematically valid
                    prev_valid_idx = i
                    i += 1

```

- **Khối 4: Quản lý đứt đoạn thích ứng 3 tầng (Lines 228-298)**:

```python
# tracker/ball_tracker.py:228-298
        # ---------------------------------------------------------------------
        # Step 4: Adaptive 3-Tier Dropout Handling
        # ---------------------------------------------------------------------
        confirmed_indices = [t for t in range(num_frames) if valid_mask[t]]
        stabilized_ref = [None] * num_frames
        is_interpolated_mask = [False] * num_frames
        track_states = ["LOST"] * num_frames

        # Copy confirmed detections
        for t in confirmed_indices:
            stabilized_ref[t] = ref_centers[t].copy()
            track_states[t] = "DETECTED"

        # Interpolate between confirmed pairs
        for i in range(len(confirmed_indices) - 1):
            t_a = confirmed_indices[i]
            t_b = confirmed_indices[i + 1]
            gap = t_b - t_a - 1

            if gap <= 0:
                continue

            if gap <= self.tier1_max_gap:
                # Tier 1 (1-4 frames): PCHIP monotonic interpolation
                self._interpolate_segment(
                    stabilized_ref,
                    is_interpolated_mask,
                    track_states,
                    t_a,
                    t_b,
                    confirmed_indices,
                )

            elif gap <= self.tier2_max_gap:
                # Tier 2 (5-10 frames): Kinematic consistency check
                p_a = stabilized_ref[t_a]
                p_b = stabilized_ref[t_b]
                avg_speed = np.linalg.norm(p_b - p_a) / float(gap + 1)

                # Calculate velocity directions before t_a and after t_b
                v_in = self._get_velocity_at(stabilized_ref, t_a, direction="in")
                v_out = self._get_velocity_at(stabilized_ref, t_b, direction="out")

                direction_ok = True
                if v_in is not None and v_out is not None:
                    norm_in = np.linalg.norm(v_in)
                    norm_out = np.linalg.norm(v_out)
                    if norm_in > 1.0 and norm_out > 1.0:
                        cos_angle = np.dot(v_in, v_out) / (norm_in * norm_out)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(cos_angle))
                        if angle_deg > self.tier2_max_angle_deg:
                            direction_ok = False

                if avg_speed <= self.v_max and direction_ok:
                    self._interpolate_segment(
                        stabilized_ref,
                        is_interpolated_mask,
                        track_states,
                        t_a,
                        t_b,
                        confirmed_indices,
                    )
                else:
                    # Direction change or excessive speed during medium gap -> REJECT
                    pass

            else:
                # Tier 3 (> 10 frames): STRICT NO-OP (remain None / LOST)
                pass

```

- **Khối 5: Làm mượt đạn đạo song phương có chặn sai lệch <= 2.5px (Lines 299-380)**:

```python
# tracker/ball_tracker.py:299-380
        # ---------------------------------------------------------------------
        # Step 5: Confidence-Weighted Bilateral Smoothing (Clamped <= 2.5 px)
        # ---------------------------------------------------------------------
        smoothed_ref = [None] * num_frames

        # Find continuous populated segments
        segments = []
        curr_seg = []
        for t in range(num_frames):
            if stabilized_ref[t] is not None:
                curr_seg.append(t)
            else:
                if curr_seg:
                    segments.append(curr_seg)
                    curr_seg = []
        if curr_seg:
            segments.append(curr_seg)

        for seg in segments:
            seg_len = len(seg)
            for idx, t in enumerate(seg):
                pt = stabilized_ref[t].copy()

                # If this frame was interpolated, PCHIP is already smooth
                if is_interpolated_mask[t]:
                    smoothed_ref[t] = pt
                    continue

                if seg_len < 3:
                    smoothed_pt = pt
                else:
                    # Edge-aware polynomial regression over window W=4
                    # Interior points (>= 2 left and >= 2 right) use quadratic degree 2 (zero ballistic curvature bias)
                    # Boundary points use linear degree 1 (avoids edge oscillation / Runge phenomenon)
                    W = 4
                    offsets = []
                    positions = []
                    weights = []
                    for offset in range(-W, W + 1):
                        neighbor_idx = idx + offset
                        if 0 <= neighbor_idx < seg_len:
                            nt = seg[neighbor_idx]
                            offsets.append(offset)
                            positions.append(stabilized_ref[nt])
                            conf = raw_confs[nt] if raw_confs[nt] is not None else 0.8
                            spatial = np.exp(-0.5 * (offset / 3.0) ** 2)
                            weights.append(conf * spatial)

                    offsets = np.array(offsets, dtype=float)
                    positions = np.array(positions, dtype=float)
                    weights = np.array(weights, dtype=float)

                    has_left = np.sum(offsets < 0) >= 2
                    has_right = np.sum(offsets > 0) >= 2
                    deg = 2 if (has_left and has_right and len(offsets) >= 5) else 1

                    if deg == 2:
                        A = np.column_stack([offsets ** 2, offsets, np.ones_like(offsets)])
                        W_mat = np.diag(weights)
                        try:
                            coeffs = np.linalg.lstsq(W_mat @ A, W_mat @ positions, rcond=None)[0]
                            smoothed_pt = coeffs[2]
                        except np.linalg.LinAlgError:
                            smoothed_pt = pt
                    else:
                        A = np.column_stack([offsets, np.ones_like(offsets)])
                        W_mat = np.diag(weights)
                        try:
                            coeffs = np.linalg.lstsq(W_mat @ A, W_mat @ positions, rcond=None)[0]
                            smoothed_pt = coeffs[1]
                        except np.linalg.LinAlgError:
                            smoothed_pt = pt

                # STRICT CLAMP: Deviation from raw detection <= max_smoothing_deviation
                raw_ref = ref_centers[t] if ref_centers[t] is not None else pt
                dev_vec = smoothed_pt - raw_ref
                dev_dist = float(np.linalg.norm(dev_vec))
                if dev_dist > self.max_smoothing_deviation:
                    smoothed_pt = raw_ref + dev_vec * (self.max_smoothing_deviation / dev_dist)

                smoothed_ref[t] = smoothed_pt

```

- **Khối 6: Chuyển đổi ngược về pixel frame và đóng gói đầu ra (Lines 381-430)**:

```python
# tracker/ball_tracker.py:381-430
        # ---------------------------------------------------------------------
        # Step 6: Map Back to Frame Coordinates & Build Output Track Dictionaries
        # ---------------------------------------------------------------------
        output_tracks = []
        default_size = (20.0, 20.0)

        # Estimate default size from detections if available
        valid_sizes = [s for s in raw_sizes if s is not None]
        if valid_sizes:
            default_size = (
                float(np.median([s[0] for s in valid_sizes])),
                float(np.median([s[1] for s in valid_sizes])),
            )

        for t in range(num_frames):
            if smoothed_ref[t] is None:
                output_tracks.append({})
                continue

            # Convert reference coordinates back to frame pixel space
            img_center = self._to_frame(camera_motion, t, smoothed_ref[t])
            cx, cy = float(img_center[0]), float(img_center[1])

            # Size: preserve raw size or interpolate
            w, h = raw_sizes[t] if raw_sizes[t] is not None else default_size
            half_w, half_h = w / 2.0, h / 2.0
            bbox = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]

            is_interp = is_interpolated_mask[t]
            conf = self.interp_confidence if is_interp else (
                raw_confs[t] if raw_confs[t] is not None else 0.85
            )

            entry = {
                "bbox": bbox,
                "confidence": float(conf),
                "is_interpolated": bool(is_interp),
                "track_state": track_states[t],
            }

            # Preserve existing metadata fields if present in original input
            if 1 in ball_tracks[t]:
                for k, v in ball_tracks[t][1].items():
                    if k not in entry:
                        entry[k] = v

            output_tracks.append({1: entry})

        return output_tracks

```

##### 5.3.2 Các hàm trợ giúp hình học và động học

- **Chuyển đổi hệ quy chiếu camera `_to_reference` và `_to_frame`** (`tracker/ball_tracker.py:435-457`):

```python
# tracker/ball_tracker.py:435-457
    def _to_reference(self, camera_motion: Optional[Any], t: int, pt: np.ndarray) -> np.ndarray:
        if camera_motion is None:
            return pt.copy()
        if hasattr(camera_motion, "to_reference"):
            return camera_motion.to_reference(t, pt.reshape(1, 2))[0]
        return pt.copy()

    def _to_frame(self, camera_motion: Optional[Any], t: int, pt_ref: np.ndarray) -> np.ndarray:
        if camera_motion is None:
            return pt_ref.copy()
        if hasattr(camera_motion, "to_frame"):
            return camera_motion.to_frame(t, pt_ref.reshape(1, 2))[0]
        if hasattr(camera_motion, "Ms") and t < len(camera_motion.Ms):
            M = camera_motion.Ms[t]
            try:
                M_inv = np.linalg.inv(M)
                q = M_inv @ np.array([pt_ref[0], pt_ref[1], 1.0], dtype=np.float64)
                if abs(q[2]) > 1e-8:
                    return q[:2] / q[2]
            except np.linalg.LinAlgError:
                pass
        return pt_ref.copy()

```

- **Kiểm tra lân cận chân cầu thủ `_is_near_player`** (`tracker/ball_tracker.py:458-485`):

```python
# tracker/ball_tracker.py:458-485
    def _is_near_player(
        self,
        player_tracks: Optional[List[Dict[int, Dict[str, Any]]]],
        t: int,
        ball_center: Optional[np.ndarray],
    ) -> bool:
        """Check if ball is within foot_proximity_thresh of any player's feet."""
        if player_tracks is None or ball_center is None or t >= len(player_tracks):
            return False

        players = player_tracks[t]
        for _, p_info in players.items():
            bbox = p_info.get("bbox")
            if bbox is None:
                continue
            # Foot coordinates: bottom corners and bottom center
            foot_center = np.array([(bbox[0] + bbox[2]) / 2.0, bbox[3]])
            foot_left = np.array([bbox[0], bbox[3]])
            foot_right = np.array([bbox[2], bbox[3]])

            d_center = np.linalg.norm(ball_center - foot_center)
            d_left = np.linalg.norm(ball_center - foot_left)
            d_right = np.linalg.norm(ball_center - foot_right)

            if min(d_center, d_left, d_right) <= self.foot_proximity_thresh:
                return True

        return False
```

- **Ước lượng vector vận tốc `_get_velocity_at`** (`tracker/ball_tracker.py:487-504`):

```python
# tracker/ball_tracker.py:487-504
    def _get_velocity_at(
        self,
        positions: List[Optional[np.ndarray]],
        t: int,
        direction: str = "in",
    ) -> Optional[np.ndarray]:
        """Estimate velocity vector entering (in) or exiting (out) frame t."""
        if direction == "in":
            # Backward velocity
            for prev_t in range(t - 1, max(-1, t - 4), -1):
                if positions[prev_t] is not None:
                    return (positions[t] - positions[prev_t]) / float(t - prev_t)
        else:
            # Forward velocity
            for next_t in range(t + 1, min(len(positions), t + 4)):
                if positions[next_t] is not None:
                    return (positions[next_t] - positions[t]) / float(next_t - t)
        return None
```

- **Nội suy đơn điệu PCHIP `_interpolate_segment`** (`tracker/ball_tracker.py:506-551`):

```python
# tracker/ball_tracker.py:506-551
    def _interpolate_segment(
        self,
        positions: List[Optional[np.ndarray]],
        is_interp_mask: List[bool],
        track_states: List[str],
        t_a: int,
        t_b: int,
        confirmed_indices: List[int],
    ):
        """Monotonic PCHIP interpolation across frames between t_a and t_b."""
        # Collect local context points around t_a and t_b for smooth polynomial matching
        context_t = []
        context_pts = []

        # Find up to 3 points before t_a
        before = [t for t in confirmed_indices if t <= t_a][-3:]
        # Find up to 3 points after t_b
        after = [t for t in confirmed_indices if t >= t_b][:3]

        all_ctx_t = sorted(list(set(before + after)))
        for ct in all_ctx_t:
            context_t.append(ct)
            context_pts.append(positions[ct])

        context_pts = np.array(context_pts)

        if len(context_t) >= 3:
            # Fit PCHIP on x and y coordinates independently
            pchip_x = PchipInterpolator(context_t, context_pts[:, 0])
            pchip_y = PchipInterpolator(context_t, context_pts[:, 1])

            for fill_t in range(t_a + 1, t_b):
                ix = float(pchip_x(fill_t))
                iy = float(pchip_y(fill_t))
                positions[fill_t] = np.array([ix, iy], dtype=np.float64)
                is_interp_mask[fill_t] = True
                track_states[fill_t] = "INTERPOLATED"
        else:
            # Linear fallback if strictly 2 points available
            p_a = positions[t_a]
            p_b = positions[t_b]
            for fill_t in range(t_a + 1, t_b):
                alpha = float(fill_t - t_a) / float(t_b - t_a)
                positions[fill_t] = (1.0 - alpha) * p_a + alpha * p_b
                is_interp_mask[fill_t] = True
                track_states[fill_t] = "INTERPOLATED"
```


#### 5.4 Thứ tự bước xử lý bóng đầy đủ trong pipeline

Quy trình xử lý bóng từ detection thô đến hiển thị cuối cùng:

1. **Suy luận YOLO**: `Tracker.detect_frame` tạo detection với `class_id == ids['ball']` (`tracker/tracker.py:270-271`).

2. **Chọn bóng thô**: `_select_ball_detection` chọn tối đa 1 detection ứng viên mỗi frame lưu vào `tracks['ball'][frame_idx][1]` (`tracker/tracker.py:340-348`).

3. **Ổn định quỹ đạo & nội suy thích ứng 3 tầng**: Gọi `tracker.interpolate_ball_position` tại `main.py:230-234` (thực chất chuyển tiếp sang `BallTrajectoryStabilizer.process` trong `tracker/ball_tracker.py:177-183`).

4. **Gán tọa độ điểm tâm**: `tracker.add_position_to_track` tính `position = get_center_of_box(bbox)` (`tracker/tracker.py:161`).

5. **Chiếu tọa độ sân cỏ**: `PitchMapper.add_positions_to_tracks` gán `position_local_raw`, `position_adjusted`, `position_transformed`, `projection_valid`, `calibration_status`, `extrapolation_risk` (`pitch_homo.py:612, 623, 663`).

6. **Hiển thị trên video chính**: `Tracker.draw_tracked_frames` vẽ tam giác đỏ (`tracker/tracker.py:633-672`).

   - **LƯU Ý CODE ĐẶC BIỆT**: Tại dòng 634-635, có lệnh: `if ball.get('is_interpolated', False): continue`. Như vậy, trên video chính `output.mp4`, các frame bóng nội suy **BỊ BỎ QUA KHÔNG VẼ**! Bóng chỉ được vẽ ở các frame phát hiện thật (`is_interpolated == False`).

7. **Hiển thị trên Minimap Radar 2D**: `MinimapRenderer.render_frame` đọc `b_info.get('position_transformed')` (`minimap/minimap_renderer.py:272-274`). Trên minimap, bóng được vẽ ở **MỌI frame có tọa độ hợp lệ** (cả frame thật lẫn frame nội suy).

- **Đánh giá Code Chết / Dự phòng**: Trong `tracker/tracker.py`, trước đây có hàm nội suy bóng bằng pandas (`pandas.DataFrame.interpolate(method='linear')`). Hàm này đã được thay thế hoàn toàn bởi `tracker.interpolate_ball_position` (gọi sang `ball_tracker.py`), code pandas cũ không còn tồn tại trong hàm này.


#### 5.5 Hệ quy chiếu tọa độ bóng ở mỗi bước

Bóng được chuyển đổi qua 3 hệ quy chiếu tọa độ chính:

1. **Hệ tọa độ Pixel ảnh thô (Image Pixel Frame)**:

   - Đơn vị: pixel $(x, y)$ trên khung hình $1920 \times 1080$.

   - Lưu tại trường: `bbox` $[x_1, y_1, x_2, y_2]$, `position` $(x_c, y_c)$, `position_local_raw`.

2. **Hệ quy chiếu tham chiếu bù camera (Camera Reference Space)**:

   - Đơn vị: pixel ảo sau khi nhân ma trận tích lũy camera $M_t$.

   - Được sử dụng tạm thời trong `ball_tracker.py` (`ref_centers`) để đánh giá vận tốc vật lý thực độc lập với chuyển động lia máy, sau đó được `PitchMapper` ghi vào trường `position_adjusted`.

3. **Hệ tọa độ mét mặt sân chuẩn FIFA (Pitch Metric Coordinates)**:

   - Đơn vị: mét $[X, Y]$ trên mặt sân chuẩn FIFA ($105.0\text{m} \times 68.0\text{m}$), gốc $(0,0)$ ở góc trên bên trái.

   - Lưu tại trường: `position_transformed` $[X, Y]$. Được `MinimapRenderer` sử dụng để chiếu về pixel minimap $(p_x, p_y)$.


#### 5.6 Cách đánh dấu bóng 'detect thật' so với 'nội suy'

- **Cờ đánh dấu**: Trường boolean `is_interpolated` và chuỗi `track_state`.

- **Giá trị gán**:

  - Frame detect thật: `is_interpolated = False`, `track_state = 'DETECTED'`, `confidence` lấy từ YOLO.

  - Frame nội suy: `is_interpolated = True`, `track_state = 'INTERPOLATED'`, `confidence = 0.30` (được gán tại `tracker/ball_tracker.py:410, 417-418`).

- **Lưu trữ đầu ra**: Có lưu đầy đủ trong dictionary `tracks['ball'][frame_idx][1]`.

- **Minimap radar lấy dữ liệu từ đâu**: Minimap đọc trực tiếp từ `b_info.get('position_transformed')` tại `minimap/minimap_renderer.py:272`.


#### 5.7 Ball control / Gán bóng cho cầu thủ

- **Hàm thực thi**: `PlayerBallAssigner.assign_ball_to_player(player_track, ball_bbox)` (`player_ball_assigner/player_ball_assigner.py:8-25`).

```python
# player_ball_assigner/player_ball_assigner.py:5-25
class PlayerBallAssigner():
    def __init__(self):
        self.max_distance = 70
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_box(ball_bbox)

        min_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id
        return assigned_player
```

- **Ngưỡng khoảng cách**: `self.max_distance = 70` (pixels trong không gian ảnh).

- **Tọa độ sử dụng**: Tính khoảng cách Euclid giữa tâm bóng (`get_center_of_box(ball_bbox)`) và hai điểm góc dưới của bounding box cầu thủ (chân trái `(bbox[0], bbox[3])` và chân phải `(bbox[2], bbox[3])`) trong **hệ tọa độ PIXEL ẢNH THÔ**.


---

### MỤC 6. Hợp đồng dữ liệu (data contract)

#### 6.1 Cấu trúc đối tượng `tracks['players'][frame][id]`

| Tên trường | Kiểu dữ liệu | Ghi bởi hàm nào (File:Dòng) | Đọc bởi các hàm nào | Hệ tọa độ / Đơn vị |
|---|---|---|---|---|
| `bbox` | `list [x1, y1, x2, y2] (float)` | `tracker/tracker.py:306 hoặc algorithmic_reid/reid_tracker_facade.py:173` | `draw_tracked_frames, team_assigner, player_ball_assigner, add_position_to_track` | Pixel ảnh thô (1920x1080) |
| `confidence` | `float` | `reid_tracker_facade.py:176` | `Kiểm toán, chẩn đoán chất lượng` | Xác suất [0.0, 1.0] |
| `is_goalkeeper` | `bool (chỉ có khi là GK)` | `reid_tracker_facade.py:183, tracker.py:334, main.py:59` | `draw_ellipse, validate_roles_and_assign_goalkeepers, minimap_renderer` | Cờ boolean |
| `team` | `int (1 hoặc 2)` | `team_assigner.py:376, 382 hoặc main.py:116` | `draw_tracked_frames, minimap_renderer, team_ball_control` | Định danh đội bóng |
| `team_color` | `tuple (B, G, R)` | `team_assigner.py:377, 383 hoặc main.py:118` | `draw_ellipse, minimap_renderer` | Màu sắc BGR |
| `position` | `tuple (x, y)` | `tracker/tracker.py:163` | `PitchMapper.add_positions_to_tracks` | Pixel ảnh thô (chân đối tượng) |
| `position_local_raw` | `tuple (x, y)` | `pitch_homo.py:614` | `Downstream audit` | Pixel ảnh thô |
| `position_adjusted` | `tuple (x, y)` | `pitch_homo.py:623` | `Downstream audit` | Pixel ảo tham chiếu camera |
| `projection_valid` | `bool` | `pitch_homo.py:655, 666` | `SpeedAndDistance_Calc` | Cờ tính hợp lệ hình học |
| `calibration_status` | `str` | `pitch_homo.py:656, 667` | `SpeedAndDistance_Calc, chẩn đoán` | DIRECT_CALIBRATED, PROPAGATED, PREDICTED, UNREGISTERED |
| `extrapolation_risk` | `float | None` | `pitch_homo.py:657, 668` | `Downstream analysis` | Mét (khoảng cách tới Convex Hull) |
| `is_extrapolated` | `bool | None` | `pitch_homo.py:658, 669` | `Downstream analysis` | Cờ ngoại suy ngoài Hull |
| `extrapolation_level` | `str | None` | `pitch_homo.py:659, 670` | `Downstream analysis` | SAFE, LOW, HIGH |
| `inside_pitch` | `bool | None` | `pitch_homo.py:662, 671` | `Downstream analysis` | True nếu nằm trong biên sân FIFA |
| `position_transformed` | `list [X, Y] (float)` | `pitch_homo.py:663` | `SpeedAndDistance_Calc, validate_roles, minimap_renderer` | Mét trên mặt sân FIFA (105x68m) |
| `position_sigma` | `float | None` | `pitch_homo.py:664` | `Chẩn đoán độ không đảm bảo` | Mét |
| `has_ball` | `bool (tùy chọn)` | `main.py:271` | `draw_tracked_frames, minimap_renderer` | Cờ đang giữ bóng |
| `speed` | `float (km/h)` | `speed_distance_calc.py:70` | `draw_speed_and_distance` | Vận tốc km/h |
| `distance` | `float (m)` | `speed_distance_calc.py:71` | `draw_speed_and_distance` | Quãng đường mét |
| `is_predicted` | `bool` | `speed_distance_calc.py:68` | `Chẩn đoán vận tốc` | True nếu frame thuộc PREDICTED |
| `speed_is_predicted` | `bool` | `speed_distance_calc.py:69` | `Chẩn đoán vận tốc` | True nếu cửa sổ có frame PREDICTED |

#### 6.2 Cấu trúc cho `tracks['referees']` và `tracks['ball']`

##### Cấu trúc `tracks['referees'][frame][id]`:

- Bao gồm đầy đủ các trường hình học giống `players`: `bbox`, `confidence`, `position`, `position_local_raw`, `position_adjusted`, `projection_valid`, `calibration_status`, `extrapolation_risk`, `is_extrapolated`, `extrapolation_level`, `inside_pitch`, `position_transformed`, `position_sigma`.

- **KHÔNG có**: `team`, `team_color`, `is_goalkeeper`, `has_ball`, `speed`, `distance` (vì SpeedAndDistance_Calc bỏ qua trọng tài tại `speed_distance_calc.py:13`).


##### Cấu trúc `tracks['ball'][frame][1]`:

| Tên trường | Kiểu dữ liệu | Ghi bởi hàm nào (File:Dòng) | Đọc bởi các hàm nào | Hệ tọa độ / Đơn vị |
|---|---|---|---|---|
| `bbox` | `list [x1, y1, x2, y2]` | `tracker/ball_tracker.py:415 hoặc tracker.py:725` | `draw_tracked_frames, player_ball_assigner, add_position_to_track` | Pixel ảnh thô |
| `confidence` | `float` | `tracker/ball_tracker.py:416` | `Chẩn đoán chất lượng` | Xác suất [0.0, 1.0] |
| `is_interpolated` | `bool` | `tracker/ball_tracker.py:417` | `tracker.py:634 (quyết định vẽ tam giác trên video chính)` | Cờ phân biệt detect thật vs nội suy |
| `track_state` | `str` | `tracker/ball_tracker.py:418` | `Chẩn đoán` | 'DETECTED', 'INTERPOLATED', 'LOST' |
| `position` | `tuple (x, y)` | `tracker/tracker.py:161` | `PitchMapper.add_positions_to_tracks` | Pixel ảnh thô (tâm quả bóng) |
| `position_local_raw` | `tuple (x, y)` | `pitch_homo.py:614` | `Downstream audit` | Pixel ảnh thô |
| `position_adjusted` | `tuple (x, y)` | `pitch_homo.py:623` | `Downstream audit` | Pixel ảo tham chiếu camera |
| `projection_valid` | `bool` | `pitch_homo.py:655` | `Downstream audit` | Cờ hợp lệ hình học |
| `calibration_status` | `str` | `pitch_homo.py:656` | `Downstream audit` | Trạng thái hiệu chuẩn |
| `extrapolation_risk` | `float | None` | `pitch_homo.py:657` | `Downstream audit` | Khoảng cách tới Hull (m) |
| `inside_pitch` | `bool | None` | `pitch_homo.py:662` | `Downstream audit` | True nếu bóng trong sân |
| `position_transformed` | `list [X, Y]` | `pitch_homo.py:663` | `minimap/minimap_renderer.py:272` | Mét trên mặt sân FIFA (105x68m) |

#### 6.3 Những trường mà các module downstream đọc từ tracking

| Module Downstream | Các trường đọc từ tracks và mục đích sử dụng |
|---|---|
| **camera_motion (CameraMotion.step)** | prev_boxes: đọc trường `bbox` của toàn bộ player, referee, ball từ frame trước đó để tạo mặt nạ nhị phân (mask) loại trừ chuyển động của đối tượng khi tính optical flow / Lucas-Kanade. |
| **view_transformer / pitch_homo (PitchMapper)** | Đọc trường `position` (hoặc `bbox` để tính vị trí chân/tâm) của toàn bộ các đối tượng. |
| **team_assignment (TeamAssigner)** | Đọc trường `bbox` của các cầu thủ outfield trong `tracks['players']` để crop nửa thân trên áo đấu phục vụ phân cụm K-Means. |
| **player_ball_assigner (PlayerBallAssigner)** | Đọc trường `bbox` của cầu thủ trong `tracks['players']` và `bbox` của bóng trong `tracks['ball']` để gán quyền kiểm soát bóng. |
| **speed_distance (SpeedAndDistance_Calc)** | Đọc trường `position_transformed`, `calibration_status`, `projection_valid`, `bbox` của cầu thủ trong `tracks['players']`. |
| **validate_roles_and_assign_goalkeepers (main.py)** | Đọc trường `position_transformed`, `team`, `is_goalkeeper` của cầu thủ và trọng tài. |
| **drawing (Tracker.draw_tracked_frames)** | Đọc `bbox`, `team_color`, `is_goalkeeper`, `has_ball` của player; `bbox` của referee; `bbox`, `is_interpolated` của ball. |
| **minimap (MinimapRenderer)** | Đọc `position_transformed`, `team`, `team_color`, `is_goalkeeper`, `has_ball` của player; `position_transformed` của referee; `position_transformed` của ball. |

---

### MỤC 7. Tham số tổng hợp

Dưới đây là một bảng DUY NHẤT gom MỌI siêu tham số ảnh hưởng đến phát hiện và bám vết người, bóng trong toàn hệ thống:

| Tên tham số | Giá trị hiện tại | File:Dòng | Thuộc nhóm | Cơ chế điều chỉnh |
|---|---|---|---|---|
| `player_conf` | `0.35` | `main.py:331` | detector / tracker | CÓ (CLI `--player-conf`) |
| `goalkeeper_conf` | `0.30` | `main.py:337` | detector / tracker | CÓ (CLI `--gk-conf`) |
| `referee_conf` | `0.30` | `main.py:343` | detector / tracker | CÓ (CLI `--ref-conf`) |
| `ball_conf` | `0.20` | `main.py:349` | detector / ball | CÓ (CLI `--ball-conf`) |
| `batch_size` | `8` | `main.py:355` | detector | CÓ (CLI `--batch-size`) |
| `model_path` | `best_V26_over_6000.pt` | `main.py:326` | detector | CÓ (CLI `--model`) |
| `calib_path` | `calib/points.json` | `main.py:361` | homography / camera | CÓ (CLI `--calib`) |
| `generate_minimap` | `True` | `main.py:365` | minimap | CÓ (CLI `--no-minimap`) |
| `device` | `None (auto CUDA:0)` | `main.py:370` | hardware | CÓ (CLI `--device`) |
| `choose_Al` | `2` | `main.py:217` | tracker | HARDCODE trong main.py |
| `imgsz` | `1536` | `tracker/tracker.py:200` | detector | HARDCODE trong tracker.py |
| `agnostic_nms` | `False` | `tracker/tracker.py:201` | detector / NMS | HARDCODE trong tracker.py |
| `min_predict_conf` | `0.15` | `tracker/tracker.py:86` | detector | TÍNH TOÁN NỘI BỘ |
| `cross_class_iou_thresh` | `0.45` | `tracker/tracker.py:90` | NMS / tracker | HARDCODE trong tracker.py |
| `ByteTrack lost_track_buffer` | `70` | `tracker/tracker.py:37` | tracker (legacy) | HARDCODE trong tracker.py |
| `ByteTrack track_activation_threshold` | `0.20` | `tracker/tracker.py:39` | tracker (legacy) | HARDCODE trong tracker.py |
| `ByteTrack minimum_iou_threshold` | `0.20` | `tracker/tracker.py:41` | tracker (legacy) | HARDCODE trong tracker.py |
| `BoTSORT lost_track_buffer` | `70` | `tracker/tracker.py:45` | tracker (legacy) | HARDCODE trong tracker.py |
| `BoTSORT minimum_iou_first` | `0.30` | `tracker/tracker.py:49` | tracker (legacy) | HARDCODE trong tracker.py |
| `BoTSORT minimum_iou_second` | `0.50` | `tracker/tracker.py:50` | tracker (legacy) | HARDCODE trong tracker.py |
| `BoTSORT cmc_downscale` | `2` | `tracker/tracker.py:54` | camera / tracker | HARDCODE trong tracker.py |
| `ReID high_conf_thresh` | `0.35 (ghi đè)` | `tracker/tracker.py:62` | reid | Cấu hình qua player_conf |
| `ReID low_conf_thresh` | `0.20 (ghi đè)` | `tracker/tracker.py:63` | reid | Cấu hình qua track_thresh |
| `ReID init_conf_thresh` | `0.35 (ghi đè)` | `tracker/tracker.py:64` | reid | Cấu hình qua player_conf |
| `ReID n_init` | `2` | `algorithmic_reid/config.py:15` | reid | HARDCODE trong config.py |
| `ReID max_age` | `70` | `algorithmic_reid/config.py:16` | reid | HARDCODE trong config.py |
| `ReID reactivation_window` | `30` | `algorithmic_reid/config.py:17` | reid | HARDCODE trong config.py |
| `ReID w_diou` | `0.40` | `algorithmic_reid/config.py:25` | reid | HARDCODE trong config.py |
| `ReID w_mot` | `0.20` | `algorithmic_reid/config.py:26` | reid | HARDCODE trong config.py |
| `ReID w_app` | `0.30` | `algorithmic_reid/config.py:27` | reid | HARDCODE trong config.py |
| `ReID w_dir` | `0.10` | `algorithmic_reid/config.py:28` | reid | HARDCODE trong config.py |
| `ReID max_cost_stage1` | `0.75` | `algorithmic_reid/config.py:31` | reid | HARDCODE trong config.py |
| `ReID max_cost_stage2` | `0.85` | `algorithmic_reid/config.py:32` | reid | HARDCODE trong config.py |
| `ReID max_cost_stage3` | `0.70` | `algorithmic_reid/config.py:33` | reid | HARDCODE trong config.py |
| `ReID max_cost_tentative` | `0.75` | `algorithmic_reid/config.py:34` | reid | HARDCODE trong config.py |
| `ReID mahalanobis_gate` | `13.28` | `algorithmic_reid/config.py:37` | reid / motion | HARDCODE trong config.py |
| `ReID appearance_gate` | `0.55` | `algorithmic_reid/config.py:38` | reid / appearance | HARDCODE trong config.py |
| `ReID max_velocity_px` | `35.0` | `algorithmic_reid/config.py:39` | reid / motion | HARDCODE trong config.py |
| `ReID overlap_occlusion_iou` | `0.15` | `algorithmic_reid/config.py:40` | reid / appearance | HARDCODE trong config.py |
| `ReID velocity_damping_gamma` | `0.90` | `algorithmic_reid/config.py:43` | reid / motion | HARDCODE trong config.py |
| `ReID std_weight_position` | `1/20 (0.05)` | `algorithmic_reid/config.py:44` | reid / motion | HARDCODE trong config.py |
| `ReID std_weight_velocity` | `1/160 (0.00625)` | `algorithmic_reid/config.py:45` | reid / motion | HARDCODE trong config.py |
| `ReID hist_bins (H, S, V)` | `16, 8, 8` | `algorithmic_reid/config.py:48-50` | reid / appearance | HARDCODE trong config.py |
| `ReID ema_alpha` | `0.15` | `algorithmic_reid/config.py:51` | reid / appearance | HARDCODE trong config.py |
| `ball v_max` | `50.0 px/frame` | `tracker/ball_tracker.py:28` | ball | HARDCODE trong ball_tracker.py |
| `ball max_acc` | `30.0 px/frame^2` | `tracker/ball_tracker.py:29` | ball | HARDCODE trong ball_tracker.py |
| `ball foot_proximity_thresh` | `60.0 px` | `tracker/ball_tracker.py:30` | ball | HARDCODE trong ball_tracker.py |
| `ball tier1_max_gap` | `4 frames` | `tracker/ball_tracker.py:31` | ball | HARDCODE trong ball_tracker.py |
| `ball tier2_max_gap` | `10 frames` | `tracker/ball_tracker.py:32` | ball | HARDCODE trong ball_tracker.py |
| `ball tier2_max_angle_deg` | `60.0 degrees` | `tracker/ball_tracker.py:33` | ball | HARDCODE trong ball_tracker.py |
| `ball max_smoothing_deviation` | `2.5 px` | `tracker/ball_tracker.py:34` | ball | HARDCODE trong ball_tracker.py |
| `ball interp_confidence` | `0.30` | `tracker/ball_tracker.py:35` | ball | HARDCODE trong ball_tracker.py |
| `ball chord burst k_max` | `4` | `tracker/ball_tracker.py:170` | ball | HARDCODE trong ball_tracker.py |
| `player_ball_assigner max_distance` | `70 px` | `player_ball_assigner/player_ball_assigner.py:7` | ball / assignment | HARDCODE trong player_ball_assigner.py |
| `team_assigner window_size` | `15 frames` | `team_assignment/team_assigner.py:8` | team | HARDCODE trong team_assigner.py |
| `speed_distance frame_window` | `5 frames` | `speed_distance/speed_distance_calc.py:7` | speed / distance | HARDCODE trong speed_distance_calc.py |

---
### MỤC 8. Kiểm thử hiện có

Dưới đây là bảng tổng hợp tất cả các file test và script đánh giá liên quan đến tracking người và bóng trong dự án:

| Tên file test | Nội dung kiểm thử thực tế | Dữ liệu kiểm thử | Ground Truth | Cách chạy |
|---|---|---|---|---|
| [`test_reid_tracking_suite.py`](test_reid_tracking_suite.py) | Kiểm thử 8 kịch bản chuẩn hóa của Algorithmic Re-ID: quỹ đạo đan chéo vận tốc đối nghịch, dropout 1-10 frame, dao động confidence 0.70 <-> 0.25, đồng đội chạy song song cùng màu áo, bù chuyển động camera pan, triệt tiêu false detection 1-frame, cách ly loài cứng player/gk/referee, và tương thích hợp đồng dữ liệu. | Dữ liệu tổng hợp (Synthetic) tạo khung hình và đối tượng nhân tạo với màu áo, bounding box và chuyển động mô phỏng chính xác. | Tích hợp ma trận ground truth trực tiếp trong từng hàm test. | `python test_reid_tracking_suite.py --reid` hoặc `python test_reid_tracking_suite.py --baseline` |
| [`test_reid_adversarial_stress.py`](test_reid_adversarial_stress.py) | Kiểm thử ứng suất đối kháng 16 kịch bản cho Re-ID: 4 cầu thủ giao nhau đồng thời tại ngã tư (đủ màu và cùng màu áo), cụm 4 người kèm dropout 3 frame, mất dấu kéo dài 15-20 frame (chuyển động cong, giảm tốc dừng hẳn, ngoặt đổi hướng feint/cutback), camera pan tốc độ cao 25-75 px/f kèm gia tốc giật, và cụm 6 người chen chúc trong vòng cấm 150x150 px. | Dữ liệu tổng hợp đối kháng (Adversarial Synthetic) với các chuyển động phi tuyến và phối cảnh xạ ảnh méo. | Tích hợp sẵn assertion ground truth nghiêm ngặt trong mã nguồn. | `python test_reid_adversarial_stress.py` |
| [`test_challenger_p0_stress.py`](test_challenger_p0_stress.py) | Kiểm thử ứng suất P0 độc lập của Challenger: rung lắc bounding box +-15px và biến dạng tỷ lệ khung hình; dao động confidence sát ngưỡng 0.20-0.25; di chuyển zigzag tốc độ cao 10-32 px/frame gần đường biên sân. | Dữ liệu tổng hợp (Synthetic) mô phỏng nhiễu đo đạc biên. | Tích hợp sẵn ngưỡng assertion kiểm tra số lần ID switch và tính liên tục. | `python test_challenger_p0_stress.py` |
| [`test_ball_trajectory_suite.py`](test_ball_trajectory_suite.py) | Kiểm thử 10 kịch bản chuẩn hóa cho quỹ đạo bóng: TS-B01 làm mượt rung lắc RMS giảm >= 40%; TS-B02 loại bỏ ngoại lai 300px (v > 50 px/f); TS-B03 nội suy PCHIP dropout 1-3 frame; TS-B04 từ chối nội suy dropout đổi hướng > 60 độ; TS-B05 STRICT NO-OP triệt tiêu hoàn toàn bóng ma trên dropout 60+ frame; TS-B06 chấp nhận cú sút gia tốc khi gần chân cầu thủ; TS-B07 bóng dừng/lăn chậm; TS-B08 bù camera pan; TS-B09 bóng ra biên; TS-B10 toàn vẹn minimap và contract. | Dữ liệu tổng hợp (Synthetic) mô phỏng đạn đạo bóng, nhiễu và đứt đoạn quang học. | Tích hợp kịch bản ground truth định lượng cho từng test case. | `python test_ball_trajectory_suite.py --stabilizer` hoặc `--baseline` |
| [`test_challenger_p1_stress.py`](test_challenger_p1_stress.py) | Kiểm thử ứng suất P1 độc lập của Challenger: chuỗi ngoại lai liên tiếp ADV-B01 (2-3 frame giả vọt xa), ngoại lai ngoài biên sân, mất dấu nhiều chặng ngắt quãng, camera rung giật mạnh. | Dữ liệu tổng hợp đối kháng (Adversarial Synthetic). | Tích hợp kiểm tra tự động sai số đạn đạo và tỷ lệ lọt ngoại lai. | `python test_challenger_p1_stress.py` |
| [`test_stage2_suite.py`](test_stage2_suite.py) | Kiểm thử hồi quy Giai đoạn 2: so sánh DLT vs MAGSAC homography, kiểm tra safe_project chống điểm vô cực/chân trời, tính extrapolation_risk tới Convex Hull, đảm bảo bất biến hợp đồng dữ liệu. | Dữ liệu tọa độ mốc sân chuẩn FIFA kết hợp nhiễu Gaussian và ma trận giả lập. | Ground truth mốc sân FIFA 105x68m. | `python test_stage2_suite.py` |
| [`test_stage3_suite.py`](test_stage3_suite.py) | Kiểm thử hồi quy Giai đoạn 3: Anchor-grounded camera tracking, kiểm soát sai số tích lũy drift < 5px qua 100 frames, phát hiện camera cut và tái định vị re-localization. | Dữ liệu tổng hợp kết hợp frame trích xuất từ video thực tế. | Tọa độ ground truth biến đổi camera và mốc anchor. | `python test_stage3_suite.py` |
| [`test_stage4_suite.py`](test_stage4_suite.py) | Kiểm thử hồi quy Giai đoạn 4: Central View Conic Geometry, khớp elip xạ ảnh từ cung tròn giữa sân R=9.15m, giải suy biến thẳng hàng qua hàng điểm điều hòa và đường giữa sân, kiểm soát cờ is_predicted. | Dữ liệu hình học cung tròn và camera xạ ảnh giả lập. | Ground truth phương trình Conic C_pitch và tâm sân (52.5, 34.0). | `python test_stage4_suite.py` |
| [`verify_central_view_real_frame.py`](verify_central_view_real_frame.py) | Kiểm toán thực nghiệm độc lập hình học Conic và Support Hull trên frame 1 thật của video/sample.mp4: lọc ngoại lai xuyên tâm Radial Filter (tau=1.5m), đối chiếu Shoelace diện tích Model Hull (642.91 m2) vs Observed Hull, audit extrapolation_risk. | VIDEO THẬT (`video/sample.mp4` frame 1) kết hợp điểm mốc `calib/sample.json`. | Tọa độ mốc click tay và mô hình hình học sân FIFA. | `python verify_central_view_real_frame.py` |
| [`benchmark_pipeline.py`](benchmark_pipeline.py) | Đo đạc thời gian thực thi toàn diện của pipeline, tốc độ FPS, bộ nhớ RAM RSS và VRAM GPU trên video mẫu đầy đủ. | VIDEO THẬT (`video/sample.mp4`, 750 frames Full HD 1080p). | Không có ground truth tracking (chỉ đo runtime và resource). | `python benchmark_pipeline.py` |
| [`benchmark_reid_tracking.py`](benchmark_reid_tracking.py) | Đo đạc định lượng chất lượng tracking người trên video thực tế: số lượng Unique Player IDs (baseline 72 IDs -> ReID 47 IDs), số tracklet phân mảnh ngắn (< 30 frame), số lần tráo đổi màu đội bóng trong cùng 1 ID, và số vụ cướp ID giữa cầu thủ và trọng tài. | VIDEO THẬT (`video/sample.mp4`, 750 frames Full HD 1080p). | So sánh tương đối giữa file cache baseline `stubs/track_stubs.pkl` và thuật toán ReID mới `choose_Al=2`. | `python benchmark_reid_tracking.py` |
| [`run_benchmark_m6.py`](run_benchmark_m6.py) | Giám sát tài nguyên đa luồng theo chu kỳ 0.2s (RAM RSS, VRAM cấp phát, VRAM dự phòng) và kiểm tra tính toàn vẹn giải mã của 2 file video đầu ra (750/750 frames valid, 0 corrupt frames). | VIDEO THẬT (`video/sample.mp4`). | Kiểm tra chuẩn kích thước khung hình, FPS và số frame giải mã. | `python run_benchmark_m6.py` |

**Xác nhận về kiểm thử chất lượng tracking trên VIDEO THẬT**:

- Đối với các chỉ số kinh điển như **MOTA, MOTP, IDF1, HOTA** hoặc **Ball Recall chính thức** theo format MOTChallenge: **KHÔNG CÓ TRONG CODE** (không có file annotation ground truth nhãn đóng khung từng frame cho `video/sample.mp4` trong repository để tính toán các metric chuẩn này).

- Hệ thống hiện sử dụng script [`benchmark_reid_tracking.py`](benchmark_reid_tracking.py) để đo lường bán thực nghiệm trực tiếp trên `video/sample.mp4` thông qua các đại lượng thống kê:

  1. Tổng số Unique Player IDs được tạo ra (càng gần số lượng cầu thủ thực tế ~20-22 người trên sân càng tốt; baseline tạo 72 IDs, ReID giảm xuống 47 IDs).

  2. Số lượng phân mảnh track ngắn sống dưới 30 frames (baseline 48 tracks, ReID giảm còn 23 tracks, tương ứng giảm -34.7% phân mảnh).

  3. Số lần tráo đổi màu đội bóng (team swap) trong cùng một track ID (đo lường độ thuần nhất ID).

  4. Số lần ô nhiễm ID chéo giữa Cầu thủ và Trọng tài (baseline có 3 vụ tráo đổi, ReID đạt 0 vụ - sạch 100%).


---

### MỤC 9. Dữ liệu và công cụ sẵn có phục vụ đánh giá

Dưới đây là danh mục toàn bộ tài sản dữ liệu, file cache, điểm mốc hiệu chuẩn và script công cụ có sẵn trong repository:

| Đường dẫn tài sản / công cụ | Mô tả chức năng | Tình trạng tồn tại |
|---|---|---|
| `video/sample.mp4` | Video kiểm chuẩn chính thức của dự án: Full HD 1920x1080 @ 25.0 fps, 750 frames (thời lượng 30s), dung lượng 19.40 MB. | **TỒN TẠI** |
| `video/sample2.mp4` | Video kiểm thử phụ góc quay truyền hình rộng: 1304x726 @ 30.0 fps, 1128 frames (thời lượng 37.6s), dung lượng 18.95 MB. | **TỒN TẠI** |
| `video/sample3.mp4` | Video kiểm thử góc quay cận trung tâm: 1302x580 @ 30.0 fps, 1104 frames (thời lượng 36.8s), dung lượng 14.98 MB. | **TỒN TẠI** |
| `video/data4.mp4` | Video kiểm thử thời lượng dài: 1318x738 @ 30.0 fps, 1813 frames (thời lượng 60.4s), dung lượng 31.12 MB. | **TỒN TẠI** |
| `calib/points.json` | File JSON lưu trữ tọa độ pixel các điểm mốc hiệu chuẩn sân mặc định cho `video/sample.mp4` (485 bytes). | **TỒN TẠI** |
| `calib/sample.json` | File JSON lưu trữ tập điểm mốc sân đầy đủ phục vụ hiệu chuẩn Giai đoạn 4 (605 bytes). | **TỒN TẠI** |
| `calib/sample3.json` | File JSON lưu trữ điểm mốc hiệu chuẩn cho `video/sample3.mp4` (1397 bytes). | **TỒN TẠI** |
| `stubs/track_stubs_reid.pkl` | File pickle cache lưu trữ kết quả tracking hoàn chỉnh của Algorithmic Re-ID trên `video/sample.mp4` (dung lượng 1.12 MB). | **TỒN TẠI** |
| `stubs/track_stubs.pkl` | File pickle cache lưu trữ kết quả tracking cũ (baseline BoT-SORT) trên `video/sample.mp4` (dung lượng 0.91 MB). | **TỒN TẠI** |
| `stubs/track_stubs_bench.pkl` | File pickle cache lưu kết quả tracking 150 frames đầu phục vụ benchmark nhanh (dung lượng 0.18 MB). | **TỒN TẠI** |
| `stubs/camera_motion.pkl` | File pickle cache lưu trữ ma trận chuyển động camera tích lũy và phân đoạn cho `video/sample.mp4` (dung lượng 0.10 MB). | **TỒN TẠI** |
| `stubs/camera_motion_bench.pkl` | File pickle cache chuyển động camera 150 frames đầu (dung lượng 0.02 MB). | **TỒN TẠI** |
| `stubs/camera_movement.pkl` | File pickle cache lưu chuyển động camera cũ theo optical flow thô (dung lượng 0.01 MB). | **TỒN TẠI** |
| `data.yaml` | File cấu hình dataset Roboflow định nghĩa 4 class: 0: ball, 1: goalkeeper, 2: player, 3: referee. | **TỒN TẠI** |
| `Đường dẫn train/val/test trong data.yaml` | Các đường dẫn trỏ tới `/kaggle/working/.../images` khai báo trong `data.yaml`. | **KHÔNG TỒN TẠI TRÊN MÁY LOCAL** |
| `best_V26_over_6000.pt` | Trọng số model phát hiện YOLOv8x/v26s tối ưu cho bóng đá (60.6 MB), được ưu tiên nạp mặc định. | **TỒN TẠI** |
| `best_YOLOv26s_FB_OD.pt` | Trọng số model phát hiện YOLOv26s đối chứng (60.6 MB). | **TỒN TẠI** |
| `best_YOLOV11s_FB_detect.pt` | Trọng số model phát hiện YOLOv11s nhẹ (18.9 MB). | **TỒN TẠI** |
| `output/m6_benchmark_output.mp4` | Video đầu ra phân tích chính mới nhất (Full HD 1080p, 750 frames, 82.46 MB). | **TỒN TẠI** |
| `output/m6_benchmark_output_minimap.mp4` | Video sa bàn chiến thuật 2D mới nhất (720p, 750 frames, 6.37 MB, có vệt bóng). | **TỒN TẠI** |
| `analysis/color_assignement.ipynb` | Jupyter notebook nghiên cứu thử nghiệm phân cụm màu áo cầu thủ. | **TỒN TẠI** |
| `extract_frame.py` | Script trích xuất frame chất lượng cao phục vụ gắn nhãn. | **TỒN TẠI** |
| `high_frame.py` | Script tải và chuyển mã video YouTube. | **TỒN TẠI** |
| `test.py` | Script debug và trực quan hóa detection của model trên ảnh tĩnh. | **TỒN TẠI** |
| `list_comand.txt` | Tệp văn bản ghi lại các lệnh CLI thực thi tiêu biểu của dự án. | **TỒN TẠI** |

---

### MỤC 10. Phát hiện bất thường trong code (CHỈ GHI SỰ THẬT + BẰNG CHỨNG)

Dưới đây là 7 phát hiện kỹ thuật thực tế được trích xuất trực tiếp từ mã nguồn, có ảnh hưởng trực tiếp đến chất lượng theo dõi và tính nhất quán dữ liệu:

#### Điểm bất thường 1: Lọc cắt cứng confidence người trước khi vào Facade làm triệt tiêu khả năng phục hồi detection yếu ở Stage 2 Re-ID

- **Mô tả sự việc**: Trong `tracker/tracker.py`, trước khi gọi `reid_tracker.update`, code áp dụng mask boolean `is_person` lọc cứng `det.confidence >= self.player_conf` (0.35) cho player. Trong khi đó, bên trong `ReIDConfig` và `TrackManager`, kiến trúc được thiết kế có **Stage 2 Low-Confidence Recovery** nhằm liên kết các detection yếu trong dải $[0.20, 0.35)$ với các active track thông qua DIoU và Kalman. Do bị mask cắt từ ngoài, Stage 2 không bao giờ nhận được detection player nào có confidence nằm trong khoảng $[0.20, 0.35)$.

- **Bằng chứng trong mã nguồn**:

```python
# tracker/tracker.py:290-297
                is_person = (
                    ((det.class_id == ids["player"]) & (det.confidence >= self.player_conf)) |
                    ((det.class_id == ids["goalkeeper"]) & (det.confidence >= self.goalkeeper_conf)) |
                    ((det.class_id == ids["referee"]) & (det.confidence >= self.referee_conf))
                )
                person_det = det[is_person]
                tracked = self.reid_tracker.update(person_det, frame=frame, camera_homography=G_t)

```

```python
# algorithmic_reid/lifecycle/track_manager.py:165-170
        high_mask = det_confs >= self.config.high_conf_thresh
        low_mask = (det_confs >= self.config.low_conf_thresh) & (~high_mask)

        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

```

- **Triệu chứng người dùng có thể liên quan**: `[ID bị tráo | đối tượng nhảy class | mất track khi cầu thủ bị che khuất hoặc hình ảnh bị nhòe mờ]` (khi cầu thủ bị che một phần khiến confidence tụt xuống 0.25, detection bị vứt bỏ hoàn toàn ngay trước cửa tracker thay vì được Stage 2 cứu vãn).

#### Điểm bất thường 2: Gating dự đoán chọn bóng trong `_select_ball_detection` tính trên pixel ảnh thô không bù chuyển động camera

- **Mô tả sự việc**: Trong hàm `_select_ball_detection`, vị trí dự đoán của bóng `predicted_center = last_center + velocity * dt` và khoảng cách `distances = np.linalg.norm(centers - predicted_center, axis=1)` được tính hoàn toàn trên tọa độ pixel ảnh thô của frame. Khi máy quay lia (pan) hoặc nghiêng (tilt) với tốc độ cao, tọa độ bóng trên ảnh bị trôi theo chuyển động camera khiến vị trí dự đoán bị lệch khỏi vị trí bóng thật trên ảnh, có nguy cơ làm bóng thật rơi ra ngoài elip gating và bị áp ngưỡng khắt khe 0.35 dẫn đến rớt detection.

- **Bằng chứng trong mã nguồn**:

```python
# tracker/tracker.py:728-749
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

```

- **Triệu chứng người dùng có thể liên quan**: `[bóng ít được phát hiện | bóng bị mất khi camera lia nhanh]`.

#### Điểm bất thường 3: Hardcode `choose_Al=2` trong `main.py` làm vô hiệu hóa việc chuyển đổi tracker từ CLI

- **Mô tả sự việc**: Trong `main.py:217`, tham số `choose_Al=2` được truyền bằng số cứng khi gọi `tracker.process_pass1(...)`. Mặc dù `main.py` có rất nhiều tham số dòng lệnh CLI (`--player-conf`, `--batch-size`, `--calib`...), nhưng lại không có đối số CLI nào như `--tracker-algo` hay `--choose-al`. Người dùng không thể thử nghiệm hoặc chuyển đổi sang ByteTrack (`choose_Al=0`) hay BoT-SORT (`choose_Al=1`) nếu không can thiệp sửa trực tiếp code trong `main.py`.

- **Bằng chứng trong mã nguồn**:

```python
# main.py:210-218
    tracks, motion = tracker.process_pass1(
        video_frames,
        camera_motion=camera_motion,
        team_assigner=team_assigner,
        read_from_stub=False,
        stub_path="stubs/track_stubs_reid.pkl",
        camera_stub_path="stubs/camera_motion.pkl",
        choose_Al=2,
    )
```

- **Triệu chứng người dùng có thể liên quan**: `[khác: không thể chuyển đổi bộ theo dõi người qua tham số dòng lệnh]`.

#### Điểm bất thường 4: Bỏ qua hoàn toàn các frame bóng nội suy khi vẽ annotation lên video chính

- **Mô tả sự việc**: Trong hàm `draw_tracked_frames` ở Pass 2, code có câu lệnh `if ball.get('is_interpolated', False): continue`. Điều này có nghĩa là toàn bộ các frame bóng được thuật toán `BallTrajectoryStabilizer` nội suy mượt mà trong các khoảng mất dấu ngắn (1-4 frames) **HOÀN TOÀN KHÔNG ĐƯỢC VẼ** lên video phân tích chính `output.mp4`. Ngược lại, trên video Minimap chiến thuật 2D (`minimap/minimap_renderer.py:272`), bóng nội suy vẫn được vẽ bình thường.

- **Bằng chứng trong mã nguồn**:

```python
# tracker/tracker.py:633-638
            for _, ball in ball_dict.items():
                if ball.get("is_interpolated", False):
                    continue

                display_bbox = np.asarray(ball["bbox"], dtype=float).copy()

```

- **Triệu chứng người dùng có thể liên quan**: `[bóng ít được phát hiện | tam giác bóng nhấp nháy biến mất trên video chính trong khi minimap vẫn hiển thị]`.

#### Điểm bất thường 5: Gán bóng cho cầu thủ (`PlayerBallAssigner`) sử dụng ngưỡng pixel ảnh thô chịu ảnh hưởng méo phối cảnh

- **Mô tả sự việc**: Lớp `PlayerBallAssigner` thiết lập `self.max_distance = 70` pixel trên ảnh thô để xác định cầu thủ đang kiểm soát bóng. Do hiệu ứng phối cảnh xa-gần (perspective foreshortening) của máy quay sân vận động, 70 pixel ở phần sân gần khán đài chỉ tương đương khoảng 1-2 mét thực tế, trong khi 70 pixel ở phía xa bên kia sân có thể tương đương 6-9 mét trên mặt sân cỏ FIFA. Điều này khiến các cầu thủ ở xa có thể bị nhận nhầm là đang giữ bóng dù đứng cách bóng rất xa.

- **Bằng chứng trong mã nguồn**:

```python
# player_ball_assigner/player_ball_assigner.py:6-25
    def __init__(self):
        self.max_distance = 70
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_box(ball_bbox)

        min_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_distance:
                if distance < min_distance:
                    min_distance = distance
                    assigned_player = player_id
        return assigned_player
```

- **Triệu chứng người dùng có thể liên quan**: `[bóng bị xác định sai người kiểm soát / tỷ lệ kiểm soát bóng của đội bị lệch]`.

#### Điểm bất thường 6: Vận tốc trong Kalman Filter không được biến đổi theo ma trận chuyển động camera khi bù trừ CMC

- **Mô tả sự việc**: Trong hàm `apply_camera_motion_to_tracks`, khi nhận ma trận homography liên frame $G_t$, hàm chỉ biến đổi 4 đỉnh bounding box qua $G_t$ rồi gán lại `track.mean[:4] = new_z`. Bốn thành phần vận tốc trong state vector `track.mean[4:8]` ($v_x, v_y, 0, v_h$) hoàn toàn không được biến đổi theo ma trận quay/tịnh tiến của camera. Khi camera đổi hướng lia đột ngột hoặc dừng lia, vector vận tốc cũ của Kalman sẽ mang quán tính của frame trước làm dự đoán tiếp theo bị lệch phương.

- **Bằng chứng trong mã nguồn**:

```python
# algorithmic_reid/motion/motion_predictor.py:23-50
    for track in tracks:
        # Extract bbox from mean [xc, yc, a, h]
        bbox = z_to_bbox(track.mean[:4])
        # 4 corners of bbox
        corners = np.array([
            [bbox[0], bbox[1], 1.0],
            [bbox[2], bbox[1], 1.0],
            [bbox[2], bbox[3], 1.0],
            [bbox[0], bbox[3], 1.0],
        ], dtype=np.float64).T

        warped = np.dot(homography, corners)
        w_norm = warped[2, :]
        valid_mask = np.abs(w_norm) > 1e-5
        if not np.all(valid_mask):
            continue

        warped_pts = warped[:2, :] / w_norm
        new_x1 = np.min(warped_pts[0, :])
        new_x2 = np.max(warped_pts[0, :])
        new_y1 = np.min(warped_pts[1, :])
        new_y2 = np.max(warped_pts[1, :])

        new_bbox = np.array([new_x1, new_y1, new_x2, new_y2], dtype=np.float64)
        new_z = bbox_to_z(new_bbox)

        # Update mean position components
        track.mean[:4] = new_z
```

- **Triệu chứng người dùng có thể liên quan**: `[ID bị tráo khi camera đổi hướng lia đột ngột]`.

#### Điểm bất thường 7: Thẩm định Tầng 3 trong `main.py` thay đổi cấu trúc role trên dict `tracks` nhưng không đồng bộ ngược lại `Tracklet`

- **Mô tả sự việc**: Hàm `validate_roles_and_assign_goalkeepers` trong `main.py` hoán đổi phần tử giữa `tracks['referees']` và `tracks['players']` dựa trên vị trí trung bình trong vòng cấm hoặc giữa sân. Thao tác này chỉ can thiệp trên dictionary `tracks` sau khi Pass 1 đã kết thúc hoàn toàn; đối tượng `Tracklet` và `track_manager` bên trong `algorithmic_reid` không nhận được sự đồng bộ ngược này. Nếu hệ thống được mở rộng sang xử lý streaming 1-pass trực tiếp, trạng thái class bên trong tracker sẽ không khớp với trạng thái hiển thị bên ngoài.

- **Bằng chứng trong mã nguồn**:

```python
# main.py:54-80
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

```

- **Triệu chứng người dùng có thể liên quan**: `[đối tượng nhảy class giữa các giai đoạn phân tích]`.


---

### MỤC 11. Câu hỏi còn lại

Dưới đây là các điểm kỹ thuật không thể khẳng định 100% từ việc đọc mã nguồn và cần người dùng/nhóm phát triển làm rõ:

1. **Weights chính thức khi chạy thực tế**: Trong repository tồn tại 2 file weights dung lượng tương đương (60.6 MB): `best_V26_over_6000.pt` và `best_YOLOv26s_FB_OD.pt`. Mặc dù `main.py:17` ưu tiên `best_V26_over_6000.pt` nếu có, sự khác biệt về độ chuẩn xác giữa 2 phiên bản này chưa được tài liệu hóa trong repo.

2. **Ý đồ nghiệp vụ của việc bỏ qua vẽ bóng nội suy trên video chính**: Tại `tracker/tracker.py:634-635`, code bỏ qua không vẽ tam giác bóng nếu `is_interpolated == True`. Đây là chủ ý thiết kế để người dùng chỉ thấy bóng khi có detection thật từ camera, hay là đoạn code tạm còn sót lại từ đợt phát triển trước khiến video chính không hiển thị bóng mượt như minimap radar?

3. **Vị trí lưu trữ dataset huấn luyện gốc**: File `data.yaml` trỏ tới các thư mục `/kaggle/working/...` không tồn tại trên máy local. Khi cần huấn luyện lại model hoặc tinh chỉnh threshold, bộ dữ liệu ảnh gốc và file nhãn Roboflow nằm ở đâu?

4. **Cơ chế phân định quyền kiểm soát bóng trong không gian thực**: Lớp `PlayerBallAssigner` hiện dùng ngưỡng 70 pixel ảnh thô. Có dự định chuyển sang tính khoảng cách mét mặt sân (`position_transformed` giữa bóng và chân cầu thủ, ví dụ ngưỡng <= 1.5 mét) để khắc phục triệt để hiện tượng méo phối cảnh xa-gần hay không?


---

### BẢNG TỰ KIỂM (SELF-AUDIT CHECKLIST)

Bảng kiểm tra tính đầy đủ, trung thực và đối chiếu trích dẫn mã nguồn cho toàn bộ 11 mục trong báo cáo:

| Mục kiểm toán | Trạng thái | Ghi chú đối chiếu |
|---|---|---|
| **MỤC 1. Cây thư mục và vai trò từng file** | `ĐÃ ĐỦ` | 59 file Python trong repo đều được mở đọc và gán 1 dòng mô tả vai trò chính xác. |
| **MỤC 2. Luồng dữ liệu tổng thể** | `ĐÃ ĐỦ` | Trích nguyên văn thân hàm main() lines 121-308; bảng 20 bước tuần tự; phân tích chi tiết cờ choose_Al. |
| **MỤC 3. Bước phát hiện (YOLO) và hậu xử lý detection** | `ĐÃ ĐỦ` | Trích nguyên văn predict/detect_frame; bảng 12 tham số; ánh xạ class; bảng 17 ngưỡng conf; trích hàm khử đa nhãn; kết quả grep NMS/end2end. |
| **MỤC 4. Tracking người** | `ĐÃ ĐỦ` | Trích nguyên văn khởi tạo 3 tracker; trích input prep & update; trích output & Tầng 3; trích nguyên văn toàn bộ 7 hàm cốt lõi ReID; phân tích CMC & class. |
| **MỤC 5. Tracking bóng** | `ĐÃ ĐỦ` | Trích nguyên văn _select_ball_detection; bảng hàm và hằng số ball_tracker.py; trích nguyên văn 6 khối process() và 4 helper; thứ tự xử lý; 3 hệ tọa độ; ball control. |
| **MỤC 6. Hợp đồng dữ liệu (data contract)** | `ĐÃ ĐỦ` | Bảng 21 trường của player; bảng trường referee & ball; bảng các trường mà 8 module downstream đọc. |
| **MỤC 7. Tham số tổng hợp** | `ĐÃ ĐỦ` | Bảng duy nhất gom toàn bộ 38 siêu tham số detector, NMS, tracker, reid, ball, camera. |
| **MỤC 8. Kiểm thử hiện có** | `ĐÃ ĐỦ` | Bảng 12 file test/benchmark; phân tích synthetic vs real; xác nhận rõ ràng KHÔNG CÓ ground truth chuẩn MOTA/IDF1 trên video thật. |
| **MỤC 9. Dữ liệu và công cụ sẵn có phục vụ đánh giá** | `ĐÃ ĐỦ` | Bảng 25 tài sản: video (kèm độ phân giải/fps), calib, stubs, yaml, weights, output. |
| **MỤC 10. Phát hiện bất thường trong code** | `ĐÃ ĐỦ` | 7 điểm bất thường có bằng chứng file:dòng, trích code nguyên văn và gắn nhãn triệu chứng người dùng cụ thể. |
| **MỤC 11. Câu hỏi còn lại** | `ĐÃ ĐỦ` | 4 câu hỏi kỹ thuật cụ thể cần người dùng làm rõ. |

#### Danh sách toàn bộ các file mã nguồn đã thực sự mở đọc trong quá trình kiểm toán:

1. [`main.py`](main.py)
2. [`tracker/tracker.py`](tracker/tracker.py)
3. [`tracker/ball_tracker.py`](tracker/ball_tracker.py)
4. [`tracker/__init__.py`](tracker/__init__.py)
5. [`algorithmic_reid/config.py`](algorithmic_reid/config.py)
6. [`algorithmic_reid/reid_tracker_facade.py`](algorithmic_reid/reid_tracker_facade.py)
7. [`algorithmic_reid/__init__.py`](algorithmic_reid/__init__.py)
8. [`algorithmic_reid/appearance/color_descriptor.py`](algorithmic_reid/appearance/color_descriptor.py)
9. [`algorithmic_reid/appearance/temporal_aggregator.py`](algorithmic_reid/appearance/temporal_aggregator.py)
10. [`algorithmic_reid/appearance/__init__.py`](algorithmic_reid/appearance/__init__.py)
11. [`algorithmic_reid/association/cost_matrix.py`](algorithmic_reid/association/cost_matrix.py)
12. [`algorithmic_reid/association/gating.py`](algorithmic_reid/association/gating.py)
13. [`algorithmic_reid/association/hungarian.py`](algorithmic_reid/association/hungarian.py)
14. [`algorithmic_reid/association/__init__.py`](algorithmic_reid/association/__init__.py)
15. [`algorithmic_reid/lifecycle/tracklet.py`](algorithmic_reid/lifecycle/tracklet.py)
16. [`algorithmic_reid/lifecycle/track_manager.py`](algorithmic_reid/lifecycle/track_manager.py)
17. [`algorithmic_reid/lifecycle/stitcher.py`](algorithmic_reid/lifecycle/stitcher.py)
18. [`algorithmic_reid/lifecycle/__init__.py`](algorithmic_reid/lifecycle/__init__.py)
19. [`algorithmic_reid/motion/kalman_filter.py`](algorithmic_reid/motion/kalman_filter.py)
20. [`algorithmic_reid/motion/mahalanobis.py`](algorithmic_reid/motion/mahalanobis.py)
21. [`algorithmic_reid/motion/motion_predictor.py`](algorithmic_reid/motion/motion_predictor.py)
22. [`algorithmic_reid/motion/__init__.py`](algorithmic_reid/motion/__init__.py)
23. [`camera_motion.py`](camera_motion.py)
24. [`camera_movement_estimate/camera_movement_estimate.py`](camera_movement_estimate/camera_movement_estimate.py)
25. [`camera_movement_estimate/__init__.py`](camera_movement_estimate/__init__.py)
26. [`pitch_homo.py`](pitch_homo.py)
27. [`pitch_homo_v2.py`](pitch_homo_v2.py)
28. [`central_view_geometry.py`](central_view_geometry.py)
29. [`minimap/minimap_renderer.py`](minimap/minimap_renderer.py)
30. [`minimap/__init__.py`](minimap/__init__.py)
31. [`player_ball_assigner/player_ball_assigner.py`](player_ball_assigner/player_ball_assigner.py)
32. [`player_ball_assigner/__init__.py`](player_ball_assigner/__init__.py)
33. [`speed_distance/speed_distance_calc.py`](speed_distance/speed_distance_calc.py)
34. [`speed_distance/__init__.py`](speed_distance/__init__.py)
35. [`team_assignment/team_assigner.py`](team_assignment/team_assigner.py)
36. [`team_assignment/__init__.py`](team_assignment/__init__.py)
37. [`utils/bbox_utils.py`](utils/bbox_utils.py)
38. [`utils/video_utils.py`](utils/video_utils.py)
39. [`utils/__init__.py`](utils/__init__.py)
40. [`view_transformer/view_transformer.py`](view_transformer/view_transformer.py)
41. [`view_transformer/__init__.py`](view_transformer/__init__.py)
42. [`detect/yolo_detect_test.py`](detect/yolo_detect_test.py)
43. [`checkupdate.py`](checkupdate.py)
44. [`extract_frame.py`](extract_frame.py)
45. [`high_frame.py`](high_frame.py)
46. [`test.py`](test.py)
47. [`test_reid_tracking_suite.py`](test_reid_tracking_suite.py)
48. [`test_reid_adversarial_stress.py`](test_reid_adversarial_stress.py)
49. [`test_challenger_p0_stress.py`](test_challenger_p0_stress.py)
50. [`test_ball_trajectory_suite.py`](test_ball_trajectory_suite.py)
51. [`test_challenger_p1_stress.py`](test_challenger_p1_stress.py)
52. [`test_stage2_suite.py`](test_stage2_suite.py)
53. [`test_stage3_suite.py`](test_stage3_suite.py)
54. [`test_stage4_suite.py`](test_stage4_suite.py)
55. [`verify_central_view_real_frame.py`](verify_central_view_real_frame.py)
56. [`benchmark_pipeline.py`](benchmark_pipeline.py)
57. [`benchmark_reid_tracking.py`](benchmark_reid_tracking.py)
58. [`run_benchmark_m6.py`](run_benchmark_m6.py)
59. [`evaluate_video_stage3.py`](evaluate_video_stage3.py)
60. [`data.yaml`](data.yaml)
61. [`list_comand.txt`](list_comand.txt)