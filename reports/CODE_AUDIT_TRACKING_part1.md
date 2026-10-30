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

