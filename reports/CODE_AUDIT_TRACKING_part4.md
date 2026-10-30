# BÁO CÁO KIỂM TOÁN MÃ NGUỒN — PHẦN 4: KIỂM THỬ, BẤT THƯỜNG & BẢNG TỰ KIỂM (MỤC 8–11)
**Dự án**: Football CV Analysis | [Quay lại Mục lục Phần 1](CODE_AUDIT_TRACKING_part1.md)

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