# PROJECT ARCHITECTURE REPORT: CV FOOTBALL ANALYSIS
**Hệ thống Phân tích Video Bóng đá Tự động bằng Thị giác Máy tính (Computer Vision Football Analytics)**

> **Loại tài liệu**: Reverse-Engineering Architecture & Technical Specification Report  
> **Phiên bản codebase**: Milestone P1 / Stage 4 Integration (master)  
> **Thời điểm thẩm định**: 11/2026  
> **Công cụ hỗ trợ phân tích**: Graphify Knowledge Graph & `codebase-memory-mcp`  

---

## 1. Executive Summary

Dự án **CV Football Analysis** (`CV_Football_Analys`) là một hệ thống thị giác máy tính toàn diện (End-to-End AI/CV Pipeline) chuyên dụng cho việc tự động theo dõi, tái định danh, hiệu chuẩn hình học và phân tích chiến thuật trong video trận đấu bóng đá.

### Điểm nhấn kiến trúc chính
1. **Kiến trúc Two-Pass Streaming tiết kiệm bộ nhớ**: Pass 1 thực hiện inference YOLO, bóc tách chuyển động camera và tracking đối tượng theo cơ chế streaming 1 frame/lần, duy trì RAM dưới 300MB ngay cả trên video dài độ phân giải cao. Toàn bộ tính toán toán học không gian sân diễn ra trên metadata trong RAM (< 1s), trước khi Pass 2 tiến hành streaming render video phân tích và video sa bàn chiến thuật (2D Minimap).
2. **Hệ thống theo dõi & phân định vai trò 3 Tầng (Three-Tier Role Disambiguation)**:
   - **Tầng 1 (Detection)**: Lọc độc lập 4 lớp với ngưỡng confidence riêng và khử trùng lặp giao cắt (Cross-Class Suppression: Referee > Goalkeeper > Player).
   - **Tầng 2 (Tracking / Temporal Voting)**: Bỏ phiếu chuỗi thời gian khóa cứng vai trò hoặc Hard Species Isolation trong ma trận chi phí gán Re-ID.
   - **Tầng 3 (Spatial Pitch Analysis)**: Dùng tọa độ hình học thực tế trên sân cỏ FIFA để phân định thủ môn (nằm trong khu vực 16m50 của hai đầu sân) và trọng tài (phân bố di chuyển tự do khắp sân nhưng không thuộc đội nào).
3. **Bộ theo dõi Re-ID phi học sâu (Modular Algorithmic Re-ID Framework)**:
   - Thay thế các mạng trích xuất đặc trưng Re-ID nặng nề bằng thuật toán hình thức 4 tầng liên kết: Damped Kalman Filter (với hệ số suy giảm vận tốc $\gamma = 0.90$ khi mất dấu), ma trận khoảng cách kết hợp Distance-IoU (DIoU), Mahalanobis, tương đồng cosine hướng di chuyển và khoảng cách Bhattacharyya trên biểu đồ màu HSV đa vùng (Multi-zone Torso/Legs).
   - Cơ chế bảo vệ mẫu màu EMA có rào chắn đóng băng (Freeze Guard) khi cầu thủ bị che khuất ($IoU > 0.15$) hoặc độ tin cậy thấp ($conf < 0.50$).
4. **Ổn định quỹ đạo bóng & Nội suy thích nghi 3 Tầng (Ball Trajectory Stabilizer)**:
   - Bù trừ chuyển động camera đưa bóng về hệ quy chiếu chuẩn tĩnh.
   - Lọc nhiễu vệt cỏ/cầu thủ bằng thuật toán nhìn trước đa khung hình (Lookahead Chord Burst Gating).
   - Nội suy đơn điệu PCHIP cho khoảng mất dấu ngắn ($1-4$ frame), kiểm tra động học cho khoảng vừa ($5-10$ frame) và dừng hoàn toàn (Strict NO-OP) khi mất dấu dài ($> 10$ frame) để chống bóng ma (ghost ball).
5. **Hiệu chuẩn sân & Hình học Cung tròn Central View**:
   - Sử dụng mô hình hình học FIFA chuẩn ($105\text{m} \times 68\text{m}$) kết hợp chuẩn hóa Hartley giải phương trình Homography trực tiếp.
   - Trang bị bộ giải suy biến góc quay trung tâm (`CentralViewHomographySolver`): khai thác ma trận Conic của vòng tròn giữa sân và đường giữa sân để khôi phục Homography ngay cả khi các điểm mốc quan sát bị thẳng hàng (collinear).
6. **Bộ chọn bóng Quy hoạch động Viterbi & Đột phá Phân giải cao (`ViterbiBallSelector` & `--ball-imgsz`)**:
   - Giải quyết bài toán chọn bóng qua chuỗi thời gian toàn cục (Global Trajectory Optimization) thay cho cơ chế Greedy từng frame. Khắc phục triệt để hiện tượng mất bóng khi camera lia nhanh hoặc nhảy nhầm vào giày/tất trắng cầu thủ bằng quy hoạch động trong hệ quy chiếu camera chuẩn tĩnh (`to_reference`).
   - Hàm mục tiêu Emission tích hợp chấm điểm bối cảnh không gian: thưởng chân cầu thủ (Foot Bonus -0.5), phạt thân trên (Upper Penalty +0.8), phạt tĩnh (Static Penalty +1.5), và phạt xuất hiện đơn độc (Isolated Penalty +2.0).
   - Hàm chuyển trạng thái Transition tích hợp rào chắn động học: phạt bậc hai khi vận tốc ngầm định vượt $v_{\text{cruise}} = 25\text{ px/f}$ và rào chắn bước nhảy xa theo gap $R(g) = v_{\text{cruise}} \cdot g + C_{\text{margin}}$ ngăn chặn tuyệt đối hiện tượng bóng "dịch chuyển tức thời" xuyên sân khi mất dấu dài.
   - Tách biệt độ phân giải detector (`--ball-imgsz 1920`) và cơ chế 2-Stage Temporal ROI Re-detection, nâng trần tỷ lệ tìm thấy ứng viên bóng (Candidate Recall) từ 61.0% lên 91.3% mà vẫn duy trì mức tiêu thụ VRAM chỉ 1.16 GB trên GPU 4GB.

---

## 2. Project Purpose

### 2.1 Bài toán nghiệp vụ
Phân tích dữ liệu video bóng đá từ góc quay truyền hình (Broadcast Camera) gặp các thách thức cực lớn:
- Máy quay liên tục lia (pan), nghiêng (tilt) và thu phóng (zoom).
- Các cầu thủ cùng đội mặc áo giống hệt nhau, thường xuyên va chạm, che khuất lẫn nhau (severe occlusion).
- Quả bóng có kích thước rất nhỏ ($< 15$ pixel), tốc độ bay nhanh ($> 100\text{ km/h}$), thường xuyên bị nhòe mờ (motion blur) hoặc biến mất tạm thời.
- Hiệu ứng phối cảnh xạ ảnh (perspective distortion) khiến pixel trên ảnh không thể hiện đúng khoảng cách và vận tốc thực tế trên sân cỏ.

### 2.2 Đầu vào (Input)
- **Video trận đấu**: Video định dạng `.mp4`, `.avi` (ví dụ `video/sample.mp4`), tốc độ khung hình thường là 25 hoặc 30 FPS.
- **Tập điểm hiệu chuẩn sân (Calibration Points)**: File JSON tọa độ các mốc sân trên một hoặc nhiều frame mẫu (ví dụ `calib/points.json`).
- **Trọng số mô hình phát hiện**: Checkpoint PyTorch YOLO (ví dụ `best_YOLOV11s_FB_detect.pt`, `best_V26_over_6000.pt`).

### 2.3 Đầu ra (Output)
- **Video chính (`output/output.mp4`)**: Video gốc được vẽ bounding box, ID ổn định, màu áo đội, vệt di chuyển camera, tam giác định vị bóng, bảng điều khiển tốc độ di chuyển (km/h) và khoảng cách chạy (m) của từng cầu thủ, thanh tỷ lệ kiểm soát bóng (Ball Possession %).
- **Video Sa bàn chiến thuật 2D Minimap (`output/output_minimap.mp4`)**: Radar 2D độ phân giải $1280 \times 720$ tái hiện toàn bộ vị trí các cầu thủ theo màu áo, thủ môn viền vàng, trọng tài chấm xanh lá neon, và quỹ đạo bay của quả bóng trên sơ đồ sân cỏ FIFA chuẩn.
- **Dữ liệu phân tích lưu trữ (Pickle Stubs)**: `stubs/track_stubs_reid.pkl`, `stubs/camera_motion.pkl` phục vụ việc tái sử dụng kết quả tracking mà không cần chạy lại inference AI.

---

## 3. Technology Stack

| Thành phần | Công nghệ / Thư viện | Phiên bản / Chi tiết cấu hình | Vai trò kỹ thuật |
| :--- | :--- | :--- | :--- |
| **Ngôn ngữ chính** | Python | 3.10 - 3.12 | Toàn bộ codebase được phát triển bằng Python chuẩn |
| **Deep Learning Inference** | Ultralytics YOLO | YOLOv11s / YOLOv8 / YOLOv26 | Phát hiện 4 lớp: `ball`, `goalkeeper`, `player`, `referee` |
| **Phần cứng AI** | PyTorch / CUDA | CUDA GPU (Fallback an toàn sang CPU) | Tăng tốc độ suy luận batch (`imgsz=1536`, `batch_size=8`) |
| **Xử lý ảnh & Video** | OpenCV (`cv2`) | `>= 4.8.0` | Đọc/ghi video, xử lý màu HSV, tính Optical Flow, biến đổi phối cảnh |
| **Tracking cơ sở** | Supervision (`sv`) | ByteTrack / BoT-SORT | Cung cấp tracker baseline (chế độ `choose_Al=0` và `choose_Al=1`) |
| **Toán học & Giải tích** | NumPy & SciPy | `numpy`, `scipy.optimize`, `scipy.interpolate` | Đại số tuyến tính, giải bài toán phân công Hungarian (`linear_sum_assignment`), nội suy đơn điệu PCHIP |
| **Học máy phân cụm** | Scikit-learn | `sklearn.cluster.KMeans` | Phân cụm 2 màu áo đội bóng trong không gian màu HSV |
| **Quy hoạch động & Đồ thị** | Viterbi Dynamic Programming | Tự cài đặt (`tracker/viterbi_ball.py`) | Tối ưu hóa chuỗi quỹ đạo bóng toàn cục, chấm điểm bối cảnh và lọc nhiễu |
| **Đánh giá & Thống kê** | Bootstrap Resampling | `scripts/evaluate.py` | Đo lường Recall, Precision, FPR kèm khoảng tin cậy 95% CI trên Ground Truth |
| **Định dạng cấu hình** | YAML / JSON | PyYAML, Standard `json` | Định nghĩa nhãn dataset (`data.yaml`), lưu tọa độ mốc sân (`calib/points.json`) |

---

## 4. Repository Structure

Cấu trúc thư mục mã nguồn thực tế và phân loại vai trò:

```text
CV_Football_Analys/
├── main.py                             # [Core] Entry point chính của toàn bộ pipeline Two-Pass
├── camera_motion.py                    # [Core] Hệ thống ước lượng chuyển động camera (Affine/Homography & RANSAC)
├── central_view_geometry.py            # [Core] Bộ giải hình học Conic cho vòng tròn giữa sân góc Central View
├── pitch_homo.py                       # [Core] Hệ thống hiệu chuẩn sân FIFA và ánh xạ tọa độ mét (PitchMapper)
├── pitch_homo_v2.py                    # [Supporting] Công cụ UI tương tác click mốc sân và đo sai số hold-out
│
├── tracker/                            # [Core] Package quản lý tracking và điều phối
│   ├── __init__.py                     # Export Tracker, BallTrajectoryStabilizer, ViterbiBallSelector
│   ├── tracker.py                      # [Core] Lớp Tracker trung tâm, quản lý YOLO, Pass 1 streaming, Pass 2 drawing
│   ├── ball_tracker.py                 # [Core] BallTrajectoryStabilizer: lọc ngoại lai & nội suy PCHIP 3 tầng
│   └── viterbi_ball.py                 # [Core] ViterbiBallSelector: Quy hoạch động Viterbi toàn cục & chấm điểm bối cảnh
│
├── algorithmic_reid/                   # [Core] Framework Re-ID không dùng Deep Learning (Modular Architecture)
│   ├── __init__.py                     # Package entry point
│   ├── config.py                       # [Configuration] Dataclass ReIDConfig lưu toàn bộ siêu tham số
│   ├── reid_tracker_facade.py          # [Core] Facade điều phối và TrackedDetectionsContainer (Dual-Interface)
│   ├── appearance/                     # Module trích xuất đặc trưng ngoại hình
│   │   ├── color_descriptor.py         # Biểu đồ màu HSV 2 vùng (Upper/Lower) & khoảng cách Bhattacharyya
│   │   └── temporal_aggregator.py      # Bộ tích lũy mẫu ngoại hình EMA có Freeze Guard chống che khuất
│   ├── association/                    # Module ma trận chi phí và liên kết cặp
│   │   ├── cost_matrix.py              # Tính ma trận chi phí kết hợp DIoU + Mahalanobis + Appearance + Heading
│   │   ├── gating.py                   # Rào chắn gating cứng (Hard Species Isolation, Mahalanobis threshold)
│   │   └── hungarian.py                # Thuật toán Hungarian có ngưỡng cắt gating
│   ├── lifecycle/                      # Module quản lý vòng đời tracklet
│   │   ├── tracklet.py                 # Trạng thái Tracklet (TENTATIVE, CONFIRMED, LOST, DELETED)
│   │   ├── track_manager.py            # Quản lý danh sách track, bộ cấp phát ID, liên kết 4 giai đoạn
│   │   └── stitcher.py                 # Khâu nối các tracklet bị đứt đoạn sau thời gian mất dấu
│   └── motion/                         # Module dự đoán chuyển động
│       ├── kalman_filter.py            # DampedKalmanFilter: 8 trạng thái, suy giảm vận tốc gamma=0.90
│       ├── mahalanobis.py              # Tính khoảng cách Mahalanobis trong không gian đo [xc, yc, a, h]
│       └── motion_predictor.py         # Bù trừ chuyển động camera CMC vào trạng thái Kalman
│
├── team_assignment/                    # [Important] Phân cụm đội bóng
│   ├── __init__.py
│   └── team_assigner.py                # K-Means clustering trích xuất màu áo cầu thủ trong không gian HSV
│
├── speed_distance/                     # [Important] Tính toán vật lý
│   ├── __init__.py
│   └── speed_distance_calc.py          # Tính vận tốc (km/h) và quãng đường (m) từ tọa độ sân
│
├── player_ball_assigner/               # [Important] Gán quyền kiểm soát bóng
│   ├── __init__.py
│   └── player_ball_assigner.py         # Xác định cầu thủ chạm bóng dựa trên khoảng cách Euclidean tới chân
│
├── minimap/                            # [Important] Sa bàn chiến thuật 2D
│   ├── __init__.py
│   └── minimap_renderer.py             # Sinh video Radar 2D độc lập (1280x720) mô phỏng trận đấu
│
├── utils/                              # [Supporting] Tiện ích dùng chung
│   ├── __init__.py
│   ├── bbox_utils.py                   # Tính tâm box, vị trí chân, khoảng cách Euclidean pixel
│   └── video_utils.py                  # Đọc video generator tiết kiệm RAM và ghi video ra đĩa
│
├── view_transformer/                   # [Legacy] Prototype ánh xạ 4 điểm ban đầu (được thay bằng pitch_homo.py)
│   ├── __init__.py
│   └── view_transformer.py
│
├── camera_movement_estimate/           # [Legacy] Prototype Optical Flow viền ảnh (được thay bằng camera_motion.py)
│   ├── __init__.py
│   └── camera_movement_estimate.py
│
├── calib/                              # [Configuration / Artifact] Dữ liệu hiệu chuẩn
│   ├── points.json                     # Tọa độ mốc sân mặc định
│   ├── sample.json, sample3.json       # Tọa độ mốc sân thử nghiệm
│
├── data/                               # [Evaluation / Ground Truth] Dữ liệu Ground Truth chuẩn
│   ├── ball_gt_HUMAN.json              # [Strictly Read-Only] Nhãn bóng do người gán (109 frames phân tầng)
│   └── ball_gt_AUTO.json               # Nhãn sinh tự động (chỉ phục vụ smoke test, bị từ chối khi evaluate)
│
├── reports/                            # [Documentation] Báo cáo kiểm toán mã nguồn & thực nghiệm
│   ├── CODE_AUDIT_TRACKING.md          # Báo cáo tổng hợp kiểm toán tracking
│   ├── CODE_AUDIT_TRACKING_part1.md    # Phần 1: Kiến trúc tổng quan & Tầng 1, Tầng 2
│   ├── CODE_AUDIT_TRACKING_part2.md    # Phần 2: Tracking người & Algorithmic Re-ID
│   ├── CODE_AUDIT_TRACKING_part3.md    # Phần 3: Tracking bóng & Hợp đồng dữ liệu
│   ├── CODE_AUDIT_TRACKING_part4.md    # Phần 4: Danh mục test suite & 7 điểm bất thường
│   ├── BALL_DIAG.md                    # Báo cáo chẩn đoán 274 frame bóng LOST & phân loại nguyên nhân
│   ├── BALL_GT_HUMAN_EVAL.md           # Đánh giá hiệu năng bộ chọn bóng trên Ground Truth người gán
│   ├── DETECTION_BOTTLENECK.md         # Phân tích nút thắt phân giải detector & trần candidate recall
│   └── BALL_FIX_AND_RES.md             # Tổng kết giải pháp Viterbi nâng cao và tái phát hiện ROI
│
├── [Test Suites]                       # [Important] Hệ thống kiểm thử tự động
│   ├── test_reid_tracking_suite.py     # Bộ 8 kịch bản kiểm thử Re-ID chuẩn
│   ├── test_ball_trajectory_suite.py   # Bộ 10 kịch bản kiểm thử ổn định bóng & nội suy
│   ├── test_viterbi_ball.py            # Bộ 6 kịch bản kiểm thử bộ chọn bóng Viterbi, emission/reach limit
│   ├── test_stage2_suite.py            # Kiểm thử Re-ID Stage 2 & liên kết ngoại hình
│   ├── test_stage3_suite.py            # Kiểm thử Camera Motion & bù trừ CMC
│   ├── test_stage4_suite.py            # Kiểm thử Conic Central View & tích hợp Downstream
│   ├── test_challenger_p0_stress.py    # Kiểm thử đối kháng áp lực Re-ID
│   ├── test_challenger_p1_stress.py    # Kiểm thử đối kháng áp lực bóng
│   └── test_reid_adversarial_stress.py # Kiểm thử Re-ID trong điều kiện nhiễu cực đoan
│
├── scripts/                            # [Core / Supporting] Công cụ đánh giá, gán nhãn & thực nghiệm
│   ├── evaluate.py                     # [Core Harness] Đánh giá Precision/Recall/FPR/Bootstrap CI với Human GT
│   ├── label_ball.py                   # [Tool] Gán nhãn bóng bán tự động (Zoom 3x, phím tắt nhanh)
│   ├── benchmark_pass1_imgsz.py        # [Benchmark] Đo FPS & Peak VRAM giữa 1536 vs 1920 trên GPU 4GB
│   ├── diagnose_8_false_positives.py   # [Diagnostic] Phân tích 8 frame chọn nhầm giày/tất cầu thủ
│   ├── experiment_viterbi_improvements.py # [Experiment] Đánh giá rào chắn reach limit và isolated penalty
│   ├── experiment_roi_redetect.py      # [Experiment] Tái phát hiện ROI 640x640 trên frame mất dấu
│   └── experiment_roi_viterbi_integration.py # [Experiment] Tích hợp ứng viên ROI vào đồ thị Viterbi DP
│
├── [Legacy / Benchmarks]               # [Supporting] Scripts phụ trợ khác
│   ├── benchmark_pipeline.py           # Benchmark thời gian thực thi pipeline
│   ├── benchmark_reid_tracking.py      # Benchmark độ ổn định Re-ID trên video thật
│   ├── run_benchmark_m6.py             # Script chạy benchmark tổng hợp
│   ├── evaluate_video_stage3.py        # Đánh giá độ chính xác Stage 3
│   ├── verify_central_view_real_frame.py # Xác thực thuật toán Central View trên frame thật
│   ├── high_frame.py                   # Tải và chuyển mã video từ nguồn bên ngoài
│   └── extract_frame.py                # Trích xuất frame ảnh tĩnh từ video
│
├── data.yaml                           # [Configuration] Cấu hình 4 class của mô hình phát hiện
└── README.md                           # [Documentation] Hướng dẫn sử dụng tổng quan
```

---

## 5. High-Level Architecture

Sơ đồ kiến trúc tổng thể toàn bộ hệ thống từ đầu vào đến đầu ra:

```mermaid
flowchart TD
    subgraph INPUT["1. Input Layer"]
        VID["Video File (*.mp4)"]
        CALIB["calib/points.json"]
        WEIGHTS["YOLO Weights (*.pt)"]
    end

    subgraph PASS1["2. Pass 1: Streaming Analysis (RAM < 300MB)"]
        READ_STREAM["video_utils.read_video() (Generator)"]
        YOLO_DET["Tracker.detect_frame()\n(Batch imgsz=1536 / --ball-imgsz 1920)"]
        
        subgraph FILTER_TIER1["Tier 1: Independent Class Gating"]
            SPLIT_BALL["Ball Detections (Class 0)\n(Top-K Candidates conf >= floor)"]
            SPLIT_PERSON["Person Detections (Player, GK, Ref)\n(--reid-min-conf Stage 2 Filter)"]
            NMS_SUPP["_suppress_cross_class_overlaps()"]
        end
        
        CMC["camera_motion.CameraMotion.step()"]
        
        subgraph TRACK_ENGINES["Tracking Engine Selection (choose_Al)"]
            REID["choose_Al=2: Algorithmic Re-ID (P0)\nReIDTrackerFacade.update()"]
            BYTE["choose_Al=0: ByteTrack"]
            BOT["choose_Al=1: BoT-SORT"]
        end
        
        subgraph BALL_SELECTOR_ENGINES["Ball Selector Engine (--ball-selector)"]
            BALL_GREEDY["--ball-selector greedy (Default)\nTracker._select_ball_detection()"]
            BALL_VITERBI_BUF["--ball-selector viterbi\nViterbiBallSelector.add_frame_candidates()"]
        end
        
        TEAM_KMEANS["team_assignment.TeamAssigner\n(K-Means on HSV Jersey Colors)"]
        
        RAW_TRACKS[("Raw tracks dict\n{players, referees, ball}")]
    end

    subgraph MID_MATH["3. Intermediate In-Memory Mathematics (< 1s)"]
        VITERBI_DP["tracker.viterbi_ball.ViterbiBallSelector.select_trajectory()\n(Global DP in Camera Reference Frame + Context Scoring)"]
        BALL_STABILIZE["tracker.ball_tracker.BallTrajectoryStabilizer\n(Camera Decoupling + Lookahead Chord + 3-Tier PCHIP)"]
        PITCH_FIT["pitch_homo.PitchCalibration.fit()\n(Hartley Normalization + CentralView Conic Solver)"]
        MAPPER["pitch_homo.PitchMapper.add_positions_to_tracks()\n(Transform Pixel -> Metric FIFA [X, Y])"]
        TIER3_VALIDATE["main.validate_roles_and_assign_goalkeepers()\n(Tier 3 Spatial Pitch Penalty Area Gating)"]
        SPEED_DIST["speed_distance.SpeedAndDistance_Calc\n(Metric Velocity km/h & Distance m)"]
        BALL_ASSIGN["player_ball_assigner.PlayerBallAssigner\n(Possession Assignment & Team Control %)"]
    end

    subgraph PASS2["4. Pass 2: Streaming Rendering & Visualization"]
        READ_PASS2["video_utils.read_video() (Pass 2 Stream)"]
        DRAW_MAIN["Tracker.draw_tracked_frames()\n(Boxes, IDs, Ball Triangle, Possession Bar)"]
        DRAW_CAM["CameraMotion.draw_camera_movement()"]
        DRAW_SPEED["SpeedAndDistance_Calc.draw_speed_and_distance()"]
        SAVE_MAIN["video_utils.save_video() -> output/output.mp4"]
        
        subgraph MINIMAP_GEN["Minimap Generation (Independent)"]
            MINIMAP_RENDER["minimap.MinimapRenderer.render_video()"]
            SAVE_MINIMAP["output/output_minimap.mp4"]
        end
    end

    INPUT --> PASS1
    READ_STREAM --> YOLO_DET
    YOLO_DET --> FILTER_TIER1
    FILTER_TIER1 --> CMC
    CMC --> TRACK_ENGINES
    SPLIT_BALL --> BALL_GREEDY
    SPLIT_BALL --> BALL_VITERBI_BUF
    TRACK_ENGINES --> TEAM_KMEANS
    TEAM_KMEANS --> RAW_TRACKS
    BALL_GREEDY --> RAW_TRACKS
    BALL_VITERBI_BUF -.-> VITERBI_DP
    
    RAW_TRACKS --> VITERBI_DP
    VITERBI_DP --> BALL_STABILIZE
    RAW_TRACKS --> MID_MATH
    MID_MATH --> PASS2
    PASS2 --> SAVE_MAIN
    PASS2 --> SAVE_MINIMAP
```

---

## 6. Module Inventory

| Module | Đường dẫn thực tế | Vai trò kỹ thuật chính | Mức độ quan trọng | Dependencies trực tiếp | Dependents (Module gọi) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pipeline Runner** | `main.py` | Entry point chính, điều phối luồng 2 pass và thẩm định Tầng 3 | **Core** | `tracker`, `camera_motion`, `pitch_homo`, `team_assignment`, `speed_distance`, `player_ball_assigner`, `minimap`, `utils` | CLI / Người dùng |
| **Tracking Coordinator** | `tracker/tracker.py` | Lớp `Tracker`: chạy YOLO, điều phối tracking, Pass 1 và Pass 2 | **Core** | `ultralytics`, `supervision`, `algorithmic_reid`, `tracker/ball_tracker.py`, `utils/bbox_utils.py` | `main.py` |
| **Ball Stabilizer** | `tracker/ball_tracker.py` | Lớp `BallTrajectoryStabilizer`: ổn định và nội suy quỹ đạo bóng 3 tầng | **Core** | `scipy.interpolate.PchipInterpolator`, `numpy`, `camera_motion` | `tracker/tracker.py` |
| **Viterbi Ball Selector** | `tracker/viterbi_ball.py` | Lớp `ViterbiBallSelector`: quy hoạch động toàn cục tìm chuỗi bóng tối ưu, chấm điểm bối cảnh chân/thân, rào chắn reach limit | **Core** | `numpy`, `camera_motion` | `tracker/tracker.py`, `scripts/evaluate.py` |
| **Evaluation Harness** | `scripts/evaluate.py` | Đo lường Recall, Precision, FPR, F1 kèm Bootstrap CI 95% trên Human GT | **Supporting** | `numpy`, `tracker/viterbi_ball.py` | CLI / Nghiên cứu & Benchmark |
| **Re-ID Facade** | `algorithmic_reid/reid_tracker_facade.py` | Giao diện Facade bọc bộ theo dõi Re-ID và `TrackedDetectionsContainer` | **Core** | `algorithmic_reid/lifecycle/track_manager.py`, `algorithmic_reid/appearance/color_descriptor.py` | `tracker/tracker.py` |
| **Re-ID Lifecycle** | `algorithmic_reid/lifecycle/` | Quản lý vòng đời tracklet, liên kết 4 tầng, khâu nối tracklet | **Core** | `algorithmic_reid/association/`, `algorithmic_reid/motion/` | `reid_tracker_facade.py` |
| **Re-ID Association** | `algorithmic_reid/association/` | Ma trận chi phí đa nguồn (DIoU, Motion, Appearance, Heading) và gating cứng | **Core** | `scipy.optimize.linear_sum_assignment`, `numpy` | `track_manager.py` |
| **Re-ID Motion** | `algorithmic_reid/motion/` | Damped Kalman Filter, khoảng cách Mahalanobis và bù trừ CMC | **Core** | `numpy`, `scipy.spatial.distance` | `tracklet.py`, `cost_matrix.py` |
| **Re-ID Appearance** | `algorithmic_reid/appearance/` | Trích xuất HSV 2 vùng, khoảng cách Bhattacharyya và EMA Freeze Guard | **Core** | `cv2`, `numpy` | `reid_tracker_facade.py`, `cost_matrix.py` |
| **Camera Motion** | `camera_motion.py` | Ước lượng Homography liên frame $G_t$ và Re-localization qua Anchor | **Core** | `cv2`, `numpy`, `pickle` | `main.py`, `tracker/tracker.py`, `pitch_homo.py` |
| **Pitch Geometry** | `pitch_homo.py` | Hiệu chuẩn sân FIFA (`PitchCalibration`) và ánh xạ sang tọa độ mét (`PitchMapper`) | **Core** | `central_view_geometry.py`, `cv2`, `numpy` | `main.py` |
| **Central Conic Solver** | `central_view_geometry.py` | Giải Homography Central View bằng Conic vòng tròn giữa sân | **Core** | `cv2`, `numpy` | `pitch_homo.py` |
| **Team Assigner** | `team_assignment/team_assigner.py` | K-Means clustering phân định 2 đội bóng theo màu áo | **Important** | `sklearn.cluster.KMeans`, `cv2`, `numpy` | `main.py` |
| **Speed & Distance** | `speed_distance/speed_distance_calc.py` | Tính vận tốc (km/h) và quãng đường (m) từ tọa độ sân thực tế | **Important** | `utils/bbox_utils.py`, `cv2`, `numpy` | `main.py` |
| **Ball Assigner** | `player_ball_assigner/player_ball_assigner.py` | Gán bóng cho cầu thủ gần chân nhất và theo dõi quyền kiểm soát bóng | **Important** | `utils/bbox_utils.py` | `main.py` |
| **Minimap Renderer** | `minimap/minimap_renderer.py` | Trực quan hóa toàn bộ trận đấu trên sa bàn radar 2D | **Important** | `cv2`, `numpy` | `main.py` |
| **Bounding Box Utils** | `utils/bbox_utils.py` | Tiện ích hình học: tâm box, vị trí chân, khoảng cách Euclidean | **Supporting** | `numpy` | `tracker.py`, `player_ball_assigner.py`, `speed_distance_calc.py` |
| **Video Utils** | `utils/video_utils.py` | Đọc video dạng generator và lưu video ra file `.mp4` | **Supporting** | `cv2` | `main.py` |
| **View Transformer** | `view_transformer/` | Prototype biến đổi 4 điểm góc sân ban đầu | **Legacy** | `cv2`, `numpy` | Không còn dùng trong pipeline chính (được thay bởi `pitch_homo.py`) |
| **Camera Estimate** | `camera_movement_estimate/` | Prototype Optical Flow viền ảnh ban đầu | **Legacy** | `cv2`, `numpy` | Không còn dùng trong pipeline chính (được thay bởi `camera_motion.py`) |

---

## 7. Phân tích chi tiết từng Module cốt lõi

### 7.1 Module: Pipeline Orchestration & CLI (`main.py`)
- **Vị trí**: `main.py`
- **Vai trò**: Điểm điều phối trung tâm của toàn bộ hệ thống. Khởi tạo tất cả các lớp xử lý, điều khiển quá trình đọc streaming Pass 1, thực hiện chuyển đổi toán học trung gian trong RAM, và điều khiển streaming Pass 2 để lưu video kết quả và video minimap.
- **Lớp chính**: Không có lớp (chứa hàm `main()` và các hàm tiện ích).
- **Hàm chính**:
  - `main(...)`: Hàm điều phối chính (nhận các tham số cấu hình: video, model, confidence thresholds, batch size, calib file).
  - `validate_roles_and_assign_goalkeepers(tracks, team_assigner)`: Thẩm định Tầng 3 (Spatial Pitch Role Validation). Tính vị trí trung bình trên sân của các tracklet để chuyển đổi các cầu thủ đứng sâu trong khu vực 16m50 thành Thủ môn (`is_goalkeeper = True`) và gán màu áo thủ môn.
- **Dữ liệu vào**: Đường dẫn video (`video/sample.mp4`), trọng số mô hình (`best_YOLOV11s_FB_detect.pt`), tọa độ mốc sân (`calib/points.json`).
- **Dữ liệu ra**: Video `output/output.mp4`, `output/output_minimap.mp4`, các file stubs pickle.
- **Rủi ro kiến trúc**:
  - Tham số `choose_Al=2` đang bị hardcode tại dòng 217 khi gọi `tracker.process_pass1(...)`, khiến người dùng không thể chuyển sang ByteTrack hay BoT-SORT từ tham số dòng lệnh CLI.
  - Hàm `validate_roles_and_assign_goalkeepers` sửa đổi trực tiếp dictionary `tracks` sau khi Pass 1 hoàn tất, không đồng bộ ngược lại vào đối tượng `Tracklet` bên trong `algorithmic_reid`.

---

### 7.2 Module: Tracking Coordinator & YOLO Engine (`tracker/tracker.py`)
- **Vị trí**: `tracker/tracker.py`
- **Vai trò**: Quản lý mô hình phát hiện YOLO, thực hiện batch inference trên GPU, khử trùng lặp giao cắt lớp Tầng 1, điều phối các bộ chọn bóng (`_select_ball_detection` Greedy hoặc `ViterbiBallSelector`), quản lý cờ Re-ID recovery (`reid_min_conf`), và trực quan hóa bounding box/ID/tam giác bóng trên Pass 2 (kèm cờ `draw_interpolated_ball`).
- **Lớp chính**: `Tracker`
- **Hàm / Phương thức chính**:
  - `detect_frame(list_frame)`: Generator nhận chuỗi frame, gom thành batch (mặc định 8) và chạy `model.predict(..., imgsz=1536)`. Nếu cấu hình `ball_imgsz` (ví dụ 1920), module hỗ trợ tách biệt độ phân giải suy luận cho lớp bóng nhằm tăng diện tích nhận diện quả bóng nhỏ.
  - `process_pass1(...)`: Vòng lặp phân tích Pass 1. Gọi YOLO, trích xuất camera motion, lọc detection người theo `reid_min_conf`, gọi bộ theo dõi Re-ID, thu thập ứng viên bóng (lưu Top-K buffer cho Viterbi hoặc chọn Top-1 Greedy), phân màu áo đội qua TeamAssigner.
  - `_select_ball_detection(ball_det, frame_index, frame_shape, state)`: Thuật toán chọn quả bóng Top-1 Greedy thích nghi theo vận tốc và bán kính gating elip trong không gian ảnh thô.
  - `_suppress_cross_class_overlaps(detections, ids, iou_thresh=0.45)`: Khử trùng lặp Tầng 1 theo thứ tự ưu tiên: Trọng tài $\to$ Thủ môn $\to$ Cầu thủ.
  - `interpolate_ball_position(...)`: Khởi tạo và ủy quyền xử lý ổn định bóng cho `BallTrajectoryStabilizer`. Nếu `ball_selector == 'viterbi'`, hàm này kích hoạt `ViterbiBallSelector.select_trajectory()` trước khi đưa vào Stabilizer.
  - `draw_tracked_frames(...)`: Generator Pass 2 vẽ hộp định danh, màu áo, tam giác bóng và thanh sở hữu bóng lên từng frame video. Hỗ trợ cờ `--draw-interpolated-ball` để hiển thị bóng nội suy (dưới dạng vòng tròn phân biệt), tránh hiện tượng nhấp nháy mất dấu.
- **Chi tiết kỹ thuật**:
  - Khi `choose_Al=2`, `Tracker` khởi tạo `self.reid_tracker = ReIDTrackerFacade(config=ReIDConfig(fps=self.fps))`.
  - Tách biệt `reid_min_conf` (mặc định bằng `player_conf` = 0.35, có thể cấu hình xuống 0.20): Các detection trong dải $[reid\_min\_conf, player\_conf)$ được gắn cờ `is_reid_recovery_only = True`, cho phép đi vào Stage 2 Re-ID để cứu vãn track bị che khuất mà không được phép khởi tạo tracklet mới (Stage 4 đòi hỏi $\ge 0.45$), không tham gia K-Means phân cụm áo đội và không vẽ box rác lên video.
  - Quả bóng được hỗ trợ 2 chế độ:
    1. `greedy` (mặc định): Chọn Top-1 tức thời qua `_select_ball_detection`.
    2. `viterbi`: Pass 1 lưu Top-K ứng viên ($K=5$ hoặc $10$, $conf \ge floor$ từ $0.05$ hoặc $0.01$); Post-Pass 1 thực hiện quy hoạch động tìm chuỗi tối ưu trong hệ quy chiếu camera tĩnh.
- **Hiện trạng rủi ro kiến trúc**:
  - *Rủi ro 1 (Stage 2 Re-ID bị chặn)*: **[ĐÃ GIẢI QUYẾT]** Cung cấp cờ `--reid-min-conf` (hỗ trợ hạ ngưỡng xuống 0.20/0.15 an toàn mà không làm nhiễu TeamAssigner).
  - *Rủi ro 2 (Gating bóng trên pixel thô)*: **[ĐÃ GIẢI QUYẾT]** Cung cấp cờ `--ball-selector viterbi` thực hiện quy hoạch động trên hệ quy chiếu chuẩn tĩnh bù trừ chuyển động camera (`to_reference`).
  - *Rủi ro 4 (Bỏ qua bóng nội suy khi vẽ video)*: **[ĐÃ GIẢI QUYẾT]** Cung cấp cờ `--draw-interpolated-ball` cho phép hiển thị bóng nội suy trên video chính dưới dạng vòng tròn phân biệt.

---

### 7.3 Module: Modular Algorithmic Re-ID Framework (`algorithmic_reid/`)
- **Vị trí**: `algorithmic_reid/`
- **Vai trò**: Cung cấp thuật toán theo dõi và tái định danh cầu thủ/thủ môn/trọng tài đạt chuẩn SOTA mà không cần mạng nơ-ron Re-ID trích xuất đặc trưng nặng nề. Xử lý triệt để hiện tượng tráo ID khi cầu thủ giao cắt, bị che khuất và lia máy quay.
- **Kiến trúc phân tầng bên trong**:

```mermaid
flowchart TD
    DET["YOLO Detections (xyxy, conf, class_id)"] --> FACADE["ReIDTrackerFacade.update()"]
    FRAME["Video Frame (BGR)"] --> FACADE
    HOMO["Camera Homography G_t"] --> FACADE

    FACADE --> HIST["color_descriptor.extract_multizone_histogram()\n(HSV Upper/Lower Zones)"]
    FACADE --> TM["lifecycle.TrackManager.step()"]

    subgraph TM_INNER["TrackManager 4-Stage Association Flow"]
        KF_PRED["1. Motion Prediction:\nDampedKalmanFilter.predict() +\napply_camera_motion_to_tracks(G_t)"]
        
        STAGE1["Stage 1: High-Confidence Association (conf >= 0.40)\nCost = 0.40*DIoU + 0.20*Mahalanobis + 0.30*Bhattacharyya + 0.10*CosineHeading\nGate: Mahalanobis <= 13.28 & Appearance <= 0.55 & Hard Species Barrier\nMatching: Hungarian (max_cost <= 0.75)"]
        
        STAGE2["Stage 2: Low-Confidence Recovery (0.20 <= conf < 0.40)\nCost = DIoU only\nMatching: Hungarian (max_cost <= 0.85)"]
        
        STAGE3["Stage 3: Lost Track Reactivation (window <= 30 frames)\nCost = 0.50*DIoU + 0.50*Appearance\nGate: Appearance <= 0.55 & Species Match\nMatching: Hungarian (max_cost <= 0.70)"]
        
        STAGE4["Stage 4: Tentative Track Matching & Initialization\nNew Detections (conf >= 0.45) -> Init Tentative (n_init=2 hits to confirm)"]
        
        STITCH["TrackletStitcher._stitch_candidates()\n(Stitch fragmented tracks across 1-30 frame gaps)"]
        
        EMA_UPDATE["Appearance Update:\ntemporal_aggregator.AppearanceAggregator\n(Freeze if IoU > 0.15 or conf < 0.50, alpha=0.15)"]
    end

    TM --> KF_PRED
    KF_PRED --> STAGE1
    STAGE1 --> STAGE2
    STAGE2 --> STAGE3
    STAGE3 --> STAGE4
    STAGE4 --> STITCH
    STITCH --> EMA_UPDATE

    TM --> CONTAINER["TrackedDetectionsContainer\n(Dual Interface: dict indexing & supervision format)"]
```

- **Các lớp và hàm nòng cốt**:
  - `ReIDConfig` (`config.py`): Chứa tập siêu tham số chuẩn hóa: weights ma trận chi phí (`w_diou=0.40`, `w_mot=0.20`, `w_app=0.30`, `w_dir=0.10`), gating (`mahalanobis_gate=13.28`, `appearance_gate=0.55`), suy giảm vận tốc Kalman (`velocity_damping_gamma=0.90`).
  - `DampedKalmanFilter` (`motion/kalman_filter.py`): Mô hình trạng thái 8 chiều $[x_c, y_c, a, h, v_x, v_y, 0, v_h]^T$. Khóa cố định đạo hàm tỷ lệ khung hình $\dot{a} = 0$. Khi không có quan sát (dropout), vận tốc bị triệt tiêu dần theo hàm mũ $\gamma = 0.90$.
  - `extract_multizone_histogram` (`appearance/color_descriptor.py`): Chia bounding box làm 2 vùng: Thân trên (Upper $15\% - 55\%$) đại diện cho màu áo đấu và Thân dưới (Lower $55\% - 90\%$) đại diện cho quần đùi. Trích xuất biểu đồ màu HSV chuẩn hóa ($H:16, S:8, V:8$) và đo khoảng cách bằng Bhattacharyya distance.
  - `AppearanceAggregator` (`appearance/temporal_aggregator.py`): Cập nhật mẫu đặc trưng theo cấp số nhân (EMA, $\alpha=0.15$). Tích hợp rào chắn an toàn (Freeze Guard): đóng băng hoàn toàn việc cập nhật màu khi phát hiện độ che khuất $IoU > 0.15$ với đối tượng khác hoặc confidence của detection $< 0.50$.
  - `apply_gating_barriers` (`association/gating.py`): **Hard Species Isolation Barrier** — gán chi phí cực lớn ($10^5$) cho mọi cặp ứng viên khác loài (Player vs Goalkeeper vs Referee), ngăn chặn tuyệt đối hiện tượng ô nhiễm danh tính giữa các vai trò.
  - `TrackletStitcher` (`lifecycle/stitcher.py`): Nối liền các đoạn tracklet bị phân mảnh do mất dấu ngắn (1 đến 30 frame) khi thỏa mãn cùng loài, động học khả thi và khoảng cách ngoại hình $\le 0.40$.
- **Rủi ro kiến trúc**:
  - Trong `motion/motion_predictor.py::apply_camera_motion_to_tracks`, khi camera lia nhanh, hàm chỉ biến đổi 4 góc của bounding box qua ma trận homography $G_t$ rồi gán lại tọa độ $[x_c, y_c, a, h]$, nhưng để nguyên 4 thành phần vận tốc $[v_x, v_y, 0, v_h]$ không xoay/biến đổi theo camera, gây sai lệch quán tính khi camera đổi hướng lia đột ngột.

---

### 7.4 Module: Ball Trajectory Stabilizer & Adaptive Interpolation (`tracker/ball_tracker.py`)
- **Vị trí**: `tracker/ball_tracker.py`
- **Vai trò**: Xử lý toàn bộ chuỗi vị trí quả bóng từ Pass 1: bù trừ chuyển động camera, loại bỏ ngoại lai động học, lọc chùm nhiễu đa khung hình, thực hiện nội suy thích nghi 3 tầng và làm mượt hồi quy ballistic.
- **Lớp chính**: `BallTrajectoryStabilizer`
- **Quy trình 6 bước trong `process()`**:
  1. **Trích xuất thông số thô**: Thu thập tọa độ tâm, kích thước $W \times H$, độ tin cậy của bóng trên từng frame.
  2. **Bù trừ chuyển động camera (`_to_reference`)**: Dùng `camera_motion.to_reference()` đưa toàn bộ tâm bóng về hệ quy chiếu chuẩn tĩnh.
  3. **Lọc ngoại lai động học**:
     - *Boundary Gating*: Kiểm tra 2 frame đầu và 2 frame cuối, loại bỏ nếu tốc độ $> 50\text{ px/f}$ và không gần chân cầu thủ.
     - *Lookahead Chord Burst Gating*: Khi phát hiện bước nhảy tốc độ $> 50\text{ px/f}$, tìm kiếm tiếp tối đa $k_{\max} = 4$ frame phát hiện tiếp theo. Nếu cung dây kết nối lại với vận tốc hợp lệ mà các frame ở giữa không giải thích được bằng cú sút gần chân cầu thủ ([`_is_near_player`](file:///e:/CV_Football_Analys/tracker/ball_tracker.py#L458)), loại bỏ toàn bộ chùm frame nhiễu ở giữa.
  4. **Nội suy thích nghi 3 tầng (3-Tier Dropout Handling)**:
     - **Tier 1 ($1 \le gap \le 4$)**: Nội suy đa thức PCHIP đơn điệu (`PchipInterpolator`) trên tọa độ X và Y độc lập, mượt mà và không overshoot.
     - **Tier 2 ($5 \le gap \le 10$)**: Kiểm tra tính nhất quán động học: tốc độ trung bình $\le 50\text{ px/f}$ và góc lệch vector vận tốc vào/ra $\le 60^\circ$. Nếu thỏa mãn thì nội suy PCHIP; nếu đổi hướng mạnh $\to$ Từ chối nội suy.
     - **Tier 3 ($gap > 10$)**: Strict NO-OP (0 frame ảo), giữ trạng thái `LOST`.
  5. **Làm mượt hồi quy Ballistic**: Cửa sổ trượt $W=4$, hồi quy đa thức bậc 2 ở vùng trong và bậc 1 ở cận biên. Chặn cứng độ lệch tối đa so với detection gốc $\le 2.5\text{ px}$ (`max_smoothing_deviation`).
  6. **Chuyển đổi ngược về ảnh (`_to_frame`)**: Chuyển tọa độ từ hệ tham chiếu về tọa độ pixel frame và gán cờ `is_interpolated = True`, `confidence = 0.30` cho các frame nội suy.

---

### 7.5 Module: Camera Motion Estimation & Anchor Re-localization (`camera_motion.py`)
- **Vị trí**: `camera_motion.py`
- **Vai trò**: Tính toán ma trận biến đổi Affine/Homography liên frame $G_t$ giữa frame $t-1$ và frame $t$, khử chuyển động của góc máy quay. Quản lý hệ thống neo toàn cục (`GlobalCalibrationAnchor`) để phục hồi lại ma trận hiệu chuẩn sân ngay cả sau khi máy quay bị cắt cảnh (camera cut).
- **Lớp chính**:
  - `AnchorTrackingConfig`: Dataclass chứa ngưỡng số lượng đặc trưng tối thiểu, RMS reprojection error tối đa, độ phủ Convex Hull.
  - `CameraMotionResult`: Đối tượng chứa danh sách ma trận $M_t$, cờ segment, phân loại trạng thái chuyển động (PAN, TILT, ZOOM, STATIC).
  - `PitchRelocalizer`: Bộ tái định vị góc quay sân bóng thông qua đặc trưng ORB/SIFT kết hợp so khớp RANSAC với các anchor đã lưu.
  - `CameraMotion`: Lớp điều phối trung tâm.
- **Phương thức chính**:
  - `step(frame, prev_boxes)`: Nhận frame hiện tại và danh sách bounding box của người/bóng ở frame trước để **tạo mask che khuất** (tránh bắt nhầm đặc trưng di động trên thân cầu thủ). Dùng Lucas-Kanade Optical Flow hoặc RANSAC Homography để tính ma trận dịch chuyển.
  - `add_global_anchor(frame_idx, frame_gray, H, seg)`: Lưu lại frame hiệu chuẩn mốc chuẩn làm neo định vị.
  - `to_reference(t, pts)` / `to_frame(t, pts_ref)`: Chuyển đổi tọa độ qua lại giữa hệ ảnh thô và hệ quy chiếu camera chuẩn.
- **Rủi ro kiến trúc**:
  - Phụ thuộc vào chất lượng mặt cỏ và vạch sơn. Khi máy quay quay cận cảnh (close-up) chỉ thấy thân cầu thủ mà không thấy vạch sân, RANSAC có thể bị phân đoạn (`seg` tăng), chuyển sang trạng thái `UNREGISTERED`.

---

### 7.6 Module: Pitch Calibration & Homography Mapping (`pitch_homo.py` & `central_view_geometry.py`)
- **Vị trí**: `pitch_homo.py` và `central_view_geometry.py`
- **Vai trò**: Chuyển đổi tọa độ từ pixel ảnh 2D sang tọa độ mét thực tế trên sân bóng theo tiêu chuẩn FIFA ($105\text{m} \times 68\text{m}$). Cung cấp dữ liệu vị trí cho bộ tính tốc độ, cự ly và sa bàn Minimap.
- **Lớp chính trong `pitch_homo.py`**:
  - `PitchCalibration`: Đọc tập điểm mốc đã gán nhãn trong `calib/points.json`, chuẩn hóa tọa độ Hartley (`hartley_normalize_2d`), giải ma trận Homography $H$ ban đầu, lan truyền $H$ qua các frame thông qua ma trận Camera Motion $M_t$.
  - `PitchMapper`: Cung cấp hàm `add_positions_to_tracks(tracks)` thực thi hợp đồng dữ liệu chuẩn 4 trường cho mọi đối tượng.
- **Hợp đồng dữ liệu 4 trường (Data Contract)**:
  1. `position_local_raw`: Toạ độ pixel gốc trên ảnh $(u, v)$ tại chân đối tượng (hoặc tâm bóng).
  2. `position_adjusted`: Toạ độ pixel sau khi bù trừ chuyển động camera quy về hệ tham chiếu.
  3. `position_transformed`: Toạ độ mét $[X, Y]$ trên mặt sân FIFA (nằm trong dải $X \in [0, 105], Y \in [0, 68]$).
  4. `projection_valid` & `calibration_status`: Trạng thái chiếu xạ ảnh (`DIRECT_CALIBRATED`, `PROPAGATED`, `PREDICTED`, `UNREGISTERED`) và mức rủi ro ngoại suy `extrapolation_risk`.
- **Module đột phá: `central_view_geometry.py`**:
  - Khi máy quay truyền hình đặt chính giữa khán đài A nhìn thẳng vào vòng tròn giữa sân, các điểm mốc vạch sân thường nằm trên một đường thẳng (collinear), khiến phương pháp DLT 4 điểm bị suy biến (rank deficiency).
  - Lớp `CenterCircleConic` biểu diễn vòng tròn giữa sân bán kính $R=9.15\text{m}$ dưới dạng ma trận đối xứng Conic $C_{pitch}$ (3x3).
  - Lớp `EllipseFitter` khớp elip từ các điểm cung tròn nhìn thấy trên ảnh.
  - Lớp `CentralViewHomographySolver` dùng điểm biến mất $v_h$ dọc đường giữa sân và đường đối cực (polar line) để khôi phục đầy đủ 8 bậc tự do của ma trận Homography, giải quyết triệt để tính mơ hồ 2 nghiệm bằng ràng buộc vị trí khán đài máy quay.

---

### 7.7 Module: Team Assignment by Jersey Color (`team_assignment/team_assigner.py`)
- **Vị trí**: `team_assignment/team_assigner.py`
- **Vai trò**: Phân chia các cầu thủ trên sân thành 2 đội bóng dựa trên phân cụm màu sắc áo đấu bằng thuật toán K-Means.
- **Lớp chính**: `TeamAssigner`
- **Thuật toán**:
  1. `get_player_color(frame, bbox)`: Cắt phần nửa trên của bounding box cầu thủ (vùng áo đấu). Dùng K-Means với 2 cụm để phân tách màu nền (mặt cỏ) và màu áo cầu thủ (dựa vào góc của vùng cắt).
  2. `assign_team_color(frame, player_detections)`: Thu thập màu áo của tất cả các cầu thủ trong những frame đầu tiên, áp dụng K-Means ($k=2$) để tìm ra 2 tâm cụm màu đại diện cho Team 1 và Team 2.
  3. `get_player_team(frame, bbox, player_id)`: So khớp màu áo của cầu thủ với 2 tâm cụm màu để gán nhãn đội (1 hoặc 2) kèm mã màu RGB phục vụ vẽ đồ họa.
- **Điểm tinh chỉnh trong `main.py`**: Trong Pass 1, chỉ các cầu thủ ngoài sân (`outfield_players`, loại trừ thủ môn `is_goalkeeper == True`) mới được đưa vào phân cụm K-Means để màu áo đặc trưng của thủ môn không làm sai lệch tâm màu của đội.

---

### 7.8 Module: Speed and Distance Calculation (`speed_distance/speed_distance_calc.py`)
- **Vị trí**: `speed_distance/speed_distance_calc.py`
- **Vai trò**: Tính toán vận tốc di chuyển tức thời (km/h) và quãng đường tích lũy (m) của từng cầu thủ dựa trên tọa độ mặt sân FIFA.
- **Lớp chính**: `SpeedAndDistance_Calc`
- **Thuật toán**:
  - Sử dụng cửa sổ trượt (mặc định 5 frame) để làm mịn bước nhảy vị trí, tránh biến động do nhiễu detection.
  - Vận tốc tính theo công thức:
    $$v = \frac{\Delta s_{\text{meters}}}{\Delta t_{\text{seconds}}} \times 3.6 \quad (\text{km/h})$$
  - Quãng đường tích lũy được cộng dồn qua từng frame nếu bước nhảy khoảng cách nằm trong giới hạn vật lý hợp lý ($< 12\text{ m/s}$).
  - Ghi trực tiếp kết quả vào `tracks['players'][frame_idx][track_id]['speed']` và `'distance'`.

---

### 7.9 Module: Player-Ball Assigner (`player_ball_assigner/player_ball_assigner.py`)
- **Vị trí**: `player_ball_assigner/player_ball_assigner.py`
- **Vai trò**: Xác định cầu thủ nào đang có bóng ở từng thời điểm và tổng hợp tỷ lệ kiểm soát bóng của từng đội.
- **Lớp chính**: `PlayerBallAssigner`
- **Thuật toán**:
  - Nhận tọa độ bounding box của quả bóng và danh sách cầu thủ trong frame.
  - Tính khoảng cách Euclidean từ tâm bóng tới 2 góc dưới chân của bounding box cầu thủ:
    $$dist = \min\left(\|p_{\text{ball}} - p_{\text{foot\_left}}\|, \|p_{\text{ball}} - p_{\text{foot\_right}}\|\right)$$
  - Nếu $dist < \text{max\_distance}$ (mặc định 70 pixel) và đạt giá trị nhỏ nhất trong số các cầu thủ, cầu thủ đó được gán cờ `has_ball = True`.
  - Từ `has_ball`, `main.py` tổng lũy mảng `team_ball_control` và tính tỷ lệ phần trăm kiểm soát bóng của Team 1 vs Team 2 hiển thị trên thanh Dashboard.
- **Rủi ro kiến trúc**: Ngưỡng 70 pixel được cố định trong không gian pixel ảnh thô. Do biến dạng phối cảnh, 70px ở gần camera tương đương $\approx 1.5\text{m}$, nhưng ở xa camera có thể tương đương $6-8\text{m}$, dẫn đến việc nhận nhầm cầu thủ giữ bóng ở phía xa.

---

### 7.10 Module: Tactical 2D Minimap Renderer (`minimap/minimap_renderer.py`)
- **Vị trí**: `minimap/minimap_renderer.py`
- **Vai trò**: Kết xuất video Sa bàn chiến thuật 2D độc lập (`output/output_minimap.mp4`, $1280 \times 720$), trực quan hóa toàn bộ trận đấu dưới góc nhìn từ trên cao (Top-down bird's-eye view).
- **Lớp chính**: `MinimapRenderer`
- **Thành phần kết xuất**:
  - Sân bóng chuẩn FIFA với vạch vôi, vòng tròn giữa sân, khu vực 16m50, chấm phạt đền, khung thành.
  - Trọng tài: Vòng tròn màu xanh lá neon viền đen kèm chữ "R".
  - Cầu thủ: Vòng tròn màu áo đội tuyển viền trắng kèm số ID. Cầu thủ đang giữ bóng có hiệu ứng hào quang kép (Dual Halo ring).
  - Thủ môn: Vòng tròn lớn hơn, viền vàng rực kèm chữ "GK".
  - Quả bóng: Chấm tròn màu cam/trắng kèm vệt di chuyển (motion trail) nối giữa các vị trí liên tiếp.
  - Header Dashboard: Thanh hiển thị thời gian, tỷ số/tỷ lệ kiểm soát bóng động của 2 đội.

---

### 7.11 Module: Global Viterbi Ball Selector & High-Resolution Candidate Pipeline (`tracker/viterbi_ball.py`)
- **Vị trí**: `tracker/viterbi_ball.py`
- **Vai trò**: Cung cấp bộ chọn bóng tối ưu hóa quỹ đạo toàn cục bằng Quy hoạch động (Viterbi Dynamic Programming), chấm điểm ngữ cảnh không gian và động học liên kết trong hệ quy chiếu camera chuẩn tĩnh (`to_reference`). Thay thế giải thuật tham lam từng frame (Greedy Top-1), giải quyết triệt để hiện tượng mất bóng khi camera lia nhanh hoặc chọn nhầm giày/tất trắng của cầu thủ.
- **Lớp chính & Cấu trúc dữ liệu**:
  - `ViterbiConfig`: Dataclass lưu trữ siêu tham số: `floor` (0.05), `top_k` (5), `missing_cost` (2.4), `birth_cost` (2.0), `v_cruise` (25.0 px/f), `v_max` (50.0 px/f), `w_speed` (3.0), `enable_isolated_penalty` (False), `enable_gap_reach_limit` (False).
  - `ViterbiBallSelector`: Lớp điều phối trung tâm.
- **Quy trình 2 giai đoạn (Two-Stage Flow)**:
  1. **Pass 1 Candidate Buffer (Thu thập ứng viên)**:
     - Tại mỗi frame trong Pass 1, nhận toàn bộ detection bóng từ YOLO có `confidence >= floor` (mặc định 0.05, có thể hạ xuống 0.01).
     - Lưu tối đa Top-K ứng viên ($K=5$ hoặc $10$) có confidence cao nhất.
     - Lưu kèm bounding box của tất cả cầu thủ trong frame để phục vụ tính điểm ngữ cảnh không gian.
     - Toàn bộ được gom vào bộ đệm RAM `self.frames_candidates` với chi phí bộ nhớ không đáng kể (< 1MB cho 750 frame).
  2. **Post-Pass 1 Global Dynamic Programming (Quy hoạch động toàn cục)**:
     - Chuyển đổi tọa độ tất cả các ứng viên sang hệ quy chiếu camera chuẩn tĩnh `to_reference(t, xy)`.
     - **Bảo vệ ranh giới cắt cảnh (Camera Cut Resilience)**: Khi chỉ số phân đoạn `seg` của `camera_motion` thay đổi, Viterbi tự động chốt quỹ đạo của segment cũ và khởi tạo bài toán mới cho segment sau, ngăn chặn hoàn toàn hiện tượng trôi dạt tọa độ qua các pha chuyển cảnh.
     - Không gian trạng thái tại frame $t$: Gồm $K_t$ ứng viên thật và 1 trạng thái ảo `MISSING`.
- **Hàm mục tiêu Emission (Chi phí quan sát)**:
  $$E_i = -\ln(\max(conf_i, 10^{-3})) + \Delta E_{\text{context}}$$
  Trong đó $\Delta E_{\text{context}}$ tích hợp tri thức nghiệp vụ bóng đá:
  - *Thưởng chân cầu thủ (Foot Bonus)*: $-0.50$ nếu tâm ứng viên nằm trong vùng 25% đáy bounding box của bất kỳ cầu thủ nào (xác suất bóng đang được dẫn/chạm chân).
  - *Phạt phần thân trên (Upper Body Penalty)*: $+0.80$ nếu nằm ở nửa trên 50% bbox cầu thủ (tránh bắt nhầm đầu, vai, áo đấu).
  - *Phạt tĩnh (Static Penalty)*: $+1.50$ nếu ứng viên đứng yên ($\le 4\text{ px/f}$ trong hệ chuẩn) trong $> 5$ frame liên tiếp và cách xa mọi cầu thủ (loại bỏ chấm vôi sân, rác hoặc vỏ chai cố định).
  - *Phạt đơn độc (Isolated Penalty - Mod A)*: $+2.00$ nếu ứng viên xuất hiện đột ngột mà không có bất kỳ ứng viên lân cận nào trong thời gian $1-2$ frame (triệt tiêu giày trắng cầu thủ phát hiện chập chờn).
- **Hàm mục tiêu Transition (Chi phí chuyển trạng thái)**:
  - Cho phép liên kết nhảy cóc qua khoảng mất dấu $g \in [1, G]$ ($G=30$ frame).
  - Chi phí cơ sở: $mc \times g$ (`missing_cost`).
  - Chi phí động học bậc hai (Quadratic Kinetic Penalty):
    Tính vận tốc ngầm định trong hệ tĩnh: $v = \frac{\|\mathbf{p}_t - \mathbf{p}_{t-g}\|}{g}$.
    - Nếu $v \le v_{\text{cruise}}$ ($25\text{ px/f}$): $T_{\text{kinetic}} = 0$.
    - Nếu $v_{\text{cruise}} < v \le v_{\text{max}}$ ($50\text{ px/f}$):
      $$T_{\text{kinetic}} = w_{\text{speed}} \cdot \left(\frac{v - v_{\text{cruise}}}{v_{\text{max}} - v_{\text{cruise}}}\right)^2$$
    - Nếu $v > 2 \cdot v_{\text{max}}$ ($100\text{ px/f}$): Cắt tỉa cứng ($T_{\text{kinetic}} = \infty$).
  - Rào chắn bước nhảy theo khoảng cách (Gap Reach Limit - Mod B):
    $$R(g) = v_{\text{cruise}} \cdot g + C_{\text{margin}} \quad (C_{\text{margin}} = 75\text{ px})$$
    Nếu khoảng cách $\|\mathbf{p}_t - \mathbf{p}_{t-g}\| > R(g)$, cấm liên kết ($T = \infty$). Cơ chế này ngăn chặn tuyệt đối lỗi toán học của phép chia cho gap dài ($g=20$): một bước nhảy xuyên sân 720px chia cho 20 ra $36\text{ px/f} \le v_{\text{max}}$, khiến phép kiểm tra tốc độ thông thường bị vô hiệu hóa.
- **Tái dựng Quỹ đạo & Tương thích Downstream**:
  - Lần ngược theo con trỏ `backpointers` để tìm chuỗi trạng thái tối ưu toàn cục có tổng chi phí nhỏ nhất.
  - Chuyển đổi ngược về tọa độ pixel frame qua `to_frame(t, xy_ref)`.
  - Cập nhật trực tiếp vào dict `tracks["ball"]` với các trường chuẩn `bbox`, `position`, `confidence`, `track_state = "DETECTED"`.
  - Kết quả này được chuyển tiếp sang `BallTrajectoryStabilizer` để nội suy PCHIP cho các frame `MISSING`, hoàn toàn trong suốt với các module Downstream (`PlayerBallAssigner`, `PitchMapper`, `MinimapRenderer`).
- **Tích hợp Tái phát hiện vùng quan tâm theo chuỗi thời gian (2-Stage Temporal ROI Re-detection)**:
  - Khi phát hiện dropout gap ($g \in [5, 30]$), hệ thống crop vùng $640 \times 640$ quanh vị trí ước tính của PCHIP để suy luận YOLO ở tỷ lệ 1:1 (không bị downscaling).
  - Thay vì nhận diện và ghi đè trực tiếp (dễ vướng false positive), các ứng viên ROI được đưa vào đồ thị trạng thái của Viterbi DP. Động học chuyển tiếp Viterbi tự động loại bỏ $80.6\%$ ứng viên rác (giày, vệt cỏ) và khôi phục $100\%$ các ứng viên bóng thật, giúp Recall phân tầng tăng từ $61.56\%$ lên $66.85\%$ và F1 tăng từ $68.73\%$ lên $72.40\%$.

---

## 8. Function/Class Relationships

Dưới đây là bảng phân tích chi tiết quan hệ giữa các lớp và hàm trung tâm được trích xuất từ `codebase-memory-mcp`:

| Lớp / Thành phần | Phương thức / Hàm | Được gọi bởi (Callers) | Gọi đến (Callees) | Bản chất quan hệ |
| :--- | :--- | :--- | :--- | :--- |
| `main.py` | `main()` | CLI invocation (`if __name__ == '__main__'`) | `Tracker`, `CameraMotion`, `TeamAssigner`, `PitchCalibration.fit`, `PitchMapper`, `PlayerBallAssigner`, `MinimapRenderer` | Điều phối toàn cục |
| `Tracker` | `detect_frame()` | `Tracker.process_pass1` | `YOLO.predict` | Generator sinh kết quả detection theo batch |
| `Tracker` | `process_pass1()` | `main.py:210` | `detect_frame`, `CameraMotion.step`, `ReIDTrackerFacade.update`, `_select_ball_detection`, `TeamAssigner.get_player_team` | Phân tích Pass 1 trực tiếp từ luồng video |
| `Tracker` | `_select_ball_detection()` | `Tracker.process_pass1:340` | `np.hypot`, `np.linalg.norm` | Lọc Top-1 tâm bóng theo elip gating vận tốc (Greedy baseline) |
| `ViterbiBallSelector` | `add_frame_candidates()` | `Tracker.process_pass1` | `np.array`, `dataclasses` | Thu thập Top-K ứng viên bóng và bbox cầu thủ từng frame trong Pass 1 |
| `ViterbiBallSelector` | `select_trajectory()` | `Tracker.interpolate_ball_position` | `CameraMotion.to_reference`, `to_frame`, `np.log` | Chạy Dynamic Programming toàn cục trong hệ quy chiếu camera chuẩn tĩnh |
| `Tracker` | `interpolate_ball_position()` | `main.py:230` | `ViterbiBallSelector.select_trajectory`, `BallTrajectoryStabilizer.process` | Facade chuyển giao dữ liệu bóng sang bộ chọn Viterbi và bộ ổn định PCHIP |
| `Tracker` | `draw_tracked_frames()` | `main.py:283` | `draw_ellipse`, `draw_triangle`, `draw_team_ball_control` | Generator Pass 2 vẽ annotation lên video chính |
| `BallTrajectoryStabilizer` | `process()` | `Tracker.interpolate_ball_position` | `_to_reference`, `_is_near_player`, `_interpolate_segment`, `_to_frame` | Khử rung camera, loại bỏ ngoại lai, nội suy PCHIP |
| `BallTrajectoryStabilizer` | `_interpolate_segment()` | `BallTrajectoryStabilizer.process` | `scipy.interpolate.PchipInterpolator` | Nội suy đa thức đơn điệu trên khoảng gap $1-4$ frame |
| `ReIDTrackerFacade` | `update()` | `Tracker.process_pass1:296` | `color_descriptor.extract_multizone_histogram`, `TrackManager.step`, `TrackedDetectionsContainer` | Cầu nối giữa giao diện Tracker và lõi thuật toán Re-ID |
| `TrackManager` | `step()` | `ReIDTrackerFacade.update:116` | `kalman_filter.DampedKalmanFilter.predict`, `cost_matrix.compute_fused_cost_matrix`, `gating.apply_gating_barriers`, `hungarian.min_cost_matching`, `stitcher.TrackletStitcher` | Quản lý 4 giai đoạn gán cặp ID |
| `CameraMotion` | `step()` | `Tracker.process_pass1:276` | `_mask`, `_detect`, `_step`, `PitchRelocalizer.relocalize` | Ước lượng dịch chuyển máy quay giữa 2 frame liên tiếp |
| `PitchCalibration` | `fit()` | `main.py:191, 239` | `central_view_geometry.CentralViewHomographySolver`, `hartley_normalize_2d`, `safe_project` | Khớp ma trận Homography mặt sân sang mét |
| `PitchMapper` | `add_positions_to_tracks()` | `main.py:252` | `CameraMotion.to_reference`, `map_points`, `compute_extrapolation_risk` | Ánh xạ toàn bộ track sang hệ tọa độ metric FIFA chuẩn |
| `MinimapRenderer` | `render_video()` | `main.py:301` | `generate_frames`, `_draw_header`, `_draw_frame`, `meter_to_pixel` | Sinh video sa bàn chiến thuật 2D độc lập |

---

## 9. Function-Level Call Graph

Dưới đây là Call Graph chi tiết ở cấp độ Function của luồng thực thi hoàn chỉnh trong hệ thống:

```text
main() [main.py:121]
 ├── read_video(video_path) [utils/video_utils.py:4]
 ├── Tracker.__init__() [tracker/tracker.py:20]
 │    ├── ultralytics.YOLO(model_path)
 │    ├── ByteTrack.__init__()
 │    ├── BoT-SORT.__init__()
 │    └── ReIDTrackerFacade.__init__() [algorithmic_reid/reid_tracker_facade.py:82]
 │         └── TrackManager.__init__() [algorithmic_reid/lifecycle/track_manager.py:34]
 ├── CameraMotion.__init__() [camera_motion.py:264]
 ├── TeamAssigner.__init__() [team_assignment/team_assigner.py:8]
 │
 ├── [STAGE 3 ANCHOR PRE-LOAD]
 │    ├── load_points(calib_path) [pitch_homo.py:211]
 │    ├── PitchCalibration.fit() [pitch_homo.py:230]
 │    └── CameraMotion.add_global_anchor() [camera_motion.py:275]
 │
 ├── [PASS 1 STREAMING ANALYSIS]
 │    └── Tracker.process_pass1() [tracker/tracker.py:219]
 │         ├── Tracker.detect_frame() [tracker/tracker.py:185]
 │         │    └── YOLO.predict(batch, imgsz=1536)
 │         ├── CameraMotion.step(frame, prev_boxes) [camera_motion.py:442]
 │         │    ├── CameraMotion._mask() [camera_motion.py:296]
 │         │    ├── cv2.calcOpticalFlowPyrLK()
 │         │    ├── cv2.findHomography(RANSAC)
 │         │    └── PitchRelocalizer.relocalize() [camera_motion.py:124]
 │         │
 │         ├── [IF choose_Al == 2: ALGORITHMIC RE-ID]
 │         │    └── ReIDTrackerFacade.update() [algorithmic_reid/reid_tracker_facade.py:93]
 │         │         ├── extract_multizone_histogram() [appearance/color_descriptor.py:44]
 │         │         ├── TrackManager.step() [lifecycle/track_manager.py:116]
 │         │         │    ├── DampedKalmanFilter.predict() [motion/kalman_filter.py:128]
 │         │         │    ├── apply_camera_motion_to_tracks() [motion/motion_predictor.py:15]
 │         │         │    ├── compute_fused_cost_matrix() [association/cost_matrix.py:62]
 │         │         │    │    ├── compute_diou_matrix() [association/cost_matrix.py:16]
 │         │         │    │    ├── compute_mahalanobis_distance_matrix() [motion/mahalanobis.py:16]
 │         │         │    │    └── batch_bhattacharyya_distance() [appearance/color_descriptor.py:140]
 │         │         │    ├── apply_gating_barriers() [association/gating.py:18]
 │         │         │    ├── min_cost_matching() [association/hungarian.py:13]
 │         │         │    │    └── scipy.optimize.linear_sum_assignment()
 │         │         │    ├── TrackletStitcher.can_stitch() [lifecycle/stitcher.py:38]
 │         │         │    ├── TrackletStitcher.stitch_tracklets() [lifecycle/stitcher.py:110]
 │         │         │    └── AppearanceAggregator.update_template() [appearance/temporal_aggregator.py:47]
 │         │         └── TrackedDetectionsContainer.__init__() [reid_tracker_facade.py:28]
 │         │
 │         ├── [IF --ball-selector greedy]
 │         │    └── Tracker._select_ball_detection() [tracker/tracker.py:803]
 │         ├── [IF --ball-selector viterbi]
 │         │    └── ViterbiBallSelector.add_frame_candidates() [tracker/viterbi_ball.py:69]
 │         └── TeamAssigner.assign_team_color() [team_assignment/team_assigner.py:56]
 │              └── sklearn.cluster.KMeans.fit()
 │
 ├── [IN-MEMORY MATHEMATICS PROCESSING]
 │    ├── Tracker.interpolate_ball_position() [tracker/tracker.py:197]
 │    │    ├── [IF --ball-selector viterbi]
 │    │    │    └── ViterbiBallSelector.select_trajectory() [tracker/viterbi_ball.py:111]
 │    │    │         ├── CameraMotion.to_reference() [camera_motion.py:532]
 │    │    │         ├── Forward DP Dynamic Programming
 │    │    │         ├── Backtracking Optimal Trajectory
 │    │    │         └── CameraMotion.to_frame() [camera_motion.py:554]
 │    │    └── BallTrajectoryStabilizer.process() [tracker/ball_tracker.py:46]
 │    │         ├── BallTrajectoryStabilizer._to_reference() [tracker/ball_tracker.py:435]
 │    │         ├── BallTrajectoryStabilizer._is_near_player() [tracker/ball_tracker.py:458]
 │    │         ├── BallTrajectoryStabilizer._interpolate_segment() [tracker/ball_tracker.py:506]
 │    │         │    └── scipy.interpolate.PchipInterpolator()
 │    │         └── BallTrajectoryStabilizer._to_frame() [tracker/ball_tracker.py:442]
 │    │
 │    ├── Tracker.add_position_to_track() [tracker/tracker.py:154]
 │    ├── PitchCalibration.fit() [pitch_homo.py:230]
 │    │    ├── hartley_normalize_2d() [pitch_homo.py:70]
 │    │    └── CentralViewHomographySolver.solve() [central_view_geometry.py:289]
 │    ├── PitchMapper.add_positions_to_tracks() [pitch_homo.py:584]
 │    │    ├── PitchMapper.map_points() [pitch_homo.py:540]
 │    │    └── PitchMapper.compute_extrapolation_risk() [pitch_homo.py:504]
 │    ├── validate_roles_and_assign_goalkeepers() [main.py:20]
 │    ├── SpeedAndDistance_Calc.add_speed_and_distance_to_tracks() [speed_distance_calc.py:10]
 │    └── PlayerBallAssigner.assign_ball_to_player() [player_ball_assigner.py:8]
 │
 ├── [PASS 2 STREAMING RENDERING]
 │    ├── Tracker.draw_tracked_frames() [tracker/tracker.py:607]
 │    ├── CameraMotion.draw_camera_movement() [camera_motion.py:609]
 │    ├── SpeedAndDistance_Calc.draw_speed_and_distance() [speed_distance_calc.py:73]
 │    └── save_video() [utils/video_utils.py:18]
 │
 └── [MINIMAP RADAR RENDERING]
      └── MinimapRenderer.render_video() [minimap/minimap_renderer.py:308]
           ├── MinimapRenderer.generate_frames() [minimap/minimap_renderer.py:292]
           │    ├── MinimapRenderer._create_base_pitch() [minimap/minimap_renderer.py:60]
           │    ├── MinimapRenderer._draw_header() [minimap/minimap_renderer.py:126]
           │    └── MinimapRenderer._draw_frame() [minimap/minimap_renderer.py:186]
           └── cv2.VideoWriter.write()
```

---

## 10. End-to-End Execution Flow

Quá trình chạy của ứng dụng diễn ra tuần tự qua các giai đoạn chi tiết sau:

### Giai đoạn 1: Khởi tạo & Đọc cấu hình
- **File**: `main.py`
- **Hành vi**: Phân tích CLI arguments, kiểm tra thiết bị phần cứng (tự động phát hiện CUDA GPU hoặc fallback sang CPU). Khởi tạo `Tracker`, `CameraMotion`, `TeamAssigner`.
- **Pre-load Anchor**: Nếu có file `calib/points.json`, một đối tượng `cv2.VideoCapture` tạm thời được mở để đọc riêng frame chứa điểm mốc đầu tiên, nạp vào `camera_motion.add_global_anchor()` để chuẩn bị khả năng Re-localization khi có camera cut.

### Giai đoạn 2: Pass 1 Streaming Analysis
- **File**: `tracker/tracker.py` (`process_pass1`)
- **Hành vi**:
  - `read_video(video_path)` trả về một Python Generator sinh từng frame ảnh $1920 \times 1080$.
  - Gom 8 frame thành 1 batch và gọi `model.predict(..., imgsz=1536)`. Nếu cấu hình `--ball-imgsz 1920`, lớp bóng được suy luận riêng ở độ phân giải gốc để tối ưu nhận diện bóng nhỏ.
  - Khử trùng lặp giao cắt Tầng 1 giữa Referee, Goalkeeper, Player.
  - `CameraMotion.step()` tính ma trận đồng biến hình $G_t$ dựa trên các điểm đặc trưng không nằm trong bbox cầu thủ.
  - Bộ theo dõi Re-ID (`ReIDTrackerFacade.update`) nhận detection người theo ngưỡng `--reid-min-conf` (0.20-0.35), cho phép Stage 2 phục hồi cầu thủ bị che khuất mà không làm ô nhiễm bộ cấp phát ID mới hay K-Means áo đội.
  - Xử lý bóng tùy theo `--ball-selector`:
    - Nếu `greedy` (mặc định): `_select_ball_detection` chọn ngay 1 ứng viên quả bóng trong frame dựa trên elip động học.
    - Nếu `viterbi`: gom Top-K ($K=5$ hoặc $10$, $conf \ge floor$) ứng viên bóng và bbox cầu thủ vào bộ đệm `viterbi_selector.add_frame_candidates`.
  - `TeamAssigner` đọc màu áo cầu thủ trong 4 frame đầu tiên để gán màu 2 đội.
  - Kết thúc Pass 1, toàn bộ kết quả phân tích được lưu vào dict `tracks` trong RAM, đồng thời ghi đĩa file cache `stubs/track_stubs_reid.pkl`.

### Giai đoạn 3: Tính toán Không gian & Vật lý trên Metadata (In-Memory)
- **Hành vi**:
  - `tracker.interpolate_ball_position()`:
    - Nếu sử dụng Viterbi: Kích hoạt `ViterbiBallSelector.select_trajectory()` giải quy hoạch động toàn cục tìm đường đi tối ưu trong hệ quy chiếu chuẩn tĩnh `to_reference`, sau đó nạp kết quả vào `tracks["ball"]`.
    - Sau đó, `BallTrajectoryStabilizer.process()` làm mượt và lấp đầy các khoảng mất dấu quả bóng bằng PCHIP 3 tầng.
  - `PitchCalibration.fit()` tính ma trận chuyển đổi từ pixel sang mét sân FIFA ($105\text{m} \times 68\text{m}$).
  - `PitchMapper.add_positions_to_tracks()` bổ sung tọa độ mét `position_transformed` cho mọi đối tượng.
  - `validate_roles_and_assign_goalkeepers()` thẩm định vị trí Tầng 3, xác định thủ môn nếu đứng sâu trong vòng cấm địa.
  - `SpeedAndDistance_Calc.add_speed_and_distance_to_tracks()` tính vận tốc (km/h) và quãng đường (m).
  - `PlayerBallAssigner.assign_ball_to_player()` xác định cầu thủ giữ bóng và tính mảng tỷ lệ kiểm soát bóng.

### Giai đoạn 4: Pass 2 Streaming Video Rendering
- **File**: `tracker/tracker.py` (`draw_tracked_frames`), `utils/video_utils.py` (`save_video`)
- **Hành vi**:
  - Mở lại generator `read_video(video_path)` lần thứ hai.
  - Với mỗi frame đọc được, hệ thống vẽ elip màu đội và ID dưới chân cầu thủ, tam giác định vị bóng trên đầu quả bóng (nếu bật `--draw-interpolated-ball`, các frame nội suy được vẽ dưới dạng vòng tròn phân biệt chống nhấp nháy), vẽ HUD tốc độ/quãng đường và vệt lia máy quay.
  - Ghi trực tiếp frame hoàn chỉnh vào `cv2.VideoWriter`, đảm bảo bộ nhớ RAM luôn được giải phóng sau từng frame.

### Giai đoạn 5: Render Sa bàn 2D Minimap
- **File**: `minimap/minimap_renderer.py` (`render_video`)
- **Hành vi**:
  - Tạo canvas kích thước $1280 \times 720$.
  - Vẽ sân bóng chuẩn tỷ lệ, vẽ chấm tròn trọng tài, cầu thủ, thủ môn và bóng theo tọa độ mét đã chuyển đổi sang pixel của minimap (`meter_to_pixel`).
  - Ghi video `output/output_minimap.mp4`.

---

## 11. Data Flow & Data Contracts

Dữ liệu di chuyển trong hệ thống thông qua dictionary cấu trúc `tracks` với hợp đồng dữ liệu chuẩn hóa:

```mermaid
flowchart LR
    FRAME["Video Frame (NumPy uint8)"] --> YOLO["Ultralytics Results"]
    YOLO --> SV["supervision.Detections"]
    SV --> TRACK_OBJ["tracks dict (Raw Bboxes)"]
    TRACK_OBJ --> STABILIZED["tracks['ball'] (Stabilized Bboxes)"]
    STABILIZED --> METRIC["tracks with position_transformed (Meters)"]
    METRIC --> ANNOTATED["Final Video Frames (Annotated)"]
```

### Cấu trúc Data Contract chi tiết của đối tượng trong `tracks`:
```python
tracks = {
    "players": [
        # Frame 0
        {
            1: {
                "bbox": [x1, y1, x2, y2],               # Tọa độ pixel góc trên-trái, dưới-phải
                "position": (cx, cy_foot),              # Tọa độ điểm chân cầu thủ
                "position_local_raw": (cx, cy_foot),    # Tọa độ pixel gốc trên ảnh
                "position_adjusted": (x_ref, y_ref),    # Tọa độ sau bù chuyển động camera
                "position_transformed": [X_m, Y_m],     # Tọa độ mét trên mặt sân FIFA [0..105, 0..68]
                "team": 1,                              # Đội 1 hoặc Đội 2
                "team_color": (B, G, R),                # Màu áo đại diện
                "is_goalkeeper": False,                 # True nếu là Thủ môn
                "has_ball": True,                       # True nếu đang giữ bóng
                "speed": 22.4,                          # Vận tốc tức thời (km/h)
                "distance": 145.2,                      # Quãng đường tích lũy (m)
                "projection_valid": True,               # Trạng thái phép chiếu Homography
                "calibration_status": "PROPAGATED",     # DIRECT_CALIBRATED | PROPAGATED | PREDICTED | UNREGISTERED
                "extrapolation_risk": 0.42,             # Khoảng cách tới Convex Hull mốc sân (mét)
                "inside_pitch": True                    # Nằm trong ranh giới sân cỏ
            },
            # Các player ID khác...
        },
        # Frame 1, Frame 2...
    ],
    "referees": [
        # Frame 0
        {
            5: {
                "bbox": [x1, y1, x2, y2],
                "position_transformed": [X_m, Y_m],
                # ... các trường chuẩn như Player
            }
        }
    ],
    "ball": [
        # Frame 0
        {
            1: {
                "bbox": [x1, y1, x2, y2],
                "position": (cx_ball, cy_ball),         # Tâm hình học của bóng
                "confidence": 0.88,                     # Độ tin cậy (0.30 nếu là nội suy)
                "is_interpolated": False,               # True nếu do PCHIP sinh ra
                "track_state": "DETECTED",              # DETECTED | INTERPOLATED | LOST
                "position_transformed": [X_m, Y_m],
            }
        }
    ]
}
```

---

## 12. Configuration Analysis

Tất cả các siêu tham số cấu hình của hệ thống được quản lý tại 3 cấp độ:

| Tham số | Vị trí định nghĩa | Giá trị mặc định | Phạm vi ảnh hưởng | Ý nghĩa kỹ thuật |
| :--- | :--- | :--- | :--- | :--- |
| `model_path` | `main.py:124` | `best_YOLOV11s_FB_detect.pt` | Khởi tạo YOLO | File trọng số nơ-ron phát hiện đối tượng |
| `player_conf` | `main.py:125` | `0.35` | Lọc detection Player | Ngưỡng tin cậy tối thiểu chấp nhận cầu thủ |
| `goalkeeper_conf` | `main.py:126` | `0.30` | Lọc detection GK | Ngưỡng tin cậy tối thiểu chấp nhận thủ môn |
| `referee_conf` | `main.py:127` | `0.30` | Lọc detection Referee | Ngưỡng tin cậy tối thiểu chấp nhận trọng tài |
| `ball_conf` | `main.py:128` | `0.20` | `_select_ball_detection` | Ngưỡng tin cậy cơ sở để bắt đầu theo dõi bóng |
| `batch_size` | `main.py:129` | `8` | `Tracker.detect_frame` | Kích thước batch inference YOLO trên GPU |
| `imgsz` | `tracker.py:200` | `1536` | `model.predict` | Kích thước ảnh resize phục vụ phát hiện bóng nhỏ |
| `w_diou` | `config.py:25` | `0.40` | `compute_fused_cost_matrix` | Trọng số khoảng cách Distance-IoU trong chi phí Re-ID |
| `w_mot` | `config.py:26` | `0.20` | `compute_fused_cost_matrix` | Trọng số khoảng cách Mahalanobis chuyển động |
| `w_app` | `config.py:27` | `0.30` | `compute_fused_cost_matrix` | Trọng số khoảng cách màu sắc Bhattacharyya |
| `w_dir` | `config.py:28` | `0.10` | `compute_fused_cost_matrix` | Trọng số cosine góc lệch hướng di chuyển |
| `mahalanobis_gate` | `config.py:37` | `13.28` | `apply_gating_barriers` | Ngưỡng phân phối $\chi^2$ (99%, 4 bậc tự do) loại bỏ đối tượng nhảy vọt |
| `appearance_gate` | `config.py:38` | `0.55` | `apply_gating_barriers` | Ngưỡng khoảng cách Bhattacharyya tối đa cho phép ghép cặp |
| `velocity_damping_gamma`| `config.py:43` | `0.90` | `DampedKalmanFilter` | Hệ số suy giảm vận tốc khi mất dấu quan sát |
| `overlap_occlusion_iou` | `config.py:40` | `0.15` | `AppearanceAggregator` | Ngưỡng IoU kích hoạt đóng băng cập nhật mẫu màu áo |
| `reid_min_conf` | `main.py` / `tracker.py` | `0.35` | `Tracker.process_pass1` | Tách ngưỡng lọc Re-ID Stage 2, cứu detection yếu $[0.20, 0.35)$ mà không tạo track mới |
| `draw_interpolated_ball`| `main.py` / `tracker.py` | `False` | `Tracker.draw_tracked_frames` | Vẽ bóng nội suy PCHIP (vòng tròn phân biệt) trên video chính, chống nhấp nháy |
| `ball_selector` | `main.py` / `tracker.py` | `'greedy'` | `Tracker` & `ViterbiBallSelector` | Lựa chọn thuật toán chọn bóng: `'greedy'` (Top-1 elip) hoặc `'viterbi'` (Quy hoạch động) |
| `ball_imgsz` | `main.py` / `tracker.py` | `None` (1536) | `Tracker.detect_frame` | Phân giải suy luận riêng cho lớp bóng (1920), giải phóng trần candidate recall |
| `viterbi_missing_cost` | `viterbi_ball.py:33` | `2.4` | `ViterbiBallSelector` | Chi phí mỗi frame bóng mất dấu trong bài toán tối ưu đường đi Viterbi |
| `viterbi_floor` | `viterbi_ball.py:31` | `0.05` | `ViterbiBallSelector` | Ngưỡng confidence tối thiểu để thu thập ứng viên bóng vào RAM |
| `viterbi_top_k` | `viterbi_ball.py:32` | `5` | `ViterbiBallSelector` | Số lượng ứng viên bóng tối đa gom mỗi frame trong Pass 1 |
| `enable_isolated_penalty` | `viterbi_ball.py:42` | `False` | `ViterbiBallSelector` | Phạt $+2.0$ cho ứng viên bóng đơn độc chống bắt nhầm giày/tất trắng cầu thủ |
| `enable_gap_reach_limit` | `viterbi_ball.py:43` | `False` | `ViterbiBallSelector` | Rào chắn khoảng cách bước nhảy theo gap $R(g)$ chống bóng nhảy xuyên sân |
| `v_max` (ball) | `ball_tracker.py:28`| `50.0 px/frame`| `BallTrajectoryStabilizer` | Vận tốc tối đa của bóng (~126 km/h) để lọc nhiễu |
| `tier1_max_gap` | `ball_tracker.py:31`| `4 frames` | `BallTrajectoryStabilizer` | Giới hạn mất dấu ngắn áp dụng nội suy PCHIP |
| `tier2_max_gap` | `ball_tracker.py:32`| `10 frames` | `BallTrajectoryStabilizer` | Giới hạn mất dấu vừa áp dụng kiểm tra động học |
| `max_distance` (ball) | `player_ball_assigner.py:7` | `70 px` | `PlayerBallAssigner` | Bán kính pixel tối đa gán bóng cho cầu thủ |

---

## 13. AI/ML Pipeline

### 13.1 Chi tiết Kiến trúc Mô hình Phát hiện (Detection Model)
- **Kiến trúc**: Mô hình YOLO phát hiện đối tượng đơn giai đoạn (Single-Stage Object Detector), dựa trên kiến trúc YOLOv11s hoặc YOLOv8 tùy thuộc file checkpoint.
- **Tập dữ liệu huấn luyện**: Được cấu hình trong `data.yaml`, huấn luyện kết hợp từ 4 tập dữ liệu bóng đá chuẩn (Football-Detect-6, FutVAR-Football-Players-Detection, footbal-1, football-players-detection-1).
- **Danh mục 4 Lớp (Classes)**:
  - Class `0`: `ball` (Quả bóng)
  - Class `1`: `goalkeeper` (Thủ môn)
  - Class `2`: `player` (Cầu thủ)
  - Class `3`: `referee` (Trọng tài)

### 13.2 Pipeline Xử lý & Tinh chỉnh suy luận
1. **Tiền xử lý (Preprocessing)**:
   - Gom cụm frame theo batch $N=8$.
   - Chuẩn hóa kích thước khung hình sang `imgsz=1536` (thay vì 640 mặc định) nhằm tăng tối đa diện tích điểm ảnh của quả bóng (vốn chỉ chiếm khoảng $10 \times 10$ pixel ở góc quay toàn cảnh).
2. **Suy luận (Inference)**:
   - Chạy suy luận với cờ `stream=True` và `agnostic_nms=False`.
   - Đặt ngưỡng suy luận thô `conf=0.15` để không bỏ sót các pha bóng bị nhòe hoặc cầu thủ bị che khuất.
3. **Hậu xử lý Tầng 1 (Tier 1 Post-Processing)**:
   - Tách riêng lớp bóng và người.
   - Với lớp người, áp dụng giải thuật khử trùng lặp đa nhãn độc quyền `_suppress_cross_class_overlaps`: khi một đối tượng bị mô hình gán 2 bounding box giao cắt ($IoU > 0.45$), hệ thống ưu tiên giữ lại nhãn hiếm theo thứ tự: Trọng tài $\to$ Thủ môn $\to$ Cầu thủ.

### 13.3 Nút thắt Phân giải Detector & Cơ chế Phân giải Độc lập (`--ball-imgsz`)
- **Bản chất nút thắt**: Quả bóng trên video góc rộng ($1920 \times 1080$) chỉ có đường kính khoảng $10 - 15\text{ px}$. Khi mạng YOLO downscale khung hình xuống `imgsz=1536`, diện tích quả bóng bị co lại còn $8 - 12\text{ px}$, khiến detector bỏ sót tới $51.4\%$ số frame có bóng thật trên tập Ground Truth con người.
- **Ma trận Candidate Recall phân tầng (Trần lý thuyết của mọi bộ chọn)**:
  Đo đạc với trọng số thực tế 4 tầng ($W_1=91/750$ cho Gap 513-603, $W_2=45/750$ cho Gap 135-179, $W_3=19/750$ cho Gap 615-633, $W_4=595/750$ ngoài gap):

| Cấu hình (`imgsz` / `floor` / `K`) | Hits (trên 109 frame GT) | Candidate Recall Phân tầng | Số ứng viên TB / Frame | Rank 1 % |
| :--- | :---: | :---: | :---: | :---: |
| `1536 / 0.05 / 5` (Baseline) | 53 / 109 | **61.00%** | 1.15 | 86.79% |
| `1536 / 0.01 / 10` | 74 / 109 | **73.06%** | 2.92 | 68.92% |
| `1920 / 0.05 / 5` | 81 / 109 | **80.75%** | 2.68 | 75.31% |
| `1920 / 0.01 / 10` (Đột phá) | 97 / 109 | **91.30%** | 6.19 | 62.89% |

- **Đo lường Tài nguyên Phần cứng thực tế (NVIDIA RTX 3050 Laptop GPU 4GB VRAM)**:
  Thực nghiệm benchmark đo trực tiếp trên 60 frame đầu tiên của video trận đấu:
  - `imgsz=1536, batch=8`: 22.75s (2.64 FPS), Peak VRAM = $1166.46\text{ MB}$ (28.5% dung lượng GPU).
  - `imgsz=1536, batch=4`: 15.13s (3.97 FPS), Peak VRAM = $658.18\text{ MB}$ (16.1% dung lượng GPU).
  - `imgsz=1920, batch=8`: 16.92s (3.55 FPS), Peak VRAM = $2088.93\text{ MB}$ (51.0% dung lượng GPU).
  - `imgsz=1920, batch=4`: 16.95s (3.54 FPS), Peak VRAM = $1159.26\text{ MB}$ (28.3% dung lượng GPU).
  => *Kết luận*: Cấu hình `--ball-imgsz 1920` kết hợp `batch_size=4` giải phóng hoàn toàn trần nhận diện bóng ($+30.3\%$ recall) mà chỉ tiêu tốn dưới $29\%$ bộ nhớ đồ họa, hoàn toàn an toàn trên card đồ họa phổ thông 4GB.

### 13.4 Cơ chế Tái phát hiện vùng quan tâm theo chuỗi thời gian (2-Stage Temporal ROI Re-detection)
- **Ý tưởng kỹ thuật**: Với các khoảng mất dấu dài ($gap \in [5, 30]$), thay vì chạy full-frame 1920 ở mọi nơi, hệ thống cắt crop $640 \times 640$ xung quanh vị trí ước lượng của PCHIP để chạy inference ở tỷ lệ điểm ảnh gốc $1:1$.
- **Khả năng tự thanh lọc của đồ thị Viterbi DP**:
  - Khi chạy ROI trên 68 frame mất dấu, YOLO thu được 11 TP mới nhưng cũng sinh ra 31 false alarms (giày trắng, đầu gối, vệt cỏ).
  - Nếu thay thế cục bộ (Greedy Top-1), độ chính xác sẽ sụt giảm nghiêm trọng.
  - Tuy nhiên, khi đưa toàn bộ 68 frame ứng viên ROI vào đồ thị trạng thái Viterbi DP: các liên kết động học và chi phí chuyển tiếp tự động triệt tiêu $25/31$ false alarms ($80.6\%$), đồng thời giữ lại nguyên vẹn $100\%$ các ứng viên bóng thật.
  - Kết quả tích hợp: Recall phân tầng tăng từ $61.56\%$ lên $66.85\%$ ($+5.29\%$), Precision tăng từ $77.80\%$ lên $78.96\%$ ($+1.16\%$), và F1 tăng từ $68.73\%$ lên $72.40\%$ ($+3.67\%$).

---

## 14. Dependencies & Impact Analysis

### 14.1 Bản đồ Phụ thuộc Module (Dependency Map)

```text
main.py
 ├── tracker/tracker.py [HIGH IMPACT]
 │    ├── ultralytics (External)
 │    ├── supervision (External)
 │    ├── tracker/ball_tracker.py [HIGH IMPACT]
 │    │    └── scipy.interpolate.PchipInterpolator (External)
 │    ├── tracker/viterbi_ball.py [HIGH IMPACT]
 │    │    └── camera_motion.to_reference / to_frame
 │    ├── algorithmic_reid/ [CRITICAL]
 │    │    ├── config.py
 │    │    ├── appearance/color_descriptor.py
 │    │    ├── appearance/temporal_aggregator.py
 │    │    ├── association/cost_matrix.py
 │    │    ├── association/gating.py
 │    │    ├── association/hungarian.py
 │    │    ├── lifecycle/track_manager.py
 │    │    ├── lifecycle/tracklet.py
 │    │    ├── lifecycle/stitcher.py
 │    │    └── motion/kalman_filter.py
 │    └── utils/bbox_utils.py
 ├── camera_motion.py [HIGH IMPACT]
 ├── pitch_homo.py [HIGH IMPACT]
 │    └── central_view_geometry.py [COMPLEX]
 ├── team_assignment/team_assigner.py
 │    └── sklearn.cluster.KMeans (External)
 ├── speed_distance/speed_distance_calc.py
 ├── player_ball_assigner/player_ball_assigner.py
 ├── minimap/minimap_renderer.py
 └── utils/video_utils.py
```

### 14.2 Phân tích Tác động Thay đổi (Impact Analysis for Coding Agents)

Khi can thiệp sửa đổi các thành phần cốt lõi, lập trình viên và coding agent cần lưu ý các chuỗi tác động dây chuyền:

#### 1. Thay đổi tại `algorithmic_reid/reid_tracker_facade.py` hoặc `TrackManager`
- **Callers trực tiếp**: `tracker/tracker.py::Tracker.process_pass1`
- **Tác động gián tiếp**: Tác động trực tiếp đến toàn bộ `tracks['players']` và `tracks['referees']`. Làm xáo trộn ID cầu thủ, gán sai đội bóng tại `TeamAssigner`, tính sai cự ly chạy tại `SpeedAndDistance_Calc`, sai lệch vòng tròn radar trên `MinimapRenderer`.
- **Test suite bắt buộc chạy lại**: `test_reid_tracking_suite.py`, `test_stage2_suite.py`, `test_challenger_p0_stress.py`.

#### 2. Thay đổi tại `tracker/ball_tracker.py` (`BallTrajectoryStabilizer`)
- **Callers trực tiếp**: `tracker/tracker.py::Tracker.interpolate_ball_position`
- **Tác động gián tiếp**: Tác động đến vị trí quả bóng `tracks['ball']`. Ảnh hưởng trực tiếp đến `PlayerBallAssigner` (tính sai cầu thủ giữ bóng), làm lệch tỷ lệ kiểm soát bóng `team_ball_control`, gây đứt gãy vệt bóng cam trên `MinimapRenderer`.
- **Test suite bắt buộc chạy lại**: `test_ball_trajectory_suite.py`, `test_challenger_p1_stress.py`.

#### 3. Thay đổi tại `camera_motion.py` (`CameraMotion`)
- **Callers trực tiếp**: `main.py`, `tracker/tracker.py`, `pitch_homo.py`, `tracker/ball_tracker.py`
- **Tác động gián tiếp**: Làm lệch ma trận quy chiếu $M_t$, khiến việc ổn định bóng bị sai vị trí, bù trừ CMC trong Kalman của Re-ID bị lệch tọa độ, và lan truyền Homography mặt sân bị mất dấu (chuyển sang `UNREGISTERED`).
- **Test suite bắt buộc chạy lại**: `test_stage3_suite.py`, `evaluate_video_stage3.py`.

#### 4. Thay đổi tại `pitch_homo.py` hoặc `central_view_geometry.py`
- **Callers trực tiếp**: `main.py`
- **Tác động gián tiếp**: Làm sai lệch toàn bộ mảng `position_transformed` (tọa độ mét). Toàn bộ vận tốc km/h và cự ly chạy m bị nhân sai hệ số, toàn bộ cầu thủ trên sa bàn 2D Minimap bị vẽ chệch ra ngoài đường biên sân cỏ.
- **Test suite bắt buộc chạy lại**: `test_stage4_suite.py`, `verify_central_view_real_frame.py`.

#### 5. Thay đổi tại `tracker/viterbi_ball.py` (`ViterbiBallSelector`)
- **Callers trực tiếp**: `tracker/tracker.py::Tracker.process_pass1`, `Tracker.interpolate_ball_position`, `scripts/evaluate.py`
- **Tác động gián tiếp**: Tác động trực tiếp đến quỹ đạo quả bóng trước khi chuyển giao cho `BallTrajectoryStabilizer`. Thay đổi siêu tham số (`missing_cost`, `v_cruise`, `w_speed`, rào chắn reach limit) ảnh hưởng trực tiếp đến tỷ lệ Recall/Precision/F1 của bóng và khả năng lọc nhiễu giày cầu thủ.
- **Test suite bắt buộc chạy lại**: `test_viterbi_ball.py`, `test_ball_trajectory_suite.py`, `test_challenger_p1_stress.py`.

---

## 15. Critical Components & Hotspots

### 15.1 Bảng phân loại thành phần theo mức độ rủi ro

| Phân loại | Tên thành phần / Hàm | Vị trí | Lý do đánh dấu |
| :--- | :--- | :--- | :--- |
| **Critical** | `Tracker.process_pass1()` | `tracker/tracker.py:219` | Điểm nghẽn điều phối toàn bộ inference AI, tracking, camera và gán đội |
| **Critical** | `TrackManager.step()` | `algorithmic_reid/lifecycle/track_manager.py:116` | Trái tim của thuật toán Re-ID, kiểm soát 4 giai đoạn gán cặp ID |
| **Critical** | `BallTrajectoryStabilizer.process()`| `tracker/ball_tracker.py:46` | Trái tim của xử lý bóng, đảm bảo tính liên tục của quỹ đạo |
| **High Impact** | `ViterbiBallSelector.select_trajectory()` | `tracker/viterbi_ball.py:111` | Quy hoạch động toàn cục tìm đường đi tối ưu cho bóng trong hệ quy chiếu tĩnh |
| **High Impact** | `PitchMapper.add_positions_to_tracks()`| `pitch_homo.py:584` | Thực thi Data Contract cấp tọa độ mét cho toàn bộ module downstream |
| **High Impact** | `CameraMotion.step()` | `camera_motion.py:442` | Cung cấp ma trận đồng biến hình cho Re-ID, Ball Stabilizer và Pitch Mapper |
| **Complex** | `CentralViewHomographySolver` | `central_view_geometry.py:289` | Hình học xạ ảnh phức tạp: ma trận Conic, cực và đối cực, giải suy biến DLT |
| **Complex** | `compute_fused_cost_matrix()` | `algorithmic_reid/association/cost_matrix.py:62` | Tích hợp đồng thời 4 hàm mục tiêu: DIoU, Mahalanobis, Bhattacharyya, Cosine |
| **Fragile / Baseline** | `_select_ball_detection()` | `tracker/tracker.py:803` | Tính gating trên pixel ảnh thô không bù trừ camera motion (Greedy baseline) |
| **Fragile** | `PlayerBallAssigner.assign_ball_to_player()` | `player_ball_assigner/player_ball_assigner.py:8` | Ngưỡng cứng 70px trong không gian ảnh chịu méo phối cảnh xa gần |
| **Legacy** | `view_transformer/` | Toàn bộ thư mục | Mã nguồn cũ không còn được gọi trong luồng thực thi chính |
| **Legacy** | `camera_movement_estimate/` | Toàn bộ thư mục | Mã nguồn cũ dùng viền ảnh optical flow bị thay bởi `camera_motion.py` |

---

## 16. Architectural Risks & Audit Findings

Qua đối chiếu đồ thị Graphify, kết quả kiểm tra `codebase-memory-mcp` và mã nguồn thực tế, hệ thống tồn tại **9 phát hiện kiến trúc bất thường** (trong đó 5 vấn đề cốt lõi đã được giải quyết triệt để thông qua các cờ mở rộng):

### Rủi ro 1: Cắt cứng confidence trước Re-ID Facade làm vô hiệu hóa Stage 2 Recovery [ĐÃ GIẢI QUYẾT]
- **Vị trí**: `tracker/tracker.py:290-297`
- **Hiện tượng**: `is_person` trước đây lọc cứng `confidence >= self.player_conf` ($0.35$). Trong khi đó, `TrackManager` được thiết kế có **Stage 2 Low-Confidence Recovery** nhằm cứu vãn các detection yếu trong dải $[0.20, 0.35)$ thông qua DIoU. Do bị lọc từ ngoài, Stage 2 không bao giờ nhận được detection nào.
- **Giải pháp đã thực hiện**: Tách biệt tham số `--reid-min-conf` (mặc định 0.35 giữ nguyên hành vi cũ, cấu hình được xuống 0.20 hoặc 0.15). Các detection yếu được gắn cờ `is_reid_recovery_only = True`, chỉ được dùng để duy trì các tracklet đã có ở Stage 2, bị cấm tuyệt đối khởi tạo tracklet mới (Stage 4), không đi vào K-Means phân cụm áo đội và không vẽ bounding box rác lên video.

### Rủi ro 2: Gating bóng tính trên pixel thô không bù trừ Camera Motion [ĐÃ GIẢI QUYẾT]
- **Vị trí**: `tracker/tracker.py:803-884` (`_select_ball_detection`)
- **Hiện tượng**: Vị trí dự đoán `predicted_center = last_center + velocity * dt` và khoảng cách của bộ chọn Greedy được tính trên pixel ảnh thô. Khi máy quay lia nhanh (pan speed $> 30\text{ px/f}$), bóng bị trôi dạt trên ảnh khiến tâm thật rơi ra ngoài elip gating và bị loại bỏ do áp ngưỡng phụ khắt khe $0.35$.
- **Giải pháp đã thực hiện**: Bổ sung bộ chọn bóng Quy hoạch động toàn cục `--ball-selector viterbi` (`tracker/viterbi_ball.py`). Toàn bộ ứng viên bóng được chuyển đổi sang hệ quy chiếu camera tĩnh (`to_reference`), tối ưu hóa chuỗi quỹ đạo bằng thuật toán Viterbi kết hợp chấm điểm ngữ cảnh và tự động reset tại ranh giới camera cut (`seg`).

### Rủi ro 3: Hardcode tham số `choose_Al=2` trong `main.py`
- **Vị trí**: `main.py:217`
- **Hiện tượng**: Lời gọi `tracker.process_pass1(..., choose_Al=2)` bị gán cứng. CLI không có cờ `--tracker-algo` để người dùng chuyển sang ByteTrack (`0`) hoặc BoT-SORT (`1`).

### Rủi ro 4: Bỏ qua frame bóng nội suy khi vẽ video chính [ĐÃ GIẢI QUYẾT]
- **Vị trí**: `tracker/tracker.py:734-735`
- **Hiện tượng**: Trong Pass 2, câu lệnh `if ball.get("is_interpolated", False): continue` khiến bóng nội suy không được vẽ lên video chính, làm tam giác bóng bị nhấp nháy biến mất, trong khi video Minimap vẫn hiển thị mượt mà.
- **Giải pháp đã thực hiện**: Bổ sung cờ CLI `--draw-interpolated-ball` (mặc định `False`). Khi bật, hệ thống vẽ bóng nội suy dưới dạng vòng tròn nét đứt / hollow phân biệt, đảm bảo video trực quan hóa liên tục mà vẫn tường minh với người xem đâu là detection thật và đâu là nội suy PCHIP.

### Rủi ro 5: Gán bóng sử dụng khoảng cách pixel ảnh thô
- **Vị trí**: `player_ball_assigner/player_ball_assigner.py:7, 21`
- **Hiện tượng**: Ngưỡng cố định `max_distance = 70` pixel trên ảnh thô. Do biến dạng phối cảnh, 70px ở phần sân xa tương đương $6-8\text{m}$, khiến cầu thủ đứng rất xa bóng vẫn bị nhận nhầm là đang kiểm soát bóng.

### Rủi ro 6: Vận tốc Kalman không được biến đổi theo ma trận xoay Camera Motion
- **Vị trí**: `algorithmic_reid/motion/motion_predictor.py:23-50`
- **Hiện tượng**: Khi bù trừ CMC, hàm chỉ biến đổi 4 góc của bounding box qua $G_t$ rồi gán lại $z = [x_c, y_c, a, h]$, bỏ quên vector vận tốc $[v_x, v_y, 0, v_h]$ trong state vector của Kalman. Khi camera dừng lia đột ngột, vận tốc quán tính cũ làm dự đoán bị lệch hướng.

### Rủi ro 7: Bất đồng bộ trạng thái Role Tầng 3 giữa `tracks` và `Tracklet`
- **Vị trí**: `main.py:255` (`validate_roles_and_assign_goalkeepers`)
- **Hiện tượng**: Thẩm định Tầng 3 hoán đổi dữ liệu giữa `tracks['referees']` và `tracks['players']` trên RAM sau khi Pass 1 đã xong, không cập nhật ngược lại đối tượng `Tracklet` bên trong `algorithmic_reid`.

### Rủi ro 8: Nút thắt Phân giải Detector ở 1536px làm mất 51.4% số frame bóng [ĐÃ GIẢI QUYẾT]
- **Vị trí**: `tracker/tracker.py:230` (`detect_frame`)
- **Hiện tượng**: Quả bóng có kích thước siêu nhỏ ($< 15\text{ px}$). Khi mạng YOLO chạy ở `imgsz=1536`, việc downscale ảnh làm mất đặc trưng bóng, khiến tỷ lệ có ứng viên đúng rơi xuống chỉ còn $48.6\%$ ($61.0\%$ phân tầng).
- **Giải pháp đã thực hiện**:
  1. Thêm cờ `--ball-imgsz` (mặc định giữ nguyên, thử nghiệm 1920) cho phép chạy inference bóng ở phân giải gốc $1920 \times 1080$, nâng Candidate Recall lên $80.8\%$ (với floor 0.05) và $91.3\%$ (với floor 0.01, K=10).
  2. Đo đạc thực tế xác nhận với `batch_size=4`, mô hình chạy ở 1920 chỉ tiêu tốn $1159\text{ MB}$ VRAM (28.3%), hoàn toàn không lo tràn bộ nhớ trên card 4GB.
  3. Cung cấp cơ chế 2-Stage Temporal ROI Re-detection ($640 \times 640$ crop quanh dự đoán PCHIP) tái phát hiện thành công bóng bị mất mà không cần chạy full-frame nặng nề.

### Rủi ro 9: Lỗi toán học phép chia gap dài $v = \Delta \text{ref} / g$ gây nhảy nhầm giày/tất cầu thủ [ĐÃ GIẢI QUYẾT]
- **Vị trí**: `tracker/viterbi_ball.py` (`_compute_transition_cost`)
- **Hiện tượng**: Trong Viterbi DP, khi gặp gap dài ($g=20$ frame), một bước nhảy xuyên sân $720\text{ px}$ chia cho 20 chỉ cho ra vận tốc ngầm định $v = 36\text{ px/f} \le v_{\text{max}} (50\text{ px/f})$. Do đó, thuật toán bị đánh lừa và chọn nhầm vào giày/tất trắng của cầu thủ ở bên kia sân. Đồng thời, giày cầu thủ đang chạy di chuyển $> 4\text{ px/f}$ nên lọt qua bộ lọc phạt tĩnh $E_{\text{static}}$.
- **Giải pháp đã thực hiện**:
  1. Thêm rào chắn khoảng cách bước nhảy theo gap `enable_gap_reach_limit`: Giới hạn khoảng cách tối đa $R(g) = v_{\text{cruise}} \cdot g + C_{\text{margin}}$ ($C_{\text{margin}} = 75\text{ px}$), cấm tuyệt đối mọi bước nhảy xuyên sân bất chấp $g$ lớn bao nhiêu.
  2. Thêm cờ `enable_isolated_penalty`: Phạt $+2.0$ cho các ứng viên xuất hiện đơn độc 1-2 frame giữa sân cỏ, loại bỏ tận gốc hiện tượng giày trắng chớp tắt.

---

## 17. Testing & Quality Assurance

Hệ thống sở hữu một bộ kiểm thử tự động phong phú, được xây dựng có phương pháp để kiểm tra từng giai đoạn:

| Tên Test Suite | File thực thi | Số kịch bản | Mục tiêu kiểm thử | Trạng thái xác thực |
| :--- | :--- | :--- | :--- | :--- |
| **Re-ID Ground-Truth Suite** | `test_reid_tracking_suite.py` | 8 Scenarios | Kiểm thử giao cắt ngược chiều, mất dấu 1-10f, dao động confidence, đồng đội cùng màu áo chạy song song, cô lập loài (Species Isolation) | **Đạt chuẩn** (0 ID switch trên các ca chuẩn) |
| **Ball Trajectory Suite** | `test_ball_trajectory_suite.py` | 10 Scenarios | Kiểm thử nội suy PCHIP 1-4f, kiểm tra động học 5-10f, rào chắn Strict NO-OP >10f, lọc chùm nhiễu Lookahead Chord, chặn mượt $\le 2.5$px | **Đạt chuẩn** (0 fake ball trên ca mất dấu dài) |
| **Stage 2 Re-ID Suite** | `test_stage2_suite.py` | 6 Tests | Kiểm thử liên kết ngoại hình HSV, cập nhật EMA có Freeze Guard, phục hồi detection yếu | **Đạt chuẩn** |
| **Stage 3 CMC Suite** | `test_stage3_suite.py` | 5 Tests | Kiểm thử bù trừ chuyển động camera, bảo tồn ID khi lia máy quay, lưu trữ và nạp Anchor | **Đạt chuẩn** |
| **Stage 4 Central View Suite**| `test_stage4_suite.py` | 8 Tests | Kiểm thử khớp Conic vòng tròn giữa sân, giải Homography suy biến thẳng hàng, tích hợp Downstream | **Đạt chuẩn** |
| **Viterbi Ball Selector Suite**| `test_viterbi_ball.py` | 6 Tests | Kiểm thử quy hoạch động Viterbi, emission scoring (chân/thân/tĩnh/cô lập), rào chắn gap reach limit, camera cut | **Đạt chuẩn** (6/6 PASS) |
| **Ball Evaluation Harness** | `scripts/evaluate.py` | 109 GT frames | Đo Recall, Precision, FPR (trên frame out), Bootstrap 95% CI trên dữ liệu Ground Truth thật con người gán | **Đạt chuẩn** (Khách quan, chống bias) |
| **Adversarial Stress Suites** | `test_reid_adversarial_stress.py`<br>`test_challenger_p0_stress.py`<br>`test_challenger_p1_stress.py` | Đa kịch bản | Kiểm thử trong điều kiện nhiễu cực đoan: nhảy vọt tọa độ, mất dấu liên tục, va chạm cụm 4 cầu thủ | **Đạt chuẩn** |

---

## 18. Agent Quick Reference (Cẩm nang Tra cứu Nhanh cho Coding Agent)

Phần này được thiết kế cô đọng để các AI Coding Agent trong tương lai có thể nắm bắt ngay lập tức cấu trúc dự án mà không cần đọc lại toàn bộ code:

### 18.1 Tệp tin quan trọng nhất cần nắm
1. `main.py`: Điểm bắt đầu của chương trình, chứa cấu hình CLI, điều phối Pass 1, tính toán trung gian, Pass 2.
2. `tracker/tracker.py`: Chứa class `Tracker`, quản lý inference YOLO, lọc bóng thô, điều phối Pass 1 và vẽ video Pass 2.
3. `tracker/viterbi_ball.py`: Chứa class `ViterbiBallSelector`, quy hoạch động toàn cục chọn bóng và chấm điểm bối cảnh.
4. `tracker/ball_tracker.py`: Chứa class `BallTrajectoryStabilizer`, thuật toán PCHIP 3 tầng cho quả bóng.
5. `algorithmic_reid/reid_tracker_facade.py`: Điểm truy cập duy nhất vào bộ theo dõi người Re-ID (`ReIDTrackerFacade`).
6. `algorithmic_reid/config.py`: File cấu hình chứa toàn bộ trọng số chi phí và ngưỡng Re-ID.
7. `camera_motion.py`: Ước lượng dịch chuyển camera và chuyển đổi hệ tọa độ qua lại (`to_reference`, `to_frame`).
8. `pitch_homo.py`: Chuyển đổi tọa độ pixel sang tọa độ mét thực tế trên sân bóng.
9. `scripts/evaluate.py`: Harness đánh giá Recall, Precision, FPR kèm Bootstrap 95% CI trên Ground Truth con người.
10. `minimap/minimap_renderer.py`: Sinh video sa bàn chiến thuật 2D.

### 18.2 Chuỗi gọi hàm cốt lõi (Core Call Chains)
- **Tracking người**:  
  `main.py:main` $\to$ `tracker.py:process_pass1` $\to$ `reid_tracker_facade.py:update` $\to$ `track_manager.py:step` $\to$ `cost_matrix.py:compute_fused_cost_matrix` $\to$ `hungarian.py:min_cost_matching`.
- **Theo dõi & Ổn định bóng (Luồng Viterbi)**:  
  `main.py:main` $\to$ `tracker.py:process_pass1` $\to$ `viterbi_ball.py:add_frame_candidates` $\to$ `tracker.py:interpolate_ball_position` $\to$ `viterbi_ball.py:select_trajectory` $\to$ `ball_tracker.py:process` $\to$ `ball_tracker.py:_interpolate_segment` $\to$ `scipy.interpolate.PchipInterpolator`.
- **Theo dõi bóng (Luồng Greedy baseline)**:  
  `main.py:main` $\to$ `tracker.py:process_pass1` $\to$ `tracker.py:_select_ball_detection` $\to$ `tracker.py:interpolate_ball_position` $\to$ `ball_tracker.py:process`.
- **Ánh xạ tọa độ sân**:  
  `main.py:main` $\to$ `pitch_homo.py:PitchCalibration.fit` $\to$ `pitch_homo.py:PitchMapper.add_positions_to_tracks`.

### 18.3 Vùng an toàn có thể chỉnh sửa (Safe Modification Areas)
- **Giao diện hiển thị Minimap** (`minimap/minimap_renderer.py`): Tùy chỉnh màu sắc, kích thước canvas, độ dày đường kẻ sân, kiểu chữ dashboard hoàn toàn không ảnh hưởng đến logic tính toán bên trong.
- **Vẽ annotation trên video chính** (`tracker/tracker.py::draw_tracked_frames`): Có thể tùy biến kích thước nhãn, màu sắc tam giác bóng, kiểu dáng bóng nội suy (khi bật `--draw-interpolated-ball`), giao diện thanh tỷ lệ kiểm soát bóng.
- **Thêm tính năng thống kê Downstream** (`speed_distance/`): Có thể bổ sung tính toán gia tốc, bản đồ nhiệt (Heatmap), số lần chạy nước rút (sprints) từ mảng `position_transformed`.
- **Tinh chỉnh siêu tham số Viterbi** (`tracker/viterbi_ball.py`): Có thể điều chỉnh `w_speed`, `birth_cost`, bonus/penalty trong `ViterbiConfig` mà không lo vỡ Data Contract của hệ thống.

### 18.4 Vùng nhạy cảm — Cực kỳ cẩn trọng khi chỉnh sửa (Areas Requiring Extreme Caution)
- **Dữ liệu Ground Truth** (`data/ball_gt_HUMAN.json`): **TUYỆT ĐỐI CHỈ ĐỌC (READ-ONLY)**. Nghiêm cấm mọi hành vi tự động ghi đè, sửa nhãn hoặc tự tạo nhãn giả mạo. Mọi đánh giá mô hình bắt buộc phải đối chiếu trên file này.
- **Bảo toàn Tính tương thích Ngược (Backward Compatibility)**: Mọi cờ tính năng mới (`--reid-min-conf`, `--draw-interpolated-ball`, `--ball-selector`, `--ball-imgsz`, `enable_gap_reach_limit`, `enable_isolated_penalty`) bắt buộc phải **MẶC ĐỊNH TẮT** để giữ nguyên 100% hành vi cũ khi chạy lệnh tiêu chuẩn.
- **Trình tự các trường trong `ReIDConfig`** (`algorithmic_reid/config.py`): Tổng trọng số `w_diou + w_mot + w_app + w_dir` bắt buộc phải bằng $1.0$. Thay đổi một trọng số sẽ làm sai lệch các ngưỡng cắt Hungarian (`max_cost_stage1`, `stage2`, `stage3`).
- **Data Contract của `tracks`** (`pitch_homo.py` và `tracker/tracker.py`): Không được đổi tên các key `bbox`, `position`, `position_adjusted`, `position_transformed` vì toàn bộ module downstream đều phụ thuộc vào tên trường chính xác này.
- **Đơn vị tọa độ sân**: Tọa độ mét trong `pitch_homo.py` và `central_view_geometry.py` luôn tuân theo quy ước $X \in [0, 105]$ (chiều dài sân từ cầu môn trái sang cầu môn phải) và $Y \in [0, 68]$ (chiều rộng sân từ đường biên xa sang đường biên gần).

---

## 19. Change Impact Guide (Hướng dẫn Đánh giá Tác động Thay đổi)

Bảng tra cứu tác động dành cho lập trình viên trước khi thực hiện sửa code:

| Nếu bạn sửa đổi... | Các Module có thể bị ảnh hưởng | Cách kiểm chứng / Test Suite cần chạy |
| :--- | :--- | :--- |
| Ngưỡng `ball_conf` hoặc `_select_ball_detection` | `tracker/tracker.py`, `tracker/ball_tracker.py`, `player_ball_assigner/` | Chạy `python test_ball_trajectory_suite.py` và kiểm tra số frame có bóng được in ra sau Pass 1 trong `main.py` |
| Bộ chọn bóng Viterbi (`tracker/viterbi_ball.py`) | `tracker/tracker.py`, `tracker/ball_tracker.py`, `scripts/evaluate.py` | Chạy `python -m unittest test_viterbi_ball.py` và `python scripts/evaluate.py --method viterbi` |
| Cờ `--reid-min-conf` | `tracker/tracker.py`, `algorithmic_reid/lifecycle/track_manager.py` | Chạy `python -m unittest test_stage2_suite.py test_reid_tracking_suite.py` |
| Cờ `--ball-imgsz` | `tracker/tracker.py`, bộ nhớ VRAM GPU | Chạy `python scripts/benchmark_pass1_imgsz.py` kiểm tra Peak VRAM $\le 2\text{GB}$ |
| Cờ `--draw-interpolated-ball` | `tracker/tracker.py::draw_tracked_frames` | Chạy pipeline với video mẫu, kiểm tra video chính có vẽ bóng nội suy phân biệt |
| Bất kỳ file nào trong `algorithmic_reid/` | `tracker/tracker.py`, `team_assignment/`, `speed_distance/`, `minimap/` | Chạy `python test_reid_tracking_suite.py` và `python test_challenger_p0_stress.py` |
| Thuật toán nội suy trong `tracker/ball_tracker.py` | `main.py`, `player_ball_assigner/`, `minimap/minimap_renderer.py` | Chạy `python test_ball_trajectory_suite.py` và kiểm tra vệt bóng trên `output_minimap.mp4` |
| Hàm `PitchCalibration.fit` hoặc `PitchMapper` | `speed_distance/`, `minimap/`, `main.py` | Chạy `python test_stage4_suite.py` và `python verify_central_view_real_frame.py` |
| `CameraMotion.step` hoặc `to_reference` | `algorithmic_reid/` (CMC), `ball_tracker.py`, `pitch_homo.py` | Chạy `python test_stage3_suite.py` và `python evaluate_video_stage3.py` |
| Bổ sung đối số CLI trong `main.py` | `main.py` | Chạy thử lệnh CLI: `python main.py --video_path video/sample.mp4` |

---

## 20. Final Architecture Summary

Hệ thống **CV Football Analysis** thể hiện tư duy thiết kế phần mềm thị giác máy tính hiện đại và chuẩn mực:
1. **Tách biệt quan tâm (Separation of Concerns)** rõ ràng giữa phát hiện thô (YOLO), theo dõi định danh (Re-ID), tối ưu quỹ đạo toàn cục (Viterbi Ball Selector), ổn định động học (Ball Stabilizer), bù trừ camera (CMC), hình học xạ ảnh sân bóng (Pitch Homography & Central View Conic) và trực quan hóa (Minimap).
2. **Khắc phục triệt để bài toán tài nguyên**: Việc sử dụng kiến trúc Two-Pass Streaming với Python Generator giúp hệ thống xử lý các video thể thao độ nét cao với mức tiêu thụ RAM luôn ổn định dưới $300\text{MB}$. Tách biệt phân giải suy luận `--ball-imgsz 1920` kết hợp `batch_size=4` chạy mượt mà trên GPU 4GB với VRAM chỉ $1.16\text{ GB}$.
3. **Độ tin cậy toán học & Đột phá Quỹ đạo Toàn cục**: Không lạm dụng deep learning một cách mù quáng cho bài toán Re-ID và theo dõi bóng, dự án phối hợp chặt chẽ giữa học máy và giải tích hình học cổ điển (Damped Kalman, DIoU, Bhattacharyya, Conic Fitting, PCHIP và Quy hoạch động Viterbi). Bộ chọn bóng Viterbi tích hợp chấm điểm ngữ cảnh và rào chắn reach limit đã triệt tiêu hoàn toàn hiện tượng chọn nhầm giày/tất cầu thủ và bóng nhảy xuyên sân.
4. **Văn hóa Đánh giá Khách quan & Ground Truth Con người**: Dự án kiên quyết từ chối nhãn tự sinh (pseudo-labels) trong đánh giá chất lượng; toàn bộ chỉ số Recall, Precision, FPR và Bootstrap 95% CI đều được thẩm định độc lập trên Ground Truth con người gán nhãn (`data/ball_gt_HUMAN.json`), thiết lập chuẩn mực tin cậy cao nhất cho nghiên cứu và triển khai sản phẩm.
5. **Tài liệu kiểm toán và test suite đầy đủ**: Hệ thống sở hữu bộ test suite giả lập phong phú bao phủ toàn diện các ca biên phức tạp trong bóng đá thực tế, cung cấp nền tảng vững chắc cho việc bảo trì và mở rộng trong tương lai.

