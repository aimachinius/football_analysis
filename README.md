# ⚽ Football CV Analysis

> Real-time football video analysis: multi-player tracking with algorithmic Re-ID, ball trajectory stabilization, team assignment, pitch calibration, and a tactical 2D minimap — powered by YOLOv8/v26 + CUDA.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Player Tracking & Re-ID** | Algorithmic Re-ID (HSV appearance + DIoU + Mahalanobis + Hungarian) — no deep metric model required |
| **Ball Trajectory Stabilization** | Kalman-filter interpolation across occlusion gaps (`ball_tracker.py`) |
| **Team Assignment** | KMeans clustering on jersey color; goalkeeper / referee class separation |
| **Pitch Calibration** | FIFA-standard homography (`pitch_homo.py`) + Central View Conic Geometry (`central_view_geometry.py`) |
| **Speed & Distance** | Real-world speed (km/h) and cumulative distance per player via projected pitch coordinates |
| **Tactical Minimap** | Top-down 2D minimap rendered in real time alongside annotated video |

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          PASS 1 — Inference                     │
│                                                                 │
│  Video Input ──► YOLOv8/v26 Detection (imgsz=1536, batch=8)    │
│                       │                                         │
│                       ▼                                         │
│         Algorithmic ReID Tracker ◄── Camera Motion Comp.       │
│                       │                                         │
│                       ▼                                         │
│          Team Assignment (KMeans)                               │
│                       │                                         │
│                       ▼                                         │
│     Ball Trajectory Stabilizer (Kalman + Interpolation)        │
│                       │                                         │
│                       ▼                                         │
│    Pitch Calibration (Homography + Conic Geometry)             │
│                       │                                         │
│                       ▼                                         │
│          Speed / Distance Calculator                            │
└────────────────────────────┬────────────────────────────────────┘
                             │  intermediate data
┌────────────────────────────▼────────────────────────────────────┐
│                          PASS 2 — Rendering                     │
│                                                                 │
│       Draw Annotations on Frames ──► Output Video (main)       │
│                                  └──► Minimap Video (2D)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Requirements

- **Python** ≥ 3.10
- **CUDA** 11.8+ (GPU recommended; CPU fallback available but very slow)

### Python Dependencies

```
ultralytics>=8.3.0
supervision>=0.30.0
torch>=2.7.0+cu118
torchvision>=0.22.0+cu118
numpy>=2.2.0
scipy>=1.16.0
opencv-python>=4.13.0
```

---

## 🚀 Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd CV_Football_Analys

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install ultralytics supervision scipy opencv-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Download model weights (not tracked in git — obtain separately)
#    Place *.pt files in the project root:
#      best_V26_over_6000.pt   (primary detection model, ~60 MB)
```

---

## ▶️ Usage

### Basic

```bash
python main.py output/result.mp4
```

### Full Options

```bash
python main.py output/result.mp4 \
    --video video/sample.mp4 \
    --model best_V26_over_6000.pt \
    --player-conf 0.35 \
    --gk-conf 0.30 \
    --ref-conf 0.30 \
    --ball-conf 0.20 \
    --batch-size 8 \
    --calib calib/sample.json \
    --device 0
```

| Argument | Default | Description |
|---|---|---|
| `output` | (required) | Path for the annotated output video |
| `--video` | `video/sample.mp4` | Input video file |
| `--model` | `best_V26_over_6000.pt` | YOLO detection weights |
| `--player-conf` | `0.35` | Detection confidence threshold — players |
| `--gk-conf` | `0.30` | Detection confidence threshold — goalkeepers |
| `--ref-conf` | `0.30` | Detection confidence threshold — referees |
| `--ball-conf` | `0.20` | Detection confidence threshold — ball |
| `--batch-size` | `8` | Inference batch size (reduce if VRAM limited) |
| `--calib` | `calib/sample.json` | Pitch calibration point file |
| `--device` | `0` | CUDA device index (`cpu` for CPU-only) |

---

## 📊 Performance Benchmarks

Measured on a consumer GPU (CUDA 11.8), **750-frame 1080p video**, YOLOv26x `imgsz=1536 batch=8`:

| Metric | Value |
|---|---|
| Total runtime | ~98 s (1.63 min) |
| Throughput | **7.65 FPS** |
| VRAM used / reserved | 1,166 MB / 2,102 MB |
| Peak RAM | ~802 MB |

---

## 📂 Key Output Files

| File | Description |
|---|---|
| `output/gd4.mp4` | Annotated match video with tracking overlays |
| `output/gd4_minimap.mp4` | Synchronized 2D tactical minimap |
| `output/benchmark_run_optimized.mp4` | Benchmark annotated output |
| `output/benchmark_run_optimized_minimap.mp4` | Benchmark minimap output |
| `reports/CODE_AUDIT_TRACKING.md` | Full technical audit report |

---

## 📁 Project Structure

```
CV_Football_Analys/
├── main.py                      # Entry point
├── camera_motion.py             # Camera motion compensation
├── pitch_homo.py                # FIFA pitch homography
├── central_view_geometry.py     # Conic geometry & support hull
├── data.yaml                    # Class definitions (ball/gk/player/referee)
├── tracker/
│   ├── tracker.py               # Main tracker (761 lines)
│   └── ball_tracker.py          # Ball trajectory stabilizer
├── algorithmic_reid/            # Player Re-ID package
│   ├── config.py
│   ├── reid_tracker_facade.py
│   ├── appearance/              # HSV histogram features
│   ├── association/             # DIoU + Mahalanobis + Hungarian
│   ├── lifecycle/               # Tracklet, TrackManager, Stitcher
│   └── motion/                  # DampedKalmanFilter, CMC
├── minimap/                     # 2D minimap renderer
├── speed_distance/              # Speed & distance calculator
├── team_assignment/             # KMeans team clustering
├── player_ball_assigner/        # Ball-to-player assignment
├── utils/                       # Shared bbox / video utilities
├── calib/                       # Calibration point files (JSON)
├── reports/                     # Technical audit reports
└── output/                      # Generated videos
```

---

## 📝 License

Private / Internal use. Contact the project owner for licensing terms.

