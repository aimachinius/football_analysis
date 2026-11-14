"""
algorithmic_reid.config - Centralized Hyperparameters and Configuration.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ReIDConfig:
    # Frame rate
    fps: float = 25.0

    # Lifecycle parameters
    n_init: int = 2                       # Consecutive hits to confirm a tentative track (suppresses 1-frame noise)
    max_age: int = 70                     # Max frames to retain lost tracks in buffer
    reactivation_window: int = 30         # Max frame gap for Stage 3 lost track reactivation / stitching
    max_coast_frames: int = 15            # Max frames to output coasting predicted bbox during dropouts to stop flickering
    co_association_window: int = 15       # Window to include recently occluded/lost tracks in Stage 1 co-association

    # Confidence thresholds
    high_conf_thresh: float = 0.40        # Stage 1 matching minimum confidence
    low_conf_thresh: float = 0.20         # Stage 2 low-confidence recovery floor
    init_conf_thresh: float = 0.45        # Minimum confidence to initialize a new tentative track

    # Cost weights (sum to 1.0)
    w_diou: float = 0.40                  # Weight for Distance-IoU cost
    w_mot: float = 0.20                   # Weight for Mahalanobis motion cost
    w_app: float = 0.30                   # Weight for Appearance Bhattacharyya cost
    w_dir: float = 0.10                   # Weight for Heading/Direction cosine cost

    # Matching rejection thresholds
    max_cost_stage1: float = 0.75         # Reject Stage 1 match if cost > 0.75
    max_cost_stage2: float = 0.85         # Reject Stage 2 match if cost > 0.85
    max_cost_stage3: float = 0.70         # Reject Stage 3 reactivation match if cost > 0.70
    max_cost_tentative: float = 0.75      # Reject tentative track match if cost > 0.75

    # Gating barriers
    mahalanobis_gate: float = 13.28       # Chi-squared 99% threshold for 4-DoF
    appearance_gate: float = 0.55         # Maximum allowed Bhattacharyya distance (hard appearance barrier)
    max_velocity_px: float = 35.0         # Max sprint displacement cap (px/frame)
    overlap_occlusion_iou: float = 0.15   # Bbox IoU above which appearance EMA is frozen
    max_spawn_overlap_iou: float = 0.35   # Strict spawn guard: suppress rogue spawns (e.g. ID 63) inside collision clusters
    enable_team_gating: bool = True       # Enforce hard cross-team barrier (Team 1 != Team 2)

    # Advanced Multi-Direction Extensions (Directions A, B, C)
    use_pitch_kinematics: bool = False    # Direction A: 2D Pitch-Metric Kinematics via Homography
    use_deep_reid: bool = False           # Direction B: Deep Semantic Re-ID Extractor (MobileNetV3)
    use_gta: bool = True                  # Direction C: Global Tracklet Association Post-Processing (Enabled by default)
    w_pitch: float = 0.20                 # Weight for Direction A Metric Pitch Cost
    w_deep: float = 0.35                  # Weight for Direction B Deep Semantic Embedding Cost

    # Kalman filter parameters
    velocity_damping_gamma: float = 0.90  # Exponential decay factor for unobserved track velocity
    std_weight_position: float = 1.0 / 20.0
    std_weight_velocity: float = 1.0 / 160.0

    # Appearance feature extraction
    hist_bins_h: int = 16
    hist_bins_s: int = 8
    hist_bins_v: int = 8
    ema_alpha: float = 0.15               # Template update learning rate

    # Species class mapping
    class_ball: int = 0
    class_goalkeeper: int = 1
    class_player: int = 2
    class_referee: int = 3
