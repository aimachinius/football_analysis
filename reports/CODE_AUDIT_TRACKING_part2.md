# BÁO CÁO KIỂM TOÁN MÃ NGUỒN — PHẦN 2: TRACKING NGƯỜI (MỤC 4)
**Dự án**: Football CV Analysis | [Quay lại Mục lục Phần 1](CODE_AUDIT_TRACKING_part1.md)

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
