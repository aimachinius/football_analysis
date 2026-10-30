# BÁO CÁO KIỂM TOÁN MÃ NGUỒN — PHẦN 3: TRACKING BÓNG & DATA CONTRACT (MỤC 5–7)
**Dự án**: Football CV Analysis | [Quay lại Mục lục Phần 1](CODE_AUDIT_TRACKING_part1.md)

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
