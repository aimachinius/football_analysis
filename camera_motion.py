"""
camera_motion.py - Ước lượng chuyển động camera (pan / tilt / zoom) bằng homography và Anchor-Grounded Tracking.

Giai đoạn 3 (Anchor-Grounded Tracking, Kiểm soát Drift và Re-localization sau Camera Cut):
  1. Mô hình homography (8 bậc tự do), đúng với camera quay + zoom quanh một tâm cố định.
  2. Anchor-Grounded Tracking:
     - Phân định Global Calibration Anchor (chuẩn sân FIFA) vs Tracking Anchor (tham chiếu ngắn hạn).
     - Cơ chế Anchor Refresh định kỳ (60 frames) hoặc khi độ bao phủ không gian suy giảm.
     - Tích luỹ ma trận M_t = M_{t-1} @ G_t = T_{t -> ref} với G_t = T_{t -> t-1}.
  3. Feature Tracking trên mặt sân với lọc Forward-Backward PyrLK:
     - Che toàn bộ bounding box cầu thủ / trọng tài / bóng (dilated) và overlay tỉ số.
     - Kiểm tra 2 chiều (forward-backward) với ngưỡng sai số khắt khe <= 0.5 px.
     - Ước lượng G_t bằng cv2.USAC_MAGSAC với rào chắn sai số RMS <= 1.5 px và độ bao phủ Convex Hull >= 20%.
  4. Quản lý trạng thái và Prediction Horizon:
     - Khi tracking suy giảm tạm thời (motion blur / cluster cầu thủ): coasting với mô hình vận tốc không đổi
       G_hat = G_{t-1} tối đa 10 frames (trạng thái PREDICTED).
     - Quá 10 frames hoặc phát hiện camera cut: chuyển trạng thái sang UNREGISTERED (H_t = None, không xuất toạ độ giả).
  5. Phát hiện Camera Cut & Tái định vị (PitchRelocalizer):
     - Phát hiện cắt cảnh / replay dựa trên kết hợp sụt giảm inlier đột ngột và biến động ngoại hình lớn (diff > cut_diff).
     - Re-localization Engine: trích xuất ORB/SIFT descriptors trên vùng sân cỏ, đối chiếu thư viện Global Anchor
       qua USAC_MAGSAC (yêu cầu >= 15 matches, >= 12 inliers, RMS <= 2.0 px). Tái định vị thành công đưa frame
       trở lại trạng thái PROPAGATED.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set

import cv2
import numpy as np


@dataclass
class AnchorTrackingConfig:
    grid: tuple = (8, 5)              # chia frame thành lưới ô để điểm đặc trưng phân bố đều
    corners_per_cell: int = 5
    quality: float = 0.005
    min_distance: int = 7
    block_size: int = 7
    lk_win: int = 21
    lk_levels: int = 3
    fb_thresh: float = 0.5            # sai số forward-backward tối đa (pixel)
    ransac_thresh: float = 1.0        # ngưỡng RANSAC / USAC_MAGSAC (pixel)
    min_inliers: int = 30             # số inlier tối thiểu để chấp nhận homography
    min_inlier_ratio: float = 0.60    # tỉ lệ inlier tối thiểu
    min_convex_hull_coverage: float = 0.20  # tỉ lệ diện tích bao lồi inlier / diện tích frame
    max_reproj_rms_px: float = 1.5    # sai số reprojection RMS tối đa trên inliers (px)
    max_pred_horizon: int = 10        # số frame coasting tối đa trước khi rơi vào UNREGISTERED
    anchor_refresh_interval: int = 60 # chu kỳ làm mới tracking anchor (frame)
    box_dilate: int = 12              # nới hộp đối tượng khi che
    static_masks: tuple = ()          # vùng overlay cố định, tỉ lệ (x1, y1, x2, y2), vd ((0, 0, .3, .12),)
    max_log_scale: float = 0.05       # |log(zoom)| tối đa mỗi frame
    max_persp: float = 5e-4
    cut_diff: float = 8.0             # khác biệt xám trung bình để coi là cắt cảnh
    cut_min_tracked: int = 8          # số điểm theo dõi thành công tối thiểu
    cut_min_inlier_ratio: float = 0.15 # tỉ lệ inlier tối thiểu tránh cắt cảnh
    # Cấu hình Re-localization
    reloc_min_matches: int = 15       # số điểm tương đồng tối thiểu để thử fit
    reloc_min_inliers: int = 12       # số inlier tối thiểu để xác nhận tái định vị
    reloc_max_reproj_rms_px: float = 2.0  # sai số RMS tối đa khi tái định vị
    reloc_thresh_px: float = 2.0      # ngưỡng USAC_MAGSAC pixel cho tái định vị


# Tương thích ngược với tên lớp cũ
CameraMotionConfig = AnchorTrackingConfig


@dataclass
class GlobalCalibrationAnchor:
    """Điểm neo hiệu chuẩn toàn cục (Global Calibration Anchor) có quan hệ với toạ độ sân FIFA chuẩn."""
    frame_idx: int
    H_pitch: np.ndarray               # Ma trận 3x3: toạ độ pixel anchor -> mét trên sân FIFA
    gray: np.ndarray                  # Ảnh xám của frame neo
    keypoints: list                   # Danh sách cv2.KeyPoint
    descriptors: np.ndarray           # Descriptors (ORB hoặc SIFT)
    seg: int = 0
    convex_hull: Optional[np.ndarray] = None


@dataclass
class TrackingAnchor:
    """Điểm neo theo dõi ngắn hạn (Tracking Anchor) phục vụ reset template và chống dồn tích méo điểm."""
    frame_idx: int
    T_to_global: np.ndarray           # Ma trận 3x3: toạ độ pixel tracking anchor -> pixel global reference
    global_anchor_idx: int
    step_count: int = 0


class PitchRelocalizer:
    """
    Module tái định vị toàn cục (Global Re-localization Engine).
    Quản lý thư viện Global Calibration Anchors và tìm kiếm phép biến đổi ánh xạ từ frame hiện tại
    về frame neo khi tracking bị mất hoặc sau khi cắt cảnh.
    """
    def __init__(self, cfg: Optional[AnchorTrackingConfig] = None):
        self.cfg = cfg or AnchorTrackingConfig()
        # Dùng ORB với số lượng đặc trưng dồi dào
        self.detector = cv2.ORB_create(nfeatures=1500, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.anchors: Dict[int, GlobalCalibrationAnchor] = {}

    def add_anchor(self, frame_idx: int, gray: np.ndarray, H_pitch: np.ndarray,
                   mask: Optional[np.ndarray] = None, seg: int = 0,
                   convex_hull: Optional[np.ndarray] = None) -> bool:
        """Đăng ký một frame neo toàn cục vào thư viện."""
        if gray is None or H_pitch is None or not np.all(np.isfinite(H_pitch)):
            return False
        kp, des = self.detector.detectAndCompute(gray, mask)
        if des is None or len(kp) < self.cfg.reloc_min_matches:
            return False
        self.anchors[frame_idx] = GlobalCalibrationAnchor(
            frame_idx=frame_idx,
            H_pitch=H_pitch.copy(),
            gray=gray.copy(),
            keypoints=kp,
            descriptors=des,
            seg=seg,
            convex_hull=convex_hull
        )
        return True

    def relocalize(self, gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[Tuple[int, np.ndarray, Dict[str, Any]]]:
        """
        Tìm kiếm frame neo phù hợp nhất cho frame hiện tại.
        Trả về: (anchor_frame_idx, T_curr_to_anchor, info_dict) hoặc None.
        T_curr_to_anchor ánh xạ toạ độ đồng nhất: x_anchor ~ T_curr_to_anchor @ x_curr.
        """
        if not self.anchors or gray is None:
            return None

        kp_curr, des_curr = self.detector.detectAndCompute(gray, mask)
        if des_curr is None or len(kp_curr) < self.cfg.reloc_min_matches:
            return None

        best_result = None
        best_score = float("inf")

        for a_idx, anchor in self.anchors.items():
            if anchor.descriptors is None or len(anchor.descriptors) < self.cfg.reloc_min_matches:
                continue

            matches = self.matcher.knnMatch(des_curr, anchor.descriptors, k=2)
            good_matches = []
            for m_pair in matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < self.cfg.reloc_min_matches:
                continue

            src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([anchor.keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            flag = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
            T_mat, inliers_mask = cv2.findHomography(src_pts, dst_pts, flag, self.cfg.reloc_thresh_px,
                                                    maxIters=2500, confidence=0.999)

            if T_mat is None or inliers_mask is None or not np.all(np.isfinite(T_mat)):
                continue

            mask_bool = (inliers_mask.ravel() == 1)
            n_inl = int(mask_bool.sum())
            if n_inl < self.cfg.reloc_min_inliers:
                continue

            # Tính sai số RMS reprojection trên inliers
            src_inl = src_pts[mask_bool]
            dst_inl = dst_pts[mask_bool]

            src_homo = np.hstack([src_inl, np.ones((n_inl, 1), dtype=np.float64)])
            proj_dst = (T_mat @ src_homo.T).T
            if np.any(np.abs(proj_dst[:, 2]) < 1e-9):
                continue
            proj_dst = proj_dst[:, :2] / proj_dst[:, 2:3]

            rms = float(np.sqrt(np.mean(np.sum((proj_dst - dst_inl) ** 2, axis=1))))
            if rms > self.cfg.reloc_max_reproj_rms_px:
                continue

            # Kiểm tra hình học sơ bộ
            A = T_mat[:2, :2] / T_mat[2, 2]
            det_val = float(np.linalg.det(A))
            if det_val <= 0.2 or det_val >= 5.0:
                continue

            score = rms / np.sqrt(n_inl)
            if score < best_score:
                best_score = score
                best_result = (
                    a_idx,
                    T_mat / T_mat[2, 2],
                    {
                        "anchor_idx": a_idx,
                        "n_matches": len(good_matches),
                        "n_inliers": n_inl,
                        "reproj_rms": rms,
                        "det": det_val
                    }
                )

        return best_result


@dataclass
class CameraMotionResult:
    Ms: np.ndarray                    # (T, 3, 3) frame t -> tham chiếu của đoạn
    seg: np.ndarray                   # (T,) chỉ số đoạn (tăng sau mỗi lần cắt cảnh)
    diag: list                        # chẩn đoán từng frame
    status: np.ndarray = field(default_factory=lambda: np.array([]))  # (T,) trạng thái từng frame

    def __post_init__(self):
        if len(self.status) == 0 and len(self.diag) > 0:
            st = [d.get("status", "PROPAGATED") for d in self.diag]
            self.status = np.array(st, dtype=object)

    def to_reference(self, t, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        q = self.Ms[t] @ np.vstack([pts.T, np.ones(len(pts))])
        return (q[:2] / q[2]).T

    def pan_zoom(self, t, size):
        """(dx, dy, zoom) của tâm frame t so với tham chiếu, dùng để vẽ/chẩn đoán."""
        w, h = size
        c = np.array([[w / 2.0, h / 2.0]])
        d = self.to_reference(t, c)[0] - c[0]
        A = self.Ms[t][:2, :2] / self.Ms[t][2, 2]
        return float(d[0]), float(d[1]), float(np.sqrt(abs(np.linalg.det(A))))

    def get_status(self, t: int) -> str:
        """Lấy trạng thái hiệu chuẩn / tracking tại frame t."""
        if t < len(self.status):
            return str(self.status[t])
        if t < len(self.diag):
            return str(self.diag[t].get("status", "PROPAGATED"))
        return "UNREGISTERED"

    def is_registered(self, t: int) -> bool:
        """Kiểm tra frame t có được định vị trong hệ quy chiếu hợp lệ không."""
        return self.get_status(t) != "UNREGISTERED"

    @property
    def cuts(self):
        return [t for t, d in enumerate(self.diag) if d.get("cut")]


def boxes_from_tracks(tracks):
    """Gom bbox người + bóng theo từng frame để che khi chọn điểm đặc trưng."""
    n = len(tracks["players"])
    out = []
    for t in range(n):
        b = []
        for key in ("players", "referees", "ball"):
            if key in tracks and t < len(tracks[key]):
                b += [v["bbox"] for v in tracks[key][t].values() if v.get("bbox") is not None and len(v["bbox"]) == 4]
        out.append(b)
    return out


class CameraMotion:
    def __init__(self, cfg: Optional[AnchorTrackingConfig] = None,
                 relocalizer: Optional[PitchRelocalizer] = None):
        self.cfg = cfg or AnchorTrackingConfig()
        self.relocalizer = relocalizer or PitchRelocalizer(self.cfg)
        self._crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        self.annotated_frames: Set[int] = set()
        self.reset()

    def set_relocalizer(self, relocalizer: PitchRelocalizer):
        self.relocalizer = relocalizer

    def add_global_anchor(self, frame_idx: int, gray: np.ndarray, H_pitch: np.ndarray,
                           mask: Optional[np.ndarray] = None, seg: int = 0,
                           convex_hull: Optional[np.ndarray] = None) -> bool:
        """Đăng ký global calibration anchor vào relocalizer."""
        self.annotated_frames.add(frame_idx)
        return self.relocalizer.add_anchor(frame_idx, gray, H_pitch, mask=mask, seg=seg, convex_hull=convex_hull)

    def reset(self):
        """Khởi tạo lại toàn bộ trạng thái của chuỗi theo dõi."""
        self.Ms: List[np.ndarray] = []
        self.segs: List[int] = []
        self.diag: List[Dict[str, Any]] = []
        self.status_list: List[str] = []
        self.M: np.ndarray = np.eye(3, dtype=np.float64)
        self.seg: int = 0
        self.prev: Optional[np.ndarray] = None
        self.G_prev: np.ndarray = np.eye(3, dtype=np.float64)
        self.pred_count: int = 0
        self.tracking_anchor: Optional[TrackingAnchor] = None

    # ------------------------------------------------------------------ chọn điểm
    def _mask(self, shape, boxes):
        h, w = shape[:2]
        m = np.full((h, w), 255, np.uint8)
        d = self.cfg.box_dilate
        for b in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            m[max(0, y1 - d):min(h, y2 + d), max(0, x1 - d):min(w, x2 + d)] = 0
        for fx1, fy1, fx2, fy2 in self.cfg.static_masks:
            m[int(fy1 * h):int(fy2 * h), int(fx1 * w):int(fx2 * w)] = 0
        return m

    def _detect(self, gray, mask):
        cfg = self.cfg
        h, w = gray.shape
        gx, gy = cfg.grid
        pts = []
        for r in range(gy):
            for c in range(gx):
                x0, x1 = c * w // gx, (c + 1) * w // gx
                y0, y1 = r * h // gy, (r + 1) * h // gy
                p = cv2.goodFeaturesToTrack(
                    gray[y0:y1, x0:x1], maxCorners=cfg.corners_per_cell, qualityLevel=cfg.quality,
                    minDistance=cfg.min_distance, mask=mask[y0:y1, x0:x1], blockSize=cfg.block_size)
                if p is not None:
                    pts.append(p.reshape(-1, 2) + np.array([x0, y0], dtype=np.float32))
        if not pts:
            return None
        return np.concatenate(pts).astype(np.float32).reshape(-1, 1, 2)

    # ------------------------------------------------------------------ kiểm tra hình học
    def _sane(self, G):
        if G is None or not np.all(np.isfinite(G)) or abs(G[2, 2]) < 1e-9:
            return False
        A = G[:2, :2] / G[2, 2]
        det = np.linalg.det(A)
        if det <= 0:
            return False
        if abs(np.log(np.sqrt(det))) > self.cfg.max_log_scale:
            return False
        return abs(G[2, 0] / G[2, 2]) < self.cfg.max_persp and abs(G[2, 1] / G[2, 2]) < self.cfg.max_persp

    def _compute_reproj_rms(self, src: np.ndarray, dst: np.ndarray, G: np.ndarray) -> float:
        """Tính sai số RMS reprojection (pixel) giữa src -> dst qua G."""
        src_homo = np.hstack([src, np.ones((len(src), 1), dtype=np.float64)])
        proj = (G @ src_homo.T).T
        if np.any(np.abs(proj[:, 2]) < 1e-9):
            return float("inf")
        proj_pts = proj[:, :2] / proj[:, 2:3]
        return float(np.sqrt(np.mean(np.sum((proj_pts - dst) ** 2, axis=1))))

    def _compute_hull_coverage(self, pts: np.ndarray, shape: Tuple[int, int]) -> float:
        """Tính tỉ lệ diện tích bao lồi của các điểm trên diện tích toàn khung hình."""
        if len(pts) < 3:
            return 0.0
        hull = cv2.convexHull(pts.astype(np.float32))
        area = cv2.contourArea(hull)
        h, w = shape[:2]
        return float(area / (w * h))

    # ------------------------------------------------------------------ một bước tracking
    def _step(self, prev, curr, boxes_prev):
        """
        Trả về (G, info):
        - G ánh xạ toạ độ frame hiện tại -> frame trước: x_{t-1} ~ G_t @ x_t.
        - info: thông tin chẩn đoán (n_pts, n_ok, n_inl, reproj_rms, hull_coverage, cut, mode).
        """
        cfg = self.cfg
        info = dict(mode="hold", n_pts=0, n_ok=0, n_inl=0, reproj_rms=float("inf"),
                    hull_coverage=0.0, cut=False)
        mask = self._mask(prev.shape, boxes_prev)
        pts = self._detect(prev, mask)

        n_ok = 0
        if pts is not None and len(pts) >= 8:
            lk = dict(winSize=(cfg.lk_win, cfg.lk_win), maxLevel=cfg.lk_levels, criteria=self._crit)
            nxt, st1, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(curr, prev, nxt, None, **lk)

            # Rào chắn Forward-Backward Error <= cfg.fb_thresh (0.5 px)
            fb_err = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
            ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_err <= cfg.fb_thresh)
            n_ok = int(ok.sum())
            info.update(n_pts=len(pts), n_ok=n_ok)

            if n_ok >= 8:
                src, dst = nxt[ok].reshape(-1, 2), pts[ok].reshape(-1, 2)
                flag = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
                G, inl = cv2.findHomography(src, dst, flag, cfg.ransac_thresh,
                                            maxIters=2000, confidence=0.999)

                if G is not None and inl is not None:
                    inl_mask = (inl.ravel() == 1)
                    n_inl = int(inl_mask.sum())
                    inl_ratio = float(n_inl / n_ok) if n_ok > 0 else 0.0

                    if n_inl >= 4:
                        reproj_rms = self._compute_reproj_rms(src[inl_mask], dst[inl_mask], G)
                        hull_cov = self._compute_hull_coverage(src[inl_mask], prev.shape)
                    else:
                        reproj_rms = float("inf")
                        hull_cov = 0.0

                    info.update(n_inl=n_inl, inlier_ratio=inl_ratio,
                                reproj_rms=reproj_rms, hull_coverage=hull_cov)

                    # Kiểm tra các điều kiện chất lượng
                    passes_quality = (
                        n_inl >= cfg.min_inliers and
                        inl_ratio >= cfg.min_inlier_ratio and
                        reproj_rms <= cfg.max_reproj_rms_px and
                        hull_cov >= cfg.min_convex_hull_coverage and
                        self._sane(G)
                    )

                    if passes_quality:
                        info.update(mode="homography")
                        return G / G[2, 2], info

                    # Fallback sang affine cục bộ nếu điểm đủ tốt
                    A, inl2 = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                          ransacReprojThreshold=cfg.ransac_thresh)
                    if A is not None and inl2 is not None:
                        n_inl2 = int(inl2.sum())
                        if n_inl2 >= max(15, cfg.min_inliers // 2):
                            G_aff = np.vstack([A, [0.0, 0.0, 1.0]])
                            if self._sane(G_aff):
                                info.update(mode="similarity", n_inl=n_inl2)
                                return G_aff, info

        # Phát hiện cắt cảnh: Đòi hỏi cả hai điều kiện:
        # 1. Khác biệt ngoại hình ảnh lớn (diff > cut_diff)
        # 2. VÀ sụt giảm theo dõi trầm trọng (n_ok < cut_min_tracked hoặc inlier ratio cực thấp)
        small = lambda g: cv2.resize(g, (80, 45), interpolation=cv2.INTER_AREA).astype(np.float32)
        diff = float(np.abs(small(prev) - small(curr)).mean())
        is_cut = (diff > cfg.cut_diff) and (
            n_ok < cfg.cut_min_tracked or
            (info.get("n_inl", 0) < cfg.cut_min_tracked and info.get("inlier_ratio", 0.0) < cfg.cut_min_inlier_ratio)
        )

        if is_cut:
            info.update(mode="cut", cut=True, diff=diff)
            return None, info

        info.update(mode="degraded", diff=diff)
        return None, info

    def step(self, frame, boxes_prev=None):
        """
        Xử lý từng frame trực tiếp theo cơ chế Anchor-Grounded Tracking & State Machine.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()
        t = len(self.Ms)
        bp = boxes_prev if boxes_prev is not None else []
        mask = self._mask(gray.shape, bp)

        if self.prev is None:
            # Frame đầu tiên: Thiết lập anchor ban đầu
            info = dict(mode="init", n_pts=0, n_ok=0, n_inl=0, cut=False, pred_count=0)
            self.M = np.eye(3, dtype=np.float64)
            self.tracking_anchor = TrackingAnchor(frame_idx=0, T_to_global=np.eye(3), global_anchor_idx=0, step_count=0)
            status = "DIRECT_CALIBRATED" if 0 in self.annotated_frames else "PROPAGATED"
            info["status"] = status
            self.pred_count = 0
        else:
            prev_status = self.status_list[-1] if self.status_list else "PROPAGATED"

            # Nếu frame trước đang ở trạng thái UNREGISTERED: cố gắng tái định vị trước
            if prev_status == "UNREGISTERED":
                reloc = self.relocalizer.relocalize(gray, mask)
                if reloc is not None:
                    anchor_idx, T_curr_to_a, r_info = reloc
                    self.M = T_curr_to_a.copy()
                    self.tracking_anchor = TrackingAnchor(frame_idx=t, T_to_global=self.M.copy(),
                                                          global_anchor_idx=anchor_idx, step_count=0)
                    self.pred_count = 0
                    status = "PROPAGATED"
                    info = dict(mode="relocated", relocated=True, reloc_info=r_info, status=status, cut=False)
                    print(f"[CameraMotion RE-LOCALIZE] Frame {t}: Tái định vị THÀNH CÔNG từ trạng thái UNREGISTERED về Anchor {anchor_idx} "
                          f"(inliers={r_info['n_inliers']}, RMS={r_info['reproj_rms']:.2f}px). Khôi phục tracking.")
                    self.Ms.append(self.M.copy())
                    self.segs.append(self.seg)
                    self.status_list.append(status)
                    self.diag.append(info)
                    self.prev = gray
                    return self.M, info

            G, info = self._step(self.prev, gray, bp)
            info["pred_count"] = self.pred_count

            if info.get("cut", False):
                # =============================================================
                # TRƯỜNG HỢP 1: CAMERA CUT ĐƯỢC PHÁT HIỆN
                # =============================================================
                old_seg = self.seg
                self.seg += 1
                self.pred_count = 0

                # Thử tái định vị (Re-localization) qua thư viện Global Anchor
                reloc = self.relocalizer.relocalize(gray, mask)
                if reloc is not None:
                    anchor_idx, T_curr_to_a, r_info = reloc
                    self.M = T_curr_to_a.copy()
                    self.tracking_anchor = TrackingAnchor(frame_idx=t, T_to_global=self.M.copy(),
                                                          global_anchor_idx=anchor_idx, step_count=0)
                    status = "PROPAGATED"
                    info.update(mode="relocated", relocated=True, reloc_info=r_info, status=status)
                    print(f"[CameraMotion RE-LOCALIZE] Frame {t}: Tái định vị THÀNH CÔNG sau camera cut về Anchor {anchor_idx} "
                          f"(inliers={r_info['n_inliers']}, RMS={r_info['reproj_rms']:.2f}px). Khôi phục tracking.")
                else:
                    self.M = np.eye(3, dtype=np.float64)
                    status = "UNREGISTERED"
                    info.update(status=status)
                    print(f"[CameraMotion WARNING] Camera Cut tại frame {t} (Phân đoạn {old_seg} -> {self.seg}). "
                          f"Chưa có anchor khớp -> Chuyển trạng thái UNREGISTERED.")

            elif G is not None and prev_status != "UNREGISTERED":
                # =============================================================
                # TRƯỜNG HỢP 2: TRACKING THÀNH CÔNG & ĐẠT CHUẨN CHẤT LƯỢNG
                # =============================================================
                self.G_prev = G.copy()
                self.pred_count = 0

                # Tích luỹ M_t = M_{t-1} @ G_t
                self.M = self.M @ G
                self.M = self.M / self.M[2, 2]

                # Kiểm tra cơ chế Anchor Refresh (sau mỗi 60 frame hoặc độ bao phủ suy giảm)
                if self.tracking_anchor is not None:
                    self.tracking_anchor.step_count += 1
                    if (self.tracking_anchor.step_count >= self.cfg.anchor_refresh_interval or
                            info.get("hull_coverage", 1.0) < 0.25):
                        # Làm mới Tracking Anchor tại frame hiện tại
                        self.tracking_anchor = TrackingAnchor(
                            frame_idx=t,
                            T_to_global=self.M.copy(),
                            global_anchor_idx=self.tracking_anchor.global_anchor_idx,
                            step_count=0
                        )
                        info["anchor_refreshed"] = True

                status = "DIRECT_CALIBRATED" if t in self.annotated_frames else "PROPAGATED"
                info["status"] = status

            else:
                # =============================================================
                # TRƯỜNG HỢP 3: TRACKING SUY GIẢM (BLUR / OCCLUSION)
                # =============================================================
                if prev_status == "UNREGISTERED":
                    self.M = np.eye(3, dtype=np.float64)
                    status = "UNREGISTERED"
                    info.update(mode="unregistered", status=status, pred_count=self.pred_count)
                elif self.pred_count < self.cfg.max_pred_horizon:
                    # Trong prediction horizon (<= 10 frames): coasting với G_{t-1}
                    self.pred_count += 1
                    G_hat = self.G_prev.copy()
                    self.M = self.M @ G_hat
                    self.M = self.M / self.M[2, 2]
                    status = "PREDICTED"
                    info.update(mode="predicted", status=status, pred_count=self.pred_count)
                else:
                    # Vượt quá prediction horizon: thử tái định vị, nếu thất bại rơi vào UNREGISTERED
                    reloc = self.relocalizer.relocalize(gray, mask)
                    if reloc is not None:
                        anchor_idx, T_curr_to_a, r_info = reloc
                        self.M = T_curr_to_a.copy()
                        self.tracking_anchor = TrackingAnchor(frame_idx=t, T_to_global=self.M.copy(),
                                                              global_anchor_idx=anchor_idx, step_count=0)
                        self.pred_count = 0
                        status = "PROPAGATED"
                        info.update(mode="relocated", relocated=True, reloc_info=r_info, status=status)
                    else:
                        self.M = np.eye(3, dtype=np.float64)
                        status = "UNREGISTERED"
                        info.update(mode="unregistered", status=status, pred_count=self.pred_count)

        self.Ms.append(self.M.copy())
        self.segs.append(self.seg)
        self.status_list.append(status)
        self.diag.append(info)
        self.prev = gray
        return self.M, info

    def to_reference(self, t: int, pts: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi toạ độ từ frame t sang hệ quy chiếu tham chiếu của đoạn.
        Tương thích trực tiếp với interface của CameraMotionResult.
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        if t < len(self.Ms):
            M = self.Ms[t]
        elif len(self.Ms) > 0:
            M = self.Ms[-1]
        else:
            M = np.eye(3, dtype=np.float64)
        q = M @ np.vstack([pts.T, np.ones(len(pts))])
        return (q[:2] / q[2]).T

    def get_result(self, stub_path=None):
        """Lấy kết quả chuyển động camera sau khi duyệt hết video."""
        result = CameraMotionResult(
            np.asarray(self.Ms),
            np.asarray(self.segs),
            self.diag,
            status=np.asarray(self.status_list, dtype=object)
        )
        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path) or ".", exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(result, f)
        return result

    # ------------------------------------------------------------------ toàn video (tương thích ngược)
    def estimate(self, frames, boxes_per_frame=None, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)
        if isinstance(frames, tuple):          # read_video() trả (frames, fps)
            frames = frames[0]

        self.reset()
        for t, frame in enumerate(frames):
            bp = boxes_per_frame[t - 1] if boxes_per_frame is not None and t - 1 < len(boxes_per_frame) else []
            self.step(frame, bp)

        return self.get_result(stub_path)

    # ------------------------------------------------------------------ vẽ
    @staticmethod
    def draw_camera_movement(frames, motion: CameraMotionResult):
        for t, frame in enumerate(frames):
            # Tối ưu bộ nhớ: Alpha blend trực tiếp trên vùng ROI (620x140) thay vì copy toàn bộ frame 1080p
            sub_roi = frame[0:140, 0:620]
            white_box = np.full_like(sub_roi, 255)
            cv2.addWeighted(white_box, 0.6, sub_roi, 0.4, 0, sub_roi)
            frame[0:140, 0:620] = sub_roi
            if t < len(motion.Ms):
                dx, dy, z = motion.pan_zoom(t, (frame.shape[1], frame.shape[0]))
                d = motion.diag[t]
                st = motion.get_status(t)
                lines = [
                    f"Pan X: {dx:.1f}px  Pan Y: {dy:.1f}px",
                    f"Zoom: x{z:.3f}   seg: {int(motion.seg[t])}   Status: {st}",
                    f"mode: {d.get('mode', 'N/A')}  inliers: {d.get('n_inl', 0)}  RMS: {d.get('reproj_rms', 0.0):.2f}px"
                ]
            else:
                lines = ["(no camera data)"]
            for i, s in enumerate(lines):
                cv2.putText(frame, s, (10, 35 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            yield frame
