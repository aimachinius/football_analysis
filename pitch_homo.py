"""
pitch_homography.py - Chuẩn hoá pixel -> mét trên sân FIFA (105m x 68m).
Phiên bản Giai đoạn 2:
  1. Robust Homography với cv2.USAC_MAGSAC (OpenCV 4.13+) và DLT baseline.
  2. Rào chắn số học safe_project (Normalized Horizon Guard, finite check, matrix health check).
  3. Chuẩn hoá Hartley 2D và kiểm tra số điều kiện (Hartley condition number).
  4. Hợp đồng dữ liệu 4 trường: projection_valid, calibration_status, extrapolation_risk, inside_pitch.
  5. Bảo toàn tương thích ngược cho downstream modules (MinimapRenderer, SpeedAndDistance_Calc, Tracker).
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any, Set

import cv2
import numpy as np

PITCH_L, PITCH_W = 105.0, 68.0


def pitch_landmarks(L=PITCH_L, W=PITCH_W):
    cy = W / 2.0
    lm = {
        "corner_TL": (0, 0), "corner_BL": (0, W), "corner_TR": (L, 0), "corner_BR": (L, W),
        "halfway_T": (L / 2, 0), "halfway_B": (L / 2, W), "center_spot": (L / 2, cy),
        "circle_T": (L / 2, cy - 9.15), "circle_B": (L / 2, cy + 9.15),
        "circle_L": (L / 2 - 9.15, cy), "circle_R": (L / 2 + 9.15, cy),
    }
    pa, six, gp = 40.32 / 2, 18.32 / 2, 7.32 / 2
    d = float(np.sqrt(9.15 ** 2 - 5.5 ** 2))          # giao điểm cung D với vạch vòng cấm
    for side, sgn, x0 in (("L", 1.0, 0.0), ("R", -1.0, L)):
        X = lambda dx, s=sgn, o=x0: o + s * dx
        lm.update({
            f"{side}_pen_T_goalline": (X(0), cy - pa), f"{side}_pen_T_front": (X(16.5), cy - pa),
            f"{side}_pen_B_front": (X(16.5), cy + pa), f"{side}_pen_B_goalline": (X(0), cy + pa),
            f"{side}_six_T_goalline": (X(0), cy - six), f"{side}_six_T_front": (X(5.5), cy - six),
            f"{side}_six_B_front": (X(5.5), cy + six), f"{side}_six_B_goalline": (X(0), cy + six),
            f"{side}_pen_spot": (X(11), cy),
            f"{side}_arc_T": (X(16.5), cy - d), f"{side}_arc_B": (X(16.5), cy + d),
            f"{side}_arc_front": (X(20.15), cy),
            f"{side}_post_T": (X(0), cy - gp), f"{side}_post_B": (X(0), cy + gp),
        })
    return {k: (float(v[0]), float(v[1])) for k, v in lm.items()}


PITCH = pitch_landmarks()


def pitch_wireframe(L=PITCH_L, W=PITCH_W, n=60):
    """Danh sách polyline (toạ độ mét) của các đường kẻ sân, để vẽ overlay kiểm tra."""
    cy = W / 2.0
    lines = [np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], float),
             np.array([[L / 2, 0], [L / 2, W]], float)]
    th = np.linspace(0, 2 * np.pi, n)
    lines.append(np.stack([L / 2 + 9.15 * np.cos(th), cy + 9.15 * np.sin(th)], 1))
    a = np.arccos(5.5 / 9.15)
    ta = np.linspace(-a, a, n // 2)
    for sgn, x0 in ((1.0, 0.0), (-1.0, L)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            lines.append(np.array([[x0, cy - half], [x0 + sgn * depth, cy - half],
                                   [x0 + sgn * depth, cy + half], [x0, cy + half]], float))
        lines.append(np.stack([x0 + sgn * 11 + sgn * 9.15 * np.cos(ta), cy + 9.15 * np.sin(ta)], 1))
    return lines


# ============================================================================
# Chuẩn hoá Hartley & Rào chắn số học
# ============================================================================
def hartley_normalize_2d(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chuẩn hoá tọa độ 2D theo Hartley:
    - Trọng tâm dịch về gốc tọa độ (0, 0).
    - Khoảng cách trung bình tới gốc tọa độ bằng sqrt(2).
    Trả về:
      T: ma trận biến đổi 3x3
      pts_norm: mảng toạ độ đã chuẩn hoá (N, 2)
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.eye(3, dtype=np.float64), pts
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    pts_norm = (T @ pts_homo.T).T[:, :2]
    return T, pts_norm


def compute_hartley_condition_number(H: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    """
    Tính số điều kiện (Condition Number) của ma trận Homography đã chuẩn hoá Hartley:
    H_norm = T_dst @ H @ inv(T_src).
    Trả về kappa = sigma_max / sigma_min.
    """
    if H is None or not np.all(np.isfinite(H)):
        return float("inf")
    T_src, _ = hartley_normalize_2d(src_pts)
    T_dst, _ = hartley_normalize_2d(dst_pts)
    try:
        inv_T_src = np.linalg.inv(T_src)
        H_norm = T_dst @ H @ inv_T_src
        norm_val = np.linalg.norm(H_norm)
        if norm_val < 1e-12 or not np.all(np.isfinite(H_norm)):
            return float("inf")
        H_norm = H_norm / norm_val
        s = np.linalg.svd(H_norm, compute_uv=False)
        if s[-1] < 1e-12:
            return float("inf")
        return float(s[0] / s[-1])
    except Exception:
        return float("inf")


def safe_project(H: np.ndarray, pts: np.ndarray, eps_horizon: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chiếu an toàn toạ độ 2D qua ma trận Homography 3x3 với đầy đủ rào chắn số học:
    1. Kiểm tra ma trận H hữu hạn, det khác 0 và không suy biến rank theo tiêu chí tương đối bất biến scale.
    2. Chuẩn hoá ma trận theo chuẩn Frobenius: H_norm = H / ||H||_F (bảo đảm tính bất biến tuyệt đối theo scale và sign).
    3. Kiểm tra toạ độ đầu vào hữu hạn (không NaN, không Inf).
    4. Rào chắn vanishing line / horizon bất biến:
       Mẫu số chuẩn hoá: |w_i| / (||h_3||_2 * ||x_tilde_i||_2) > eps_horizon.
    5. Kiểm tra toạ độ đầu ra hữu hạn.

    Trả về:
      coords: (N, 2) toạ độ đã chiếu (float64). Các điểm không hợp lệ có giá trị np.nan.
      valid_mask: (N,) mảng boolean xác định điểm nào hoàn toàn hợp lệ.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    coords = np.full((n, 2), np.nan, dtype=np.float64)
    valid_mask = np.zeros(n, dtype=bool)

    if n == 0:
        return coords, valid_mask

    if H is None or not isinstance(H, np.ndarray) or H.shape != (3, 3):
        return coords, valid_mask

    # 1. Kiểm tra ma trận H hữu hạn
    if not np.all(np.isfinite(H)):
        return coords, valid_mask

    # 2. Chuẩn hoá Frobenius để bất biến hoàn toàn với mọi phép nhân tỷ lệ H -> c * H (c != 0)
    norm_H = np.linalg.norm(H)
    if not np.isfinite(norm_H) or norm_H <= 0.0:
        return coords, valid_mask

    H_norm = H / norm_H

    # Kiểm tra tính khả nghịch và độ suy biến theo tỷ lệ SVD tương đối bất biến scale: sigma_min / sigma_max
    s = np.linalg.svd(H_norm, compute_uv=False)
    if s[0] <= 0.0 or (s[-1] / s[0]) < 1e-12:
        return coords, valid_mask

    # 3. Kiểm tra toạ độ đầu vào hữu hạn
    finite_input = np.all(np.isfinite(pts), axis=1)
    if not np.any(finite_input):
        return coords, valid_mask

    h3 = H_norm[2, :]  # vector dòng thứ 3 của ma trận đã chuẩn hoá
    norm_h3 = np.linalg.norm(h3)
    if norm_h3 < 1e-12:
        return coords, valid_mask

    # 4. Tính toạ độ thuần nhất và kiểm tra vanishing line chuẩn hoá
    pts_homo = np.hstack([pts, np.ones((n, 1), dtype=np.float64)])  # (N, 3)
    norm_x = np.linalg.norm(pts_homo, axis=1)                        # (N,)

    q = (H_norm @ pts_homo.T).T  # (N, 3)
    w = q[:, 2]

    # Tiêu chí chuẩn hoá đường chân trời (Normalized Horizon Guard)
    # normalized_w = |w| / (||h3||_2 * ||x_tilde||_2)
    # Hoàn toàn bất biến với c * H và -H
    denom_norm = norm_h3 * norm_x
    normalized_w = np.zeros(n, dtype=np.float64)
    valid_denom = denom_norm > 1e-12
    normalized_w[valid_denom] = np.abs(w[valid_denom]) / denom_norm[valid_denom]

    is_projectable = finite_input & (normalized_w > eps_horizon)

    if np.any(is_projectable):
        proj_x = q[is_projectable, 0] / w[is_projectable]
        proj_y = q[is_projectable, 1] / w[is_projectable]
        finite_output = np.isfinite(proj_x) & np.isfinite(proj_y)

        idx_valid = np.where(is_projectable)[0][finite_output]
        coords[idx_valid, 0] = proj_x[finite_output]
        coords[idx_valid, 1] = proj_y[finite_output]
        valid_mask[idx_valid] = True

    return coords, valid_mask


def _apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Wrapper tương thích ngược, tự động gọi safe_project."""
    coords, _ = safe_project(H, pts)
    return coords


# ============================================================================
# Hiệu chuẩn
# ============================================================================
def load_points(path: str) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """JSON {"frame": {"landmark": [x, y], ...}, ...} -> {int: {str: (x, y)}}"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): {n: (float(p[0]), float(p[1])) for n, p in v.items()} for k, v in raw.items()}


class PitchCalibration:
    def __init__(self, H_by_seg: Dict[int, np.ndarray], report: Dict[int, Dict[str, Any]],
                 convex_hulls: Optional[Dict[int, np.ndarray]] = None,
                 annotated_frames: Optional[Set[int]] = None,
                 model_convex_hulls: Optional[Dict[int, np.ndarray]] = None):
        self.H_by_seg = H_by_seg          # {seg: H pixel_tham_chiếu -> mét}
        self.report = report              # {seg: chi_tiet_danh_gia}
        self.convex_hulls = convex_hulls or {}  # {seg: np.ndarray shape (K, 1, 2) toạ độ mét của bao lồi quan sát thực tế}
        self.model_convex_hulls = model_convex_hulls or {}  # {seg: np.ndarray shape (K, 1, 2) toạ độ mét của bao lồi mô hình}
        self.annotated_frames = annotated_frames or set()

    @staticmethod
    def fit(points_by_frame: Dict[int, Dict[str, Any]],
            motion=None,
            method: str = "magsac",
            threshold: float = 2.0,
            max_iters: int = 2000,
            confidence: float = 0.999,
            arc_points_by_frame: Optional[Dict[int, np.ndarray]] = None,
            frame_images_by_frame: Optional[Dict[int, np.ndarray]] = None) -> PitchCalibration:
        """
        Hiệu chuẩn Homography từ các điểm mốc quan sát được.
        Hỗ trợ:
          - method='magsac' (USAC_MAGSAC, robust đối với ngoại lai/click lệch).
          - method='dlt' (Direct Linear Transformation / method=0, baseline so sánh).
          - arc_points_by_frame: (tuỳ chọn) dict chứa tập điểm cung tròn {frame_idx: np.ndarray (N, 2)}
          - frame_images_by_frame: (tuỳ chọn) dict chứa khung hình {frame_idx: image_bgr} để tự động trích xuất cung tròn.
        threshold: Ngưỡng sai số tính bằng MÉT trên hệ toạ độ đích (sân bóng FIFA).
        """
        groups = defaultdict(lambda: ([], [], []))
        groups_arc = {}
        groups_frame = {}
        annotated_frames = set(points_by_frame.keys())

        if arc_points_by_frame is not None:
            for f, arc_p in arc_points_by_frame.items():
                seg = int(motion.seg[f]) if motion is not None else 0
                groups_arc[seg] = np.asarray(arc_p, dtype=np.float32)

        if frame_images_by_frame is not None:
            for f, frm in frame_images_by_frame.items():
                seg = int(motion.seg[f]) if motion is not None else 0
                groups_frame[seg] = frm

        for f, pts in points_by_frame.items():
            seg = int(motion.seg[f]) if motion is not None else 0
            for name, xy in pts.items():
                if name in ("_arc_points_", "arc_points", "center_circle_arc"):
                    groups_arc[seg] = np.asarray(xy, dtype=np.float32)
                    continue
                if name not in PITCH:
                    raise KeyError(f"Landmark không có trong PITCH: {name}")
                ref = motion.to_reference(f, [xy])[0] if motion is not None else np.asarray(xy, float)
                src, dst, names = groups[seg]
                src.append(ref)
                dst.append(PITCH[name])
                names.append(f"{name}@{f}")

        H_by_seg, report, convex_hulls, model_convex_hulls = {}, {}, {}, {}
        for seg, (src, dst, names) in groups.items():
            src, dst = np.asarray(src, float), np.asarray(dst, float)
            if len(src) < 4:
                report[seg] = dict(error=f"Đoạn {seg}: cần >= 4 điểm, hiện có {len(src)}")
                continue

            # Kiểm tra suy biến thẳng hàng trên toạ độ sân chuẩn
            sv = np.linalg.svd(dst - dst.mean(0), compute_uv=False)
            if sv[-1] / sv[0] < 0.05:
                # [GIAI ĐOẠN 4]: Tự động kích hoạt Central View Conic Solver
                # Khai thác hình học vòng tròn giữa sân & đường giữa sân khi DLT thất bại do thẳng hàng
                central_names = {"circle_T", "circle_B", "center_spot", "halfway_T", "halfway_B", "circle_L", "circle_R"}
                named_dict = {names[i].split("@")[0]: (float(src[i][0]), float(src[i][1])) for i in range(len(names))}
                if len(set(named_dict.keys()) & central_names) >= 3:
                    seg_arc_pts = groups_arc.get(seg)
                    seg_frame = groups_frame.get(seg)
                    try:
                        from central_view_geometry import CentralViewHomographySolver
                        solver = CentralViewHomographySolver(reproj_threshold_m=threshold)
                        H_cv, cv_report = solver.solve_from_landmarks_and_arc(
                            named_dict,
                            arc_points=seg_arc_pts,
                            frame_image=seg_frame
                        )
                        if H_cv is not None:
                            H_by_seg[seg] = H_cv
                            report[seg] = cv_report
                            # 1. Observed Support Hull: Dựng từ các điểm THỰC TẾ quan sát được
                            obs_pts = cv_report.get("observed_support_points_pitch")
                            if obs_pts is not None and len(obs_pts) >= 3:
                                obs_hull = cv2.convexHull(np.asarray(obs_pts, dtype=np.float32))
                            else:
                                obs_hull = cv2.convexHull(dst.astype(np.float32))
                            convex_hulls[seg] = obs_hull

                            # 2. Model-Derived Support Hull: Dựng từ mô hình vòng tròn FIFA 32 điểm lý thuyết
                            model_pts = cv_report.get("model_derived_support_points_pitch")
                            if model_pts is not None and len(model_pts) >= 3:
                                model_hull = cv2.convexHull(np.asarray(model_pts, dtype=np.float32))
                            else:
                                model_hull = obs_hull
                            model_convex_hulls[seg] = model_hull

                            # Ghi nhận diện tích thực tế của cả 2 vùng vào report
                            cv_report["observed_support_hull_area_m2"] = float(cv2.contourArea(obs_hull))
                            cv_report["model_support_hull_area_m2"] = float(cv2.contourArea(model_hull))
                            continue
                        else:
                            report[seg] = dict(error=f"Đoạn {seg}: CentralViewHomographySolver từ chối ({cv_report.get('error', 'Không tìm được nghiệm')})")
                            continue
                    except Exception as e:
                        report[seg] = dict(error=f"Đoạn {seg}: Lỗi ngoại lệ trong CentralViewHomographySolver: {str(e)}")
                        continue

                report[seg] = dict(error=f"Đoạn {seg}: các điểm gần như thẳng hàng trên sân")
                continue

            # Thực hiện ước lượng Homography
            method_lower = str(method).lower()
            if method_lower in ("magsac", "usac_magsac"):
                flag = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
                H, inlier_mask = cv2.findHomography(src, dst, flag, threshold,
                                                    maxIters=max_iters, confidence=confidence)
            elif method_lower in ("dlt", "0", "linear"):
                H, inlier_mask = cv2.findHomography(src, dst, 0)
            else:
                raise ValueError(f"Phương pháp không hợp lệ: {method}. Chọn 'magsac' hoặc 'dlt'.")

            if H is None or not np.all(np.isfinite(H)):
                report[seg] = dict(error=f"Đoạn {seg}: cv2.findHomography không tìm được ma trận hợp lệ")
                continue

            mask = inlier_mask.ravel() if inlier_mask is not None else np.ones(len(src), dtype=int)
            inliers = np.where(mask == 1)[0]

            if len(inliers) < 4:
                report[seg] = dict(error=f"Đoạn {seg}: không đủ inlier ({len(inliers)} < 4) với threshold={threshold}m")
                continue

            # Kiểm tra số điều kiện Hartley
            cond_num = compute_hartley_condition_number(H, src[inliers], dst[inliers])
            if cond_num > 1e8:
                report[seg] = dict(error=f"Đoạn {seg}: ma trận Homography suy biến số học (Hartley cond={cond_num:.1e})")
                continue

            # Tính sai số fit trên toạ độ sân (mét)
            proj_dst, val_dst = safe_project(H, src)
            fit_err = np.linalg.norm(proj_dst - dst, axis=1)

            # Tính sai số reprojection ngược trên pixel ảnh
            try:
                H_inv = np.linalg.inv(H)
                proj_src, val_src = safe_project(H_inv, dst)
                reproj_err_px = np.linalg.norm(proj_src - src, axis=1)
            except Exception:
                reproj_err_px = np.full(len(src), np.nan)

            inlier_fit_err = fit_err[inliers]
            inlier_reproj_px = reproj_err_px[inliers]

            rep = dict(
                method=method_lower,
                threshold_m=threshold,
                n_total=len(src),
                n_inliers=len(inliers),
                inlier_ratio=float(len(inliers) / len(src)),
                hartley_condition=float(cond_num),
                rms_fit_m=float(np.sqrt((inlier_fit_err ** 2).mean())),
                mean_fit_m=float(inlier_fit_err.mean()),
                max_fit_m=float(inlier_fit_err.max()),
                worst_fit=names[int(inliers[inlier_fit_err.argmax()])],
                rms_reproj_px=float(np.sqrt((inlier_reproj_px ** 2).mean())) if np.all(np.isfinite(inlier_reproj_px)) else None,
                mean_reproj_px=float(inlier_reproj_px.mean()) if np.all(np.isfinite(inlier_reproj_px)) else None,
            )

            # Đánh giá Leave-One-Out (LOO) trên các điểm inliers
            if len(inliers) >= 6:
                loo = []
                src_in, dst_in = src[inliers], dst[inliers]
                for i in range(len(inliers)):
                    keep = np.arange(len(inliers)) != i
                    if method_lower in ("magsac", "usac_magsac"):
                        Hi, _ = cv2.findHomography(src_in[keep], dst_in[keep],
                                                   cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else 0,
                                                   threshold, maxIters=max_iters, confidence=confidence)
                    else:
                        Hi, _ = cv2.findHomography(src_in[keep], dst_in[keep], 0)
                    if Hi is not None:
                        pred_i, _ = safe_project(Hi, src_in[i:i + 1])
                        loo.append(np.linalg.norm(pred_i[0] - dst_in[i]))
                if len(loo) == len(inliers):
                    loo = np.asarray(loo)
                    rep.update(loo_mean_m=float(loo.mean()), loo_max_m=float(loo.max()),
                               worst_loo=names[int(inliers[loo.argmax()])])

            # Tính Convex Hull trên hệ toạ độ sân chuẩn từ các inlier
            dst_inliers_f32 = dst[inliers].astype(np.float32)
            hull = cv2.convexHull(dst_inliers_f32)
            convex_hulls[seg] = hull
            model_convex_hulls[seg] = hull

            H_by_seg[seg] = H
            report[seg] = rep

        return PitchCalibration(H_by_seg, report, convex_hulls=convex_hulls, annotated_frames=annotated_frames,
                                model_convex_hulls=model_convex_hulls)


# ============================================================================
# Ánh xạ & Hợp đồng dữ liệu
# ============================================================================
class PitchMapper:
    def __init__(self, calib: PitchCalibration, motion, pixel_sigma: float = 2.0, margin_m: float = 3.0,
                 L: float = PITCH_L, W: float = PITCH_W, use_model_hull: bool = False):
        self.calib = calib
        self.motion = motion
        self.pixel_sigma = pixel_sigma
        self.margin = margin_m
        self.L = L
        self.W = W
        self.use_model_hull = use_model_hull

    def Ht(self, t: int) -> Optional[np.ndarray]:
        if self.motion is None:
            return self.calib.H_by_seg.get(0)
        status = self.get_calibration_status(t)
        if status == "UNREGISTERED":
            return None
        seg = int(self.motion.seg[t]) if t < len(self.motion.seg) else 0
        H0 = self.calib.H_by_seg.get(seg)
        if H0 is None:
            H0 = self.calib.H_by_seg.get(0)
        if H0 is None:
            return None
        return H0 @ self.motion.Ms[t]

    def get_calibration_status(self, t: int) -> str:
        """
        Xác định trạng thái hiệu chuẩn của frame t:
        - UNREGISTERED: đoạn video chưa có Homography hoặc tracking thất bại vượt quá horizon.
        - DIRECT_CALIBRATED: frame được đánh dấu landmark thủ công.
        - PREDICTED: camera motion ở chế độ hold / coasting.
        - PROPAGATED: camera motion tracking bình thường từ frame mốc.
        """
        seg = int(self.motion.seg[t]) if (self.motion is not None and t < len(self.motion.seg)) else 0
        if seg not in self.calib.H_by_seg and 0 not in self.calib.H_by_seg:
            return "UNREGISTERED"

        if self.motion is not None:
            if hasattr(self.motion, "get_status"):
                m_st = self.motion.get_status(t)
                if m_st == "UNREGISTERED":
                    return "UNREGISTERED"
                if t in self.calib.annotated_frames:
                    return "DIRECT_CALIBRATED"
                if m_st in ("PREDICTED", "predicted"):
                    return "PREDICTED"
                if m_st in ("PROPAGATED", "propagated", "homography", "similarity", "relocated"):
                    return "PROPAGATED"
            elif hasattr(self.motion, "status") and t < len(self.motion.status):
                m_st = str(self.motion.status[t])
                if m_st == "UNREGISTERED":
                    return "UNREGISTERED"
                if t in self.calib.annotated_frames:
                    return "DIRECT_CALIBRATED"
                if m_st in ("PREDICTED", "predicted"):
                    return "PREDICTED"
                if m_st in ("PROPAGATED", "propagated"):
                    return "PROPAGATED"

        if seg not in self.calib.H_by_seg or self.calib.H_by_seg[seg] is None:
            return "UNREGISTERED"

        if t in self.calib.annotated_frames or (t == 0 and 0 in self.calib.H_by_seg and seg == 0):
            return "DIRECT_CALIBRATED"

        mode = "track"
        if self.motion is not None and hasattr(self.motion, "diag") and t < len(self.motion.diag):
            mode = self.motion.diag[t].get("mode", "track")
            if self.motion.diag[t].get("status") == "UNREGISTERED":
                return "UNREGISTERED"

        if mode in ("hold", "init", "predicted", "PREDICTED") and t > 0:
            return "PREDICTED"

        return "PROPAGATED"

    def compute_extrapolation_risk(self, seg: int, pts_pitch: np.ndarray, use_model_hull: Optional[bool] = None) -> np.ndarray:
        """
        Tính khoảng cách Euclid (mét) từ điểm trên sân đến Convex Hull của các landmark quan sát được.
        - Điểm trong hoặc trên biên Convex Hull: 0.0m.
        - Điểm ngoài Convex Hull: khoảng cách ngắn nhất > 0.0m.
        - Không có hull hoặc điểm không hợp lệ: np.nan.
        - use_model_hull: nếu True, dùng model_convex_hulls; nếu False, dùng convex_hulls (quan sát thực tế đã lọc ngoại lai);
          nếu None (mặc định), sử dụng cấu hình self.use_model_hull của mapper instance.
        """
        if use_model_hull is None:
            use_model_hull = self.use_model_hull
        pts_pitch = np.asarray(pts_pitch, dtype=np.float64).reshape(-1, 2)
        n = len(pts_pitch)
        risks = np.full(n, np.nan, dtype=np.float64)
        hulls_dict = self.calib.model_convex_hulls if use_model_hull else self.calib.convex_hulls
        hull = hulls_dict.get(seg)
        if hull is None or len(hull) < 3:
            hull = self.calib.convex_hulls.get(seg)
        if hull is None or len(hull) < 3:
            return risks

        hull_f32 = hull.astype(np.float32)
        for i, pt in enumerate(pts_pitch):
            if not np.all(np.isfinite(pt)):
                continue
            # dist > 0 bên trong, == 0 trên biên, < 0 bên ngoài
            dist = cv2.pointPolygonTest(hull_f32, (float(pt[0]), float(pt[1])), measureDist=True)
            risks[i] = 0.0 if dist >= 0.0 else float(-dist)
        return risks

    def is_strictly_inside_pitch(self, x: float, y: float) -> bool:
        """Kiểm tra toạ độ có nằm hoàn toàn trong sân FIFA chuẩn [0, 105] x [0, 68] không."""
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
        return (0.0 <= x <= self.L) and (0.0 <= y <= self.W)

    def map_points(self, t: int, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ánh xạ an toàn (N, 2) pixel sang (N, 2) mét và tính sai số position_sigma.
        Trả về (coords_m, sigma_m). Nếu lỗi số học hoặc chưa hiệu chuẩn: coords là NaN.
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        n = len(pts)
        calib_status = self.get_calibration_status(t)
        if calib_status == "UNREGISTERED":
            return np.full((n, 2), np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)

        Ht = self.Ht(t)
        if Ht is None:
            return np.full((n, 2), np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)

        q0, valid0 = safe_project(Ht, pts)
        qx, valid_x = safe_project(Ht, pts + [1.0, 0.0])
        qy, valid_y = safe_project(Ht, pts + [0.0, 1.0])

        sigma = np.full(n, np.nan, dtype=np.float64)
        valid_all = valid0 & valid_x & valid_y
        if np.any(valid_all):
            scale = np.maximum(
                np.linalg.norm(qx[valid_all] - q0[valid_all], axis=1),
                np.linalg.norm(qy[valid_all] - q0[valid_all], axis=1)
            )
            sg = self.pixel_sigma * scale
            if calib_status == "PREDICTED":
                pred_count = 1
                if self.motion is not None and hasattr(self.motion, "diag") and t < len(self.motion.diag):
                    pred_count = self.motion.diag[t].get("pred_count", 1)
                sg = sg + 0.5 * pred_count + 1.0
            elif calib_status == "PROPAGATED":
                sg = sg + 0.02 * np.sqrt(max(1, t))
            sigma[valid_all] = sg

        return q0, sigma

    def inside(self, xy: np.ndarray) -> np.ndarray:
        """Hỗ trợ tương thích ngược với margin_m."""
        xy = np.asarray(xy).reshape(-1, 2)
        m = self.margin
        return (-m <= xy[:, 0]) & (xy[:, 0] <= self.L + m) & (-m <= xy[:, 1]) & (xy[:, 1] <= self.W + m)

    def add_positions_to_tracks(self, tracks: Dict[str, List[Dict[int, Dict[str, Any]]]]):
        """
        Thực thi hợp đồng dữ liệu chuẩn 4 trường cho mọi đối tượng trong tracks:
          1. projection_valid: bool (Phép chiếu số học qua homography có hợp lệ không)
          2. calibration_status: str (DIRECT_CALIBRATED, PROPAGATED, PREDICTED, UNREGISTERED)
          3. extrapolation_risk: float | None (Khoảng cách mét tới Convex Hull của landmark; None nếu chưa calib)
          4. inside_pitch: bool | None (True/False nếu có toạ độ toàn cục hợp lệ; None nếu UNREGISTERED hoặc invalid)

        Quy ước toạ độ:
          - position_local_raw: toạ độ pixel ảnh (u, v) tại chân đối tượng (bottom-center bbox hoặc ball center).
          - position_adjusted: toạ độ pixel hệ tham chiếu (sau bù trừ chuyển động camera motion).
          - position_transformed: toạ độ mét [X, Y] trên mặt sân FIFA (None khi UNREGISTERED hoặc phép chiếu hỏng).
          - Khi projection_valid == True và calibration_status != 'UNREGISTERED':
            position_transformed = [X, Y] vẫn được lưu ngay cả khi inside_pitch == False (phục vụ cầu thủ ném biên).
        """
        for obj, per_frame in tracks.items():
            for t, frame in enumerate(per_frame):
                if not frame:
                    continue

                calib_status = self.get_calibration_status(t)
                seg = int(self.motion.seg[t]) if (self.motion is not None and t < len(self.motion.seg)) else 0

                infos = list(frame.values())
                pts_list = []
                for info in infos:
                    b = info["bbox"]
                    if info.get("position") is None:
                        info["position"] = ((b[0] + b[2]) / 2.0, (b[3] + b[1]) / 2.0) if obj == "ball" \
                            else ((b[0] + b[2]) / 2.0, float(b[3]))
                    info["position_local_raw"] = tuple(info["position"])
                    pts_list.append(info["position"])

                pts_arr = np.asarray(pts_list, dtype=np.float64).reshape(-1, 2)

                # Bù trừ chuyển động camera đồng loạt
                if self.motion is not None:
                    adj_pts = self.motion.to_reference(t, pts_arr)
                    for i, info in enumerate(infos):
                        info["position_adjusted"] = tuple(adj_pts[i])
                else:
                    for i, info in enumerate(infos):
                        info["position_adjusted"] = tuple(pts_arr[i])

                if calib_status == "UNREGISTERED":
                    for info in infos:
                        info["projection_valid"] = False
                        info["calibration_status"] = "UNREGISTERED"
                        info["extrapolation_risk"] = None
                        info["is_extrapolated"] = None
                        info["extrapolation_level"] = None
                        info["inside_pitch"] = None
                        info["position_transformed"] = None
                        info["position_sigma"] = None
                    continue

                # Ánh xạ toạ độ đồng loạt (vectorized map_points)
                xy_batch, sg_batch = self.map_points(t, pts_arr)
                risk_batch = self.compute_extrapolation_risk(seg, xy_batch)

                for i, info in enumerate(infos):
                    coords = xy_batch[i]
                    sigma_val = sg_batch[i]
                    is_valid = bool(np.all(np.isfinite(coords)))

                    if is_valid:
                        X, Y = float(coords[0]), float(coords[1])
                        in_pitch = self.is_strictly_inside_pitch(X, Y)
                        r_val = risk_batch[i]
                        risk = float(r_val) if np.isfinite(r_val) else None

                        info["projection_valid"] = True
                        info["calibration_status"] = calib_status
                        info["extrapolation_risk"] = risk
                        info["is_extrapolated"] = bool(risk > 0.0) if risk is not None else None
                        info["extrapolation_level"] = (
                            None if risk is None else ("SAFE" if risk == 0.0 else ("LOW" if risk <= 5.0 else "HIGH"))
                        )
                        info["inside_pitch"] = in_pitch
                        info["position_transformed"] = [X, Y]
                        info["position_sigma"] = float(sigma_val) if np.isfinite(sigma_val) else None
                    else:
                        info["projection_valid"] = False
                        info["calibration_status"] = calib_status
                        info["extrapolation_risk"] = None
                        info["is_extrapolated"] = None
                        info["extrapolation_level"] = None
                        info["inside_pitch"] = None
                        info["position_transformed"] = None
                        info["position_sigma"] = None


# ============================================================================
# Kiểm tra / minh chứng
# ============================================================================
def draw_pitch_overlay(frame: np.ndarray, mapper: PitchMapper, t: int,
                       color: Tuple[int, int, int] = (0, 255, 255), thickness: int = 2) -> np.ndarray:
    """Vẽ đường kẻ sân mô hình lên frame t. Nếu đường vẽ trùng vạch thật: hiệu chuẩn tốt."""
    Ht = mapper.Ht(t)
    if Ht is None:
        return frame
    try:
        Hinv = np.linalg.inv(Ht)
    except Exception:
        return frame

    out = frame.copy()
    for line in pitch_wireframe(mapper.L, mapper.W):
        px, valid = safe_project(Hinv, line)
        for i in range(len(px) - 1):
            if valid[i] and valid[i + 1]:
                cv2.line(out, tuple(np.round(px[i]).astype(int)), tuple(np.round(px[i + 1]).astype(int)),
                         color, thickness, cv2.LINE_AA)
    return out


def landmark_errors(mapper: PitchMapper, annotations: Dict[int, Dict[str, Tuple[float, float]]]):
    """annotations: {frame: {landmark: (x, y)}} - các điểm KHÔNG dùng khi hiệu chuẩn.
    Sai số (mét) giữa vị trí dự đoán và toạ độ chuẩn của landmark."""
    errs, rows = [], []
    for f, pts in annotations.items():
        for name, xy in pts.items():
            pred, _ = mapper.map_points(f, [xy])
            if np.all(np.isfinite(pred[0])):
                e = float(np.linalg.norm(pred[0] - np.asarray(PITCH[name])))
                errs.append(e)
                rows.append((f, name, e))
    e = np.asarray(errs)
    stats = dict(n=len(e), mean_m=float(np.nanmean(e)), median_m=float(np.nanmedian(e)),
                 p95_m=float(np.nanpercentile(e, 95)), max_m=float(np.nanmax(e))) if len(e) else {}
    return stats, rows


# ============================================================================
# Công cụ click điểm (chạy tay trên máy có màn hình)
#   python pitch_homography.py pick video/sample.mp4 --frame 0 --out calib/points.json
# ============================================================================
def _pick(video, frame_idx, out):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Không đọc được frame {frame_idx} của {video}")
    clicks, canvas = [], img.copy()
    done = [False]

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(canvas, str(len(clicks)), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif event == cv2.EVENT_RBUTTONDOWN:
            done[0] = True

    cv2.namedWindow("pick", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("pick", on_mouse)
    print("Click chuột TRÁI để chọn điểm. Click chuột PHẢI để kết thúc.")
    
    while not done[0]:
        cv2.imshow("pick", canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 27, 32, ord("q"), ord("Q")):
            break
        if cv2.getWindowProperty("pick", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    print("Tên landmark hợp lệ:\n  " + ", ".join(sorted(PITCH)))
    data = {}
    for i, (x, y) in enumerate(clicks, 1):
        while True:
            name = input(f"Điểm {i} ({x}, {y}) là landmark nào? (Enter = bỏ qua): ").strip()
            if not name or name in PITCH:
                break
            print("  Tên không hợp lệ.")
        if name:
            data[name] = [x, y]
    try:
        with open(out, "r", encoding="utf-8") as f:
            allp = json.load(f)
    except FileNotFoundError:
        allp = {}
    allp[str(frame_idx)] = data
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(allp, f, indent=1)
    print(f"Đã lưu {len(data)} điểm cho frame {frame_idx} vào {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick"])
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out", default="calib/points.json")
    a = ap.parse_args()
    _pick(a.video, a.frame, a.out)
