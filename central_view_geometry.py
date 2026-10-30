"""
central_view_geometry.py - Khai thác Hình học Cung tròn Central Views & Đường giữa sân.

Giai đoạn 4 (Central View Geometry & Downstream Integration):
  1. Mô hình hoá toán học Conic của Vòng tròn giữa sân (Center Circle) bán kính FIFA R = 9.15m
     và Đường giữa sân (Halfway Line) X = 52.5m.
  2. Khớp Elip xạ ảnh (Projective Ellipse Fitting) từ các điểm cung tròn giữa sân bị che khuất
     hoặc quan sát một phần.
  3. Giải bài toán hiệu chuẩn Central View khi các điểm mốc thông thường bị thẳng hàng (collinear):
     - Xác định điểm biến mất (Vanishing point v_h) dọc theo đường giữa sân qua tỉ số kép điều hoà (Harmonic Tetrad).
     - Xác định đường đối cực (Polar line l_polar = C_img @ v_h).
     - Giao cắt đường đối cực với Conic để thu nhận 2 điểm tiếp tuyến cực biên x_L và x_R.
     - Tạo lập hệ 5 điểm mốc chữ thập (Cross-configuration) đầy đủ bậc tự do (rank 8).
  4. Giải toả tính mơ hồ 2 nghiệm (Two-fold Ambiguity Resolution) dựa trên ràng buộc hình học
     khán đài máy quay (camera gantry) và định hướng mặt cỏ FIFA.
  5. Rào chắn góc mở cung tròn (Degenerate Arc Guard): Từ chối an toàn khi cung tròn quá hẹp (< 60 độ)
     hoặc không đủ bậc tự do số học.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import numpy as np

# Thông số sân FIFA chuẩn
PITCH_L: float = 105.0
PITCH_W: float = 68.0
CENTER_X: float = PITCH_L / 2.0   # 52.5 m
CENTER_Y: float = PITCH_W / 2.0   # 34.0 m
CIRCLE_R: float = 9.15            # 9.15 m


class CenterCircleConic:
    """
    Biểu diễn toán học Conic của Vòng tròn giữa sân trên mặt phẳng chuẩn FIFA:
    (X - Xc)^2 + (Y - Yc)^2 = R^2
    dưới dạng ma trận thuần nhất đối xứng C_pitch (3x3).
    """
    Xc: float = CENTER_X
    Yc: float = CENTER_Y
    R: float = CIRCLE_R

    @classmethod
    def get_pitch_conic_matrix(cls) -> np.ndarray:
        """
        Trả về ma trận 3x3 đối xứng C_pitch:
        [  1,   0,  -Xc ]
        [  0,   1,  -Yc ]
        [ -Xc, -Yc, Xc^2 + Yc^2 - R^2 ]
        """
        c33 = cls.Xc**2 + cls.Yc**2 - cls.R**2
        return np.array([
            [1.0, 0.0, -cls.Xc],
            [0.0, 1.0, -cls.Yc],
            [-cls.Xc, -cls.Yc, c33]
        ], dtype=np.float64)

    @classmethod
    def get_halfway_line_pitch(cls) -> np.ndarray:
        """Đường giữa sân trên mặt phẳng: X = 52.5 -> 1*X + 0*Y - 52.5 = 0"""
        return np.array([1.0, 0.0, -cls.Xc], dtype=np.float64)

    @classmethod
    def get_cardinal_points_pitch(cls) -> Dict[str, Tuple[float, float]]:
        """Toạ độ mét 5 điểm chuẩn trên vòng tròn giữa sân."""
        return {
            "center_spot": (cls.Xc, cls.Yc),
            "circle_T": (cls.Xc, cls.Yc - cls.R),
            "circle_B": (cls.Xc, cls.Yc + cls.R),
            "circle_L": (cls.Xc - cls.R, cls.Yc),
            "circle_R": (cls.Xc + cls.R, cls.Yc),
            "halfway_T": (cls.Xc, 0.0),
            "halfway_B": (cls.Xc, PITCH_W)
        }


class EllipseFitter:
    """
    Khớp elip từ các điểm trên ảnh và chuyển đổi sang biểu diễn Conic 3x3.
    """
    @staticmethod
    def fit(points: np.ndarray, min_angular_span_deg: float = 60.0) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Khớp elip từ tập điểm 2D (N, 2).
        Yêu cầu:
          - N >= 5 điểm.
          - Độ bao phủ góc mở (angular span) >= min_angular_span_deg (mặc định 60 độ).
        Trả về:
          - C_img: ma trận đối xứng 3x3 đã chuẩn hoá.
          - params: dict chứa (center, axes, angle_deg, angular_span, residual_rms).
        """
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 5:
            return None

        # Kiểm tra suy biến thẳng hàng sơ bộ
        sv = np.linalg.svd(pts - pts.mean(0), compute_uv=False)
        if sv[0] <= 0 or (sv[-1] / sv[0]) < 0.05:
            return None

        try:
            (cx, cy), (d1, d2), angle = cv2.fitEllipse(pts)
        except Exception:
            return None

        if not (np.isfinite(cx) and np.isfinite(cy) and d1 > 0 and d2 > 0):
            return None

        # Tính góc mở của cung điểm quan sát so với tâm elip
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        angles_sorted = np.sort(angles)
        # Tìm khoảng hở lớn nhất giữa 2 điểm liên tiếp trên vòng tròn
        diffs = np.diff(angles_sorted)
        wrap_diff = (angles_sorted[0] + 2 * np.pi) - angles_sorted[-1]
        max_gap = max(float(np.max(diffs)), float(wrap_diff))
        angular_span = np.rad2deg(2 * np.pi - max_gap)

        if angular_span < min_angular_span_deg:
            return None

        # Chuyển đổi tham số OpenCV sang ma trận Conic 3x3
        a = max(d1, d2) / 2.0
        b = min(d1, d2) / 2.0
        if b / a < 0.05:  # Elip quá dẹt (suy biến thành đoạn thẳng)
            return None

        th_rad = np.deg2rad(angle)
        cos_t, sin_t = np.cos(th_rad), np.sin(th_rad)
        R_mat = np.array([[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        T_mat = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        M = R_mat @ T_mat

        C_can = np.diag([1.0 / (d1 / 2.0)**2, 1.0 / (d2 / 2.0)**2, -1.0])
        C_img = M.T @ C_can @ M
        norm_val = np.linalg.norm(C_img)
        if norm_val <= 0 or not np.all(np.isfinite(C_img)):
            return None
        C_img = C_img / norm_val

        # Tính sai số hình học (khoảng cách điểm tới elip xấp xỉ)
        pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
        alg_res = np.abs(np.sum(pts_homo * (C_img @ pts_homo.T).T, axis=1))

        params = {
            "center": (float(cx), float(cy)),
            "axes": (float(d1), float(d2)),
            "angle_deg": float(angle),
            "angular_span_deg": float(angular_span),
            "rms_algebraic": float(np.sqrt(np.mean(alg_res**2)))
        }
        return C_img, params


def extract_circle_arc_points_from_image(
    image: np.ndarray,
    xT: np.ndarray,
    xB: np.ndarray,
    xC: np.ndarray,
    sample_stride: int = 10,
    intensity_threshold: int = 170,
    sat_threshold: int = 80,
    horiz_factor: float = 4.8
) -> Optional[np.ndarray]:
    """
    Trích xuất các điểm biên thuộc cung tròn giữa sân từ khung hình ảnh thực tế.
    Sử dụng vị trí mốc đường giữa sân (circle_T, circle_B, center_spot) để giới hạn
    vùng tìm kiếm (ROI) xung quanh vòng tròn giữa sân, loại trừ dải vạch giữa sân
    để chỉ thu thập các điểm nằm trên cung tròn.

    LƯU Ý KỸ THUẬT QUAN TRỌNG:
    Trong camera truyền hình thể thao góc nghiêng (oblique angle), elip vòng tròn bị co ngắn
    chiều dọc (semi-minor axis b ~ r_approx) nhưng giữ nguyên hoặc mở rộng theo chiều ngang
    (semi-major axis a ~ 4.0 - 4.5 * r_approx). Do đó, horiz_factor phải đủ lớn (>= 4.5)
    để bao trùm toàn bộ bề ngang của elip, tránh hiện tượng cắt cụt nhánh trái/phải làm lệch tâm.
    """
    if image is None or len(image.shape) != 3:
        return None

    H_img, W_img = image.shape[:2]
    # Bán kính xấp xỉ trên ảnh theo trục dọc (semi-minor axis b)
    r_approx = np.linalg.norm(xB - xT) / 2.0
    if r_approx < 10.0:
        return None

    # Bounding box dự kiến của vòng tròn giữa sân với hệ số mở rộng ngang horiz_factor
    cx, cy = float(xC[0]), float(xC[1])
    x_min = max(0, int(cx - horiz_factor * r_approx))
    x_max = min(W_img, int(cx + horiz_factor * r_approx))
    y_min = max(0, int(min(xT[1], xB[1]) - 0.25 * r_approx))
    y_max = min(H_img, int(max(xT[1], xB[1]) + 0.25 * r_approx))

    if x_max <= x_min or y_max <= y_min:
        return None

    roi = image[y_min:y_max, x_min:x_max]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Vạch sơn trắng trên mặt cỏ: cường độ sáng cao và độ bão hoà màu thấp
    white_mask = (gray_roi >= intensity_threshold) & (hsv_roi[:, :, 1] <= sat_threshold)

    # Loại bỏ dải vạch giữa sân (halfway line strip) đi qua xT, xC, xB
    halfway_strip_half_w = max(4, int(0.12 * r_approx))
    local_cx = int(cx - x_min)
    col_start = max(0, local_cx - halfway_strip_half_w)
    col_end = min(roi.shape[1], local_cx + halfway_strip_half_w)
    white_mask[:, col_start:col_end] = False

    ys, xs = np.where(white_mask)
    if len(xs) < 20:
        return None

    # Chuyển toạ độ cục bộ trong ROI về toạ độ ảnh toàn cục
    global_xs = xs + x_min
    global_ys = ys + y_min
    all_pts = np.column_stack([global_xs, global_ys])

    # Lấy mẫu đều để tránh quá tải điểm và giảm nhiễu cụm
    if len(all_pts) > 150:
        step = max(1, len(all_pts) // 150)
        all_pts = all_pts[::step]

    return all_pts.astype(np.float32)


class CentralViewHomographySolver:
    """
    Thuật toán Giải quyết Hiệu chuẩn Central Views (Khung hình Trung tâm Sân bóng).
    Khai thác quan hệ xạ ảnh giữa Vòng tròn giữa sân và Đường giữa sân để khắc phục
    triệt để lỗi suy biến thẳng hàng (collinearity) của phương pháp DLT truyền thống.
    """
    def __init__(self, min_arc_span_deg: float = 60.0, reproj_threshold_m: float = 2.0):
        self.min_arc_span = min_arc_span_deg
        self.threshold_m = reproj_threshold_m
        self.pitch_conic = CenterCircleConic.get_pitch_conic_matrix()
        self.cardinals_pitch = CenterCircleConic.get_cardinal_points_pitch()

    def solve_from_landmarks_and_arc(
        self,
        landmarks: Dict[str, Tuple[float, float]],
        arc_points: Optional[np.ndarray] = None,
        frame_image: Optional[np.ndarray] = None,
        image_shape: Tuple[int, int] = (1080, 1920),
        allow_inverted_view: bool = False
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Ước lượng Homography H (pixel -> pitch mét) trong trường hợp Central View:
        Đầu vào:
          - landmarks: dict toạ độ pixel của các điểm quan sát được (ví dụ: circle_T, center_spot, circle_B, halfway_T, v.v.).
          - arc_points: (K, 2) mảng toạ độ pixel của các điểm trên cung tròn giữa sân (nếu có).
          - frame_image: (H, W, 3) ảnh khung hình video gốc (tuỳ chọn, dùng để trích xuất cung tròn nếu arc_points chưa có).
          - allow_inverted_view: bool (mặc định False, áp dụng Broadcast Gantry Prior nhìn từ khán đài).
        Trả về:
          - (H, report_dict) nếu thành công, hoặc (None, error_dict) nếu suy biến/thất bại.
        """
        report: Dict[str, Any] = {
            "method": "central_view_conic",
            "observed_landmarks": sorted(list(landmarks.keys())),
            "derived_points": {}
        }

        # 1. Kiểm tra các điểm trên đường giữa sân
        has_T = "circle_T" in landmarks
        has_B = "circle_B" in landmarks
        has_C = "center_spot" in landmarks

        # Thu thập các điểm đã biết
        src_points = []
        dst_points = []
        names = []

        # Kiểm tra nếu đã có sẵn cả circle_L và circle_R trong landmarks
        if "circle_L" in landmarks and "circle_R" in landmarks and has_T and has_B:
            # Đã có đủ 4 điểm cực biên chữ thập không thẳng hàng!
            for k in ["circle_T", "circle_B", "center_spot", "circle_L", "circle_R", "halfway_T", "halfway_B"]:
                if k in landmarks:
                    src_points.append(landmarks[k])
                    dst_points.append(self.cardinals_pitch[k])
                    names.append(k)

            src_arr = np.asarray(src_points, dtype=np.float64)
            dst_arr = np.asarray(dst_points, dtype=np.float64)
            report["arc_source"] = "cardinal_landmarks_present"
            return self._fit_and_resolve(src_arr, dst_arr, names, report, image_shape, allow_inverted_view=allow_inverted_view)

        # 2. Nếu thiếu circle_L hoặc circle_R (tình huống điển hình khi DLT thất bại do thẳng hàng):
        # Yêu cầu bắt buộc: Phải có 3 điểm thẳng hàng (circle_T, center_spot, circle_B) để lập hàng điểm điều hoà
        if not (has_T and has_B and has_C):
            report["error"] = "Central View yêu cầu tối thiểu 3 điểm circle_T, center_spot, circle_B trên đường giữa sân."
            return None, report

        xT = np.asarray(landmarks["circle_T"], dtype=np.float64)
        xB = np.asarray(landmarks["circle_B"], dtype=np.float64)
        xC = np.asarray(landmarks["center_spot"], dtype=np.float64)

        # 3. Khớp Conic C_img của Vòng tròn giữa sân
        C_img = None
        # 3a. Trích xuất arc points từ frame_image nếu chưa có arc_points
        if arc_points is None and frame_image is not None:
            extracted_arc = extract_circle_arc_points_from_image(frame_image, xT, xB, xC)
            if extracted_arc is not None and len(extracted_arc) >= 5:
                arc_points = extracted_arc
                report["arc_source"] = "extracted_from_image"
                report["n_arc_points"] = len(extracted_arc)

        if arc_points is not None and len(arc_points) >= 5:
            fit_res = EllipseFitter.fit(arc_points, min_angular_span_deg=self.min_arc_span)
            if fit_res is not None:
                C_img, ell_params = fit_res
                report["ellipse_params"] = ell_params
                if "arc_source" not in report:
                    report["arc_source"] = "provided_arc_points"
                    report["n_arc_points"] = len(arc_points)

        # 3b. Nếu không có arc_points rời rạc nhưng có các mốc cung trong landmarks:
        if C_img is None:
            circle_pts = [xT, xB]
            if "circle_L" in landmarks:
                circle_pts.append(np.asarray(landmarks["circle_L"], dtype=np.float64))
            if "circle_R" in landmarks:
                circle_pts.append(np.asarray(landmarks["circle_R"], dtype=np.float64))
            for k, pt in landmarks.items():
                if "arc" in k:
                    circle_pts.append(np.asarray(pt, dtype=np.float64))

            if len(circle_pts) >= 5:
                fit_res = EllipseFitter.fit(np.asarray(circle_pts), min_angular_span_deg=self.min_arc_span)
                if fit_res is not None:
                    C_img, ell_params = fit_res
                    report["ellipse_params"] = ell_params
                    report["arc_source"] = "landmark_arc_points"

        if C_img is None:
            report["error"] = (
                f"Không thể fit elip vòng tròn giữa sân (cần arc_points hoặc frame_image để trích xuất cung tròn với góc mở >= {self.min_arc_span}°). "
                f"Năm điểm mốc thẳng hàng trên đường giữa sân không thể tự cung cấp hình dạng elip."
            )
            report["status"] = "REJECTED_MISSING_ARC"
            return None, report

        # 4. Xác định điểm biến mất (Vanishing Point v_h) dọc đường giữa sân qua Hàng điểm điều hoà:
        # Cross-ratio CR(xT, xB; xC, v_h) = -1
        d_line = xB - xT
        len_sq = float(np.dot(d_line, d_line))
        if len_sq < 1.0:
            report["error"] = "circle_T và circle_B trùng nhau trên ảnh."
            return None, report

        lam_C = float(np.dot(xC - xT, d_line) / len_sq)
        denom_inf = 2.0 * lam_C - 1.0
        if abs(denom_inf) < 1e-6:
            # Phối cảnh affine (tâm nằm đúng chính giữa) -> v_h ở vô tận
            vh_homo = np.array([d_line[0], d_line[1], 0.0], dtype=np.float64)
        else:
            lam_inf = lam_C / denom_inf
            vh = (1.0 - lam_inf) * xT + lam_inf * xB
            vh_homo = np.array([vh[0], vh[1], 1.0], dtype=np.float64)

        # 5. Đường đối cực l_polar của v_h đối với Conic C_img:
        # l_polar = C_img @ v_h
        l_polar = C_img @ vh_homo
        norm_lp = np.linalg.norm(l_polar[:2])
        if norm_lp < 1e-9:
            report["error"] = "Đường đối cực của v_h bị suy biến."
            return None, report
        l_polar = l_polar / norm_lp

        # 6. Giao cắt đường đối cực l_polar với Conic C_img để tìm 2 điểm tiếp tuyến xL và xR:
        # Đường thẳng đi qua xC với vector chỉ phương d_vec vuông góc với pháp tuyến l_polar
        d_vec = np.array([-l_polar[1], l_polar[0]], dtype=np.float64)
        d_vec = d_vec / np.linalg.norm(d_vec)

        xC_homo = np.array([xC[0], xC[1], 1.0], dtype=np.float64)
        D_homo = np.array([d_vec[0], d_vec[1], 0.0], dtype=np.float64)

        A_quad = float(D_homo @ C_img @ D_homo)
        B_quad = float(2.0 * (xC_homo @ C_img @ D_homo))
        C_quad = float(xC_homo @ C_img @ xC_homo)

        disc = B_quad**2 - 4.0 * A_quad * C_quad
        if disc <= 0:
            report["error"] = "Đường đối cực không cắt elip vòng tròn giữa sân (biện luận hình học vô nghiệm)."
            return None, report

        sqrt_disc = np.sqrt(disc)
        t1 = (-B_quad + sqrt_disc) / (2.0 * A_quad)
        t2 = (-B_quad - sqrt_disc) / (2.0 * A_quad)

        pt1 = xC + t1 * d_vec
        pt2 = xC + t2 * d_vec

        # Phân định đâu là circle_L và đâu là circle_R:
        # Trên sân bóng FIFA nhìn từ khán đài chính, trục X tăng từ trái qua phải
        pt_left, pt_right = (pt1, pt2) if pt1[0] < pt2[0] else (pt2, pt1)

        # 7. Xây dựng tập 5 điểm chữ thập hoàn chỉnh
        # Giữ riêng biệt các mốc quan sát trực tiếp và các điểm suy ra bằng hình học
        src_points = [xT, xB, xC, pt_left, pt_right]
        dst_points = [
            self.cardinals_pitch["circle_T"],
            self.cardinals_pitch["circle_B"],
            self.cardinals_pitch["center_spot"],
            self.cardinals_pitch["circle_L"],
            self.cardinals_pitch["circle_R"]
        ]
        names = ["circle_T", "circle_B", "center_spot", "circle_L_conic", "circle_R_conic"]

        # Thêm halfway_T và halfway_B nếu có trong landmarks quan sát
        if "halfway_T" in landmarks:
            src_points.append(landmarks["halfway_T"])
            dst_points.append(self.cardinals_pitch["halfway_T"])
            names.append("halfway_T")
        if "halfway_B" in landmarks:
            src_points.append(landmarks["halfway_B"])
            dst_points.append(self.cardinals_pitch["halfway_B"])
            names.append("halfway_B")

        src_arr = np.asarray(src_points, dtype=np.float64)
        dst_arr = np.asarray(dst_points, dtype=np.float64)

        report["derived_points"] = {
            "circle_L": [float(pt_left[0]), float(pt_left[1])],
            "circle_R": [float(pt_right[0]), float(pt_right[1])]
        }
        report["conic_recovered_points"] = report["derived_points"]

        return self._fit_and_resolve(
            src_arr, dst_arr, names, report, image_shape,
            allow_inverted_view=allow_inverted_view,
            raw_arc_points=arc_points
        )

    def _fit_and_resolve(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        names: List[str],
        report: Dict[str, Any],
        image_shape: Tuple[int, int],
        allow_inverted_view: bool = False,
        raw_arc_points: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Fit Homography từ tập điểm đầy đủ bậc tự do và Giải toả tính mơ hồ 2 nghiệm (Two-fold Ambiguity).
        """
        flag = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
        H, inlier_mask = cv2.findHomography(src, dst, flag, self.threshold_m, maxIters=3000, confidence=0.999)

        if H is None or not np.all(np.isfinite(H)):
            report["error"] = "USAC_MAGSAC không tìm được nghiệm Homography từ các điểm conic."
            return None, report

        # -------------------------------------------------------------
        # GIẢI TOẢ TÍNH MƠ HỒ 2 NGHIỆM & RÀO CHẮN MẪU SỐ BẤT BIẾN SCALE/SIGN
        # -------------------------------------------------------------
        # Kiểm tra phép chiếu thuận src -> pitch
        src_homo = np.hstack([src, np.ones((len(src), 1), dtype=np.float64)])
        proj_pitch = (H @ src_homo.T).T
        w_vals = proj_pitch[:, 2]

        # 1. Kiểm tra tính nhất quán dấu của mẫu số (Cheirality / Line at infinity guard):
        # Tất cả các điểm nhìn thấy phải cùng nằm ở một phía của đường chân trời (cùng dấu w_i).
        pos_count = int(np.sum(w_vals > 1e-7))
        neg_count = int(np.sum(w_vals < -1e-7))

        if pos_count > 0 and neg_count > 0:
            report["error"] = "Nghiệm Homography cắt qua đường chân trời trong vùng quan sát (mẫu số w_i đổi dấu)."
            return None, report

        # 2. Bất biến xạ ảnh H ~ -H:
        # Nếu toàn bộ mẫu số âm, đổi dấu ma trận H -> -H để đưa về biểu diễn chuẩn tắc w_i > 0
        if neg_count > 0:
            H = -H
            proj_pitch = -proj_pitch
            w_vals = -w_vals

        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
        else:
            H = H / np.linalg.norm(H)

        proj_pitch = proj_pitch[:, :2] / proj_pitch[:, 2:3]
        fit_errors = np.linalg.norm(proj_pitch - dst, axis=1)
        rms_fit = float(np.sqrt(np.mean(fit_errors**2)))

        if rms_fit > 3.0:
            report["error"] = f"Sai số khớp trên sân quá cao (RMS = {rms_fit:.2f}m > 3.0m)."
            return None, report

        # 3. Tiền nghiệm góc quay khán đài truyền hình (Standard Broadcast Gantry Prior):
        # Camera truyền hình đặt trên cao nhìn chúc xuống:
        # - circle_T (nửa sân xa) có Y trên sân nhỏ hơn circle_B (nửa sân gần).
        # - circle_L (nửa sân trái) có X trên sân nhỏ hơn circle_R (nửa sân phải).
        if not allow_inverted_view:
            idx_T = names.index("circle_T") if "circle_T" in names else None
            idx_B = names.index("circle_B") if "circle_B" in names else None
            if idx_T is not None and idx_B is not None:
                if proj_pitch[idx_T, 1] >= proj_pitch[idx_B, 1]:
                    report["error"] = "Nghiệm Homography bị lộn ngược hướng sân (nghiệm đối xứng giả bị loại bỏ theo Broadcast Gantry Prior)."
                    return None, report

        # Tính số điều kiện Hartley
        from pitch_homo import compute_hartley_condition_number, safe_project
        cond = compute_hartley_condition_number(H, src, dst)
        if cond > 1e6:
            report["error"] = f"Số điều kiện Hartley quá cao (cond = {cond:.1e}), nghiệm kém ổn định số học."
            return None, report

        # -------------------------------------------------------------
        # 4. PHÂN ĐỊNH RẠCH RÒI 2 VÙNG HỖ TRỢ HIỆU CHUẨN:
        #    - Observed Support: Vùng hỗ trợ quan sát thực tế
        #    - Model-Derived Support: Vùng hỗ trợ mô hình hình học suy ra
        # -------------------------------------------------------------
        observed_names = report.get("observed_landmarks", [])

        # 4a. Observed Support: Chỉ gồm các điểm landmark THỰC TẾ quan sát được
        #     và các điểm cung tròn thực sự được phát hiện từ ảnh chiếu sang sân
        observed_pts_list = []
        for k in observed_names:
            if k in self.cardinals_pitch:
                observed_pts_list.append(self.cardinals_pitch[k])

        if raw_arc_points is not None and len(raw_arc_points) > 0:
            proj_arc, val_arc = safe_project(H, raw_arc_points)
            if np.any(val_arc):
                valid_proj = proj_arc[val_arc]
                # Lọc ngoại lai robust (Robust Radial Inlier Filter):
                # Chỉ giữ các điểm arc có khoảng cách tới tâm vòng tròn giữa sân (52.5, 34.0)
                # nằm trong dải dung sai vật lý hợp lý xung quanh bán kính chuẩn R=9.15m (dung sai 1.5m
                # bao trùm bề rộng nét vẽ vạch sơn 12cm và độ không chắc chắn do góc nghiêng phối cảnh).
                # Loại bỏ dứt điểm các điểm nhiễu ngoại lai (giày/tất cầu thủ, biển quảng cáo, bóng phản xạ).
                center_pitch = np.array([CENTER_X, CENTER_Y], dtype=np.float64)
                radial_dists = np.linalg.norm(valid_proj - center_pitch, axis=1)
                radial_inliers = np.abs(radial_dists - CIRCLE_R) <= 1.5
                n_inliers = int(np.sum(radial_inliers))
                report["n_arc_inliers"] = n_inliers
                report["n_arc_outliers"] = int(len(valid_proj) - n_inliers)
                if n_inliers >= 5:
                    observed_pts_list.extend(valid_proj[radial_inliers].tolist())
                else:
                    observed_pts_list.extend(valid_proj.tolist())

        observed_support_arr = np.asarray(observed_pts_list, dtype=np.float64)

        # 4b. Model-Derived Support: Gồm 32 điểm trên chu vi vòng tròn FIFA 9.15m
        #     (đa giác 32 cạnh có diện tích lý thuyết chuẩn xác = 261.34 m2)
        #     cộng các điểm mốc đường giữa sân ĐÃ QUAN SÁT THỰC SỰ
        model_pts_list = []
        for k in ["halfway_T", "halfway_B", "circle_T", "circle_B", "center_spot"]:
            if k in observed_names and k in self.cardinals_pitch:
                model_pts_list.append(self.cardinals_pitch[k])
        thetas = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        for th in thetas:
            model_pts_list.append((CENTER_X + CIRCLE_R * np.cos(th), CENTER_Y + CIRCLE_R * np.sin(th)))
        model_support_arr = np.asarray(model_pts_list, dtype=np.float64)

        report.update({
            "status": "SUCCESS",
            "n_points": len(src),
            "rms_fit_m": rms_fit,
            "mean_fit_m": float(np.mean(fit_errors)),
            "max_fit_m": float(np.max(fit_errors)),
            "hartley_condition": float(cond),
            "two_fold_ambiguity_resolved": True,
            "support_points_pitch": observed_support_arr,
            "observed_support_points_pitch": observed_support_arr,
            "model_derived_support_points_pitch": model_support_arr,
            "theoretical_circle_32gon_area_m2": 261.34
        })
        return H, report
