"""
pitch_homography.py - Chuẩn hoá pixel -> mét trên sân, thay cho view_transformer.py.

Khác bản cũ:
  1. Hiệu chuẩn bằng các LANDMARK CÓ TÊN trên vạch sân (kích thước FIFA 105 x 68 m),
     không cần nhìn thấy 4 góc sân. Dùng nhiều điểm (>= 6) + bình phương tối thiểu,
     báo cáo sai số fit và sai số leave-one-out (mét) để phát hiện điểm click sai.
  2. Hiệu chuẩn được từ NHIỀU frame: điểm click ở frame t được đưa về hệ tham chiếu
     bằng M_t của camera_motion, nên phủ rộng hơn -> giảm sai số ngoại suy.
  3. Không còn bài test "nằm trong tứ giác 4 điểm" (làm mất dữ liệu khi camera xoay);
     chỉ loại điểm nằm quá xa ngoài sân (margin_m).
  4. Mỗi vị trí có độ bất định position_sigma (mét), tính từ Jacobian của homography:
     cầu thủ ở xa camera thì 1 pixel = nhiều mét hơn. Dùng được cho id_refiner.
  5. Có công cụ kiểm tra: vẽ lại đường kẻ sân lên video, và đo sai số trên các điểm
     giữ lại (hold-out) - đây là minh chứng độ chính xác trên video thật.

Toạ độ sân: x chạy 0..105 từ TRÁI sang PHẢI trong frame hiệu chuẩn; y chạy 0..68 từ
cạnh sân XA camera (trên ảnh) tới cạnh sân GẦN camera (dưới ảnh).
Landmark tiền tố L_ = nửa sân bên trái ảnh, R_ = bên phải.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

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
# Hiệu chuẩn
# ============================================================================
def load_points(path):
    """JSON {"frame": {"landmark": [x, y], ...}, ...} -> {int: {str: (x, y)}}"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): {n: (float(p[0]), float(p[1])) for n, p in v.items()} for k, v in raw.items()}


def _apply_h(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    q = H @ np.vstack([pts.T, np.ones(len(pts))])
    return (q[:2] / q[2]).T


class PitchCalibration:
    def __init__(self, H_by_seg, report):
        self.H_by_seg = H_by_seg      # {seg: H pixel_tham_chiếu -> mét}
        self.report = report

    @staticmethod
    def fit(points_by_frame, motion=None):
        groups = defaultdict(lambda: ([], [], []))
        for f, pts in points_by_frame.items():
            seg = int(motion.seg[f]) if motion is not None else 0
            for name, xy in pts.items():
                if name not in PITCH:
                    raise KeyError(f"Landmark không có trong PITCH: {name}")
                ref = motion.to_reference(f, [xy])[0] if motion is not None else np.asarray(xy, float)
                src, dst, names = groups[seg]
                src.append(ref)
                dst.append(PITCH[name])
                names.append(f"{name}@{f}")

        H_by_seg, report = {}, {}
        for seg, (src, dst, names) in groups.items():
            src, dst = np.asarray(src, float), np.asarray(dst, float)
            if len(src) < 4:
                report[seg] = dict(error=f"Đoạn {seg}: cần >= 4 điểm, hiện có {len(src)}")
                continue
            sv = np.linalg.svd(dst - dst.mean(0), compute_uv=False)
            if sv[-1] / sv[0] < 0.05:
                report[seg] = dict(error=f"Đoạn {seg}: các điểm gần như thẳng hàng trên sân")
                continue
            H, _ = cv2.findHomography(src, dst, 0)
            fit_err = np.linalg.norm(_apply_h(H, src) - dst, axis=1)
            rep = dict(n=len(src), rms_fit_m=float(np.sqrt((fit_err ** 2).mean())),
                       max_fit_m=float(fit_err.max()), worst_fit=names[int(fit_err.argmax())])
            if len(src) >= 6:
                loo = []
                for i in range(len(src)):
                    keep = np.arange(len(src)) != i
                    Hi, _ = cv2.findHomography(src[keep], dst[keep], 0)
                    loo.append(np.linalg.norm(_apply_h(Hi, src[i:i + 1])[0] - dst[i]))
                loo = np.asarray(loo)
                rep.update(loo_mean_m=float(loo.mean()), loo_max_m=float(loo.max()),
                           worst_loo=names[int(loo.argmax())])
            H_by_seg[seg] = H
            report[seg] = rep
        return PitchCalibration(H_by_seg, report)


# ============================================================================
# Ánh xạ
# ============================================================================
class PitchMapper:
    def __init__(self, calib: PitchCalibration, motion, pixel_sigma=2.0, margin_m=3.0,
                 L=PITCH_L, W=PITCH_W):
        self.calib, self.motion = calib, motion
        self.pixel_sigma, self.margin = pixel_sigma, margin_m
        self.L, self.W = L, W

    def Ht(self, t):
        H0 = self.calib.H_by_seg.get(int(self.motion.seg[t]))
        return None if H0 is None else H0 @ self.motion.Ms[t]

    def map_points(self, t, pts):
        """(xy_mét (N,2), sigma_mét (N,)); NaN nếu đoạn của frame t chưa hiệu chuẩn."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        Ht = self.Ht(t)
        if Ht is None:
            return np.full((len(pts), 2), np.nan), np.full(len(pts), np.nan)
        q0 = _apply_h(Ht, pts)
        qx = _apply_h(Ht, pts + [1.0, 0.0])
        qy = _apply_h(Ht, pts + [0.0, 1.0])
        scale = np.maximum(np.linalg.norm(qx - q0, axis=1), np.linalg.norm(qy - q0, axis=1))
        sigma = self.pixel_sigma * scale
        if self.motion.diag[t]["mode"] in ("hold", "init") and t > 0:
            sigma = sigma + 1.0                      # camera mất theo dõi: tăng bất định
        return q0, sigma

    def inside(self, xy):
        m = self.margin
        return (-m <= xy[:, 0]) & (xy[:, 0] <= self.L + m) & (-m <= xy[:, 1]) & (xy[:, 1] <= self.W + m)

    def add_positions_to_tracks(self, tracks):
        """Ghi 'position', 'position_adjusted' (pixel hệ tham chiếu), 'position_transformed'
        ([x, y] mét hoặc None) và 'position_sigma' cho mọi đối tượng, cùng tên khoá như pipeline cũ."""
        for obj, per_frame in tracks.items():
            for t, frame in enumerate(per_frame):
                for info in frame.values():
                    b = info["bbox"]
                    if info.get("position") is None:
                        info["position"] = ((b[0] + b[2]) / 2.0, (b[3] + b[1]) / 2.0) if obj == "ball" \
                            else ((b[0] + b[2]) / 2.0, float(b[3]))
                    p = np.asarray(info["position"], dtype=np.float64)
                    info["position_adjusted"] = tuple(self.motion.to_reference(t, [p])[0])
                    xy, sg = self.map_points(t, [p])
                    ok = bool(np.all(np.isfinite(xy)) and self.inside(xy)[0])
                    info["position_transformed"] = xy[0].tolist() if ok else None
                    info["position_sigma"] = float(sg[0]) if ok else None


# ============================================================================
# Kiểm tra / minh chứng
# ============================================================================
def draw_pitch_overlay(frame, mapper: PitchMapper, t, color=(0, 255, 255), thickness=2):
    """Vẽ đường kẻ sân mô hình lên frame t. Nếu đường vẽ trùng vạch thật: hiệu chuẩn tốt."""
    Ht = mapper.Ht(t)
    if Ht is None:
        return frame
    Hinv = np.linalg.inv(Ht)
    out = frame.copy()
    for line in pitch_wireframe(mapper.L, mapper.W):
        q = Hinv @ np.vstack([line.T, np.ones(len(line))])
        valid = q[2] > 1e-6
        px = (q[:2] / np.where(valid, q[2], 1.0)).T
        valid &= np.all(np.abs(px) < 1e5, axis=1)
        for i in range(len(px) - 1):
            if valid[i] and valid[i + 1]:
                cv2.line(out, tuple(np.round(px[i]).astype(int)), tuple(np.round(px[i + 1]).astype(int)),
                         color, thickness, cv2.LINE_AA)
    return out


def landmark_errors(mapper: PitchMapper, annotations):
    """annotations: {frame: {landmark: (x, y)}} - các điểm KHÔNG dùng khi hiệu chuẩn.
    Sai số (mét) giữa vị trí dự đoán và toạ độ chuẩn của landmark."""
    errs, rows = [], []
    for f, pts in annotations.items():
        for name, xy in pts.items():
            pred, _ = mapper.map_points(f, [xy])
            e = float(np.linalg.norm(pred[0] - np.asarray(PITCH[name])))
            errs.append(e)
            rows.append((f, name, e))
    e = np.asarray(errs)
    stats = dict(n=len(e), mean_m=float(np.nanmean(e)), median_m=float(np.nanmedian(e)),
                 p95_m=float(np.nanpercentile(e, 95)), max_m=float(np.nanmax(e))) if len(e) else {}
    return stats, rows


# ============================================================================
# Công cụ click điểm trực quan với Minimap (chạy tay trên máy có màn hình)
#   python pitch_homography.py pick video/sample.mp4 --frame 0 --out calib/points.json
# ============================================================================
def _get_screen_size():
    """Lấy độ phân giải màn hình để căn giữa các cửa sổ OpenCV."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if sw > 0 and sh > 0:
            return sw, sh
    except Exception:
        pass
    return 1920, 1080


def _open_centered_window(win_name, img, win_w, win_h, sw, sh):
    """Tạo cửa sổ với kích thước cố định và đặt ngay chính giữa màn hình."""
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.resizeWindow(win_name, win_w, win_h)
    x = max(0, (sw - win_w) // 2)
    y = max(0, (sh - win_h) // 2)
    cv2.moveWindow(win_name, x, y)


def _pick_on_minimap(picked_names, sw, sh):
    """Hiển thị Minimap sân bóng gọn gàng ở giữa màn hình, chỉ hiện các chấm điểm (không hiện chữ)."""
    scale = 7.0  # Tỉ lệ gọn gàng: sân rộng ~815x546 px
    pad_x, pad_y = 40, 35
    W_img = int(PITCH_L * scale + 2 * pad_x)
    H_img = int(PITCH_W * scale + 2 * pad_y)

    def m2px(mx, my):
        return int(round(pad_x + mx * scale)), int(round(pad_y + my * scale))

    # Tạo nền sân cỏ
    base = np.full((H_img, W_img, 3), (38, 105, 42), dtype=np.uint8)
    for i in range(10):
        x1 = pad_x + int(i * (PITCH_L / 10.0) * scale)
        x2 = pad_x + int((i + 1) * (PITCH_L / 10.0) * scale)
        if i % 2 == 0:
            cv2.rectangle(base, (x1, pad_y), (x2, pad_y + int(PITCH_W * scale)), (44, 118, 48), -1)

    # Vẽ đường kẻ sân
    for line in pitch_wireframe(PITCH_L, PITCH_W):
        pts = np.array([m2px(p[0], p[1]) for p in line], dtype=np.int32)
        cv2.polylines(base, [pts], False, (235, 235, 235), 2, cv2.LINE_AA)

    # Vẽ khung thành 2 bên
    cy = PITCH_W / 2.0
    gp = 7.32 / 2.0
    cv2.rectangle(base, m2px(-2.2, cy - gp), m2px(0, cy + gp), (180, 180, 255), 2)
    cv2.rectangle(base, m2px(PITCH_L, cy - gp), m2px(PITCH_L + 2.2, cy + gp), (180, 180, 255), 2)

    # Tính toạ độ pixel của 39 landmark trên minimap
    lm_px = {name: m2px(mx, my) for name, (mx, my) in PITCH.items()}

    # Vẽ các chấm tròn landmark (không vẽ bất kỳ chữ nào lên sân)
    for name, (px, py) in lm_px.items():
        if name in picked_names:
            cv2.circle(base, (px, py), 6, (0, 220, 0), -1, cv2.LINE_AA)
            cv2.circle(base, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
        elif "post" in name:
            cv2.circle(base, (px, py), 5, (200, 80, 255), -1, cv2.LINE_AA)
            cv2.circle(base, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(base, (px, py), 5, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(base, (px, py), 6, (255, 255, 255), 1, cv2.LINE_AA)

    state = {"hover": None, "selected": None, "cancelled": False}
    snap_radius = 20.0

    def find_nearest(x, y):
        best_name, best_d = None, snap_radius
        for name, (px, py) in lm_px.items():
            d = float(np.hypot(x - px, y - py))
            if d < best_d:
                best_d = d
                best_name = name
        return best_name

    def on_minimap_mouse(event, x, y, *_):
        state["hover"] = find_nearest(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["hover"] is not None:
                state["selected"] = state["hover"]
        elif event == cv2.EVENT_RBUTTONDOWN:
            state["cancelled"] = True

    win_name = "MINIMAP - Click TRAI chon diem | Click PHAI (hoac ESC) de huy"
    _open_centered_window(win_name, base, W_img, H_img, sw, sh)
    cv2.setMouseCallback(win_name, on_minimap_mouse)

    while state["selected"] is None and not state["cancelled"]:
        view = base.copy()
        hv = state["hover"]
        if hv is not None:
            hx, hy = lm_px[hv]
            # Chỉ vẽ vòng tròn sáng vàng quanh chấm đang trỏ chuột, không vẽ chữ che tầm nhìn
            cv2.circle(view, (hx, hy), 11, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(view, (hx, hy), 4, (0, 255, 255), -1, cv2.LINE_AA)

        cv2.imshow(win_name, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            state["cancelled"] = True
            break
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            state["cancelled"] = True
            break

    cv2.destroyWindow(win_name)
    cv2.waitKey(1)
    return state["selected"]


def _pick(video, frame_idx, out):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Không đọc được frame {frame_idx} của {video}")

    sw, sh = _get_screen_size()
    ih, iw = img.shape[:2]
    # Phóng cửa sổ video vừa khoảng 80% màn hình và giữ nguyên tỉ lệ khung hình
    scale_v = min((sw * 0.80) / iw, (sh * 0.80) / ih)
    win_vw, win_vh = int(iw * scale_v), int(ih * scale_v)

    picked = []
    pending_click = [None]
    done = [False]
    busy_minimap = [False]

    def redraw_video():
        canvas = img.copy()
        for idx, (_, x, y) in enumerate(picked, 1):
            cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, str(idx), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def on_video_mouse(event, x, y, *_):
        if busy_minimap[0]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click[0] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            done[0] = True

    win_video = "VIDEO - Click TRAI chon diem | Z: Xoa diem cuoi | Click PHAI: Luu & Thoat"
    _open_centered_window(win_video, redraw_video(), win_vw, win_vh, sw, sh)
    cv2.setMouseCallback(win_video, on_video_mouse)

    print("Click chuột TRÁI trên Video -> Chọn chấm tương ứng trên Minimap.")
    print("Nhấn phím Z (hoặc Backspace) để xóa điểm vừa chọn. Click chuột PHẢI để Lưu & Kết thúc.")

    while not done[0]:
        canvas = redraw_video()
        cv2.imshow(win_video, canvas)

        if pending_click[0] is not None:
            vx, vy = pending_click[0]
            pending_click[0] = None
            busy_minimap[0] = True

            temp_canvas = canvas.copy()
            cv2.circle(temp_canvas, (vx, vy), 7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win_video, temp_canvas)
            cv2.waitKey(1)

            already_picked_names = {item[0] for item in picked}
            chosen_name = _pick_on_minimap(already_picked_names, sw, sh)
            busy_minimap[0] = False

            if chosen_name is not None:
                picked = [item for item in picked if item[0] != chosen_name]
                picked.append((chosen_name, vx, vy))
                print(f"  + Điểm {len(picked)}: {chosen_name} <- ({vx}, {vy})")

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, ord("q"), ord("Q")):
            break
        elif key in (8, 127, ord("z"), ord("Z")):
            if picked:
                removed = picked.pop()
                print(f"  - Đã xóa: {removed[0]}")
        if cv2.getWindowProperty(win_video, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    data = {name: [x, y] for name, x, y in picked}
    try:
        with open(out, "r", encoding="utf-8") as f:
            allp = json.load(f)
    except FileNotFoundError:
        allp = {}
    allp[str(frame_idx)] = data
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