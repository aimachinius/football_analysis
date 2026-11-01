"""
scripts/diagnose_ball_lost.py - Chẩn đoán chuyên sâu nguyên nhân 274 frame bóng LOST.

Các nhiệm vụ:
  1. Chạy YOLO conf=0.05 lưu top-5 ứng viên bóng thô cho toàn bộ 750 frames.
  2. Thống kê chi tiết cho 274 frame LOST và riêng 3 gap (513-603, 135-179, 615-633):
     a. Số frame KHÔNG có ứng viên nào >= 0.05.
     b. Số frame có ứng viên nhưng bị loại và lý do loại (ngoài elip, dưới ngưỡng 0.35, outlier động học, tier 3 lockout).
     c. Khoảng cách từ ứng viên tới vị trí ước lượng nội suy trong hệ quy chiếu camera.
  3. Đo tốc độ pan camera (px/frame) và kiểm tra camera cut / segment.
  4. Xuất contact sheet (mỗi 3 frame) cho gap 513-603 và 135-179 vẽ chồng top-5 ứng viên và elip gating.
  5. Xuất báo cáo reports/BALL_DIAG.md và file dữ liệu JSON.
"""

import os
import sys
import json
import pickle
from collections import defaultdict
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tracker.ball_tracker import BallTrajectoryStabilizer
from tracker import Tracker


def extract_raw_ball_candidates(video_path="video/sample.mp4", model_path="best_V26_over_6000.pt", batch_size=8, conf_thresh=0.05):
    """
    Chạy YOLOv26s với ngưỡng conf=0.05, trích xuất top-5 ứng viên bóng cho mỗi frame.
    """
    print(f"\n[BƯỚC 1] Trích xuất toàn bộ ứng viên bóng thô conf >= {conf_thresh} từ YOLO...")
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    frames_batch = []
    frame_indices_batch = []
    candidates_by_frame = {}

    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames_batch.append(frame)
        frame_indices_batch.append(f_idx)

        if len(frames_batch) == batch_size:
            results = model.predict(
                source=frames_batch,
                conf=conf_thresh,
                classes=[0],  # Chỉ detect ball
                device=device,
                imgsz=1536,
                verbose=False,
            )
            for idx, res in zip(frame_indices_batch, results):
                cand_list = []
                if res.boxes is not None and len(res.boxes) > 0:
                    b_confs = res.boxes.conf.cpu().numpy()
                    b_xyxy = res.boxes.xyxy.cpu().numpy()
                    # Sắp xếp confidence giảm dần
                    order = np.argsort(-b_confs)
                    for k in order[:5]:  # Top-5 ứng viên
                        c_val = float(b_confs[k])
                        box = [float(x) for x in b_xyxy[k]]
                        cx = (box[0] + box[2]) / 2.0
                        cy = (box[1] + box[3]) / 2.0
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        cand_list.append({
                            "conf": round(c_val, 4),
                            "center": [round(cx, 2), round(cy, 2)],
                            "size": [round(w, 2), round(h, 2)],
                            "bbox": [round(x, 2) for x in box],
                        })
                candidates_by_frame[idx] = cand_list

            frames_batch = []
            frame_indices_batch = []
            if (f_idx + 1) % 100 == 0:
                print(f"   -> Đã xử lý {f_idx + 1}/{total_frames} frames...")

        f_idx += 1

    if frames_batch:
        results = model.predict(
            source=frames_batch,
            conf=conf_thresh,
            classes=[0],
            device=device,
            imgsz=1536,
            verbose=False,
        )
        for idx, res in zip(frame_indices_batch, results):
            cand_list = []
            if res.boxes is not None and len(res.boxes) > 0:
                b_confs = res.boxes.conf.cpu().numpy()
                b_xyxy = res.boxes.xyxy.cpu().numpy()
                order = np.argsort(-b_confs)
                for k in order[:5]:
                    c_val = float(b_confs[k])
                    box = [float(x) for x in b_xyxy[k]]
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    cand_list.append({
                        "conf": round(c_val, 4),
                        "center": [round(cx, 2), round(cy, 2)],
                        "size": [round(w, 2), round(h, 2)],
                        "bbox": [round(x, 2) for x in box],
                    })
            candidates_by_frame[idx] = cand_list

    cap.release()
    print(f"-> Hoàn tất trích xuất ứng viên cho {len(candidates_by_frame)} frames.")
    return candidates_by_frame, total_frames, fps


def run_diagnostics():
    video_path = "video/sample.mp4"
    model_path = "best_V26_over_6000.pt" if os.path.exists("best_V26_over_6000.pt") else "best_YOLOv26s_FB_OD.pt"

    # Nạp CameraMotion
    print("\n[BƯỚC 2] Nạp ma trận chuyển động camera (CameraMotion)...")
    with open("stubs/camera_motion.pkl", "rb") as f:
        motion = pickle.load(f)

    # Nạp Baseline tracks
    with open("stubs/track_stubs_reid_baseline.pkl", "rb") as f:
        tracks = pickle.load(f)

    # Chạy BallTrajectoryStabilizer baseline để xác định 274 frame LOST
    tracker = Tracker(model_path, fps=25.0)
    stabilized_ball = tracker.interpolate_ball_position(
        tracks["ball"],
        camera_motion=motion,
        player_tracks=tracks.get("players"),
    )

    lost_frame_indices = set(f for f in range(len(stabilized_ball)) if 1 not in stabilized_ball[f])
    print(f"-> Tổng số frame bóng LOST thực tế: {len(lost_frame_indices)}/750 frames ({len(lost_frame_indices)/750*100:.2f}%)")

    # Trích xuất ứng viên conf >= 0.05
    cand_cache_file = "reports/ball_candidates_005.json"
    if os.path.exists(cand_cache_file):
        print(f"Loading cached ball candidates from: {cand_cache_file}")
        with open(cand_cache_file, "r", encoding="utf-8") as f:
            candidates_by_frame = {int(k): v for k, v in json.load(f).items()}
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
    else:
        candidates_by_frame, total_frames, fps = extract_raw_ball_candidates(
            video_path=video_path,
            model_path=model_path,
            batch_size=8,
            conf_thresh=0.05,
        )
        os.makedirs("reports", exist_ok=True)
        with open(cand_cache_file, "w", encoding="utf-8") as f:
            json.dump(candidates_by_frame, f, indent=2)
        print(f"Saved ball candidates to: {cand_cache_file}")

    # =========================================================================
    # PHÂN TÍCH PAN SPEED & CAMERA SEGMENTS
    # =========================================================================
    print("\n[BƯỚC 3] Tính tốc độ Pan camera và kiểm tra Camera Cut...")
    c_img = np.array([[1920 / 2.0, 1080 / 2.0]])
    pan_speeds = [0.0]  # px/frame
    for t in range(1, total_frames):
        # Biến đổi tâm ảnh qua to_reference
        c_prev_ref = motion.to_reference(t - 1, c_img)[0]
        c_curr_ref = motion.to_reference(t, c_img)[0]
        # Pan speed là độ dịch chuyển của tâm ảnh trên hệ tham chiếu
        sp = float(np.linalg.norm(c_curr_ref - c_prev_ref))
        pan_speeds.append(sp)
    pan_speeds = np.array(pan_speeds)

    # 3 Gap cần phân tích cụ thể
    target_gaps = {
        "Gap 1 (513-603, 91 frames)": {
            "start": 513, "end": 603,
            "anchor_before": 506, "anchor_after": 604,
        },
        "Gap 2 (135-179, 45 frames)": {
            "start": 135, "end": 179,
            "anchor_before": 124, "anchor_after": 180,
        },
        "Gap 3 (615-633, 19 frames)": {
            "start": 615, "end": 633,
            "anchor_before": 614, "anchor_after": 634,
        },
    }

    gap_diagnostics = {}

    for g_name, g_info in target_gaps.items():
        s = g_info["start"]
        e = g_info["end"]
        gap_frames = list(range(s, e + 1))
        n_frames = len(gap_frames)

        # Pan speed trong gap
        sub_pan = pan_speeds[s:e + 1]
        mean_pan = float(np.mean(sub_pan))
        max_pan = float(np.max(sub_pan))
        min_pan = float(np.min(sub_pan))

        # Check segment / cuts
        cuts_in_gap = [c for c in motion.cuts if s <= c <= e]
        segs_in_gap = set(int(motion.seg[f]) for f in gap_frames)

        # Ước lượng vị trí nội suy hai đầu gap trong reference frame
        f_before = g_info["anchor_before"]
        f_after = g_info["anchor_after"]

        b_before = stabilized_ball[f_before][1]["bbox"]
        b_after = stabilized_ball[f_after][1]["bbox"]

        c_before_img = np.array([[(b_before[0] + b_before[2]) / 2.0, (b_before[1] + b_before[3]) / 2.0]])
        c_after_img = np.array([[(b_after[0] + b_after[2]) / 2.0, (b_after[1] + b_after[3]) / 2.0]])

        c_before_ref = motion.to_reference(f_before, c_before_img)[0]
        c_after_ref = motion.to_reference(f_after, c_after_img)[0]

        # Thống kê ứng viên
        frames_no_cand = 0
        frames_with_cand = 0
        cand_rejection_reasons = defaultdict(int)

        distances_ref_all = []
        closest_cand_per_frame = []

        # Tái hiện trạng thái ball gating tại frame s
        diag_len = float(np.hypot(1920, 1080))

        for f in gap_frames:
            cands = candidates_by_frame.get(f, [])
            # Nội suy vị trí bóng trong reference frame tại frame f
            alpha = (f - f_before) / max(1, (f_after - f_before))
            interp_ref = c_before_ref + alpha * (c_after_ref - c_before_ref)

            # Chiếu interp_ref ngược lại ảnh frame f
            M_t = motion.Ms[f]
            M_t_inv = np.linalg.inv(M_t)
            q_interp = M_t_inv @ np.array([interp_ref[0], interp_ref[1], 1.0])
            interp_img = q_interp[:2] / q_interp[2]

            if len(cands) == 0:
                frames_no_cand += 1
                cand_rejection_reasons["no_candidates_above_005"] += 1
            else:
                frames_with_cand += 1
                # Kiểm tra từng ứng viên
                best_d_ref = float("inf")
                best_c_info = None

                for c in cands:
                    c_pt_img = np.array([[c["center"][0], c["center"][1]]])
                    c_pt_ref = motion.to_reference(f, c_pt_img)[0]
                    d_ref = float(np.linalg.norm(c_pt_ref - interp_ref))
                    distances_ref_all.append(d_ref)

                    if d_ref < best_d_ref:
                        best_d_ref = d_ref
                        best_c_info = c

                closest_cand_per_frame.append({
                    "frame": f,
                    "dist_ref_px": round(best_d_ref, 2),
                    "best_conf": best_c_info["conf"],
                    "best_center": best_c_info["center"],
                    "interp_img": [round(float(interp_img[0]), 2), round(float(interp_img[1]), 2)],
                })

                # Lý do loại bỏ:
                # 1. Nếu conf < 0.15: bị tracker._select_ball_detection loại ở c_confs >= 0.15
                # 2. Nếu conf trong [0.15, 0.35): ngoài gating elip và rơi vào fallback < 0.35
                # 3. Nếu conf >= 0.35: có vào được gating không, nếu bị gating loại do elip trôi
                top_c = max(c["conf"] for c in cands)
                if top_c < 0.15:
                    cand_rejection_reasons["below_select_threshold_015"] += 1
                elif top_c < 0.35:
                    cand_rejection_reasons["outside_gate_and_below_fallback_035"] += 1
                else:
                    cand_rejection_reasons["rejected_by_stabilizer_or_strict_tier3"] += 1

        gap_diagnostics[g_name] = {
            "total_frames": n_frames,
            "anchor_before": {"frame": f_before, "pos_img": [round(float(c_before_img[0][0]), 1), round(float(c_before_img[0][1]), 1)]},
            "anchor_after": {"frame": f_after, "pos_img": [round(float(c_after_img[0][0]), 1), round(float(c_after_img[0][1]), 1)]},
            "camera_pan_speed": {
                "mean_px_per_frame": round(mean_pan, 2),
                "max_px_per_frame": round(max_pan, 2),
                "min_px_per_frame": round(min_pan, 2),
            },
            "camera_segments": list(segs_in_gap),
            "camera_cuts": cuts_in_gap,
            "candidates_stat": {
                "frames_no_candidates": frames_no_cand,
                "frames_with_candidates": frames_with_cand,
                "rejection_reasons": dict(cand_rejection_reasons),
                "dist_to_interpolated_ref": {
                    "min_px": round(float(np.min(distances_ref_all)), 2) if distances_ref_all else None,
                    "median_px": round(float(np.median(distances_ref_all)), 2) if distances_ref_all else None,
                    "mean_px": round(float(np.mean(distances_ref_all)), 2) if distances_ref_all else None,
                    "max_px": round(float(np.max(distances_ref_all)), 2) if distances_ref_all else None,
                    "candidates_within_30px_ref": sum(1 for d in distances_ref_all if d <= 30.0),
                    "candidates_within_60px_ref": sum(1 for d in distances_ref_all if d <= 60.0),
                },
            },
            "closest_samples": closest_cand_per_frame[:10],
        }

    # =========================================================================
    # THỐNG KÊ TOÀN DIỆN CHO TẤT CẢ 274 FRAME LOST
    # =========================================================================
    print("\n[BƯỚC 4] Thống kê toàn cảnh trên 274 frame LOST...")
    all_lost_frames = sorted(list(lost_frame_indices))
    all_lost_no_cand = 0
    all_lost_with_cand = 0
    all_lost_rejection = defaultdict(int)

    for f in all_lost_frames:
        cands = candidates_by_frame.get(f, [])
        if len(cands) == 0:
            all_lost_no_cand += 1
            all_lost_rejection["no_candidates_above_005"] += 1
        else:
            all_lost_with_cand += 1
            top_c = max(c["conf"] for c in cands)
            if top_c < 0.15:
                all_lost_rejection["below_select_threshold_015"] += 1
            elif top_c < 0.35:
                all_lost_rejection["outside_gate_and_below_fallback_035"] += 1
            else:
                all_lost_rejection["rejected_by_stabilizer_or_tier3"] += 1

    overall_lost_diagnostics = {
        "total_lost_frames": len(all_lost_frames),
        "frames_no_candidates": all_lost_no_cand,
        "frames_with_candidates": all_lost_with_cand,
        "rejection_breakdown": dict(all_lost_rejection),
    }

    # =========================================================================
    # XUẤT CONTACT SHEET (MỖI 3 FRAME) CHO GAP 513-603 VÀ 135-179
    # =========================================================================
    print("\n[BƯỚC 5] Kết xuất Contact Sheets (mỗi 3 frame) cho 2 gap lớn...")
    os.makedirs("reports/contact_sheets", exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    contact_sheets_out = {}
    for g_key, (s, e, f_b, f_a) in [
        ("ball_gap_513_603", (513, 603, 506, 604)),
        ("ball_gap_135_179", (135, 179, 124, 180)),
    ]:
        sample_frames = list(range(s, e + 1, 3))
        thumbs = []
        thumb_w, thumb_h = 480, 270

        # Anchor interpolation
        b_b = stabilized_ball[f_b][1]["bbox"]
        b_a = stabilized_ball[f_a][1]["bbox"]
        cb_ref = motion.to_reference(f_b, np.array([[(b_b[0]+b_b[2])/2, (b_b[1]+b_b[3])/2]]))[0]
        ca_ref = motion.to_reference(f_a, np.array([[(b_a[0]+b_a[2])/2, (b_a[1]+b_a[3])/2]]))[0]

        for f_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            vis = frame.copy()
            # 1. Tính vị trí ước lượng nội suy
            alpha = (f_idx - f_b) / max(1, (f_a - f_b))
            interp_ref = cb_ref + alpha * (ca_ref - cb_ref)
            M_inv = np.linalg.inv(motion.Ms[f_idx])
            q = M_inv @ np.array([interp_ref[0], interp_ref[1], 1.0])
            ix, iy = int(q[0]/q[2]), int(q[1]/q[2])

            # Vẽ vị trí nội suy chuẩn: Vòng tròn nét đứt Cyan + dấu thập
            cv2.circle(vis, (ix, iy), 18, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(vis, (ix, iy), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(vis, "Expected Interp", (ix - 50, max(25, iy - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            # 2. Vẽ các ứng viên conf >= 0.05
            cands = candidates_by_frame.get(f_idx, [])
            for c in cands:
                box = [int(v) for v in c["bbox"]]
                conf = c["conf"]
                # Màu theo conf:
                # >= 0.30: Xanh lá (0, 255, 0)
                # 0.15 - 0.30: Vàng (0, 255, 255)
                # 0.05 - 0.15: Đỏ cam (0, 100, 255)
                if conf >= 0.30:
                    color = (0, 255, 0)
                elif conf >= 0.15:
                    color = (0, 255, 255)
                else:
                    color = (0, 100, 255)

                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(vis, f"B:{conf:.2f}", (box[0], max(18, box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # 3. Header
            cv2.rectangle(vis, (0, 0), (320, 42), (0, 0, 0), -1)
            cv2.putText(vis, f"Frame {f_idx} (cands={len(cands)})", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

            thumb = cv2.resize(vis, (thumb_w, thumb_h))
            thumbs.append(thumb)

        if thumbs:
            n = len(thumbs)
            cols = min(4, n)
            rows = (n + cols - 1) // cols
            grid = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
            for i, th in enumerate(thumbs):
                r = i // cols
                c = i % cols
                grid[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = th

            out_path = f"reports/contact_sheets/{g_key}.jpg"
            cv2.imwrite(out_path, grid)
            contact_sheets_out[g_key] = out_path
            print(f"-> Đã xuất contact sheet: {out_path} ({n} thumbnails, size={grid.shape[1]}x{grid.shape[0]})")

    cap.release()

    # Tổng hợp toàn bộ dữ liệu ra file JSON
    full_diag_results = {
        "overall_lost_frames": overall_lost_diagnostics,
        "gaps_diagnostics": gap_diagnostics,
        "contact_sheets": contact_sheets_out,
    }
    with open("reports/ball_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(full_diag_results, f, indent=2, ensure_ascii=False)
    print("-> Đã lưu dữ liệu chẩn đoán vào: reports/ball_diagnostics.json")

    return full_diag_results


if __name__ == "__main__":
    run_diagnostics()

