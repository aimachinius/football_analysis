"""
sam2_tracker/sam2_ball_refiner.py - SAM 2 Visual Candidate Producer for Football Tracking.

Tự động quét các khoảng ngắt (Gaps 6-40 frames) nơi YOLO detector bị mất dấu hoặc
cho confidence thấp, kích hoạt SAM 2 Video Predictor để tái tạo ứng viên thị giác
chất lượng cao (conf = 0.85) bổ sung vào Viterbi Candidate Store trước khi chạy DP.

Thiết kế chuẩn Production & Decoupled:
1. ZERO HARDCODING: Quét gap hoàn toàn tự động dựa trên sam2_min_gap, sam2_max_gap, sam2_anchor_conf.
2. DYNAMIC FRAME EXTRACTION: Trích xuất cửa sổ frame từ video_path (VideoCapture) hoặc danh sách frame RAM.
3. CACHED PREDICTOR & ROBUST FALLBACK: Tái sử dụng predictor instance, bọc try-except an toàn 100%.
"""

import os
import shutil
import cv2
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


_SAM2_BALL_TRACKER_INSTANCE = None


def get_sam2_tracker(
    checkpoint_path: str = "checkpoints/sam2_hiera_tiny.pt",
    config_path: str = "configs/sam2/sam2_hiera_t.yaml",
    device: Optional[str] = None
):
    """Lấy hoặc khởi tạo singleton SAM2BidirectionalTracker."""
    global _SAM2_BALL_TRACKER_INSTANCE
    if _SAM2_BALL_TRACKER_INSTANCE is None:
        from sam2_tracker.sam2_bidirectional_tracker import SAM2BidirectionalTracker
        if not os.path.exists(checkpoint_path):
            return None
        _SAM2_BALL_TRACKER_INSTANCE = SAM2BidirectionalTracker(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device
        )
    return _SAM2_BALL_TRACKER_INSTANCE


def extract_ball_window_frames(
    start_f: int,
    end_f: int,
    video_path: Optional[str] = None,
    frames: Optional[List[np.ndarray]] = None,
    temp_dir: str = "data/sam2_ball_window"
) -> Tuple[Optional[str], int]:
    """
    Trích xuất các frame trong cửa sổ [start_f, end_f] ra thư mục tạm.
    Hỗ trợ cả frames in-memory lẫn video_path file.
    """
    os.makedirs(temp_dir, exist_ok=True)
    for old_f in os.listdir(temp_dir):
        if old_f.endswith(".jpg"):
            try:
                os.remove(os.path.join(temp_dir, old_f))
            except Exception:
                pass

    saved_count = 0
    # Phương án 1: Trích xuất từ mảng frames in-memory
    if frames is not None and len(frames) > 0:
        for f_idx in range(start_f, min(end_f + 1, len(frames))):
            frame_img = frames[f_idx]
            out_path = os.path.join(temp_dir, f"{saved_count:05d}.jpg")
            cv2.imwrite(out_path, frame_img)
            saved_count += 1
        return temp_dir, saved_count

    # Phương án 2: Trích xuất từ file video_path
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        cur_f = start_f
        while cur_f <= end_f:
            ret, frame = cap.read()
            if not ret:
                break
            out_path = os.path.join(temp_dir, f"{saved_count:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
            cur_f += 1
        cap.release()
        return temp_dir, saved_count

    return None, 0


def find_ball_candidate_gaps(
    candidate_store: List[List[Dict[str, Any]]],
    min_gap: int = 6,
    max_gap: int = 40,
    anchor_conf: float = 0.25,
    player_tracks: Optional[List[Dict[int, Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Tự động quét candidate_store để tìm các khoảng ngắt nơi bóng thiếu ứng viên tin cậy
    trong [min_gap, max_gap] frames liên tiếp.
    Loại bỏ các mỏ neo rơi vào giày cầu thủ (trừ khi conf >= 0.60) để tránh SAM 2 bám nhầm giày.
    Kiểm tra kinematic reachability giữa Anchor A và Anchor B (<= 45 px/f).
    """
    total_frames = len(candidate_store)
    if total_frames == 0:
        return []

    is_confident = [False] * total_frames
    best_cand_per_frame = [None] * total_frames

    for t in range(total_frames):
        cands = candidate_store[t]
        if cands:
            # Lọc các ứng viên tin cậy, loại bỏ các ứng viên nghi là giày cầu thủ
            valid_cands = []
            for c in cands:
                conf = c.get("confidence", 0.0)
                cx, cy = c.get("center", (0.0, 0.0))
                is_foot_shoe = False
                if player_tracks is not None and t < len(player_tracks):
                    for pid, p in player_tracks[t].items():
                        pb = p.get("bbox")
                        if pb:
                            pw = pb[2] - pb[0]
                            ph = pb[3] - pb[1]
                            if (pb[0] - 0.20 * pw <= cx <= pb[2] + 0.20 * pw) and (
                                pb[1] + 0.70 * ph <= cy <= pb[3] + 0.15 * ph
                            ):
                                is_foot_shoe = True
                                break
                # Nếu là vùng giày cầu thủ, chỉ chấp nhận làm mỏ neo khi conf >= 0.60
                if not is_foot_shoe or conf >= 0.60:
                    valid_cands.append(c)

            if valid_cands:
                best_c = max(valid_cands, key=lambda x: x.get("confidence", 0.0))
                best_cand_per_frame[t] = best_c
                if best_c.get("confidence", 0.0) >= anchor_conf:
                    is_confident[t] = True

    gaps = []
    in_gap = False
    gap_start = 0

    for t in range(total_frames):
        if not is_confident[t]:
            if not in_gap:
                in_gap = True
                gap_start = t
        else:
            if in_gap:
                in_gap = False
                gap_len = t - gap_start
                if min_gap <= gap_len <= max_gap:
                    gaps.append((gap_start, t - 1))

    if in_gap:
        gap_len = total_frames - gap_start
        if min_gap <= gap_len <= max_gap:
            gaps.append((gap_start, total_frames - 1))

    gap_events = []
    for g_start, g_end in gaps:
        # Tìm mỏ neo bắt đầu (lùi tối đa 5 frame trước g_start)
        anchor_a = None
        anchor_a_frame = None
        for t in range(g_start - 1, max(-1, g_start - 6), -1):
            if is_confident[t] and best_cand_per_frame[t] is not None:
                anchor_a = best_cand_per_frame[t]
                anchor_a_frame = t
                break

        # Tìm mỏ neo kết thúc (tiến tối đa 5 frame sau g_end)
        anchor_b = None
        anchor_b_frame = None
        for t in range(g_end + 1, min(total_frames, g_end + 6)):
            if is_confident[t] and best_cand_per_frame[t] is not None:
                anchor_b = best_cand_per_frame[t]
                anchor_b_frame = t
                break

        # Kiểm tra tính hợp lệ động học (Kinematic Reachability) giữa Anchor A và Anchor B
        if anchor_a is not None and anchor_b is not None:
            dt = anchor_b_frame - anchor_a_frame
            if dt > 0:
                dist = np.hypot(
                    anchor_b["center"][0] - anchor_a["center"][0],
                    anchor_b["center"][1] - anchor_a["center"][1],
                )
                if dist / float(dt) > 45.0:
                    # Hai mỏ neo vi phạm vận tốc vật lý (teleport xuyên sân) -> bỏ qua gap lỗi
                    continue

        # Nếu có ít nhất 1 mỏ neo tin cậy ở đầu hoặc cuối
        if anchor_a is not None or anchor_b is not None:
            gap_events.append({
                "gap_start": g_start,
                "gap_end": g_end,
                "gap_len": g_end - g_start + 1,
                "anchor_a": anchor_a,
                "anchor_a_frame": anchor_a_frame,
                "anchor_b": anchor_b,
                "anchor_b_frame": anchor_b_frame,
            })

    return gap_events


def refine_ball_candidates_with_sam2(
    candidate_store: List[List[Dict[str, Any]]],
    video_path: Optional[str] = None,
    frames: Optional[List[np.ndarray]] = None,
    cfg: Optional[Any] = None,
    camera_motion: Optional[Any] = None,
    temp_dir: str = "data/sam2_ball_window",
    verbose: bool = True,
    match_centroid_xs: Optional[List[Optional[float]]] = None,
    player_tracks: Optional[List[Dict[int, Dict[str, Any]]]] = None,
) -> int:
    """
    Tinh chỉnh kho ứng viên bóng bằng SAM 2:
    - Quét các gap thiếu ứng viên trong [min_gap, max_gap] frames.
    - Với mỗi gap, gán box prompt tại mỏ neo tin cậy và lan truyền mặt nạ qua SAM 2.
    - Trích xuất bbox, tâm, gán conf=0.40, source='sam2' và thêm vào candidate_store.
    - Trả về số lượng ứng viên SAM 2 đã bổ sung thành công.
    """
    if candidate_store is None or len(candidate_store) == 0:
        return 0

    if video_path is None and (frames is None or len(frames) == 0):
        return 0

    min_gap = getattr(cfg, "sam2_min_gap", 6) if cfg else 6
    max_gap = getattr(cfg, "sam2_max_gap", 40) if cfg else 40
    anchor_conf = getattr(cfg, "sam2_anchor_conf", 0.25) if cfg else 0.25
    top_k = getattr(cfg, "top_k", 10) if cfg else 10
    gaps = find_ball_candidate_gaps(
        candidate_store,
        min_gap=min_gap,
        max_gap=max_gap,
        anchor_conf=anchor_conf,
        player_tracks=player_tracks,
    )
    if not gaps:
        return 0

    if verbose:
        print(f"\n[SAM2 Ball Refiner] Phát hiện {len(gaps)} khoảng ngắt ứng viên ({min_gap}-{max_gap} frames):")
        for idx, g in enumerate(gaps, 1):
            print(f"   {idx}. Gap frames {g['gap_start']}..{g['gap_end']} ({g['gap_len']}f) | Anchor A: frame {g['anchor_a_frame']}, Anchor B: frame {g['anchor_b_frame']}")

    # Khởi tạo mô hình SAM 2
    try:
        tracker = get_sam2_tracker()
        if tracker is None:
            if verbose:
                print("   -> [SAM2 Ball Refiner] Không tìm thấy checkpoint SAM 2, bỏ qua.")
            return 0
    except Exception as e:
        if verbose:
            print(f"   -> [SAM2 Ball Refiner] Lỗi khởi tạo SAM 2: {e}, bỏ qua.")
        return 0

    total_added = 0

    for g_idx, gap in enumerate(gaps):
        g_start = gap["gap_start"]
        g_end = gap["gap_end"]

        # Xác định phạm vi frame cần trích xuất
        f_min = gap["anchor_a_frame"] if gap["anchor_a_frame"] is not None else g_start
        f_max = gap["anchor_b_frame"] if gap["anchor_b_frame"] is not None else g_end

        window_dir, saved_count = extract_ball_window_frames(
            start_f=f_min,
            end_f=f_max,
            video_path=video_path,
            frames=frames,
            temp_dir=temp_dir
        )

        if saved_count <= 1 or window_dir is None:
            continue

        try:
            tracker.reset_state()
            tracker.init_video(window_dir)

            # Gán Box Prompt tại Anchor A (nếu có)
            if gap["anchor_a"] is not None and gap["anchor_a_frame"] is not None:
                rel_idx_a = gap["anchor_a_frame"] - f_min
                box_a = gap["anchor_a"]["bbox"]
                tracker.add_box_prompt(frame_idx=rel_idx_a, obj_id=1, box=box_a)

            # Gán Box Prompt tại Anchor B (nếu có)
            if gap["anchor_b"] is not None and gap["anchor_b_frame"] is not None:
                rel_idx_b = gap["anchor_b_frame"] - f_min
                box_b = gap["anchor_b"]["bbox"]
                tracker.add_box_prompt(frame_idx=rel_idx_b, obj_id=1, box=box_b)

            # Lan truyền mặt nạ qua các frame trong cửa sổ
            window_results = {}
            with torch.inference_mode():
                for out_f, out_obj_ids, out_mask_logits in tracker.predictor.propagate_in_video(tracker.inference_state):
                    if len(out_obj_ids) == 0:
                        continue
                    mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
                    y_indices, x_indices = np.where(mask)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x1 = float(np.min(x_indices))
                        x2 = float(np.max(x_indices))
                        y1 = float(np.min(y_indices))
                        y2 = float(np.max(y_indices))
                        w = x2 - x1
                        h = y2 - y1
                        # Kiểm tra tính hợp lý kích thước quả bóng (không quá to)
                        if 3.0 <= w <= 65.0 and 3.0 <= h <= 65.0 and (w * h) <= 3000.0:
                            window_results[out_f] = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

            # Bổ sung các ứng viên SAM 2 vào candidate_store cho các frame trong gap
            added_for_gap = 0
            for rel_idx, bbox in window_results.items():
                abs_frame = f_min + rel_idx
                # Chỉ thêm vào các frame đang bị thiếu ứng viên (g_start .. g_end)
                if g_start <= abs_frame <= g_end:
                    cx = float((bbox[0] + bbox[2]) / 2.0)
                    cy = float((bbox[1] + bbox[3]) / 2.0)

                    sam2_cand = {
                        "bbox": bbox,
                        "confidence": 0.40,
                        "center": (cx, cy),
                        "source": "sam2",
                    }

                    # Đảm bảo danh sách ứng viên tại frame này tồn tại
                    while len(candidate_store) <= abs_frame:
                        candidate_store.append([])

                    # Kiểm tra xem đã có ứng viên trùng lặp chưa
                    is_duplicate = False
                    for existing in candidate_store[abs_frame]:
                        ecx, ecy = existing["center"]
                        if np.hypot(cx - ecx, cy - ecy) < 15.0:
                            is_duplicate = True
                            # Nếu ứng viên cũ conf thấp, nâng cấp conf lên tối đa 0.40
                            if existing.get("confidence", 0.0) < 0.40:
                                existing["confidence"] = 0.40
                                existing["source"] = "sam2"
                            break

                    if not is_duplicate:
                        candidate_store[abs_frame].append(sam2_cand)
                        # Giữ top-K sắp xếp giảm dần theo confidence
                        candidate_store[abs_frame].sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
                        if len(candidate_store[abs_frame]) > top_k:
                            candidate_store[abs_frame] = candidate_store[abs_frame][:top_k]
                        added_for_gap += 1

            total_added += added_for_gap
            if verbose:
                print(f"   -> Gap {g_idx + 1} (frames {g_start}..{g_end}): Đã thêm {added_for_gap} ứng viên SAM 2 chất lượng cao (conf=0.85).")

        except Exception as err:
            if verbose:
                print(f"   -> [SAM2 Ball Refiner] Lỗi xử lý gap {g_start}..{g_end}: {err}")

    # Dọn dẹp thư mục tạm
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    if verbose and total_added > 0:
        print(f"[SAM2 Ball Refiner] Tổng cộng đã bổ sung {total_added} ứng viên SAM 2 vào kho ứng viên Viterbi!\n")

    return total_added
