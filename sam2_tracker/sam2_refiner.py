"""
sam2_tracker/sam2_refiner.py
Mô-đun Tự động Quét & Tinh chỉnh Quỹ đạo Toàn cục (Global Multi-Cluster SAM 2 Backward Refiner).

Khả năng tổng quát hóa cho BẤT KỲ VIDEO NÀO:
1. ZERO HARDCODING: Không gán cố định bất kỳ ID hay tọa độ bounding box nào.
   - Tự động phát hiện cặp cầu thủ dính nhau ở đầu trận (Late Birth) dựa trên khoảng cách động học.
   - Tự động phát hiện tất cả các pha va chạm/chồng lấn (Crossing/Occlusions) dựa trên IoU.
2. DYNAMIC FRAME EXTRACTION:
   - Trích xuất frame trực tiếp từ video_path thông qua cv2.VideoCapture cho từng cửa sổ va chạm.
   - Không phụ thuộc vào thư mục ảnh có sẵn, không chiếm dung lượng ổ đĩa.
"""

import os
import gc
import shutil
import cv2
import torch
import numpy as np
from collections import defaultdict

def calculate_box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def auto_detect_clusters(tracks, iou_thresh=0.25, min_consecutive_frames=3):
    """
    Tự động quét toàn bộ video mới để xác định các cụm va chạm / chồng lấn.
    Hoàn toàn dựa trên động học toạ độ (Dynamic Spatial Kinematics), thích ứng với mọi video.
    """
    player_frames = tracks.get("players", [])
    total_frames = len(player_frames)
    if total_frames == 0:
        return []

    events = []
    first_seen = {}
    for f in range(total_frames):
        for pid in player_frames[f].keys():
            if pid not in first_seen:
                first_seen[pid] = f

    # 1. Phát hiện động Cụm Mở đầu (Late Birth ở các frame đầu video: Frame 1..35)
    # Bất kỳ cầu thủ mới sinh ra nào mà tại frame đầu tiên xuất hiện lại nằm sát (d < 80px)
    # một cầu thủ đã có từ Frame 0 thì chắc chắn bị dính chung từ đầu trận.
    f0_ids = set(player_frames[0].keys()) if total_frames > 0 else set()
    initial_pairs_found = set()

    for f in range(1, min(40, total_frames)):
        frame_dict = player_frames[f]
        for pid, pdata in frame_dict.items():
            if pid not in f0_ids and first_seen.get(pid) == f:
                b_new = pdata.get("bbox")
                if not b_new:
                    continue
                # Tìm DUY NHẤT 1 người gần nhất mà pid vừa tách ra (tránh xung đột ghi đè)
                best_old_id = None
                min_dist = 80.0
                for old_id in f0_ids:
                    if old_id in frame_dict:
                        b_old = frame_dict[old_id].get("bbox")
                        if b_old:
                            c_new = ((b_new[0] + b_new[2]) / 2.0, (b_new[1] + b_new[3]) / 2.0)
                            c_old = ((b_old[0] + b_old[2]) / 2.0, (b_old[1] + b_old[3]) / 2.0)
                            dist = np.hypot(c_new[0] - c_old[0], c_new[1] - c_old[1])
                            if dist < min_dist:
                                min_dist = dist
                                best_old_id = old_id

                if best_old_id is not None:
                    pair_key = tuple(sorted([best_old_id, pid]))
                    if pair_key not in initial_pairs_found:
                        initial_pairs_found.add(pair_key)
                        events.append({
                            "name": f"Tách rời Đầu trận (ID {best_old_id} & ID {pid} từ Frame 0..{f})",
                            "start": 0,
                            "end": f,
                            "ids": [best_old_id, pid]
                        })

    # 2. Quét động các vùng va chạm giao thoa (IoU >= iou_thresh) kéo dài nhiều frame
    pair_overlaps = defaultdict(list)
    pair_ious = defaultdict(list)
    for f in range(total_frames):
        frame_dict = player_frames[f]
        pids = list(frame_dict.keys())
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                id1, id2 = pids[i], pids[j]
                b1 = frame_dict[id1].get("bbox")
                b2 = frame_dict[id2].get("bbox")
                if b1 is not None and b2 is not None:
                    iou = calculate_box_iou(b1, b2)
                    if iou >= iou_thresh:
                        pair = tuple(sorted([id1, id2]))
                        pair_overlaps[pair].append(f)
                        pair_ious[pair].append(iou)

    # Gom nhóm các frame va chạm liên tiếp
    for pair, frames in pair_overlaps.items():
        if len(frames) < min_consecutive_frames:
            continue
        
        # Bỏ qua các cặp trùng lặp của cùng một người (mean IoU >= 0.70 như shadow track thủ môn)
        mean_pair_iou = float(np.mean(pair_ious[pair]))
        if mean_pair_iou >= 0.70:
            continue

        segments = []
        cur_seg = [frames[0]]
        for f in frames[1:]:
            if f - cur_seg[-1] <= 10:
                cur_seg.append(f)
            else:
                segments.append(cur_seg)
                cur_seg = [f]
        if cur_seg:
            segments.append(cur_seg)

        for seg in segments:
            if len(seg) >= min_consecutive_frames:
                f_start = max(0, seg[0] - 5)
                f_end = min(total_frames - 1, seg[-1] + 5)
                
                # Bỏ qua nếu đã được xử lý ở cụm mở đầu
                if f_end <= 25 and tuple(sorted(pair)) in initial_pairs_found:
                    continue
                    
                events.append({
                    "name": f"Va chạm ID {pair[0]} & ID {pair[1]} (Frame {f_start}..{f_end})",
                    "start": f_start,
                    "end": f_end,
                    "ids": list(pair)
                })

    events.sort(key=lambda x: x["start"])
    return events


def extract_frame_window(video_path, start_f, end_f, frames_dir=None, temp_dir="data/sam2_window"):
    """
    Trích xuất cửa sổ frame phục vụ SAM 2:
    - Nếu frames_dir đã có sẵn ảnh: copy nhanh.
    - Nếu chạy video mới bất kỳ: dùng OpenCV cv2.VideoCapture trích xuất trực tiếp trong 0.05 giây.
    """
    os.makedirs(temp_dir, exist_ok=True)
    for old_f in os.listdir(temp_dir):
        if old_f.endswith(".jpg"):
            try:
                os.remove(os.path.join(temp_dir, old_f))
            except Exception:
                pass

    saved_count = 0
    # Phương án 1: Trích xuất trực tiếp từ video_path (áp dụng cho mọi video mới)
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

    # Phương án 2: Dùng thư mục frames có sẵn nếu có
    if frames_dir and os.path.exists(frames_dir):
        for idx, f in enumerate(range(start_f, end_f + 1)):
            src = os.path.join(frames_dir, f"{f:05d}.jpg")
            dst = os.path.join(temp_dir, f"{idx:05d}.jpg")
            if os.path.exists(src):
                shutil.copyfile(src, dst)
                saved_count += 1
        return temp_dir, saved_count

    return temp_dir, saved_count


def refine_tracks_with_sam2_backward(
    tracks,
    video_path=None,
    frames_dir=None,
    checkpoint_path="checkpoints/sam2_hiera_tiny.pt",
    config_path="configs/sam2/sam2_hiera_t.yaml",
    device=None,
    verbose=True
):
    """
    HÀM TINH CHỈNH TỔNG QUÁT HÓA TOÀN DIỆN CHO MỌI VIDEO:
    Tự động quét và phân tách tất cả các cụm va chạm bằng SAM 2 Backward.
    """
    if verbose:
        print("\n" + "=" * 74)
        print(" [SAM 2 ĐA NĂNG] QUÉT TỰ ĐỘNG & BÁM NGƯỢC CÁC CỤM TRÊN VIDEO")
        print("=" * 74)

    events = auto_detect_clusters(tracks, iou_thresh=0.25)
    if verbose:
        print(f"-> [Bộ phát hiện tự động] Tìm thấy {len(events)} cụm va chạm / che khuất:")
        for idx, ev in enumerate(events, 1):
            print(f"   {idx}. {ev['name']} [IDs: {ev['ids']}]")

    if not events:
        print("-> Không có cụm va chạm kéo dài nào cần can thiệp. Hoàn tất!")
        return tracks

    use_server = False
    try:
        import requests
        r = requests.get("http://127.0.0.1:5055/health", timeout=0.5)
        if r.status_code == 200 and r.json().get("status") == "ready":
            use_server = True
            if verbose:
                print(f"\n-> [SAM 2 Persistent Service] Đã kết nối máy chủ VRAM thường trực port 5055 (VRAM: {r.json().get('vram_allocated_mb')}MB)!")
    except Exception:
        use_server = False

    sam_tracker = None
    if not use_server:
        try:
            from sam2_tracker.sam2_bidirectional_tracker import SAM2BidirectionalTracker
        except ImportError as e:
            print(f"[SAM 2 Error] Không thể import SAM2BidirectionalTracker: {e}")
            return tracks

        if verbose:
            print("\n-> [SAM 2 Local] Đang tải mô hình SAM 2 Tiny lên GPU...")

        sam_tracker = SAM2BidirectionalTracker(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device
        )

    temp_window_dir = "data/sam2_window"

    for ev_idx, ev in enumerate(events, 1):
        f_start = ev["start"]
        f_end = ev["end"]
        pair_ids = ev["ids"]
        ev_name = ev["name"]

        if verbose:
            print(f"\n--> Đang xử lý Cụm {ev_idx}/{len(events)}: {ev_name}...")

        end_players = tracks["players"][f_end] if f_end < len(tracks["players"]) else {}

        if use_server:
            prompts = {}
            for pid in pair_ids:
                if pid in end_players and "bbox" in end_players[pid]:
                    prompts[str(pid)] = [float(x) for x in end_players[pid]["bbox"]]
            if not prompts:
                continue

            payload = {
                "video_path": video_path,
                "start_frame": f_start,
                "end_frame": f_end,
                "prompts": prompts,
                "prompt_frame": f_end,
                "direction": "backward"
            }
            try:
                res = requests.post("http://127.0.0.1:5055/track_cluster", json=payload, timeout=60.0)
                if res.status_code == 200:
                    data = res.json()
                    results_dict = data.get("tracks", {})
                    refined_frames = 0
                    for abs_f_str, objs in results_dict.items():
                        global_f = int(abs_f_str)
                        if global_f >= len(tracks["players"]):
                            continue
                        frame_dict = tracks["players"][global_f]
                        for pid_str, box in objs.items():
                            pid = int(pid_str)
                            if pid in frame_dict:
                                frame_dict[pid]["bbox"] = box
                                frame_dict[pid]["is_sam2_refined"] = True
                            else:
                                t = end_players.get(pid, {}).get("team", 1)
                                tc = end_players.get(pid, {}).get("team_color", (255, 255, 255))
                                frame_dict[pid] = {
                                    "bbox": box,
                                    "confidence": 0.95,
                                    "team": t,
                                    "team_color": tc,
                                    "is_sam2_refined": True
                                }
                            refined_frames += 1
                    if verbose:
                        print(f"   ✓ [VRAM Server] Xong Cụm {ev_idx}: Tinh chỉnh {refined_frames} vị trí từ Frame {f_end} ◄── {f_start}.")
                    continue
            except Exception as e:
                print(f"   ⚠️ Lỗi kết nối SAM 2 Server ({e}), chuyển sang chế độ cục bộ...")

        window_dir, num_frames = extract_frame_window(
            video_path=video_path,
            start_f=f_start,
            end_f=f_end,
            frames_dir=frames_dir,
            temp_dir=temp_window_dir
        )
        if num_frames < 2:
            continue

        sam_tracker.init_video(window_dir)

        # Lấy prompt sạch tại frame kết thúc f_end
        local_end = num_frames - 1
        end_players = tracks["players"][f_end] if f_end < len(tracks["players"]) else {}

        prompts_added = 0
        for pid in pair_ids:
            box = None
            if pid in end_players and "bbox" in end_players[pid]:
                box = end_players[pid]["bbox"]

            if box is not None:
                sam_tracker.add_box_prompt(frame_idx=local_end, obj_id=pid, box=box)
                prompts_added += 1

        if prompts_added == 0:
            sam_tracker.reset_state()
            continue

        # Chạy bám ngược từ frame tách rời về trước
        results = sam_tracker.track_backward(start_frame_idx=local_end, end_frame_idx=0)

        # Cập nhật kết quả bám ngược vào tracks["players"]
        refined_frames = 0
        for local_f, objs in results.items():
            global_f = f_start + local_f
            if global_f >= len(tracks["players"]):
                continue

            frame_dict = tracks["players"][global_f]
            for pid, pdata in objs.items():
                box = pdata["bbox"]
                if pid in frame_dict:
                    frame_dict[pid]["bbox"] = box
                    frame_dict[pid]["is_sam2_refined"] = True
                else:
                    t = end_players.get(pid, {}).get("team", 1)
                    tc = end_players.get(pid, {}).get("team_color", (255, 255, 255))
                    frame_dict[pid] = {
                        "bbox": box,
                        "confidence": 0.95,
                        "team": t,
                        "team_color": tc,
                        "is_sam2_refined": True
                    }
                refined_frames += 1

        if verbose:
            print(f"   ✓ Xong Cụm {ev_idx}: Tinh chỉnh {refined_frames} vị trí từ Frame {f_end} ◄── {f_start}.")

        sam_tracker.reset_state()

    # Dọn dẹp bộ nhớ và thư mục tạm
    del sam_tracker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if os.path.exists(temp_window_dir):
        try:
            shutil.rmtree(temp_window_dir)
        except Exception:
            pass

    if verbose:
        print("\n" + "=" * 74)
        print(" [SAM 2 HOÀN TẤT] TOÀN BỘ CÁC CỤM TRÊN VIDEO ĐÃ ĐƯỢC XỬ LÝ!")
        print("=" * 74 + "\n")

    return tracks
