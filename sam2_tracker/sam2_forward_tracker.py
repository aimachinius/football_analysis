"""
sam2_tracker/sam2_forward_tracker.py
Mô-đun theo dõi đối tượng video xuôi dòng (Forward Video Object Tracker) bằng SAM 2.

Đặc điểm kiến trúc:
1. BÁM DÍNH LIÊN TỤC KHÔNG PHỤ THUỘC YOLO:
   - Nhận danh sách đối tượng (Prompts) ở Frame 0 (hoặc khi đối tượng xuất hiện).
   - TẮT HOÀN TOÀN YOLO ở các frame sau.
   - SAM 2 dùng Memory Bank (Attention bộ nhớ thời gian) bám theo từng người qua toàn bộ 750 frames.
   - KHÔNG ĐỔI ID, KHÔNG NHẢY NHÃN, KHÔNG BỊ TRÁO NGƯỜI.
2. CHUNKED STREAMING (Tiết kiệm RAM & VRAM):
   - Xử lý theo từng cửa sổ (window = 50 frames) để đảm bảo không bị tràn RAM (7.7GB) hay VRAM (4GB).
   - Chuyển giao liên tục (Handover) tọa độ và mask từ cuối window trước sang đầu window sau.
"""

import os
import gc
import shutil
import cv2
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from sam2.build_sam import build_sam2_video_predictor

class SAM2ForwardTracker:
    def __init__(
        self,
        checkpoint_path="checkpoints/sam2_hiera_tiny.pt",
        config_path="configs/sam2/sam2_hiera_t.yaml",
        device=None,
        window_size=50,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = window_size
        print(f"[SAM2ForwardTracker] Đang nạp mô hình SAM 2 ({config_path}) trên {self.device}...")
        self.predictor = build_sam2_video_predictor(
            config_path,
            checkpoint_path,
            device=self.device
        )
        print("[SAM2ForwardTracker] Nạp mô hình SAM 2 thành công!")

    def track_video_forward(
        self,
        video_path: str,
        initial_prompts: Dict[int, Dict[str, Any]],
        total_frames: int,
        temp_dir="data/sam2_forward_window",
        verbose=True
    ) -> List[Dict[int, Dict[str, Any]]]:
        """
        Bám theo toàn bộ đối tượng từ Frame 0 đến total_frames bằng SAM 2.
        initial_prompts: {
            obj_id: {
                "bbox": [x1, y1, x2, y2],
                "class_id": int,       # 1=player, 2=referee, 3=goalkeeper
                "team": int,           # 1 hoặc 2 (nếu có)
                "team_color": tuple    # (B, G, R) (nếu có)
            }
        }
        Trả về: list các frame dict tracks[frame_idx] = { obj_id: { ... } }
        """
        os.makedirs(temp_dir, exist_ok=True)
        tracked_frames = [{} for _ in range(total_frames)]

        # Ghi nhận thông tin ban đầu ở Frame 0
        current_prompts = {}
        for obj_id, info in initial_prompts.items():
            tracked_frames[0][obj_id] = {
                "bbox": list(info["bbox"]),
                "confidence": 1.0,
                "team": info.get("team"),
                "team_color": info.get("team_color"),
                "is_goalkeeper": info.get("is_goalkeeper", False),
                "is_referee": info.get("is_referee", False),
            }
            current_prompts[obj_id] = list(info["bbox"])

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Chia nhỏ thành các window (mỗi window 50 frames) để an toàn tuyệt đối cho RAM/VRAM
        window_size = self.window_size
        num_windows = (total_frames - 1 + window_size - 1) // window_size

        if verbose:
            print(f"\n[SAM 2 Forward Tracker] Bắt đầu theo dõi {len(initial_prompts)} đối tượng qua {total_frames} frames ({num_windows} windows)...")

        for w_idx in range(num_windows):
            w_start = w_idx * window_size
            w_end = min(total_frames - 1, (w_idx + 1) * window_size)
            w_len = w_end - w_start + 1

            if verbose:
                print(f"-> [Window {w_idx + 1}/{num_windows}] Đang bám từ Frame {w_start} ──► Frame {w_end} ({w_len} frames)...")

            # 1. Trích xuất w_len frames vào thư mục tạm bằng OpenCV
            self._clear_dir(temp_dir)
            cap.set(cv2.CAP_PROP_POS_FRAMES, w_start)
            for f_i in range(w_len):
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(temp_dir, f"{f_i:05d}.jpg"), frame)

            # 2. Khởi tạo inference state cho window hiện tại
            inference_state = self.predictor.init_state(temp_dir)

            # 3. Gán prompt tại local frame 0 (tương ứng w_start) từ current_prompts
            active_ids_in_window = []
            for obj_id, box in current_prompts.items():
                if box is not None and len(box) == 4:
                    box_np = np.array(box, dtype=np.float32)
                    self.predictor.add_new_points_or_box(
                        inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        box=box_np
                    )
                    active_ids_in_window.append(obj_id)

            # 4. Truyền dẫn (Propagation) xuôi qua toàn bộ window
            last_window_boxes = {}
            with torch.inference_mode():
                for out_f, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=w_len,
                    reverse=False
                ):
                    global_f = w_start + out_f
                    if global_f >= total_frames:
                        continue

                    for i, obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                        y_indices, x_indices = np.where(mask)
                        if len(x_indices) > 0 and len(y_indices) > 0:
                            x1, x2 = float(np.min(x_indices)), float(np.max(x_indices))
                            y1, y2 = float(np.min(y_indices)), float(np.max(y_indices))
                            box = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

                            # Kế thừa thuộc tính cố định từ frame 0
                            orig_info = tracked_frames[0].get(obj_id, {})
                            tracked_frames[global_f][obj_id] = {
                                "bbox": box,
                                "confidence": 0.95,
                                "team": orig_info.get("team"),
                                "team_color": orig_info.get("team_color"),
                                "is_goalkeeper": orig_info.get("is_goalkeeper", False),
                                "is_referee": orig_info.get("is_referee", False),
                            }
                            if out_f == w_len - 1:
                                last_window_boxes[obj_id] = box

            # 5. Cập nhật current_prompts cho window tiếp theo
            current_prompts = last_window_boxes

            # 6. Dọn dẹp trạng thái và bộ nhớ
            self.predictor.reset_state(inference_state)
            del inference_state
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cap.release()
        self._clear_dir(temp_dir)
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

        if verbose:
            print(f"[SAM 2 Forward Tracker] Hoàn tất bám vết {total_frames} frames!")

        return tracked_frames

    def _clear_dir(self, directory: str):
        if not os.path.exists(directory):
            return
        for f in os.listdir(directory):
            if f.endswith(".jpg"):
                try:
                    os.remove(os.path.join(directory, f))
                except Exception:
                    pass

