"""
sam2_tracker/sam2_bidirectional_tracker.py
Mô-đun theo dõi độc lập hai chiều (Bidirectional Video Object Tracker) bằng SAM 2.

Đặc điểm thiết kế:
- Hoạt động độc lập (Decoupled), không phụ thuộc hay sửa đổi main.py/pipeline hiện tại.
- Hỗ trợ cả 2 chiều:
  1. Forward Tracking: Bám đối tượng xuôi theo thời gian qua các pha che khuất.
  2. Backward Tracking: Đặc trị pha tranh chấp dính chặt ở đầu trận (như Frame 0..20):
     Lấy vị trí khi đã tách rời ở Frame 20, bám ngược về Frame 0 để phân rã 2 cá thể độc lập.
- Tối ưu VRAM: Chạy tốt trên GPU 4GB VRAM (như RTX 3050 Laptop).
"""

import os
import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor

class SAM2BidirectionalTracker:
    def __init__(
        self,
        checkpoint_path="checkpoints/sam2_hiera_tiny.pt",
        config_path="configs/sam2/sam2_hiera_t.yaml",
        device=None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[SAM2Tracker] Đang tải mô hình SAM 2 ({config_path}) trên {self.device}...")
        self.predictor = build_sam2_video_predictor(
            config_path,
            checkpoint_path,
            device=self.device
        )
        print("[SAM2Tracker] Tải mô hình thành công!")

    def init_video(self, video_frames_dir):
        """Khởi tạo trạng thái bộ nhớ cho thư mục ảnh frame."""
        print(f"[SAM2Tracker] Khởi tạo video state từ {video_frames_dir}...")
        self.inference_state = self.predictor.init_state(video_frames_dir)
        self.num_frames = self.inference_state["num_frames"]
        print(f"[SAM2Tracker] Đã nạp {self.num_frames} frames vào Video State.")
        return self.inference_state

    def reset_state(self):
        """Reset bộ nhớ trạng thái."""
        if hasattr(self, "inference_state"):
            self.predictor.reset_state(self.inference_state)

    def add_box_prompt(self, frame_idx, obj_id, box):
        """
        Gán prompt bounding box cho đối tượng obj_id tại frame_idx.
        box: [x1, y1, x2, y2]
        """
        box_np = np.array(box, dtype=np.float32)
        self.predictor.add_new_points_or_box(
            self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box_np
        )

    def track_backward(self, start_frame_idx, end_frame_idx=0):
        """
        Bám ngược thời gian từ start_frame_idx lùi về end_frame_idx.
        Trả về dict: { frame_idx: { obj_id: { 'bbox': [x1, y1, x2, y2], 'mask': np.ndarray } } }
        """
        max_frames = start_frame_idx - end_frame_idx + 1
        print(f"[SAM2Tracker] Đang chạy Backward Tracking: từ Frame {start_frame_idx} ◄── Frame {end_frame_idx} ({max_frames} frames)...")

        results = {}
        with torch.inference_mode():
            for out_f, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frames,
                reverse=True
            ):
                results[out_f] = {}
                for i, obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    y_indices, x_indices = np.where(mask)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x1, x2 = float(np.min(x_indices)), float(np.max(x_indices))
                        y1, y2 = float(np.min(y_indices)), float(np.max(y_indices))
                        results[out_f][obj_id] = {
                            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                            "mask": mask
                        }
        return results

    def track_forward(self, start_frame_idx=0, max_frames=None):
        """
        Bám xuôi thời gian từ start_frame_idx tiến về trước.
        """
        if max_frames is None:
            max_frames = self.num_frames - start_frame_idx
        print(f"[SAM2Tracker] Đang chạy Forward Tracking: từ Frame {start_frame_idx} ──► Frame {start_frame_idx + max_frames - 1}...")

        results = {}
        with torch.inference_mode():
            for out_f, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frames,
                reverse=False
            ):
                results[out_f] = {}
                for i, obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    y_indices, x_indices = np.where(mask)
                    if len(x_indices) > 0 and len(y_indices) > 0:
                        x1, x2 = float(np.min(x_indices)), float(np.max(x_indices))
                        y1, y2 = float(np.min(y_indices)), float(np.max(y_indices))
                        results[out_f][obj_id] = {
                            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                            "mask": mask
                        }
        return results

