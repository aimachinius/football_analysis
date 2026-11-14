"""
sam2_server.py
Máy chủ SAM 2 Thường trực trong VRAM (Persistent SAM 2 VRAM Service).

Mục tiêu:
- Tải mô hình SAM 2 Tiny lên GPU VRAM một lần duy nhất lúc khởi động.
- Giữ mô hình thường trực trong bộ nhớ qua tất cả các lượt chạy video (sample, sample2, sample3).
- Cung cấp API nội bộ tốc độ cao qua http://127.0.0.1:5055:
    GET  /health          -> Kiểm tra trạng thái máy chủ và VRAM GPU
    POST /track_backward  -> Xử lý phân rã cụm va chạm cầu thủ ngược thời gian
    POST /track_ball_gap  -> Xử lý bám bóng xuyên qua khoảng mất dấu
"""

import os
import sys
import gc
import time
import shutil
import cv2
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

app = FastAPI(title="SAM 2 Persistent VRAM Service", version="1.0.0")

# Biến toàn cục giữ instance SAM 2 trong VRAM
PREDICTOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/sam2_hiera_tiny.pt"
CONFIG_PATH = "configs/sam2/sam2_hiera_t.yaml"

def get_predictor():
    global PREDICTOR
    if PREDICTOR is None:
        print(f"[SAM2 Server] Đang tải mô hình SAM 2 lên {DEVICE}...")
        from sam2.build_sam import build_sam2_video_predictor
        PREDICTOR = build_sam2_video_predictor(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
        print("[SAM2 Server] SAM 2 đã nạp thành công và sẵn sàng trong VRAM!")
    return PREDICTOR


@app.on_event("startup")
def startup_event():
    get_predictor()


@app.get("/health")
def health_check():
    vram_mb = 0
    if torch.cuda.is_available():
        vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
    return {
        "status": "ready",
        "device": DEVICE,
        "vram_allocated_mb": vram_mb,
        "model": "sam2_hiera_tiny",
        "pid": os.getpid()
    }


class ClusterTrackRequest(BaseModel):
    video_path: str
    start_frame: int
    end_frame: int
    # Box prompts tại frame mốc (thường là end_frame khi bám ngược)
    # {obj_id_str: [x1, y1, x2, y2]}
    prompts: Dict[str, List[float]]
    prompt_frame: int  # Frame chứa prompt (ví dụ: end_frame)
    direction: str = "backward"  # "backward" hoặc "forward"
    temp_dir: str = "data/sam2_server_window"


@app.post("/track_cluster")
def track_cluster(req: ClusterTrackRequest):
    """
    Nhận yêu cầu bám vết cụm va chạm (backward hoặc forward) cho một khoảng frame.
    Trả về: {str(frame_idx): {str(obj_id): [x1, y1, x2, y2]}}
    """
    predictor = get_predictor()
    start_f = req.start_frame
    end_f = req.end_frame

    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=400, detail=f"Không tìm thấy video: {req.video_path}")

    # Trích xuất frames cho cửa sổ va chạm
    os.makedirs(req.temp_dir, exist_ok=True)
    for old_f in os.listdir(req.temp_dir):
        if old_f.endswith(".jpg"):
            try:
                os.remove(os.path.join(req.temp_dir, old_f))
            except Exception:
                pass

    cap = cv2.VideoCapture(req.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    cur_f = start_f
    saved_count = 0
    while cur_f <= end_f:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(req.temp_dir, f"{saved_count:05d}.jpg"), frame)
        saved_count += 1
        cur_f += 1
    cap.release()

    if saved_count == 0:
        raise HTTPException(status_code=400, detail="Không trích xuất được frame nào từ video")

    # Khởi tạo trạng thái video state trong VRAM
    inference_state = predictor.init_state(req.temp_dir)
    predictor.reset_state(inference_state)

    rel_prompt_idx = req.prompt_frame - start_f
    rel_prompt_idx = max(0, min(saved_count - 1, rel_prompt_idx))

    # Đưa prompt box vào SAM 2
    for obj_id_str, box in req.prompts.items():
        obj_id = int(obj_id_str)
        box_np = np.array(box, dtype=np.float32)
        predictor.add_new_points_or_box(
            inference_state,
            frame_idx=rel_prompt_idx,
            obj_id=obj_id,
            box=box_np
        )

    results = {}
    reverse = (req.direction == "backward")

    # Bám vết qua các frame
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=rel_prompt_idx,
        reverse=reverse
    ):
        abs_frame_idx = start_f + out_frame_idx
        results[str(abs_frame_idx)] = {}

        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            if mask.any():
                ys, xs = np.where(mask)
                x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                results[str(abs_frame_idx)][str(out_obj_id)] = [x1, y1, x2, y2]

    # Giải phóng video state để dọn VRAM
    predictor.reset_state(inference_state)
    del inference_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"status": "success", "tracks": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sam2_server:app", host="127.0.0.1", port=5055, reload=False, log_level="info")

