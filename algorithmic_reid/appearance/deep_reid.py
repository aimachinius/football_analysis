"""
algorithmic_reid.appearance.deep_reid - Deep Semantic Re-ID Feature Extractor (Direction B).

Uses a lightweight MobileNetV3-Small backbone stripped of classifier head
to extract 576-dimensional L2-normalized deep semantic embeddings for player crops.
Runs at ~14ms per frame on CUDA GPU.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import cv2
from typing import List, Optional


class DeepReIDExtractor:
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        model = models.mobilenet_v3_small(weights="DEFAULT")
        model.classifier = nn.Identity()
        self.model = model.to(self.device).eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_crops(self, frame: np.ndarray, bboxes: np.ndarray) -> List[Optional[np.ndarray]]:
        """
        Extract deep embeddings for an array of bounding boxes in a frame.
        Returns list of (576,) unit-normalized vectors.
        """
        if frame is None or len(bboxes) == 0:
            return []

        h_frame, w_frame = frame.shape[:2]
        crops = []
        valid_indices = []

        for i, b in enumerate(bboxes):
            x1 = max(0, min(int(b[0]), w_frame - 1))
            y1 = max(0, min(int(b[1]), h_frame - 1))
            x2 = max(0, min(int(b[2]), w_frame))
            y2 = max(0, min(int(b[3]), h_frame))

            if (x2 - x1) < 4 or (y2 - y1) < 4:
                crops.append(None)
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                crops.append(None)
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(crop_rgb)
            valid_indices.append(i)

        out_features: List[Optional[np.ndarray]] = [None] * len(bboxes)
        if not valid_indices:
            return out_features

        tensor_list = [self.transform(crops[idx]) for idx in valid_indices]
        batch_tensors = torch.stack(tensor_list).to(self.device)

        feats = self.model(batch_tensors)
        feats = nn.functional.normalize(feats, p=2, dim=1)
        feats_np = feats.cpu().numpy()

        for k, real_idx in enumerate(valid_indices):
            out_features[real_idx] = feats_np[k].astype(np.float32)

        return out_features


def deep_cosine_distance(f1: Optional[np.ndarray], f2: Optional[np.ndarray]) -> float:
    """
    Compute normalized cosine distance between two 576-dim L2-normalized embeddings.
    Range: [0.0, 1.0]. Returns 0.5 if either feature is None.
    """
    if f1 is None or f2 is None:
        return 0.5
    cos_sim = float(np.dot(f1, f2))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return (1.0 - cos_sim) / 2.0


def batch_deep_cosine_distance(
    track_features: List[Optional[np.ndarray]],
    det_features: List[Optional[np.ndarray]],
) -> np.ndarray:
    """
    Compute pairwise cosine distance matrix between tracks (M) and detections (N).
    """
    M = len(track_features)
    N = len(det_features)
    if M == 0 or N == 0:
        return np.empty((M, N), dtype=np.float64)

    dist_mat = np.full((M, N), 0.5, dtype=np.float64)
    valid_t = [i for i, f in enumerate(track_features) if f is not None]
    valid_d = [j for j, f in enumerate(det_features) if f is not None]

    if not valid_t or not valid_d:
        return dist_mat

    t_mat = np.stack([track_features[i] for i in valid_t])  # (m, 576)
    d_mat = np.stack([det_features[j] for j in valid_d])    # (n, 576)

    sim = np.dot(t_mat, d_mat.T)  # (m, n)
    sim = np.clip(sim, -1.0, 1.0)
    sub_dist = (1.0 - sim) / 2.0

    for idx_t, real_t in enumerate(valid_t):
        for idx_d, real_d in enumerate(valid_d):
            dist_mat[real_t, real_d] = sub_dist[idx_t, idx_d]

    return dist_mat

