"""
algorithmic_reid.reid_tracker_facade - Drop-in Tracker Facade.

Accepts sv.Detections and optional camera homography M_t / G_t,
maintains tracking with algorithmic Re-ID, and outputs standardized tracks dict
compatible with tracker/tracker.py and main.py contracts:
  {
      "players": {track_id: {"bbox": [x1, y1, x2, y2], "confidence": float, ...}},
      "referees": {track_id: {"bbox": [x1, y1, x2, y2], "confidence": float, ...}},
      "tracked_det": sv.Detections(...)
  }
"""

import numpy as np
import supervision as sv
from typing import Optional, Dict, Any, List
from .config import ReIDConfig
from .lifecycle.track_manager import TrackManager
from .lifecycle.tracklet import TrackletState


class TrackedDetectionsContainer:
    """
    Dual-interface container:
      - Dictionary indexing: res["players"], res["referees"], res["tracked_det"]
      - sv.Detections attributes: res.xyxy, res.confidence, res.class_id, res.tracker_id
    """
    def __init__(
        self,
        players: Dict[int, Dict[str, Any]],
        referees: Dict[int, Dict[str, Any]],
        tracked_det: sv.Detections,
    ):
        self._dict = {
            "players": players,
            "referees": referees,
            "tracked_det": tracked_det,
        }
        self.tracked_det = tracked_det

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._dict[key]
        return self.tracked_det[key]

    def __contains__(self, key):
        return key in self._dict

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

    def get(self, key, default=None):
        return self._dict.get(key, default)

    @property
    def xyxy(self):
        return self.tracked_det.xyxy

    @property
    def confidence(self):
        return self.tracked_det.confidence

    @property
    def class_id(self):
        return self.tracked_det.class_id

    @property
    def tracker_id(self):
        return self.tracked_det.tracker_id

    def __len__(self):
        return len(self.tracked_det)


class ReIDTrackerFacade:
    def __init__(self, config: Optional[ReIDConfig] = None, fps: float = 25.0):
        if config is None:
            config = ReIDConfig(fps=fps)
        else:
            config.fps = fps
        self.config = config
        self.manager = TrackManager(config=self.config)

    def reset(self) -> None:
        self.manager.reset()

    def update(
        self,
        detections: sv.Detections,
        frame: Optional[np.ndarray] = None,
        camera_homography: Optional[np.ndarray] = None,
    ) -> TrackedDetectionsContainer:
        """
        Update tracker with current frame detections.
        detections: sv.Detections containing xyxy, confidence, class_id
        frame: optional BGR image for multi-zone appearance extraction
        camera_homography: optional 3x3 homography matrix (e.g. from Stage 3 CameraMotion)
        """
        if len(detections) == 0:
            active_tracks = self.manager.step(
                det_boxes=np.empty((0, 4), dtype=np.float32),
                det_confs=np.empty((0,), dtype=np.float32),
                det_classes=np.empty((0,), dtype=np.int32),
                frame=frame,
                camera_homography=camera_homography,
            )
            empty_det = sv.Detections.empty()
            empty_det.tracker_id = np.empty((0,), dtype=np.int32)
            return TrackedDetectionsContainer(players={}, referees={}, tracked_det=empty_det)

        det_boxes = detections.xyxy.astype(np.float32)
        det_confs = (
            detections.confidence.astype(np.float32)
            if detections.confidence is not None
            else np.ones(len(det_boxes), dtype=np.float32)
        )
        det_classes = (
            detections.class_id.astype(np.int32)
            if detections.class_id is not None
            else np.full(len(det_boxes), self.config.class_player, dtype=np.int32)
        )

        # On the very first frame, confirm high-confidence tracks immediately
        if self.manager.frame_count == 0:
            for k in range(len(det_boxes)):
                if det_confs[k] >= self.config.high_conf_thresh:
                    # Pre-initialize as confirmed
                    tid = self.manager._allocate_id()
                    feat = None
                    if frame is not None:
                        from .appearance.color_descriptor import extract_multizone_histogram
                        feat = extract_multizone_histogram(frame, det_boxes[k])
                    track = self.manager.tracks
                    from .lifecycle.tracklet import Tracklet
                    t = Tracklet(
                        track_id=tid,
                        bbox=det_boxes[k],
                        confidence=det_confs[k],
                        class_id=det_classes[k],
                        feature=feat,
                        kf=self.manager.kf,
                        aggregator=self.manager.aggregator,
                        n_init=self.config.n_init,
                        max_age=self.config.max_age,
                        start_frame=0,
                    )
                    t.state = TrackletState.CONFIRMED
                    self.manager.tracks.append(t)

        active_tracks = self.manager.step(
            det_boxes=det_boxes,
            det_confs=det_confs,
            det_classes=det_classes,
            frame=frame,
            camera_homography=camera_homography,
        )

        players = {}
        referees = {}

        out_boxes = []
        out_confs = []
        out_classes = []
        out_tids = []

        for track in active_tracks:
            bbox = track.bbox.tolist()
            tid = int(track.track_id)
            cls_id = int(track.class_id)
            conf = float(track.confidence)

            entry = {"bbox": bbox, "confidence": conf}

            if cls_id == self.config.class_referee:
                referees[tid] = entry
            elif cls_id == self.config.class_goalkeeper:
                entry["is_goalkeeper"] = True
                players[tid] = entry
            else:
                players[tid] = entry

            out_boxes.append(bbox)
            out_confs.append(conf)
            out_classes.append(cls_id)
            out_tids.append(tid)

        if len(out_boxes) > 0:
            tracked_det = sv.Detections(
                xyxy=np.array(out_boxes, dtype=np.float32),
                confidence=np.array(out_confs, dtype=np.float32),
                class_id=np.array(out_classes, dtype=np.int32),
                tracker_id=np.array(out_tids, dtype=np.int32),
            )
        else:
            tracked_det = sv.Detections.empty()
            tracked_det.tracker_id = np.empty((0,), dtype=np.int32)

        return TrackedDetectionsContainer(players=players, referees=referees, tracked_det=tracked_det)
