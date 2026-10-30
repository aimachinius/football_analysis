from transformers.utils.generic import merge_with_config_defaults
import pickle
import cv2
import numpy as np
import sys
import os
sys.path.append("../")
from utils import measure_distance, measure_xy_distance
class CameraMovementEstimate():
    def __init__(self, frame):
        self.minimum_distance = 5
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1
        self.feature = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )
        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def _adjust_positions_to_track(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
            
        # Nếu người dùng truyền trực tiếp read_video() -> nhận được tuple (generator, fps)
        if isinstance(frames, tuple):
            frames = frames[0]

        camera_move = []
        old_gray = None
        old_features = None

        for frame_num, frame in enumerate(frames):
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if old_gray is None:
                old_gray = frame_gray
                old_features = cv2.goodFeaturesToTrack(old_gray, **self.feature)
                camera_move.append([0, 0])
                continue

            new_feature, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            if new_feature is not None and old_features is not None:
                for i, (new, old) in enumerate(zip(new_feature, old_features)):
                    new_feature_point = new.ravel()
                    old_feature_point = old.ravel()
                    distance = measure_distance(new_feature_point, old_feature_point)

                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x, camera_movement_y = measure_xy_distance(
                            old_feature_point, new_feature_point
                        )

            if max_distance > self.minimum_distance:
                camera_move.append([camera_movement_x, camera_movement_y])
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.feature)
            else:
                camera_move.append([0, 0])

            old_gray = frame_gray.copy()

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_move, f)

        return camera_move
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            if frame_num < len(camera_movement_per_frame):
                x_movement, y_movement = camera_movement_per_frame[frame_num]
            else:
                x_movement, y_movement = 0.0, 0.0

            frame = cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )

            frame = cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
            )
            yield frame





