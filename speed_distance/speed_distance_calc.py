import cv2 
from utils import measure_distance, get_foot_position


class SpeedAndDistance_Calc:
    def __init__(self, fps):
        self.frame_window = 5
        self.frame_rate = fps

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        for object_name, object_tracks in tracks.items():
            if object_name == "ball" or object_name == "referees":
                continue
            number_of_frames = len(object_tracks)
            for frame_num in range(0, number_of_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)
                if last_frame <= frame_num:
                    continue
                for track_id, _ in object_tracks[frame_num].items():
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # RÀO CHẮN GIAI ĐOẠN 3: Kiểm tra toàn bộ các frame trung gian trong cửa sổ tính toán
                    # Nếu bất kỳ frame nào bị mất dấu, UNREGISTERED hoặc phép chiếu không hợp lệ -> BỎ QUA cửa sổ này
                    # Tuyệt đối không tính vận tốc hay cộng dồn quãng đường nhảy vọt qua khoảng mất hiệu chuẩn / cắt cảnh
                    window_valid = True
                    for f_mid in range(frame_num, last_frame + 1):
                        if track_id not in object_tracks[f_mid]:
                            window_valid = False
                            break
                        f_info = object_tracks[f_mid][track_id]
                        if (
                            f_info.get("calibration_status") == "UNREGISTERED"
                            or not f_info.get("projection_valid", True)
                            or f_info.get("position_transformed") is None
                        ):
                            window_valid = False
                            break

                    if not window_valid:
                        continue

                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    if start_position is None or end_position is None:
                        continue
                    distance_covered = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    has_predicted = any(
                        object_tracks[f_mid][track_id].get("calibration_status") == "PREDICTED"
                        for f_mid in range(frame_num, last_frame + 1)
                    )

                    if object_name not in total_distance:
                        total_distance[object_name] = {}
                    if track_id not in total_distance[object_name]:
                        total_distance[object_name][track_id] = 0
                    total_distance[object_name][track_id] += distance_covered
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object_name][frame_num_batch]:
                            continue
                        is_frame_pred = (tracks[object_name][frame_num_batch][track_id].get("calibration_status") == "PREDICTED")
                        tracks[object_name][frame_num_batch][track_id]['is_predicted'] = is_frame_pred
                        tracks[object_name][frame_num_batch][track_id]['speed_is_predicted'] = has_predicted
                        tracks[object_name][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object_name][frame_num_batch][track_id]['distance'] = total_distance[object_name][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        for frame_num, frame in enumerate(frames):
            text_entries = []
            for object_name, object_tracks in tracks.items():
                if object_name == "ball" or object_name == "referees":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    # Không vẽ HUD tốc độ/cự ly lên người thủ môn
                    if track_info.get("is_goalkeeper", False):
                        continue
                    if "speed" in track_info:
                        speed = track_info.get("speed", None)
                        distance = track_info.get("distance", None)

                        if speed is None or distance is None:
                            continue

                        bbox = track_info["bbox"]
                        position = get_foot_position(bbox)
                        pos = [int(position[0]), int(position[1]) + 40]
                        text_entries.append({
                            "pos": pos,
                            "speed": speed,
                            "distance": distance
                        })

            # Cơ chế Chống đè chữ HUD (Anti-Collision HUD):
            # Sắp xếp theo trục Y và dịch chuyển vị trí nếu 2 cầu thủ đứng sát nhau
            text_entries.sort(key=lambda item: item["pos"][1])
            placed_boxes = []

            for entry in text_entries:
                x, y = entry["pos"][0], entry["pos"][1]
                box_w, box_h = 90, 42
                for bx, by in placed_boxes:
                    if abs(x - bx) < box_w and abs(y - by) < box_h:
                        y = by + box_h + 4

                placed_boxes.append((x, y))
                cv2.putText(frame, f"{entry['speed']:.2f} km/h", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, f"{entry['distance']:.2f} m", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            yield frame
