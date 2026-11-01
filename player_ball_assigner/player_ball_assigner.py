import sys
sys.path.append('../')
from utils import get_center_of_box, measure_distance

class PlayerBallAssigner():
    def __init__(self, hysteresis: int = 0):
        self.max_distance = 70
        self.hysteresis = int(hysteresis)
        self.current_player = -1
        self.candidate_player = -1
        self.candidate_count = 0

    def reset(self):
        self.current_player = -1
        self.candidate_player = -1
        self.candidate_count = 0

    def assign_ball_to_player(self, players, ball_bbox):
        if ball_bbox is None:
            raw_assigned_player = -1
        else:
            ball_position = get_center_of_box(ball_bbox)
            min_distance = 99999
            raw_assigned_player = -1

            for player_id, player in players.items():
                player_bbox = player['bbox']

                distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
                distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
                distance = min(distance_left, distance_right)

                if distance < self.max_distance:
                    if distance < min_distance:
                        min_distance = distance
                        raw_assigned_player = player_id

        # Hysteresis <= 1: giữ nguyên 100% hành vi cũ
        if self.hysteresis <= 1:
            return raw_assigned_player

        # Khi bật hysteresis (N >= 2): chủ bóng chỉ đổi khi ứng viên mới thắng liên tục >= N frame
        if raw_assigned_player != self.current_player:
            if raw_assigned_player == self.candidate_player:
                self.candidate_count += 1
            else:
                self.candidate_player = raw_assigned_player
                self.candidate_count = 1

            if self.candidate_count >= self.hysteresis:
                self.current_player = self.candidate_player
                self.candidate_player = -1
                self.candidate_count = 0
        else:
            self.candidate_player = -1
            self.candidate_count = 0

        # Nếu chủ bóng hiện tại không xuất hiện trong frame, trả về -1
        if self.current_player != -1 and self.current_player not in players:
            return -1

        return self.current_player
