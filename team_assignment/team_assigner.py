from collections import defaultdict, deque
from sklearn.cluster import KMeans
import numpy as np
import cv2


class TeamAssigner:
    def __init__(self, window_size=15):
        """
        window_size: Số frame gần nhất dùng để bỏ phiếu đa số (Majority Voting).
                     15 frame tương đương ~0.5s ở video 30fps.
        """
        self.team_colors = {}
        self.player_team_dict = {}  # Lưu team hiện tại (kết quả đa số)
        self.player_team_history = defaultdict(lambda: deque(maxlen=window_size))
        self.window_size = window_size
        self.kmeans = None

    def get_player_color(self, frame, bbox):
        h_frame, w_frame = frame.shape[:2]
        x1 = max(0, min(int(bbox[0]), w_frame - 1))
        y1 = max(0, min(int(bbox[1]), h_frame - 1))
        x2 = max(0, min(int(bbox[2]), w_frame))
        y2 = max(0, min(int(bbox[3]), h_frame))

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return None

        img = frame[y1:y2, x1:x2]
        top_half = img[0 : int(img.shape[0] / 2), :]

        if top_half.size == 0 or top_half.shape[0] < 2 or top_half.shape[1] < 2:
            return None

        top_half_small = cv2.resize(top_half, (24, 24), interpolation=cv2.INTER_AREA)
        data = top_half_small.reshape(-1, 3).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            data, 2, None, criteria, 1, cv2.KMEANS_PP_CENTERS
        )

        labels_img = labels.reshape(24, 24)
        corner = [
            labels_img[0, 0],
            labels_img[0, -1],
            labels_img[-1, 0],
            labels_img[-1, -1],
        ]
        not_player = max(set(corner), key=corner.count)
        player_cluster = 1 - not_player
        player_color = centers[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detected):
        player_colors = []
        for _, player in player_detected.items():
            bbox = player["bbox"]
            color = self.get_player_color(frame, bbox)
            if color is not None:
                player_colors.append(color)

        if len(player_colors) < 2:
            return

        player_colors = np.array(player_colors)

        # Nếu có từ 6 người trở lên, phân thành 3 cụm để tách riêng ngoại lai
        # (trọng tài áo đen hoặc thủ môn), sau đó chọn 2 cụm đông người nhất làm 2 đội.
        if len(player_colors) >= 6:
            kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
            kmeans.fit(player_colors)
            counts = np.bincount(kmeans.labels_, minlength=3)
            top2 = np.argsort(counts)[-2:]  # 2 cụm đông nhất
            self.team_colors[1] = tuple(map(int, kmeans.cluster_centers_[top2[0]]))
            self.team_colors[2] = tuple(map(int, kmeans.cluster_centers_[top2[1]]))
        else:
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
            kmeans.fit(player_colors)
            self.team_colors[1] = tuple(map(int, kmeans.cluster_centers_[0]))
            self.team_colors[2] = tuple(map(int, kmeans.cluster_centers_[1]))

        self.kmeans = kmeans

    def get_player_team(self, frame, player_bbox, player_id):
        player_color = self.get_player_color(frame, player_bbox)

        if player_color is not None and 1 in self.team_colors and 2 in self.team_colors:
            d1 = np.linalg.norm(player_color - self.team_colors[1])
            d2 = np.linalg.norm(player_color - self.team_colors[2])
            raw_team_id = 1 if d1 < d2 else 2

            self.player_team_history[player_id].append(raw_team_id)

        if not self.player_team_history[player_id]:
            return self.player_team_dict.get(player_id, 1)

        votes = self.player_team_history[player_id]
        team_id = max(set(votes), key=votes.count)

        self.player_team_dict[player_id] = team_id

        return team_id
