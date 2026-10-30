import os
import cv2
import numpy as np


class MinimapRenderer:
    """
    MinimapRenderer: Sa bàn chiến thuật 2D (Tactical Radar) cho bóng đá.
    Hiển thị toàn bộ cầu thủ 2 đội, thủ môn, trọng tài và quả bóng chuyển động
    đồng bộ 100% thời gian thực trên mặt sân cỏ chuẩn FIFA (105m x 68m).
    """

    def __init__(
        self,
        fps=25.0,
        canvas_width=1280,
        canvas_height=720,
        pitch_length=105.0,
        pitch_width=68.0,
    ):
        self.fps = fps
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Khu vực Header Dashboard phía trên
        self.header_height = 70

        # Kích thước và tọa độ mặt sân bóng trên canvas
        self.pitch_h = 580
        self.pitch_w = int(round(self.pitch_h * (pitch_length / pitch_width)))  # ~895px
        self.pitch_x0 = (self.canvas_width - self.pitch_w) // 2
        self.pitch_y0 = self.header_height + (self.canvas_height - self.header_height - self.pitch_h) // 2

        # Màu sắc phong cách sân cỏ xanh cổ điển (BGR)
        self.COLOR_BG = (28, 40, 32)            # Nền viền ngoài tối
        self.COLOR_GRASS_1 = (46, 125, 50)       # Cỏ sọc xanh tươi
        self.COLOR_GRASS_2 = (38, 107, 42)       # Cỏ sọc xanh đậm
        self.COLOR_LINES = (245, 245, 245)       # Vạch kẻ sân trắng
        self.COLOR_HEADER = (18, 26, 20)         # Nền thanh dashboard

        # Vẽ sẵn mặt sân cố định (Base Pitch Canvas) một lần duy nhất để tối ưu tốc độ
        self.base_pitch = self._create_base_pitch()

    def meter_to_pixel(self, x_m, y_m):
        """Quy đổi tọa độ mét trên sân (105x68) sang tọa độ pixel trên canvas."""
        px = int(round(self.pitch_x0 + (x_m / self.pitch_length) * self.pitch_w))
        py = int(round(self.pitch_y0 + (y_m / self.pitch_width) * self.pitch_h))
        return px, py

    def _create_base_pitch(self):
        """Tạo ảnh tĩnh mặt sân cỏ xanh với đầy đủ vạch kẻ FIFA chuẩn."""
        img = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        img[:] = self.COLOR_BG

        # 1. Vẽ các sọc cỏ cắt xen kẽ (Lawn Stripes)
        num_stripes = 15
        stripe_w = self.pitch_w / num_stripes
        for i in range(num_stripes):
            x1 = int(round(self.pitch_x0 + i * stripe_w))
            x2 = int(round(self.pitch_x0 + (i + 1) * stripe_w))
            color = self.COLOR_GRASS_1 if i % 2 == 0 else self.COLOR_GRASS_2
            cv2.rectangle(img, (x1, self.pitch_y0), (x2, self.pitch_y0 + self.pitch_h), color, -1)

        # 2. Vạch đường biên ngoài sân
        x0, y0 = self.pitch_x0, self.pitch_y0
        pw, ph = self.pitch_w, self.pitch_h
        x_max, y_max = x0 + pw, y0 + ph
        cv2.rectangle(img, (x0, y0), (x_max, y_max), self.COLOR_LINES, 2, cv2.LINE_AA)

        # 3. Đường giữa sân & Vòng tròn trung tâm
        mid_x = x0 + pw // 2
        mid_y = y0 + ph // 2
        cv2.line(img, (mid_x, y0), (mid_x, y_max), self.COLOR_LINES, 2, cv2.LINE_AA)
        cv2.circle(img, (mid_x, mid_y), 4, self.COLOR_LINES, -1, cv2.LINE_AA)

        r_center_circle = int(round((9.15 / self.pitch_length) * pw))
        cv2.circle(img, (mid_x, mid_y), r_center_circle, self.COLOR_LINES, 2, cv2.LINE_AA)

        # 4. Khu vực vòng cấm địa (16.5m) và 5m50 hai đầu sân
        def draw_goal_area(is_left=True):
            sgn = 1 if is_left else -1
            base_x = x0 if is_left else x_max

            # Vòng cấm 16.5m
            pen_w = int(round((16.5 / self.pitch_length) * pw)) * sgn
            pen_y1 = int(round(y0 + ((34.0 - 20.16) / self.pitch_width) * ph))
            pen_y2 = int(round(y0 + ((34.0 + 20.16) / self.pitch_width) * ph))
            cv2.rectangle(img, (base_x, pen_y1), (base_x + pen_w, pen_y2), self.COLOR_LINES, 2, cv2.LINE_AA)

            # Khu vực 5m50 (Six-yard box)
            six_w = int(round((5.5 / self.pitch_length) * pw)) * sgn
            six_y1 = int(round(y0 + ((34.0 - 9.16) / self.pitch_width) * ph))
            six_y2 = int(round(y0 + ((34.0 + 9.16) / self.pitch_width) * ph))
            cv2.rectangle(img, (base_x, six_y1), (base_x + six_w, six_y2), self.COLOR_LINES, 2, cv2.LINE_AA)

            # Chấm phạt đền 11m
            spot_x = int(round(base_x + int(round((11.0 / self.pitch_length) * pw)) * sgn))
            cv2.circle(img, (spot_x, mid_y), 3, self.COLOR_LINES, -1, cv2.LINE_AA)

            # Cung phạt đền (D-arc)
            arc_angles = (-53, 53) if is_left else (127, 233)
            cv2.ellipse(
                img,
                (spot_x, mid_y),
                (r_center_circle, r_center_circle),
                0,
                arc_angles[0],
                arc_angles[1],
                self.COLOR_LINES,
                2,
                cv2.LINE_AA,
            )

            # Cầu môn (Goal net ngoài vạch vôi)
            goal_depth = 12 * (-sgn)
            goal_y1 = int(round(y0 + ((34.0 - 3.66) / self.pitch_width) * ph))
            goal_y2 = int(round(y0 + ((34.0 + 3.66) / self.pitch_width) * ph))
            cv2.rectangle(img, (base_x, goal_y1), (base_x + goal_depth, goal_y2), (180, 180, 180), 2, cv2.LINE_AA)

        draw_goal_area(is_left=True)
        draw_goal_area(is_left=False)

        # 5. Cung phạt góc (Corner Arcs: chuẩn FIFA 1.0m)
        r_corner = int(round((1.0 / self.pitch_length) * pw))
        cv2.ellipse(img, (x0, y0), (r_corner, r_corner), 0, 0, 90, self.COLOR_LINES, 2, cv2.LINE_AA)
        cv2.ellipse(img, (x0, y_max), (r_corner, r_corner), 0, 270, 360, self.COLOR_LINES, 2, cv2.LINE_AA)
        cv2.ellipse(img, (x_max, y0), (r_corner, r_corner), 0, 90, 180, self.COLOR_LINES, 2, cv2.LINE_AA)
        cv2.ellipse(img, (x_max, y_max), (r_corner, r_corner), 0, 180, 270, self.COLOR_LINES, 2, cv2.LINE_AA)

        return img

    def _draw_header(self, canvas, frame_idx, team_ball_control, team_colors):
        """Vẽ thanh Dashboard phía trên: Thời gian trận đấu và % kiểm soát bóng của 2 đội."""
        # Nền Header bar
        cv2.rectangle(canvas, (0, 0), (self.canvas_width, self.header_height), self.COLOR_HEADER, -1)
        cv2.line(canvas, (0, self.header_height), (self.canvas_width, self.header_height), (50, 70, 55), 2)

        # 1. Tiêu đề chính giữa & Thời gian trận đấu (MM:SS)
        elapsed_sec = int(round(frame_idx / self.fps))
        mm = elapsed_sec // 60
        ss = elapsed_sec % 60
        time_text = f"TIME {mm:02d}:{ss:02d}"
        cv2.putText(canvas, time_text, (self.canvas_width // 2 - 65, 42), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # 2. Tính tỉ lệ % kiểm soát bóng (O(1) bằng precomputed cumulative sum)
        team_1_pct = 50.0
        team_2_pct = 50.0
        if team_ball_control is not None and len(team_ball_control) > 0:
            if not hasattr(self, "_cached_ctrl_id") or self._cached_ctrl_id != id(team_ball_control):
                arr = np.asarray(team_ball_control)
                self._cached_c1 = np.cumsum(arr == 1)
                self._cached_c2 = np.cumsum(arr == 2)
                self._cached_ctrl_id = id(team_ball_control)

            idx = min(frame_idx, len(self._cached_c1) - 1)
            c1 = int(self._cached_c1[idx])
            c2 = int(self._cached_c2[idx])
            total = c1 + c2
            if total > 0:
                team_1_pct = (c1 / total) * 100.0
                team_2_pct = (c2 / total) * 100.0

        # Lấy màu sắc đại diện của 2 đội
        c_team1 = team_colors.get(1, (60, 220, 100))
        c_team2 = team_colors.get(2, (230, 230, 230))
        if isinstance(c_team1, (np.ndarray, list)):
            c_team1 = tuple(int(x) for x in c_team1)
        if isinstance(c_team2, (np.ndarray, list)):
            c_team2 = tuple(int(x) for x in c_team2)

        # 3. Huy hiệu Đội 1 (Bên trái)
        cv2.circle(canvas, (self.pitch_x0 + 15, 35), 10, c_team1, -1, cv2.LINE_AA)
        cv2.circle(canvas, (self.pitch_x0 + 15, 35), 10, (255, 255, 255), 1, cv2.LINE_AA)
        t1_label = f"TEAM 1: {team_1_pct:.1f}%"
        cv2.putText(canvas, t1_label, (self.pitch_x0 + 35, 42), cv2.FONT_HERSHEY_DUPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)

        # 4. Huy hiệu Đội 2 (Bên phải)
        right_box_x = self.pitch_x0 + self.pitch_w - 200
        cv2.circle(canvas, (right_box_x + 15, 35), 10, c_team2, -1, cv2.LINE_AA)
        cv2.circle(canvas, (right_box_x + 15, 35), 10, (255, 255, 255), 1, cv2.LINE_AA)
        t2_label = f"TEAM 2: {team_2_pct:.1f}%"
        cv2.putText(canvas, t2_label, (right_box_x + 35, 42), cv2.FONT_HERSHEY_DUPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)

    def render_frame(self, frame_idx, tracks, team_ball_control, prev_ball_pos=None):
        """Vẽ toàn bộ sa bàn 2D của đúng một frame (đối tượng, cầu thủ, bóng, dashboard)."""
        canvas = self.base_pitch.copy()

        # Tìm màu áo 2 đội
        team_colors = {}
        for f in tracks.get("players", []):
            for p in f.values():
                t = p.get("team")
                tc = p.get("team_color")
                if t in (1, 2) and tc is not None and t not in team_colors:
                    team_colors[t] = tc
            if 1 in team_colors and 2 in team_colors:
                break

        # 1. Vẽ Header Dashboard
        self._draw_header(canvas, frame_idx, team_ball_control, team_colors)

        # 2. Vẽ Trọng tài (Referees)
        referees = tracks.get("referees", [])
        if frame_idx < len(referees):
            for rid, rinfo in referees[frame_idx].items():
                pos = rinfo.get("position_transformed")
                if pos is not None and np.all(np.isfinite(pos)):
                    px, py = self.meter_to_pixel(pos[0], pos[1])
                    # Vòng tròn màu xanh lá neon đặc trưng
                    cv2.circle(canvas, (px, py), 9, (0, 235, 130), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 9, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(canvas, "R", (px - 5, py + 4), cv2.FONT_HERSHEY_DUPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Vẽ Cầu thủ & Thủ môn (Players & Goalkeepers)
        players = tracks.get("players", [])
        if frame_idx < len(players):
            for pid, pinfo in players[frame_idx].items():
                pos = pinfo.get("position_transformed")
                if pos is None or not np.all(np.isfinite(pos)):
                    continue

                px, py = self.meter_to_pixel(pos[0], pos[1])
                t = pinfo.get("team", 1)
                t_color = pinfo.get("team_color", team_colors.get(t, (0, 0, 255)))
                if isinstance(t_color, (np.ndarray, list)):
                    t_color = tuple(int(c) for c in t_color)

                is_gk = pinfo.get("is_goalkeeper", False)
                has_ball = pinfo.get("has_ball", False)

                # Hiệu ứng hào quang Halo nếu đang kiểm soát bóng
                if has_ball:
                    cv2.circle(canvas, (px, py), 18, (0, 230, 255), 2, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 22, (0, 230, 255), 1, cv2.LINE_AA)

                # Thể hiện Thủ môn vs Cầu thủ outfield
                if is_gk:
                    # Thủ môn: vòng tròn lớn hơn viền vàng rực
                    cv2.circle(canvas, (px, py), 13, (0, 215, 255), 3, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 11, t_color, -1, cv2.LINE_AA)
                    cv2.putText(canvas, "GK", (px - 7, py + 4), cv2.FONT_HERSHEY_DUPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    # Cầu thủ bình thường: vòng tròn mang màu áo đội
                    cv2.circle(canvas, (px, py), 11, t_color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 11, (255, 255, 255), 1, cv2.LINE_AA)

                    # Số ID cầu thủ
                    tid_str = str(pid)
                    font_scale = 0.35 if len(tid_str) <= 2 else 0.28
                    offset_x = 4 if len(tid_str) == 1 else (7 if len(tid_str) == 2 else 9)
                    # Chọn màu chữ tương phản (đen hoặc trắng)
                    luminance = 0.299 * t_color[2] + 0.587 * t_color[1] + 0.114 * t_color[0]
                    text_color = (0, 0, 0) if luminance > 130 else (255, 255, 255)
                    cv2.putText(canvas, tid_str, (px - offset_x, py + 4), cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, 1, cv2.LINE_AA)

        # 4. Vẽ Quả bóng (Ball)
        balls = tracks.get("ball", [])
        curr_ball_pos = None

        # prev_ball_pos có thể là tuple (pos, last_frame) hoặc pos trực tiếp
        if isinstance(prev_ball_pos, tuple) and len(prev_ball_pos) == 2 and isinstance(prev_ball_pos[1], (int, type(None))):
            last_ball_pos, last_ball_frame = prev_ball_pos
        else:
            last_ball_pos, last_ball_frame = prev_ball_pos, None

        if frame_idx < len(balls):
            ball_dict = balls[frame_idx]
            if 1 in ball_dict:
                b_info = ball_dict[1]
                pos = b_info.get("position_transformed")
                if pos is not None and np.all(np.isfinite(pos)):
                    curr_ball_pos = self.meter_to_pixel(pos[0], pos[1])

                    # Vẽ vệt di chuyển ngắn (Motion trail) có kiểm tra bước nhảy
                    if last_ball_pos is not None:
                        dt = (frame_idx - last_ball_frame) if last_ball_frame is not None else 1
                        max_jump_px = (15.0 / self.pitch_length) * self.pitch_w
                        dist = float(np.hypot(curr_ball_pos[0] - last_ball_pos[0], curr_ball_pos[1] - last_ball_pos[1]))
                        if dt <= 3 and dist <= max_jump_px:
                            cv2.line(canvas, last_ball_pos, curr_ball_pos, (0, 165, 255), 2, cv2.LINE_AA)

                    # Chấm tròn bóng màu cam/trắng nổi bật
                    cv2.circle(canvas, curr_ball_pos, 7, (0, 140, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, curr_ball_pos, 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, curr_ball_pos, 7, (0, 0, 0), 1, cv2.LINE_AA)

        new_ball_state = (curr_ball_pos, frame_idx) if curr_ball_pos is not None else (last_ball_pos, last_ball_frame)
        return canvas, new_ball_state

    def generate_frames(self, tracks, team_ball_control, total_frames=None):
        """Generator sinh từng frame Minimap liên tục (tiết kiệm RAM tối đa)."""
        if total_frames is None:
            total_frames = max(
                len(tracks.get("players", [])),
                len(tracks.get("referees", [])),
                len(tracks.get("ball", [])),
            )

        prev_ball_pos = None
        for frame_idx in range(total_frames):
            frame, prev_ball_pos = self.render_frame(
                frame_idx, tracks, team_ball_control, prev_ball_pos=prev_ball_pos
            )
            yield frame

    def render_video(self, tracks, team_ball_control, output_path, total_frames=None):
        """
        Render toàn bộ video Minimap và lưu ra file MP4.
        Tốc độ render siêu nhanh (> 120 FPS) vì không cần giải mã lại video gốc.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if total_frames is None:
            total_frames = max(
                len(tracks.get("players", [])),
                len(tracks.get("referees", [])),
                len(tracks.get("ball", [])),
            )

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.canvas_width, self.canvas_height),
        )

        try:
            print(f"-> Đang render video Minimap ({total_frames} frames, {self.fps} FPS) -> {output_path}")
            prev_ball_pos = None
            for frame_idx in range(total_frames):
                frame, prev_ball_pos = self.render_frame(
                    frame_idx, tracks, team_ball_control, prev_ball_pos=prev_ball_pos
                )
                writer.write(frame)
            print(f"-> Render Minimap hoàn tất thành công: {output_path}")
        finally:
            writer.release()

