import cv2 
from pathlib import Path
import numpy as np
def read_video(video_path):
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    def generate_frames():
        try:
            while True:
                ret,frame = capture.read()
                if not ret:
                    break
                yield frame
        finally:
            capture.release()
    return generate_frames(),fps

def save_video(output_video_frames, output_path,fps=30):
    frames = iter(output_video_frames)
    first_frame = next(frames, None)

    if first_frame is None:
        raise ValueError("Video ko co frame !")

    height, width = first_frame.shape[:2]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    try:
        writer.write(first_frame)

        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()