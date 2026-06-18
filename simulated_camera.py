# -*- coding: utf-8 -*-
"""
Simulated conveyor camera for testing without real hardware.

Features:
- Downloads/uses conveyor dataset via kagglehub
- Finds a usable video file automatically
- OpenCV-like interface:
    - connect()
    - read() -> (ret, frame)
    - release()
    - isOpened()
- FPS throttling
- Loops automatically
"""

import os
import time
import cv2


class simulated_camera:
    def __init__(self, fps=30,VIDEO_PATH=None):
        self.fps = fps
        self.frame_delay = 1.0 / max(1, fps)
        self.last_frame_time = 0.0
        
        if VIDEO_PATH is None:
            self.video_path = "Camera_Settings/simulated_camera_video/simulated_conveyor.mp4"
        else:
            self.video_path = VIDEO_PATH
    
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(
                f"Simulated camera video not found: {self.video_path}"
            )
    
        self.cap = cv2.VideoCapture(self.video_path)
    
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open simulated camera video: {self.video_path}"
            )


    def connect(self, fps=None):
        if fps is not None:
            self.fps = fps
            self.frame_delay = 1.0 / max(1, fps)
        return self

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        now = time.time()
        elapsed = now - self.last_frame_time

        if elapsed < self.frame_delay:
            time.sleep(self.frame_delay - elapsed)

        self.last_frame_time = time.time()

        ret, frame = self.cap.read()

        # Loop video automatically
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if not ret or frame is None:
            return False, None
        frame = cv2.resize(frame,(640,640))
        return True, frame

    def get_frame(self):
        return self.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def reset(self):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def __del__(self):
        self.release()