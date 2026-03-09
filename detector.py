import cv2
import numpy as np
from config import MOTION_THRESHOLD, BLUR_SIZE


class MotionDetector:
    def __init__(self):
        self.prev_gray = None

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, np.zeros_like(gray)

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        changed_pixels = cv2.countNonZero(mask)

        return changed_pixels >= MOTION_THRESHOLD, mask
