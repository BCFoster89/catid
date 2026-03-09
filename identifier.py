import cv2
import numpy as np
from config import (
    ORANGE_HSV_LOWER, ORANGE_HSV_UPPER,
    BROWN_HSV_LOWER, BROWN_HSV_UPPER,
    MIN_COLOR_PIXELS,
)


class CatIdentifier:
    def identify(self, frame, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "unknown"

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        crop = frame[y:y+h, x:x+w]

        if crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        orange_mask = cv2.inRange(
            hsv,
            np.array(ORANGE_HSV_LOWER),
            np.array(ORANGE_HSV_UPPER),
        )
        brown_mask = cv2.inRange(
            hsv,
            np.array(BROWN_HSV_LOWER),
            np.array(BROWN_HSV_UPPER),
        )

        orange_pixels = cv2.countNonZero(orange_mask)
        brown_pixels = cv2.countNonZero(brown_mask)

        if orange_pixels < MIN_COLOR_PIXELS and brown_pixels < MIN_COLOR_PIXELS:
            return "unknown"

        return "orange" if orange_pixels >= brown_pixels else "brown"
