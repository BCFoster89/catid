import csv
import os
from datetime import datetime

import cv2

from config import CAPTURES_DIR, LOG_FILE


class CatLogger:
    def __init__(self):
        os.makedirs(CAPTURES_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "cat", "image_path"])

    def log(self, frame, cat_id):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{cat_id}.jpg"
        image_path = os.path.join(CAPTURES_DIR, filename)

        cv2.imwrite(image_path, frame)

        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), cat_id, image_path])

        return image_path
