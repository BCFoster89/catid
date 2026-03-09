import time

import cv2
import numpy as np
from picamera2 import Picamera2

from config import RESOLUTION, FRAMERATE, COOLDOWN_SECONDS
from detector import MotionDetector
from identifier import CatIdentifier
from logger import CatLogger


def main():
    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": RESOLUTION, "format": "BGR888"},
        controls={"FrameRate": FRAMERATE},
    )
    camera.configure(config)
    camera.start()

    detector = MotionDetector()
    identifier = CatIdentifier()
    logger = CatLogger()

    last_capture_time = 0.0

    print("Cat monitor running. Press Ctrl+C to stop.")

    try:
        while True:
            frame = camera.capture_array()
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            motion, mask = detector.detect(frame)

            if motion and (time.time() - last_capture_time) >= COOLDOWN_SECONDS:
                time.sleep(0.5)
                photo = camera.capture_array()
                photo = cv2.rotate(photo, cv2.ROTATE_180)
                cat_id, bbox = identifier.identify(photo, mask)
                if bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(photo, (x, y), (x + w, y + h), (0, 255, 0), 2)
                image_path = logger.log(photo, cat_id)
                last_capture_time = time.time()
                print(f"Motion detected — cat: {cat_id} — saved {image_path}")

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
