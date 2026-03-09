import time

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

            motion, mask = detector.detect(frame)

            if motion and (time.time() - last_capture_time) >= COOLDOWN_SECONDS:
                cat_id = identifier.identify(frame, mask)
                image_path = logger.log(frame, cat_id)
                last_capture_time = time.time()
                print(f"Motion detected — cat: {cat_id} — saved {image_path}")

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
