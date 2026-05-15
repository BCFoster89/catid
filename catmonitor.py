import os
import secrets
import threading

import cv2
from picamera2 import Picamera2

import stream_server
from config import (
    FRAMERATE,
    JPEG_QUALITY,
    RESOLUTION,
    STREAM_PORT,
    TIMELAPSE_DIR,
    TIMELAPSE_INTERVAL,
    TOKEN_FILE,
)
from timelapse import TimelapseSaver


class FrameBuffer:
    def __init__(self):
        self._frame = None
        self._lock = threading.Lock()

    def update(self, jpeg_bytes):
        with self._lock:
            self._frame = jpeg_bytes

    def read(self):
        with self._lock:
            return self._frame


def load_or_create_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    token = secrets.token_urlsafe(16)
    with open(TOKEN_FILE, "w") as f:
        f.write(token)
    return token


def main():
    token = load_or_create_token()
    buffer = FrameBuffer()

    stream_server.init(buffer, token)
    flask_thread = threading.Thread(
        target=lambda: stream_server.app.run(host="0.0.0.0", port=STREAM_PORT),
        daemon=True,
    )
    flask_thread.start()

    timelapse = TimelapseSaver(buffer, TIMELAPSE_INTERVAL, TIMELAPSE_DIR)
    timelapse.start()

    camera = Picamera2()
    cam_config = camera.create_video_configuration(
        main={"size": RESOLUTION, "format": "RGB888"},
        controls={"FrameRate": FRAMERATE},
    )
    camera.configure(cam_config)
    camera.start()

    print(f"Stream live at  http://localhost:{STREAM_PORT}/{token}")
    print(f"Share token:    {token}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            frame = camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                buffer.update(jpeg.tobytes())
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
