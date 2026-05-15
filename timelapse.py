import os
import threading
import time
from datetime import datetime


class TimelapseSaver:
    def __init__(self, frame_buffer, interval, output_dir):
        self._buffer = frame_buffer
        self._interval = interval
        self._output_dir = output_dir
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while True:
            time.sleep(self._interval)
            frame = self._buffer.read()
            if frame is None:
                continue
            now = datetime.now()
            day_dir = os.path.join(self._output_dir, now.strftime("%Y%m%d"))
            os.makedirs(day_dir, exist_ok=True)
            path = os.path.join(day_dir, now.strftime("%H%M%S") + ".jpg")
            with open(path, "wb") as f:
                f.write(frame)
            print(f"Timelapse: saved {path}")
