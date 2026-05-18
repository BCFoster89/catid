import argparse
import os
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time

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


def _stable_subdomain(token):
    safe = re.sub(r'[^a-z0-9]', '', token.lower())[:20]
    return f"cat{safe}"


def _try_tunnel(cmd, pattern, timeout):
    url_holder = [None]
    url_ready = threading.Event()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def _read():
        for line in proc.stdout:
            m = re.search(pattern, line)
            if m:
                url_holder[0] = m.group(0).rstrip(".")
                url_ready.set()

    threading.Thread(target=_read, daemon=True).start()
    url_ready.wait(timeout=timeout)
    return proc, url_holder[0]


def _start_public_tunnel(port, subdomain):
    proc, url = _try_tunnel(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
         "-R", f"{subdomain}:80:localhost:{port}", "serveo.net"],
        r'https://\S+\.serveo\.net', 20,
    )
    if url:
        return proc, url

    proc.terminate()
    return _try_tunnel(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
         "-R", f"80:localhost:{port}", "nokey@localhost.run"],
        r'https://\S+\.lhr\.life', 15,
    )


_tunnel_proc = None


def _tunnel_monitor(port, token):
    global _tunnel_proc
    subdomain = _stable_subdomain(token)
    while True:
        try:
            proc, url = _start_public_tunnel(port, subdomain)
            _tunnel_proc = proc
            if url:
                if "serveo.net" in url:
                    print("Public tunnel:   connected (stable URL).")
                else:
                    print(f"Public link (serveo unavailable, URL may change): {url}/{token}")
            else:
                print("Tunnel failed, retrying in 30s...")
                proc.terminate()
                _tunnel_proc = None
                time.sleep(30)
                continue
            proc.wait()
            _tunnel_proc = None
            print("Tunnel dropped, reconnecting in 5s...")
            time.sleep(5)
        except Exception as e:
            print(f"Tunnel error: {e}; retrying in 30s...")
            _tunnel_proc = None
            time.sleep(30)


def _get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def load_or_create_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            return f.read().strip()
    token = secrets.token_urlsafe(16)
    with open(TOKEN_FILE, "w") as f:
        f.write(token)
    return token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Delete all timelapse frames and exit")
    args = parser.parse_args()

    if args.clear:
        if os.path.exists(TIMELAPSE_DIR):
            shutil.rmtree(TIMELAPSE_DIR)
            print(f"Cleared {TIMELAPSE_DIR}/")
        else:
            print("No timelapse directory found.")
        return

    token = load_or_create_token()
    buffer = FrameBuffer()

    stream_server.init(buffer, token, FRAMERATE, TIMELAPSE_DIR)
    flask_thread = threading.Thread(
        target=lambda: stream_server.app.run(host="0.0.0.0", port=STREAM_PORT, threaded=True),
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

    threading.Thread(target=lambda: _tunnel_monitor(STREAM_PORT, token), daemon=True).start()

    subdomain = _stable_subdomain(token)
    lan_ip = _get_lan_ip()
    print(f"Stream live at  http://localhost:{STREAM_PORT}/{token}")
    if lan_ip:
        print(f"On your network: http://{lan_ip}:{STREAM_PORT}/{token}")
    print(f"Public link:     https://{subdomain}.serveo.net/{token}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            try:
                frame = camera.capture_array()
            except Exception as e:
                print(f"Camera error: {e}; retrying in 1s...")
                time.sleep(1)
                continue
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                buffer.update(jpeg.tobytes())
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        if _tunnel_proc:
            _tunnel_proc.terminate()
        camera.stop()


if __name__ == "__main__":
    main()
