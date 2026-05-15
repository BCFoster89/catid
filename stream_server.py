import time

from flask import Flask, Response, abort

app = Flask(__name__)
_buffer = None
_token = None
_framerate = 15


def init(frame_buffer, token, framerate=15):
    global _buffer, _token, _framerate
    _buffer = frame_buffer
    _token = token
    _framerate = framerate


def _mjpeg_generator():
    while True:
        frame = _buffer.read()
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(1 / _framerate)


@app.route("/<token>")
def viewer(token):
    if token != _token:
        abort(404)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Feed</title>
  <style>
    body {{ margin: 0; background: #000; display: flex; justify-content: center; align-items: center; height: 100vh; }}
    img {{ max-width: 100%; max-height: 100vh; }}
  </style>
</head>
<body>
  <img src="/{token}/feed" alt="live feed">
</body>
</html>"""
    return html


@app.route("/<token>/feed")
def feed(token):
    if token != _token:
        abort(404)
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
