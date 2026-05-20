import glob
import io
import json
import logging
import os
import time
import zipfile

from flask import Flask, Response, abort, redirect, send_from_directory

app = Flask(__name__)
_buffer = None
_token = None
_framerate = 15
_timelapse_dir = "timelapse"
_public_url_file = "public_url.txt"


def init(frame_buffer, token, framerate=15, timelapse_dir="timelapse", public_url_file="public_url.txt"):
    global _buffer, _token, _framerate, _timelapse_dir, _public_url_file
    _buffer = frame_buffer
    _token = token
    _framerate = framerate
    _timelapse_dir = os.path.abspath(timelapse_dir)
    _public_url_file = public_url_file
    logging.getLogger("werkzeug").setLevel(logging.ERROR)


@app.route("/")
def root():
    if _token:
        return redirect(f"/{_token}")
    abort(404)


@app.route("/<token>/url")
def current_url(token):
    if token != _token:
        abort(404)
    try:
        with open(_public_url_file) as f:
            url = f.read().strip()
        if url:
            return redirect(url)
    except FileNotFoundError:
        pass
    return Response(
        "<!doctype html><html><body style='font-family:sans-serif;padding:2em'>"
        "<p>Tunnel is currently offline. Try again in a moment.</p></body></html>",
        mimetype="text/html",
    )


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
  <title>Cat Cam</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #000; display: flex; flex-direction: column; height: 100vh; }}
    #feed {{ width: 100%; flex: 1; object-fit: contain; min-height: 0; }}
    #bar {{ display: flex; justify-content: center; align-items: center; gap: 12px; padding: 8px; background: #111; flex-shrink: 0; }}
    button {{ background: #333; color: #fff; border: none; border-radius: 6px; padding: 8px 18px; font-size: 15px; cursor: pointer; }}
    button:active {{ background: #555; }}
    #tl-info {{ color: #aaa; font-size: 13px; font-family: sans-serif; min-width: 80px; text-align: center; }}
  </style>
</head>
<body>
  <img id="feed" alt="live feed">
  <div id="bar">
    <button id="tl-btn" onclick="startTimelapse()">&#9654; Timelapse</button>
    <button id="pause-btn" style="display:none" onclick="togglePause()">&#9646;&#9646; Pause</button>
    <span id="tl-info"></span>
    <button id="live-btn" style="display:none" onclick="stopTimelapse()">&#10005; Back to Live</button>
    <button onclick="location.href='/{token}/timelapse-download'">&#8595; Download</button>
  </div>
  <script>
    var img = document.getElementById('feed');
    var tlBtn = document.getElementById('tl-btn');
    var pauseBtn = document.getElementById('pause-btn');
    var liveBtn = document.getElementById('live-btn');
    var tlInfo = document.getElementById('tl-info');
    var live = true;
    var paused = false;
    var frames = [];
    var idx = 0;
    var timer = null;

    function refresh() {{
      if (!live) return;
      var n = new Image();
      n.onload = function() {{ img.src = n.src; if (live) refresh(); }};
      n.onerror = function() {{ if (live) setTimeout(refresh, 200); }};
      n.src = '/{token}/snapshot?' + Date.now();
    }}

    function startTimelapse() {{
      fetch('/{token}/timelapse-list')
        .then(function(r) {{ return r.json(); }})
        .then(function(files) {{
          if (!files.length) {{ tlInfo.textContent = 'No frames yet'; return; }}
          frames = files;
          idx = 0;
          live = false;
          paused = false;
          tlBtn.style.display = 'none';
          pauseBtn.style.display = '';
          liveBtn.style.display = '';
          playFrame();
        }});
    }}

    function playFrame() {{
      if (live || paused) return;
      var n = new Image();
      n.onload = function() {{
        if (live || paused) return;
        img.src = n.src;
        tlInfo.textContent = (idx + 1) + ' / ' + frames.length;
        idx = (idx + 1) % frames.length;
        timer = setTimeout(playFrame, 50);
      }};
      n.onerror = function() {{ if (!live && !paused) setTimeout(playFrame, 200); }};
      n.src = '/{token}/timelapse/' + frames[idx] + '?' + Date.now();
    }}

    function togglePause() {{
      paused = !paused;
      pauseBtn.innerHTML = paused ? '&#9654; Play' : '&#9646;&#9646; Pause';
      if (!paused) playFrame();
    }}

    function stopTimelapse() {{
      live = true;
      paused = false;
      clearTimeout(timer);
      pauseBtn.style.display = 'none';
      pauseBtn.innerHTML = '&#9646;&#9646; Pause';
      liveBtn.style.display = 'none';
      tlBtn.style.display = '';
      tlInfo.textContent = '';
      refresh();
    }}

    refresh();
  </script>
</body>
</html>"""
    return html


@app.route("/<token>/timelapse-download")
def timelapse_download(token):
    if token != _token:
        abort(404)
    files = sorted(glob.glob(os.path.join(_timelapse_dir, "*", "*.jpg")))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for f in files:
            arcname = os.path.relpath(f, _timelapse_dir).replace(os.sep, "/")
            zf.write(f, arcname)
    buf.seek(0)
    return Response(
        buf.read(),
        mimetype="application/zip",
        headers={"Content-Disposition": "attachment; filename=\"timelapse.zip\""},
    )


@app.route("/<token>/timelapse-list")
def timelapse_list(token):
    if token != _token:
        abort(404)
    files = sorted(glob.glob(os.path.join(_timelapse_dir, "*", "*.jpg")))
    rel = [os.path.relpath(f, _timelapse_dir).replace(os.sep, "/") for f in files]
    return Response(json.dumps(rel), mimetype="application/json")


@app.route("/<token>/timelapse/<path:filename>")
def timelapse_image(token, filename):
    if token != _token:
        abort(404)
    return send_from_directory(_timelapse_dir, filename)


@app.route("/<token>/snapshot")
def snapshot(token):
    if token != _token:
        abort(404)
    frame = _buffer.read()
    if frame is None:
        abort(503)
    return Response(
        frame,
        mimetype="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/<token>/feed")
def feed(token):
    if token != _token:
        abort(404)
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )
