import os
import threading
import time
from collections import deque
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, request, send_from_directory

from camera import build_camera
from config import settings
from leds import LedController, ShutterButton

app = Flask(__name__)

os.makedirs(settings.captures_dir, exist_ok=True)

capture_lock = threading.Lock()
captures_lock = threading.Lock()
recent_captures = deque(maxlen=100)
started_at = time.time()
last_error = {"message": None}

camera = build_camera(settings)
leds = LedController(
    settings.led_ir_pin,
    settings.led_white_pin,
    mock=settings.mock,
    ir_active_high=settings.led_ir_active_high,
    white_active_high=settings.led_white_active_high,
)
leds.set_mode(LedController.MODE_IR)


def _start_camera():
    global camera
    try:
        camera.start()
        time.sleep(0.3)
        last_error["message"] = None
    except Exception as exc:
        last_error["message"] = "camera start failed: %s" % exc
        from camera import MockCameraSource

        camera = MockCameraSource(
            (settings.preview_width, settings.preview_height),
            (settings.still_width, settings.still_height),
            settings.stream_fps,
            settings.jpeg_quality,
        )
        camera.start()


def _perform_capture(source="api"):
    with capture_lock:
        leds.set_mode(LedController.MODE_OFF)
        time.sleep(settings.white_settle_ms / 1000.0)
        leds.set_mode(LedController.MODE_WHITE)
        time.sleep(settings.white_flash_ms / 1000.0)
        try:
            still = camera.capture_still_jpeg()
        finally:
            leds.set_mode(LedController.MODE_IR)
        stamp = datetime.now(timezone.utc).astimezone()
        filename = "capture_%s.jpg" % stamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = os.path.join(settings.captures_dir, filename)
        with open(path, "wb") as handle:
            handle.write(still)

    if settings.ir_settle_ms:
        time.sleep(settings.ir_settle_ms / 1000.0)

    meta = {
        "filename": filename,
        "url": "/captures/%s" % filename,
        "width": settings.still_width,
        "height": settings.still_height,
        "bytes": len(still),
        "captured_at": stamp.isoformat(),
        "source": source,
        "camera": camera.name,
    }
    with captures_lock:
        recent_captures.appendleft(meta)
    return meta


def _on_shutter():
    try:
        _perform_capture(source="button")
    except Exception as exc:
        last_error["message"] = "shutter capture failed: %s" % exc


shutter = None
if settings.shutter_enabled:
    try:
        shutter = ShutterButton(
            settings.shutter_pin,
            _on_shutter,
            mock=settings.mock,
            bounce_time=settings.shutter_bounce_s,
        )
    except Exception as exc:
        last_error["message"] = "shutter init failed: %s" % exc


def _mjpeg_generator():
    interval = settings.frame_interval
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        frame = camera.latest_jpeg()
        if frame:
            yield boundary + frame + b"\r\n"
        time.sleep(interval)


@app.after_request
def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/")
def index():
    return jsonify(
        {
            "service": "retina-camera",
            "endpoints": ["/viewfinder", "/snapshot", "/capture", "/captures", "/led", "/health"],
        }
    )


@app.route("/viewfinder")
def viewfinder():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot")
def snapshot():
    frame = camera.latest_jpeg()
    if frame is None:
        return jsonify({"error": "no frame available"}), 503
    return Response(frame, mimetype="image/jpeg")


@app.route("/capture", methods=["POST", "GET"])
def capture():
    try:
        return jsonify(_perform_capture(source="api"))
    except Exception as exc:
        last_error["message"] = str(exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/captures")
def captures_index():
    limit = request.args.get("limit", 20, type=int)
    with captures_lock:
        items = list(recent_captures)[: max(1, limit)]
    return jsonify({"captures": items, "count": len(items)})


@app.route("/captures/<path:name>")
def captures(name):
    return send_from_directory(settings.captures_dir, name)


@app.route("/led", methods=["GET", "POST"])
def led():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        pulse = payload.get("pulse")
        if pulse in ("ir", "white"):
            return jsonify(leds.pulse(pulse, int(payload.get("ms", 400))))
        mode = payload.get("mode")
        if mode:
            leds.set_mode(mode)
        if payload.get("brightness") is not None and hasattr(camera, "set_brightness"):
            camera.set_brightness(payload["brightness"])
    return jsonify(leds.status())


@app.route("/health")
def health():
    with captures_lock:
        last = recent_captures[0] if recent_captures else None
    return jsonify(
        {
            "ok": camera.latest_jpeg() is not None,
            "uptime_s": round(time.time() - started_at, 1),
            "camera": camera.status(),
            "led": leds.status(),
            "shutter": shutter.status() if shutter else {"enabled": False, "pin": settings.shutter_pin},
            "last_capture": last,
            "last_error": last_error["message"],
        }
    )


def main():
    _start_camera()
    app.run(
        host=settings.host,
        port=settings.port,
        threaded=True,
        debug=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        camera.stop()
        leds.close()
        if shutter:
            shutter.close()
