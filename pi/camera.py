import threading
import time

import cv2
import numpy as np

try:
    from picamera2 import Picamera2

    PICAMERA_AVAILABLE = True
except Exception:
    PICAMERA_AVAILABLE = False


class BaseCamera:
    name = "base"

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def latest_jpeg(self):
        raise NotImplementedError

    def capture_still_jpeg(self, timeout=8.0):
        raise NotImplementedError

    def status(self):
        raise NotImplementedError


class PiCameraSource(BaseCamera):
    name = "picamera2"

    def __init__(self, preview_size, still_size, fps, jpeg_quality):
        if not PICAMERA_AVAILABLE:
            raise RuntimeError("picamera2 is not available on this host")
        self.preview_size = tuple(preview_size)
        self.still_size = tuple(still_size)
        self.fps = max(1, int(fps))
        self.jpeg_quality = int(jpeg_quality)
        self._picam2 = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()
        self._latest = None
        self._frame_id = 0
        self._capture_requested = threading.Event()
        self._still_ready = threading.Event()
        self._pending_still = None
        self._error = None

    def start(self):
        self._picam2 = Picamera2()
        config = self._picam2.create_preview_configuration(
            main={"size": self.still_size, "format": "RGB888"},
            lores={"size": self.preview_size, "format": "YUV420"},
        )
        self._picam2.configure(config)
        self._picam2.start()
        time.sleep(1.0)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _encode(self, bgr, quality=None):
        ok, buf = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality or self.jpeg_quality)]
        )
        return buf.tobytes() if ok else None

    def _grab_preview(self):
        try:
            frame = self._picam2.capture_array("lores")
            if frame.ndim == 3 and frame.shape[2] == 3:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            return bgr
        except Exception:
            frame = self._picam2.capture_array("main")
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return cv2.resize(bgr, self.preview_size, interpolation=cv2.INTER_AREA)

    def _grab_still(self):
        frame = self._picam2.capture_array("main")
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return bgr

    def _loop(self):
        interval = 1.0 / self.fps
        while self._running:
            started = time.time()
            try:
                if self._capture_requested.is_set():
                    self._capture_requested.clear()
                    still = self._grab_still()
                    self._pending_still = self._encode(still, quality=max(90, self.jpeg_quality))
                    self._still_ready.set()
                bgr = self._grab_preview()
                encoded = self._encode(bgr)
                if encoded is not None:
                    with self._lock:
                        self._latest = encoded
                        self._frame_id += 1
                self._error = None
            except Exception as exc:
                self._error = str(exc)
            elapsed = time.time() - started
            time.sleep(max(0.0, interval - elapsed))

    def latest_jpeg(self):
        with self._lock:
            return self._latest

    def capture_still_jpeg(self, timeout=8.0):
        self._still_ready.clear()
        self._pending_still = None
        self._capture_requested.set()
        if not self._still_ready.wait(timeout):
            raise TimeoutError("camera did not deliver a still in time")
        still, self._pending_still = self._pending_still, None
        if still is None:
            raise RuntimeError(self._error or "camera failed to capture still")
        return still

    def status(self):
        with self._lock:
            has_frame = self._latest is not None
            frame_id = self._frame_id
        return {
            "source": self.name,
            "running": self._running,
            "has_frame": has_frame,
            "frame_id": frame_id,
            "preview_size": list(self.preview_size),
            "still_size": list(self.still_size),
            "fps": self.fps,
            "error": self._error,
        }

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass


class MockCameraSource(BaseCamera):
    name = "mock"

    def __init__(self, preview_size, still_size, fps, jpeg_quality):
        self.preview_size = (int(preview_size[0]), int(preview_size[1]))
        self.still_size = (int(still_size[0]), int(still_size[1]))
        self.fps = max(1, int(fps))
        self.jpeg_quality = int(jpeg_quality)
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._latest = None
        self._frame_id = 0
        self._bright = 1.0
        self._rng = np.random.default_rng(7)

    def _render_retina(self, size, t, bright):
        w, h = size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = w * (0.5 + 0.01 * np.sin(t))
        cy = h * (0.5 + 0.01 * np.cos(t * 1.3))
        dx = (xx - cx) / (w * 0.5)
        dy = (yy - cy) / (h * 0.5)
        r = np.sqrt(dx * dx + dy * dy)
        mask = np.clip(1.0 - r, 0.0, 1.0) ** 0.35
        bgr = np.zeros((h, w, 3), np.float32)
        bgr[..., 2] = 90 + 110 * mask
        bgr[..., 1] = 25 + 45 * mask
        bgr[..., 0] = 20 + 35 * mask
        disc_x = int(cx + w * 0.16)
        disc_y = int(cy - h * 0.10)
        disc = np.clip(
            1.0 - np.sqrt(((xx - disc_x) / (w * 0.045)) ** 2 + ((yy - disc_y) / (h * 0.06)) ** 2), 0, 1
        )
        bgr[..., 2] += 120 * disc
        bgr[..., 1] += 90 * disc
        bgr[..., 0] += 60 * disc
        for k in range(9):
            ang = k * 0.62 + t * 0.15
            length = (0.35 + 0.06 * (k % 3)) * w
            ex = int(disc_x + np.cos(ang) * length)
            ey = int(disc_y + np.sin(ang) * length)
            cv2.line(bgr, (disc_x, disc_y), (ex, ey), (30, 55, 120), max(1, w // 380), cv2.LINE_AA)
        fovea = np.clip(
            1.0 - np.sqrt(((xx - cx) / (w * 0.03)) ** 2 + ((yy - cy) / (h * 0.03)) ** 2), 0, 1
        )
        bgr *= (1.0 - 0.35 * fovea)[..., None]
        bgr *= bright
        noise = self._rng.normal(0, 6, bgr.shape).astype(np.float32)
        bgr = np.clip(bgr + noise, 0, 255).astype(np.uint8)
        return bgr

    def _loop(self):
        interval = 1.0 / self.fps
        t = 0.0
        while self._running:
            started = time.time()
            with self._lock:
                brightness = self._bright
            frame = self._render_retina(self.preview_size, t, brightness)
            cv2.putText(
                frame,
                "MOCK NoIR CAMERA",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (80, 255, 160),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                time.strftime("%H:%M:%S"),
                (16, self.preview_size[1] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if ok:
                with self._lock:
                    self._latest = buf.tobytes()
                    self._frame_id += 1
            t += interval
            elapsed = time.time() - started
            time.sleep(max(0.0, interval - elapsed))

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def latest_jpeg(self):
        with self._lock:
            return self._latest

    def capture_still_jpeg(self, timeout=8.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self._render_retina(self.still_size, time.time(), 1.25)
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), max(90, self.jpeg_quality)]
            )
            if ok:
                return buf.tobytes()
            time.sleep(0.05)
        raise TimeoutError("mock camera failed to produce a still")

    def status(self):
        with self._lock:
            return {
                "source": self.name,
                "running": self._running,
                "has_frame": self._latest is not None,
                "frame_id": self._frame_id,
                "preview_size": list(self.preview_size),
                "still_size": list(self.still_size),
                "fps": self.fps,
                "error": None,
            }

    def set_brightness(self, value):
        with self._lock:
            self._bright = float(value)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)


def build_camera(settings):
    preview_size = (settings.preview_width, settings.preview_height)
    still_size = (settings.still_width, settings.still_height)
    if settings.mock or not PICAMERA_AVAILABLE:
        return MockCameraSource(preview_size, still_size, settings.stream_fps, settings.jpeg_quality)
    return PiCameraSource(preview_size, still_size, settings.stream_fps, settings.jpeg_quality)
