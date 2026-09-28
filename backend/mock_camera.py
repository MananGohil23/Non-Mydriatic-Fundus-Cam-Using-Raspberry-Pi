import threading
import time

import cv2
import numpy as np


class MockCamera:
    def __init__(self, preview_size=(1280, 720), still_size=(1920, 1080), fps=15, jpeg_quality=80):
        self.preview_size = (int(preview_size[0]), int(preview_size[1]))
        self.still_size = (int(still_size[0]), int(still_size[1]))
        self.fps = max(1, int(fps))
        self.jpeg_quality = int(jpeg_quality)
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._latest = None
        self._frame_id = 0
        self._rng = np.random.default_rng(11)

    def _render(self, size, t, bright=1.0):
        w, h = size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w * 0.5, h * 0.5
        r = np.sqrt(((xx - cx) / (w * 0.5)) ** 2 + ((yy - cy) / (h * 0.5)) ** 2)
        mask = np.clip(1.0 - r, 0.0, 1.0) ** 0.35
        bgr = np.zeros((h, w, 3), np.float32)
        bgr[..., 2] = 90 + 110 * mask
        bgr[..., 1] = 25 + 45 * mask
        bgr[..., 0] = 20 + 35 * mask
        disc_x, disc_y = int(cx + w * 0.16), int(cy - h * 0.10)
        disc = np.clip(1.0 - np.sqrt(((xx - disc_x) / (w * 0.045)) ** 2 + ((yy - disc_y) / (h * 0.06)) ** 2), 0, 1)
        bgr[..., 2] += 120 * disc
        bgr[..., 1] += 90 * disc
        bgr[..., 0] += 60 * disc
        for k in range(9):
            ang = k * 0.62 + t * 0.15
            length = (0.35 + 0.06 * (k % 3)) * w
            cv2.line(
                bgr,
                (disc_x, disc_y),
                (int(disc_x + np.cos(ang) * length), int(disc_y + np.sin(ang) * length)),
                (30, 55, 120),
                max(1, w // 380),
                cv2.LINE_AA,
            )
        bgr = np.clip(bgr * bright + self._rng.normal(0, 6, bgr.shape), 0, 255).astype(np.uint8)
        cv2.putText(bgr, "MOCK CAMERA", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 160), 2, cv2.LINE_AA)
        return bgr

    def _loop(self):
        interval = 1.0 / self.fps
        t = 0.0
        while self._running:
            start = time.time()
            frame = self._render(self.preview_size, t)
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                with self._lock:
                    self._latest = buf.tobytes()
                    self._frame_id += 1
            t += interval
            time.sleep(max(0.0, interval - (time.time() - start)))

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def latest_jpeg(self):
        with self._lock:
            return self._latest

    def capture_still_jpeg(self):
        frame = self._render(self.still_size, time.time(), bright=1.25)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), max(90, self.jpeg_quality)])
        return buf.tobytes() if ok else None

    def status(self):
        with self._lock:
            return {
                "source": "mock-backend",
                "running": self._running,
                "has_frame": self._latest is not None,
                "frame_id": self._frame_id,
                "preview_size": list(self.preview_size),
                "still_size": list(self.still_size),
                "fps": self.fps,
                "error": None,
            }

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
