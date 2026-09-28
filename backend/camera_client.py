import time

import httpx

from mock_camera import MockCamera

MJPEG_BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"


class RemoteCameraClient:
    def __init__(self, base_url, timeout=20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, read=None))

    def start(self):
        return None

    def stop(self):
        try:
            self._client.close()
        except Exception:
            pass

    def iter_mjpeg(self, stop_event):
        while not stop_event.is_set():
            try:
                with self._client.stream("GET", "%s/viewfinder" % self.base_url) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        if stop_event.is_set():
                            break
                        if chunk:
                            yield chunk
            except Exception:
                time.sleep(1.0)

    def snapshot(self):
        response = self._client.get("%s/snapshot" % self.base_url)
        response.raise_for_status()
        return response.content

    def capture(self):
        response = self._client.post("%s/capture" % self.base_url)
        response.raise_for_status()
        meta = response.json()
        image = None
        if meta.get("url"):
            fetched = self._client.get("%s%s" % (self.base_url, meta["url"]))
            fetched.raise_for_status()
            image = fetched.content
        meta["image"] = image
        return meta

    def health(self):
        try:
            response = self._client.get("%s/health" % self.base_url, timeout=5.0)
            response.raise_for_status()
            payload = response.json()
            payload["reachable"] = True
            payload["base_url"] = self.base_url
            return payload
        except Exception as exc:
            return {"reachable": False, "base_url": self.base_url, "error": str(exc)}


class MockCameraClient:
    def __init__(self, preview_size=(1280, 720), still_size=(1920, 1080), fps=15, jpeg_quality=80):
        self._camera = MockCamera(preview_size, still_size, fps, jpeg_quality)
        self._frame_interval = 1.0 / max(1, fps)

    def start(self):
        self._camera.start()

    def stop(self):
        self._camera.stop()

    def iter_mjpeg(self, stop_event):
        while not stop_event.is_set():
            frame = self._camera.latest_jpeg()
            if frame:
                yield MJPEG_BOUNDARY + frame + b"\r\n"
            time.sleep(self._frame_interval)

    def snapshot(self):
        return self._camera.latest_jpeg()

    def capture(self):
        still = self._camera.capture_still_jpeg()
        return {
            "filename": "mock_%d.jpg" % int(time.time()),
            "url": None,
            "width": self._camera.still_size[0],
            "height": self._camera.still_size[1],
            "bytes": len(still) if still else 0,
            "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": "mock-backend",
            "image": still,
        }

    def health(self):
        status = self._camera.status()
        status["reachable"] = True
        status["base_url"] = "mock"
        return status


def build_camera_client(settings):
    if settings.use_mock_camera:
        return MockCameraClient(preview_size=(960, 540), still_size=(1920, 1080))
    return RemoteCameraClient(settings.camera_base_url, settings.camera_timeout)
