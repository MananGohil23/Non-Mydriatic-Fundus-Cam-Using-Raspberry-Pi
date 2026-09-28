import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone


class CaptureStore:
    def __init__(self, max_items=50):
        self._lock = threading.Lock()
        self._items = OrderedDict()
        self._max_items = max_items

    def add(self, image_bytes, meta=None):
        capture_id = uuid.uuid4().hex[:12]
        record = {
            "id": capture_id,
            "image": image_bytes,
            "heatmap": None,
            "meta": dict(meta or {}),
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "analysis": None,
            "quality": None,
        }
        with self._lock:
            self._items[capture_id] = record
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)
        return record

    def get(self, capture_id):
        with self._lock:
            return self._items.get(capture_id)

    def update(self, capture_id, **fields):
        with self._lock:
            record = self._items.get(capture_id)
            if record is None:
                return None
            record.update(fields)
            return record

    def list(self, limit=20):
        with self._lock:
            records = list(self._items.values())
        return list(reversed(records))[:limit]


store = CaptureStore()
