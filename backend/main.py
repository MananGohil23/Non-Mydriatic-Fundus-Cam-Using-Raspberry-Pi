import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from camera_client import build_camera_client
from config import settings
from inference import build_engine
from quality import assess
from store import store

state = {}


@asynccontextmanager
async def lifespan(_app):
    engine, engine_note = build_engine(settings)
    state["engine"] = engine
    state["engine_note"] = engine_note
    camera_client = build_camera_client(settings)
    camera_client.start()
    state["camera"] = camera_client
    state["camera_mock"] = settings.use_mock_camera
    yield
    camera_client.stop()


app = FastAPI(title="RetinaScreen API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode(image_bytes):
    if not image_bytes:
        return None
    array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def _encode_jpeg(image_bgr):
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buf.tobytes() if ok else None


@app.get("/health")
def health():
    camera = state["camera"]
    engine = state["engine"]
    camera_health = camera.health()
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).astimezone().isoformat(),
        "backend": {"port": settings.port, "camera_mode": "mock" if settings.use_mock_camera else "remote"},
        "camera": camera_health,
        "model": {
            "engine": engine.name,
            "available": engine.available,
            "classes": list(settings.class_names),
            "note": state.get("engine_note"),
        },
        "quality_gate": settings.quality_enabled,
    }


@app.get("/viewfinder")
def viewfinder():
    stop = threading.Event()

    def generate():
        try:
            for chunk in state["camera"].iter_mjpeg(stop):
                yield chunk
        finally:
            stop.set()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshot")
def snapshot():
    try:
        frame = state["camera"].snapshot()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="camera snapshot failed: %s" % exc)
    if not frame:
        raise HTTPException(status_code=503, detail="no frame available from camera")
    return Response(content=frame, media_type="image/jpeg")


@app.post("/capture")
def capture():
    try:
        result = state["camera"].capture()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="capture failed: %s" % exc)
    image = result.pop("image", None)
    if not image:
        raise HTTPException(status_code=502, detail="camera returned no image data")
    decoded = _decode(image)
    quality = (
        assess(decoded, settings.blur_threshold, settings.brightness_min, settings.brightness_max)
        if decoded is not None and settings.quality_enabled
        else None
    )
    record = store.add(image, meta=result)
    store.update(record["id"], quality=quality)
    return {
        "capture_id": record["id"],
        "image_url": "/images/%s" % record["id"],
        "width": result.get("width"),
        "height": result.get("height"),
        "captured_at": result.get("captured_at") or record["created_at"],
        "source": result.get("source", "camera"),
        "quality": quality,
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(None),
    capture_id: str = Form(None),
    force: bool = Query(False),
):
    if file is not None:
        image_bytes = await file.read()
        record = store.add(image_bytes, meta={"source": "upload", "filename": file.filename})
        capture_id = record["id"]
    elif capture_id:
        record = store.get(capture_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown capture_id")
        image_bytes = record["image"]
    else:
        raise HTTPException(status_code=400, detail="provide a file upload or a capture_id")

    decoded = _decode(image_bytes)
    if decoded is None:
        raise HTTPException(status_code=400, detail="could not decode image")

    quality = (
        assess(decoded, settings.blur_threshold, settings.brightness_min, settings.brightness_max)
        if settings.quality_enabled
        else {"usable": True, "issues": [], "score": None}
    )
    store.update(capture_id, quality=quality)

    engine = state["engine"]
    started = time.time()
    analysis = None
    if quality.get("usable") or force:
        analysis = engine.analyze(decoded, with_heatmap=True)
        heatmap = analysis.pop("heatmap", None)
        if heatmap is not None:
            store.update(capture_id, heatmap=_encode_jpeg(heatmap))
    latency_ms = round((time.time() - started) * 1000, 1)
    status = "ok" if analysis is not None else "rejected"
    store.update(capture_id, analysis=analysis)

    record = store.get(capture_id)
    return {
        "capture_id": capture_id,
        "status": status,
        "image_url": "/images/%s" % capture_id,
        "heatmap_url": "/heatmaps/%s" % capture_id if record.get("heatmap") else None,
        "quality": quality,
        "analysis": analysis,
        "latency_ms": latency_ms,
        "engine": engine.name,
        "analyzed_at": datetime.now(timezone.utc).astimezone().isoformat(),
    }


@app.get("/images/{capture_id}")
def get_image(capture_id: str):
    record = store.get(capture_id)
    if record is None:
        raise HTTPException(status_code=404, detail="unknown capture_id")
    return Response(content=record["image"], media_type="image/jpeg")


@app.get("/heatmaps/{capture_id}")
def get_heatmap(capture_id: str):
    record = store.get(capture_id)
    if record is None or record.get("heatmap") is None:
        raise HTTPException(status_code=404, detail="no heatmap for this capture")
    return Response(content=record["heatmap"], media_type="image/jpeg")


@app.get("/api/captures")
def list_captures(limit: int = 20):
    records = store.list(limit=limit)
    return {
        "captures": [
            {
                "capture_id": r["id"],
                "created_at": r["created_at"],
                "image_url": "/images/%s" % r["id"],
                "heatmap_url": "/heatmaps/%s" % r["id"] if r.get("heatmap") else None,
                "quality": r.get("quality"),
                "analysis": r.get("analysis"),
            }
            for r in records
        ]
    }


@app.get("/api/config")
def api_config():
    return {
        "classes": list(settings.class_names),
        "engine": state["engine"].name,
        "camera_mode": "mock" if settings.use_mock_camera else "remote",
        "camera_base_url": settings.camera_base_url,
        "quality_gate": settings.quality_enabled,
    }


dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.isdir(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")
