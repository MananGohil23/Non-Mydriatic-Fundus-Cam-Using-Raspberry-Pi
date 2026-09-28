import os
import subprocess
import sys
import time

import httpx

BACKEND = r"D:\HTH\backend"
BASE = "http://127.0.0.1:8001"
IMAGE = r"D:\HTH\training\RETFound-main\data\test\referable\001639a390f0.png"

env = os.environ.copy()
env.update(
    {
        "INFER_MODE": "retfound",
        "CAMERA_MODE": "mock",
        "CHECKPOINT_PATH": r"D:\HTH\training\RETFound-main\output_dir\aptos_binary_lp\checkpoint-best.pth",
        "MODEL_ARCH": "vit_large_patch16_224",
        "PYTHONUNBUFFERED": "1",
    }
)

log = open(os.path.join(BACKEND, "retfound_test.out.log"), "w")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8001", "--log-level", "debug"],
    cwd=BACKEND,
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
)

try:
    ready = False
    for _ in range(90):
        time.sleep(2)
        if proc.poll() is not None:
            print("SERVER EXITED EARLY, code:", proc.returncode)
            break
        try:
            if httpx.get(BASE + "/health", timeout=5).status_code == 200:
                ready = True
                break
        except Exception:
            pass
    print("server_ready:", ready)

    health = httpx.get(BASE + "/health", timeout=10).json()
    print("engine:", health["model"]["engine"], "| load_report:", health["model"]["load_report"])

    with open(IMAGE, "rb") as handle:
        response = httpx.post(
            BASE + "/analyze", files={"file": ("test.png", handle, "image/png")}, timeout=180
        )
    print("analyze_http_status:", response.status_code)
    if response.status_code != 200:
        print("analyze_body_head:", response.text[:1000])
    else:
        result = response.json()
        print("status:", result["status"], "| engine:", result["engine"], "| latency_ms:", result["latency_ms"])
        print("quality:", result["quality"])
        print("analysis:", result["analysis"])
        print("heatmap_url:", result["heatmap_url"])
        if result["heatmap_url"]:
            heatmap = httpx.get(BASE + result["heatmap_url"], timeout=30)
            print("heatmap_bytes:", len(heatmap.content))
        print("history:", len(httpx.get(BASE + "/api/captures", timeout=10).json()["captures"]))

    with httpx.stream("GET", BASE + "/viewfinder", timeout=10) as stream:
        first = next(stream.iter_bytes())
    print("viewfinder_first_chunk_bytes:", len(first))
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    log.close()
