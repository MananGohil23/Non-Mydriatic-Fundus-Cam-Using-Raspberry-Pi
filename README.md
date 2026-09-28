# RetinaScreen — Portable Nonmydriatic Fundus Camera + AI Screening

Hackathon build: a Raspberry Pi NoIR camera streams a live IR viewfinder to a laptop
dashboard, captures a colour fundus still, and runs a retinal disease-screening model
(RETFound) over it.

```
  ┌────────────────────────┐        MJPEG / HTTP        ┌─────────────────────────┐
  │      RASPBERRY PI      │  ───────────────────────▶  │    LAPTOP — BACKEND     │
  │                        │   GET  /viewfinder         │      (FastAPI)          │
  │  NoIR cam ─ picamera2  │   POST /capture            │                         │
  │  IR LED  ─┐            │  ◀───────────────────────  │  proxies camera, runs   │
  │  White LED├─ GPIO      │                            │  quality gate + RETFound│
  │  Button  ─┘            │                            │  POST /analyze          │
  └────────────────────────┘                            └───────────┬─────────────┘
                                                                     │ JSON + image/heatmap
                                                                     ▼
                                                        ┌─────────────────────────┐
                                                        │   LAPTOP — DASHBOARD    │
                                                        │  React (Vite) :5173     │
                                                        │  viewfinder + results   │
                                                        └─────────────────────────┘
```

Only the **backend** needs to know the Pi's address. The dashboard talks to the backend
on a single origin, so there is no cross-device CORS/streaming headache in the browser.

---

## Repository layout

```
HTH/
├── pi/                      Raspberry Pi camera + LED server
│   ├── camera_server.py     Flask app: /viewfinder /snapshot /capture /led /health
│   ├── camera.py            picamera2 source + mock source
│   ├── leds.py              GPIO LED + shutter-button control
│   ├── config.py            pin numbers, resolutions (env-driven)
│   └── .env.example
├── backend/                 Laptop inference API
│   ├── main.py              FastAPI app: proxy + /capture + /analyze + history
│   ├── inference.py         stub engine + real RETFound engine (+ Grad-CAM)
│   ├── quality.py           blur / exposure / framing quality gate
│   ├── camera_client.py     remote Pi client + in-process mock camera
│   ├── mock_camera.py       synthetic retina frames (no hardware needed)
│   ├── store.py             in-memory capture store
│   └── .env.example
└── frontend/                React dashboard (Vite + Tailwind)
    └── src/RetinaScreen.jsx live viewfinder, capture, results, records
```

---

## 1. Hardware wiring

### 1.1 Camera

- Raspberry Pi 3/4/5 + **NoIR Camera Module** (no IR-cut filter) on the CSI connector,
  ribbon contacts facing the correct way (blue stiffener away from the Ethernet/USB side
  on a Pi 4/5).
- The lens is **refocused from infinity to ≈8 cm** by manually unscrewing it (per the
  Shen & Mukai 2017 build). Keep it fixed; focus a target at working distance.
- A **20-diopter condensing lens** is held in front of the eye (indirect
  ophthalmoscopy), not attached to the camera.

### 1.2 Illumination driver

Two LEDs are driven from GPIO through NPN transistors (GPIO cannot source LED current):

```
        5V ────┬───────────────┬────
               │               │
            [ R_IR ]        [ R_WHITE ]
               │               │
            IR LED           WHITE LED            (anode → resistor → 5V)
               │               │
             Q1 C             Q2 C
   BCM17 ─[1k]─ Q1 B   BCM27 ─[1k]─ Q2 B
             Q1 E             Q2 E
               │               │
              GND ─────────────┴──── (common ground with the Pi)
```

| Signal | Pi GPIO (BCM) | Physical pin | Goes to |
|---|---|---|---|
| IR LED enable | **BCM 17** | 11 | 1 kΩ → base of Q1 |
| White LED enable | **BCM 27** | 13 | 1 kΩ → base of Q2 |
| Shutter button | **BCM 18** | 12 | push-button → GND (internal pull-up) |
| Ground | — | 6 / 9 / 14 | common GND with LED supply |

**Current-limit resistor:** `R = (V_supply - V_forward) / I_target`.

| LED | V_forward (approx) | Example (5 V, target current) | Resistor |
|---|---|---|---|
| IR 850 nm (Ushio 850D) | ~1.5–1.7 V | 60 mA | ≈ 56 Ω |
| White (Ushio SMT47W) | ~3.2–3.4 V | 50 mA | ≈ 33 Ω |

Start conservative (higher resistance, lower current). The original paper measured only
50–90 µW at 850 nm and 150–170 µW white, i.e. low optical power. For multiple LEDs per
channel, replace the transistor with a small logic-level MOSFET and size the resistor
for the summed current. Never exceed the LED's rated forward current.

- **IR LED = focus/viewfinder.** The pupil does not constrict under 850 nm, so the
  live stream stays in IR.
- **White LED = capture.** Flashed for `WHITE_FLASH_MS` only, long enough to expose the
  colour still before the pupil constricts. The Pi enforces this sequence in
  `/capture`.
- **No IR-cut filter** means colour response differs from clinical fundus cameras —
  this is part of the domain gap the model is adapted to.

> Safety: do not look directly into the LEDs. Use current limiting. Keep a common ground.

### 1.3 Networking

Pi and laptop on the same network. Venue WiFi often has **client isolation** (devices
cannot see each other) — keep a phone hotspot as backup and put both machines on it.
Find the Pi's address with `hostname -I` (or `ping raspberrypi.local`).

---

## 2. Bring-up

### 2.1 Raspberry Pi (camera server)

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-kms++ python3-numpy
sudo raspi-config            # Interface Options → enable the camera, then reboot
cd pi
python3 -m pip install -r requirements.txt
cp .env.example .env         # edit pins/resolution if needed
python3 camera_server.py
```

Open `http://<pi-ip>:8000/viewfinder` in a browser to confirm the MJPEG stream.
`http://<pi-ip>:8000/health` shows camera, LED and shutter status.

### 2.2 Laptop (backend)

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # set CAMERA_BASE_URL to http://<pi-ip>:8000
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

- Interactive API docs: `http://localhost:8001/docs`
- `http://localhost:8001/health` shows whether the Pi is reachable and which model
  engine is active.

### 2.3 Laptop (dashboard)

```bash
cd frontend
npm install
cp .env.example .env            # VITE_API_BASE=http://localhost:8001
npm run dev
```

Open `http://localhost:5173`. The home screen shows camera/model readiness, the capture
screen shows the live IR viewfinder, and results show the diagnosis, confidence,
class probabilities, quality issues and the AI heatmap.

### 2.4 No hardware? Run fully mocked

Set `CAMERA_MODE=mock` in `backend/.env` (or leave `CAMERA_BASE_URL` empty). The backend
serves a synthetic retina stream and synthetic captures, so the whole dashboard is
demoable on the laptop alone. You can also use **Upload image** on the capture screen to
analyze any fundus photo without the Pi.

---

## 3. Configuration reference

### `pi/.env`

| Variable | Default | Meaning |
|---|---|---|
| `CAMERA_PORT` | 8000 | Pi HTTP port |
| `CAMERA_MOCK` | 0 | Force synthetic camera on the Pi |
| `PREVIEW_WIDTH/HEIGHT` | 1280×720 | Viewfinder resolution |
| `STREAM_FPS` | 15 | MJPEG frame rate |
| `STILL_WIDTH/HEIGHT` | 2592×1944 | Full-res still |
| `LED_IR_PIN` / `LED_WHITE_PIN` | 17 / 27 | LED driver GPIO (BCM) |
| `SHUTTER_PIN` | 18 | Physical shutter button |
| `WHITE_FLASH_MS` | 450 | White LED on-time for capture |
| `WHITE_SETTLE_MS` | 150 | Delay after switching to white before capture |
| `IR_SETTLE_MS` | 120 | Delay after switching back to IR |

### `backend/.env`

| Variable | Default | Meaning |
|---|---|---|
| `CAMERA_BASE_URL` | `http://raspberrypi.local:8000` | Pi address |
| `CAMERA_MODE` | `remote` | `remote` or `mock` |
| `INFER_MODE` | `stub` | `stub` or `retfound` |
| `CHECKPOINT_PATH` | *(empty)* | Fine-tuned RETFound `.pth` |
| `CLASS_NAMES` | `normal,referable` | Comma-separated class order |
| `INPUT_SIZE` | 224 | Model input size |
| `QUALITY_ENABLED` | 1 | Run the quality gate before inference |
| `BLUR_THRESHOLD` | 35 | Laplacian-variance blur floor |
| `BRIGHTNESS_MIN/MAX` | 18 / 245 | Exposure bounds |
| `CORS_ORIGINS` | `*` | Allowed dashboard origins |

### `frontend/.env`

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_BASE` | `http://localhost:8001` | Backend origin |

---

## 4. API

Backend (`:8001`):

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Backend + Pi + model status |
| GET | `/viewfinder` | MJPEG stream proxied from the Pi |
| GET | `/snapshot` | Single JPEG frame |
| POST | `/capture` | IR→white flash, full-res still, returns `capture_id` + quality |
| POST | `/analyze` | `multipart` with `capture_id` **or** `file`; `?force=true` bypasses the quality gate |
| GET | `/images/{id}` | Stored capture |
| GET | `/heatmaps/{id}` | Grad-CAM overlay |
| GET | `/api/captures` | Recent capture history |
| GET | `/api/config` | Active classes / engine / camera mode |

Pi (`:8000`): `/viewfinder`, `/snapshot`, `/capture`, `/led` (GET/POST `{"mode":"ir"|"white"|"off"}`), `/health`, `/captures/<name>`.

Example:

```bash
curl -X POST http://localhost:8001/capture
curl -X POST -F "capture_id=<id>" http://localhost:8001/analyze
```

---

## 5. Wiring in the real RETFound model

The backend ships with a **stub engine** so the pipeline is testable end-to-end with no
checkpoint. To switch to the real model:

1. Fine-tune RETFound (CFP variant) with `main_finetune.py`, binary `normal/referable`.
2. On the laptop: `pip install -r backend/requirements-retfound.txt`.
3. In `backend/.env`:

   ```
   INFER_MODE=retfound
   CHECKPOINT_PATH=C:\path\to\checkpoint-best.pth
   MODEL_ARCH=vit_large_patch16
   NUM_CLASSES=2
   CLASS_NAMES=normal,referable
   INPUT_SIZE=224
   ```

4. Restart uvicorn. `/health` will report `"engine": "retfound"`.

`backend/inference.py` loads the timm ViT-Large backbone, applies ImageNet
normalisation at `INPUT_SIZE`, runs a softmax head, and produces a Grad-CAM overlay from
the last transformer block. **The resize/normalisation here must match training-time
preprocessing exactly**, or accuracy collapses — keep them in sync.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| Viewfinder shows "No camera signal" | Check Pi is running and `CAMERA_BASE_URL` is correct; open `http://<pi-ip>:8000/viewfinder` directly |
| Backend `/health` says camera unreachable | Pi and laptop must be on the same network without client isolation; use a hotspot |
| `picamera2` import error on the Pi | `sudo apt install -y python3-picamera2 python3-libcamera` (do not `pip install picamera2` on Bookworm) |
| `gpiozero`/GPIO error on Pi 5 | Pi 5 uses `lgpio`; `gpiozero>=2` handles it. Ensure the user is in the `gpio` group |
| Stream works but capture fails | LED driver wiring / `WHITE_SETTLE_MS` too short; increase it |
| Analysis always "Recapture Needed" | Frame failing the quality gate — improve focus/illumination or use `?force=true` |
| `npm run dev` cannot reach API | Match `VITE_API_BASE` to the uvicorn host/port; backend CORS defaults to `*` |
