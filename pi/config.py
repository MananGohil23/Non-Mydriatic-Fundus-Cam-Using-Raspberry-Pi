import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    pass


def _bool(value, default=False):
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_or(value, fallback):
    if value is None:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    mock: bool
    preview_width: int
    preview_height: int
    still_width: int
    still_height: int
    stream_fps: int
    jpeg_quality: int
    led_ir_pin: int
    led_white_pin: int
    shutter_pin: int
    shutter_enabled: bool
    shutter_bounce_s: float
    led_ir_active_high: bool
    led_white_active_high: bool
    white_flash_ms: int
    white_settle_ms: int
    ir_settle_ms: int
    captures_dir: str

    @property
    def frame_interval(self):
        return 1.0 / max(1, self.stream_fps)


def load_settings():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    captures_dir = os.getenv("CAPTURES_DIR", "captures")
    if not os.path.isabs(captures_dir):
        captures_dir = os.path.join(base_dir, captures_dir)
    default_active_high = _bool(os.getenv("LED_ACTIVE_HIGH"), True)
    return Settings(
        host=os.getenv("CAMERA_HOST", "0.0.0.0"),
        port=_int(os.getenv("CAMERA_PORT"), 8000),
        mock=_bool(os.getenv("CAMERA_MOCK"), False),
        preview_width=_int(os.getenv("PREVIEW_WIDTH"), 1280),
        preview_height=_int(os.getenv("PREVIEW_HEIGHT"), 720),
        still_width=_int(os.getenv("STILL_WIDTH"), 2592),
        still_height=_int(os.getenv("STILL_HEIGHT"), 1944),
        stream_fps=_int(os.getenv("STREAM_FPS"), 15),
        jpeg_quality=_int(os.getenv("JPEG_QUALITY"), 80),
        led_ir_pin=_int(os.getenv("LED_IR_PIN"), 27),
        led_white_pin=_int(os.getenv("LED_WHITE_PIN"), 22),
        shutter_pin=_int(os.getenv("SHUTTER_PIN"), 17),
        shutter_enabled=_bool(os.getenv("SHUTTER_ENABLED"), True),
        shutter_bounce_s=_float(os.getenv("SHUTTER_BOUNCE_S"), 0.03),
        led_ir_active_high=_bool_or(os.getenv("LED_IR_ACTIVE_HIGH"), default_active_high),
        led_white_active_high=_bool_or(os.getenv("LED_WHITE_ACTIVE_HIGH"), default_active_high),
        white_flash_ms=_int(os.getenv("WHITE_FLASH_MS"), 200),
        white_settle_ms=_int(os.getenv("WHITE_SETTLE_MS"), 100),
        ir_settle_ms=_int(os.getenv("IR_SETTLE_MS"), 50),
        captures_dir=captures_dir,
    )


settings = load_settings()
