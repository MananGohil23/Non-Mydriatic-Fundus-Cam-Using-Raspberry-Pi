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


def _float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    camera_base_url: str
    camera_mode: str
    infer_mode: str
    checkpoint_path: str
    model_arch: str
    num_classes: int
    class_names: tuple
    input_size: int
    quality_enabled: bool
    blur_threshold: float
    brightness_min: float
    brightness_max: float
    cors_origins: tuple
    camera_timeout: float

    @property
    def use_mock_camera(self):
        return self.camera_mode.strip().lower() != "remote" or not self.camera_base_url.strip()


def load_settings():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint = os.getenv("CHECKPOINT_PATH", "").strip()
    if checkpoint and not os.path.isabs(checkpoint):
        candidate = os.path.join(base_dir, checkpoint)
        checkpoint = candidate if os.path.exists(candidate) else checkpoint
    class_names = tuple(
        name.strip() for name in os.getenv("CLASS_NAMES", "normal,referable").split(",") if name.strip()
    )
    origins = tuple(o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip())
    return Settings(
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=_int(os.getenv("BACKEND_PORT"), 8001),
        camera_base_url=os.getenv("CAMERA_BASE_URL", "http://raspberrypi.local:8000").rstrip("/"),
        camera_mode=os.getenv("CAMERA_MODE", "remote"),
        infer_mode=os.getenv("INFER_MODE", "stub").strip().lower(),
        checkpoint_path=checkpoint,
        model_arch=os.getenv("MODEL_ARCH", "vit_large_patch16"),
        num_classes=_int(os.getenv("NUM_CLASSES"), max(2, len(class_names))),
        class_names=class_names if class_names else ("normal", "referable"),
        input_size=_int(os.getenv("INPUT_SIZE"), 224),
        quality_enabled=_bool(os.getenv("QUALITY_ENABLED"), True),
        blur_threshold=_float(os.getenv("BLUR_THRESHOLD"), 35.0),
        brightness_min=_float(os.getenv("BRIGHTNESS_MIN"), 18.0),
        brightness_max=_float(os.getenv("BRIGHTNESS_MAX"), 245.0),
        cors_origins=origins if origins else ("*",),
        camera_timeout=_float(os.getenv("CAMERA_TIMEOUT"), 20.0),
    )


settings = load_settings()
