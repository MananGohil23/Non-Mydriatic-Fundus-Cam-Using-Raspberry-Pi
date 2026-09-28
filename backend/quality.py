import cv2
import numpy as np


def _score_blur(blur, threshold):
    if threshold <= 0:
        return 100.0
    return float(np.clip(100.0 * (blur / (threshold * 2.0)), 0.0, 100.0))


def _score_brightness(brightness, low, high):
    if low <= brightness <= high:
        return 100.0
    if brightness < low:
        return float(np.clip(100.0 * (brightness / max(1.0, low)), 0.0, 100.0))
    return float(np.clip(100.0 * ((255.0 - brightness) / max(1.0, 255.0 - high)), 0.0, 100.0))


def assess(image_bgr, blur_threshold=35.0, brightness_min=18.0, brightness_max=245.0):
    if image_bgr is None or image_bgr.size == 0:
        return {
            "usable": False,
            "score": 0.0,
            "blur": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "retina_fraction": 0.0,
            "issues": ["empty_image"],
        }

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    retina_fraction = float(mask.mean() / 255.0)

    issues = []
    if blur < blur_threshold:
        issues.append("too_blurry")
    if brightness < brightness_min:
        issues.append("too_dark")
    if brightness > brightness_max:
        issues.append("too_bright")
    if contrast < 12:
        issues.append("low_contrast")
    if retina_fraction < 0.05:
        issues.append("no_retina_visible")
    elif retina_fraction > 0.98:
        issues.append("no_background")

    scores = [
        _score_blur(blur, blur_threshold),
        _score_brightness(brightness, brightness_min, brightness_max),
        float(np.clip(100.0 * contrast / 60.0, 0.0, 100.0)),
        float(np.clip(100.0 * retina_fraction / 0.5, 0.0, 100.0)),
    ]
    score = round(float(np.mean(scores)), 1)
    usable = not any(
        issue in issues
        for issue in ("empty_image", "too_blurry", "too_dark", "too_bright", "no_retina_visible")
    )

    return {
        "usable": usable,
        "score": score,
        "blur": round(blur, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "retina_fraction": round(retina_fraction, 3),
        "issues": issues,
    }
