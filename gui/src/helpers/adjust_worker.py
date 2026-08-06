from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage


def _pil_to_qimage(pil_img):
    """Convert a PIL Image to a QImage (thread-safe — no QPixmap)."""
    rgb = pil_img.convert("RGB")
    data = rgb.tobytes()
    w, h = rgb.size
    return QImage(data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _apply_adjustments(pil_img, params: dict):  # noqa: C901
    """
    Apply layered adjustments to a PIL Image and return the result.

    params keys (all optional, sensible defaults):
        crop_ar       (int, int) | None   — target aspect ratio (w, h) for center-crop
        rotate        float               — CW rotation in degrees
        flip_h        bool
        flip_v        bool
        brightness    int  -100..100      — 0 = no change
        contrast      int  -100..100
        gamma         int  10..500        — stored as gamma*100; 100 = 1.00
        saturation    int  -100..100
        hue           int  -180..180      — hue shift in degrees
        sharpen       int  0..100
        blur          int  0..50
    """
    img = pil_img.copy()

    # 1. Aspect-ratio center-crop
    ar = params.get("crop_ar")
    if ar:
        aw, ah = ar
        iw, ih = img.size
        target_ratio = aw / ah
        current_ratio = iw / ih
        if current_ratio > target_ratio:
            new_w = int(ih * target_ratio)
            left = (iw - new_w) // 2
            img = img.crop((left, 0, left + new_w, ih))
        elif current_ratio < target_ratio:
            new_h = int(iw / target_ratio)
            top = (ih - new_h) // 2
            img = img.crop((0, top, iw, top + new_h))

    # 2. Rotation (positive = CW, PIL rotates CCW so negate)
    angle = params.get("rotate", 0.0)
    if abs(angle) > 0.01:
        img = img.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)

    # 3. Flip
    if params.get("flip_h", False):
        img = ImageOps.mirror(img)
    if params.get("flip_v", False):
        img = ImageOps.flip(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    # 4. Brightness  (-100..100 → PIL factor 0.0..2.0)
    b = params.get("brightness", 0) / 100.0
    if b != 0.0:
        img = ImageEnhance.Brightness(img).enhance(max(0.0, 1.0 + b))

    # 5. Contrast
    c = params.get("contrast", 0) / 100.0
    if c != 0.0:
        img = ImageEnhance.Contrast(img).enhance(max(0.0, 1.0 + c))

    # 6. Gamma  (stored as int*100; 100=1.00; applied as power 1/gamma)
    gamma = params.get("gamma", 100) / 100.0
    if abs(gamma - 1.0) > 0.005:
        arr = np.array(img, dtype=np.float32)
        arr = np.clip(np.power(arr / 255.0, 1.0 / gamma) * 255.0, 0, 255).astype(
            np.uint8
        )
        img = Image.fromarray(arr)

    # 7. Saturation
    s = params.get("saturation", 0) / 100.0
    if s != 0.0:
        img = ImageEnhance.Color(img).enhance(max(0.0, 1.0 + s))

    # 8. Hue shift  (OpenCV H is 0-179, half of 360°)
    hue_deg = params.get("hue", 0)
    if hue_deg != 0:
        arr = np.array(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.int32)
        hsv[..., 0] = (hsv[..., 0] + hue_deg // 2 + 900) % 180
        img = Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))

    # 9. Sharpen  (0..100 → PIL factor 1.0..11.0)
    sh = params.get("sharpen", 0)
    if sh > 0:
        img = ImageEnhance.Sharpness(img).enhance(1.0 + sh / 10.0)

    # 10. Blur  (0..50 → Gaussian radius 0..10)
    bl = params.get("blur", 0)
    if bl > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=bl / 5.0))

    if img.mode != "RGB":
        img = img.convert("RGB")

    # 11. White balance temperature (+100=warm/amber, -100=cool/blue)
    temp = params.get("temperature", 0)
    if temp != 0:
        arr = np.array(img, dtype=np.float32)
        t = temp / 100.0
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (1.0 + t * 0.30), 0, 255)  # R
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (1.0 - t * 0.25), 0, 255)  # B
        img = Image.fromarray(arr.astype(np.uint8))

    # 12. Tint (+100=magenta, -100=green)
    tint_v = params.get("tint", 0)
    if tint_v != 0:
        arr = np.array(img, dtype=np.float32)
        t = tint_v / 100.0
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (1.0 + t * 0.15), 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * (1.0 - t * 0.20), 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (1.0 + t * 0.10), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    # 13. Shadows (-100=crush, +100=lift)
    shadows = params.get("shadows", 0)
    if shadows != 0:
        arr = np.array(img, dtype=np.float32) / 255.0
        fade = np.clip(1.0 - arr * 2.0, 0.0, 1.0)
        arr = np.clip(arr + (shadows / 200.0) * fade, 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8))

    # 14. Highlights (-100=recover, +100=boost)
    highlights = params.get("highlights", 0)
    if highlights != 0:
        arr = np.array(img, dtype=np.float32) / 255.0
        fade = np.clip((arr - 0.5) * 2.0, 0.0, 1.0)
        arr = np.clip(arr + (highlights / 200.0) * fade, 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8))

    # 15. Vibrance — selective saturation (boosts desaturated colours more)
    vibrance = params.get("vibrance", 0)
    if vibrance != 0:
        arr = np.array(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
        s_ch = hsv[:, :, 1] / 255.0
        boost = (vibrance / 100.0) * (1.0 - s_ch)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + boost * 100.0, 0, 255)
        img = Image.fromarray(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))

    # 16. Auto white balance (gray-world assumption)
    if params.get("auto_wb", False):
        arr = np.array(img, dtype=np.float32)
        r_m = arr[:, :, 0].mean()
        g_m = arr[:, :, 1].mean()
        b_m = arr[:, :, 2].mean()
        mu = (r_m + g_m + b_m) / 3.0
        if r_m > 1:
            arr[:, :, 0] = np.clip(arr[:, :, 0] * mu / r_m, 0, 255)
        if g_m > 1:
            arr[:, :, 1] = np.clip(arr[:, :, 1] * mu / g_m, 0, 255)
        if b_m > 1:
            arr[:, :, 2] = np.clip(arr[:, :, 2] * mu / b_m, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))

    return img


class AdjustWorker(QThread):
    sig_finished = Signal(object)  # QImage
    sig_error = Signal(str)

    def __init__(self, img_path: str, params: dict, max_size: Optional[int] = None):
        super().__init__()
        self._path = img_path
        self._params = params
        self._max_size = max_size

    def run(self):
        try:
            img = Image.open(self._path)
            if self._max_size:
                w, h = img.size
                if max(w, h) > self._max_size:
                    scale = self._max_size / max(w, h)
                    img = img.resize(
                        (int(w * scale), int(h * scale)),
                        Image.Resampling.LANCZOS,
                    )
            result = _apply_adjustments(img, self._params)
            self.sig_finished.emit(_pil_to_qimage(result))
        except Exception as e:
            self.sig_error.emit(str(e))
