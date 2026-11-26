"""Decode agibot_gdk camera Image buffers to BGR."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

_log = logging.getLogger(__name__)
_warned: set[tuple[int, int]] = set()


def _warn_once(encoding: Any, color: Any, width: int, height: int) -> None:
    enc_i = int(encoding) if encoding is not None else -1
    col_i = int(color) if color is not None else -1
    key = (enc_i, col_i)
    if key in _warned:
        return
    _warned.add(key)
    _log.warning(
        "Unhandled GDK image encoding=%s color_format=%s size=%dx%d",
        encoding,
        color,
        width,
        height,
    )


def decode_gdk_image(image: Any, gdk: Any) -> np.ndarray | None:
    data = getattr(image, "data", None)
    if data is None:
        return None
    try:
        empty = not data.any() if hasattr(data, "any") else len(data) == 0
    except Exception:  # noqa: BLE001
        empty = False
    if empty:
        return None

    raw = np.asarray(data).tobytes()
    encoding = getattr(image, "encoding", None)
    jpeg_enc = getattr(gdk.Encoding, "JPEG", None)
    png_enc = getattr(gdk.Encoding, "PNG", None)
    uncompressed = getattr(gdk.Encoding, "UNCOMPRESSED", None)

    if encoding in {jpeg_enc, png_enc}:
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    if encoding != uncompressed:
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    height = int(getattr(image, "height", 0))
    width = int(getattr(image, "width", 0))
    if height <= 0 or width <= 0:
        return None

    color = getattr(image, "color_format", None)
    cf = gdk.ColorFormat
    rgb = getattr(cf, "RGB", None)
    bgr = getattr(cf, "BGR", None)
    rgba = getattr(cf, "RGBA", None)
    bgra = getattr(cf, "BGRA", None)
    gray8 = getattr(cf, "GRAY8", None)
    nv12 = getattr(cf, "NV12", None)
    nv21 = getattr(cf, "NV21", None)
    yuv420 = getattr(cf, "YUV420", None)

    try:
        if color == rgb:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if color == bgr:
            return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
        if color == bgra:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if color == rgba:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if color == gray8:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if color in {nv12, yuv420}:
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((height * 3 // 2, width))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        if color == nv21:
            yuv = np.frombuffer(raw, dtype=np.uint8).reshape((height * 3 // 2, width))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    except ValueError:
        _warn_once(encoding, color, width, height)
        return None

    _warn_once(encoding, color, width, height)
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)