"""Resize + crop pipeline aligned with data_logger (resize_save_shape → crop)."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

Crop = Tuple[Tuple[int, int], Tuple[int, int]]


def _as_int_pair(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def resolve_crop(topic_entry: dict) -> Tuple[Optional[Crop], str]:
    raw_start = topic_entry.get("crop_start_xy")
    raw_shape = topic_entry.get("crop_shape")
    if raw_start is None and raw_shape is None:
        return None, ""
    if raw_start is None or raw_shape is None:
        missing = "crop_start_xy" if raw_start is None else "crop_shape"
        return None, (
            f"크롭 설정이 반쪽입니다 — {missing} 가 없습니다. "
            "crop_start_xy 와 crop_shape 두 줄을 모두 적어야 크롭이 켜집니다"
        )

    start = _as_int_pair(raw_start)
    shape = _as_int_pair(raw_shape)
    if start is None:
        return None, f"crop_start_xy 는 [x, y] 정수 두 개여야 합니다 (지금: {raw_start!r})"
    if shape is None:
        return None, f"crop_shape 는 [H, W] 정수 두 개여야 합니다 (지금: {raw_shape!r})"
    if shape[0] <= 0 or shape[1] <= 0:
        return None, f"crop_shape 는 0 보다 커야 합니다 (지금: {shape[1]}x{shape[0]})"
    if start[0] < 0 or start[1] < 0:
        return None, f"crop_start_xy 는 0 이상이어야 합니다 (지금: x={start[0]}, y={start[1]})"
    return (start, shape), ""


def validate_crop(resize_hw: Optional[Tuple[int, int]], crop: Optional[Crop]) -> str:
    if crop is None or resize_hw is None:
        return ""
    (x, y), (ch, cw) = crop
    th, tw = resize_hw
    if y + ch > th or x + cw > tw:
        return (
            f"크롭 영역이 리사이즈 결과({tw}x{th})를 벗어납니다 — "
            f"시작 (x={x}, y={y}) 에서 {cw}x{ch} 를 자르면 "
            f"오른쪽 {x + cw}px, 아래쪽 {y + ch}px 까지 필요합니다"
        )
    return ""


def crop_image(img: np.ndarray, crop: Optional[Crop]) -> np.ndarray:
    if img is None or crop is None:
        return img
    (x, y), (ch, cw) = crop
    h, w = img.shape[:2]
    if y + ch > h or x + cw > w:
        return img
    if (x, y, ch, cw) == (0, 0, h, w):
        return img
    return np.ascontiguousarray(img[y : y + ch, x : x + cw])


def resize_then_crop(
    img: np.ndarray | None,
    *,
    resize_hw: Optional[Tuple[int, int]],
    crop: Optional[Crop],
) -> np.ndarray | None:
    """원본 → resize_save_shape → crop (data_logger 와 동일 순서)."""
    if img is None:
        return None

    out = img
    if resize_hw is not None:
        th, tw = resize_hw
        h, w = out.shape[:2]
        if (h, w) != (th, tw):
            interp = cv2.INTER_NEAREST if out.dtype == np.uint16 else cv2.INTER_AREA
            out = np.ascontiguousarray(cv2.resize(out, (tw, th), interpolation=interp))

    return crop_image(out, crop)
