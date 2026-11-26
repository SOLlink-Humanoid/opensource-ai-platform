#!/usr/bin/env python3
"""Fixed-range Turbo RGB mapping for metric depth.

Matches convert_depth_to_color.py:

    valid = finite(depth) and depth > 0 and depth != invalid_value
    scaled = clip((depth_mm - vmin) / (vmax - vmin), 0, 1)
    rgb = OpenCV COLORMAP_TURBO[round(scaled * 255)] as RGB
    invalid pixels stay black
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

import cv2
import numpy as np


def depth_to_turbo_rgb(
    depth: np.ndarray,
    vmin: float,
    vmax: float,
    invalid_value: float = 0.0,
    input_scale: float = 1.0,
) -> np.ndarray:
    """Map a metric depth image to a 3-channel uint8 Turbo RGB image."""
    values = np.asarray(depth)
    if values.ndim == 3 and values.shape[2] == 1:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"Depth image must be HxW or HxWx1, got shape {values.shape}")

    numeric = values.astype(np.float32, copy=False)
    if input_scale != 1.0:
        numeric = numeric * float(input_scale)
    valid = np.isfinite(numeric) & (numeric > 0)
    if invalid_value == 0:
        valid &= numeric != 0
    else:
        valid &= numeric != float(invalid_value)

    rgb = np.zeros((values.shape[0], values.shape[1], 3), dtype=np.uint8)
    if not np.any(valid):
        return rgb

    span = float(vmax) - float(vmin)
    if span <= 0:
        span = 1.0
    scaled = np.clip((numeric - float(vmin)) / span, 0.0, 1.0)
    indices = np.zeros(values.shape, dtype=np.uint8)
    indices[valid] = np.rint(scaled[valid] * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(indices, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return np.ascontiguousarray(rgb)


def depth_to_turbo_rgb_from_config(
    depth: np.ndarray,
    config: Union[Mapping[str, Any], Any],
) -> np.ndarray:
    """Apply a DepthPreprocessSpec or equivalent mapping object/dict."""
    if isinstance(config, Mapping):
        vmin = float(config["vmin"])
        vmax = float(config["vmax"])
        invalid_value = float(config.get("invalid_value", 0.0))
        input_scale = float(config.get("input_scale", 1.0))
    else:
        vmin = float(config.vmin)
        vmax = float(config.vmax)
        invalid_value = float(getattr(config, "invalid_value", 0.0))
        input_scale = float(getattr(config, "input_scale", 1.0))
    return depth_to_turbo_rgb(
        depth,
        vmin=vmin,
        vmax=vmax,
        invalid_value=invalid_value,
        input_scale=input_scale,
    )


def unpack_camera_config(config: tuple) -> tuple:
    """Support 4-tuple camera configs and optional depth_preprocess."""
    camera_key, obs_key, target_size, is_depth = config[:4]
    depth_preprocess: Optional[Any] = config[4] if len(config) > 4 else None
    return camera_key, obs_key, target_size, is_depth, depth_preprocess
