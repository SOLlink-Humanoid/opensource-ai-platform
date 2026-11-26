"""OpenVLA RGB input matching raw-to-openvla-converter training tiles."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

OPENVLA_TILE_WIDTH = 384
OPENVLA_TILE_HEIGHT = 256
OPENVLA_TILE_SIZE = (OPENVLA_TILE_WIDTH, OPENVLA_TILE_HEIGHT)


def bgr_to_openvla_pil(
    bgr: np.ndarray,
    *,
    tile_size: tuple[int, int] = OPENVLA_TILE_SIZE,
) -> Image.Image:
    """Crop-sized BGR frames → 384×256 letterbox (same as convert_dataset mosaic)."""
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    base = Image.fromarray(rgb)
    fitted = ImageOps.contain(base, tile_size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", tile_size, (0, 0, 0))
    x = (tile_size[0] - fitted.width) // 2
    y = (tile_size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas
