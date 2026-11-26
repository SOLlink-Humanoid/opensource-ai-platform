"""One GDK process for camera frames and joint state."""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.RLock()
_ready = False
_init_error: str | None = None
_camera: Any = None
_robot: Any = None


def gdk_error() -> str | None:
    return _init_error


def ensure_gdk() -> Any:
    """Import and initialise agibot_gdk. Raises RuntimeError on failure."""
    global _ready, _init_error
    with _lock:
        if _ready:
            import agibot_gdk

            return agibot_gdk
        try:
            import agibot_gdk
        except Exception as exc:  # noqa: BLE001
            _init_error = f"agibot_gdk import failed: {exc}"
            raise RuntimeError(_init_error) from exc
        try:
            result = agibot_gdk.gdk_init()
        except Exception as exc:  # noqa: BLE001
            _init_error = f"gdk_init failed: {exc}"
            raise RuntimeError(_init_error) from exc
        success = getattr(getattr(agibot_gdk, "GDKRes", None), "kSuccess", 0)
        if result != success:
            _init_error = f"gdk_init returned {result!r}"
            raise RuntimeError(_init_error)
        _ready = True
        _init_error = None
        return agibot_gdk


def get_camera() -> Any:
    global _camera
    gdk = ensure_gdk()
    with _lock:
        if _camera is None:
            _camera = gdk.Camera()
        return _camera


def get_robot() -> Any:
    global _robot
    gdk = ensure_gdk()
    with _lock:
        if _robot is None:
            _robot = gdk.Robot()
        return _robot


def call_gdk(fn):
    with _lock:
        return fn()
