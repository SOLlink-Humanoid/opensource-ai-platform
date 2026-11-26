"""Prefer raw camera relay when data_studio peers (camera_stream + data_logger) are up."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

_log = logging.getLogger(__name__)

_DEFAULT_DATA_LOGGER_URL = "http://127.0.0.1:8002"
_PROBE_TIMEOUT_SEC = 0.85


def data_logger_base_url() -> str:
    raw = os.environ.get("OAP_DATA_LOGGER_URL", "").strip()
    return raw or _DEFAULT_DATA_LOGGER_URL


def probe_data_logger(
    base_url: str | None = None,
    *,
    timeout_sec: float = _PROBE_TIMEOUT_SEC,
) -> bool:
    base = (base_url or data_logger_base_url()).rstrip("/")
    url = f"{base}/health"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status != 200:
                return False
            body = resp.read(4096).decode("utf-8", errors="replace")
        data = json.loads(body)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
        _log.debug("data_logger probe failed (%s): %s", url, exc)
        return False
    return isinstance(data, dict) and data.get("status") == "ok"


def resolve_studio_raw_preview(camera_service: Any) -> bool:
    """Raw MJPEG only (no YOLO annotate path) when relaying camera_stream and logger is up."""
    if os.environ.get("OAP_STUDIO_RAW_PREVIEW", "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    from .camera_stream_relay import CameraStreamRelayService

    if not isinstance(camera_service, CameraStreamRelayService):
        return False
    if not probe_data_logger():
        return False
    _log.info(
        "Studio raw preview: camera_stream relay + data_logger; YOLO overlay stream disabled"
    )
    return True