"""Head-camera frames via data_studio camera_stream (avoids a second GDK poller)."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .camera_service import (
    CameraSpec,
    FrameSnapshot,
    _gdk_poll_camera_ids,
    _process_frame,
    _specs_for_cameras,
)

_log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8003"
_PROBE_TIMEOUT_SEC = 0.85
_STREAM_READ_TIMEOUT_SEC = 30.0
_RECONNECT_DELAY_SEC = 1.0


def _env_bool(key: str, default: bool) -> bool:
    import os

    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def camera_stream_base_url() -> str:
    import os

    raw = os.environ.get("OAP_CAMERA_STREAM_URL", "").strip()
    return raw or _DEFAULT_BASE_URL


def probe_camera_stream(
    base_url: str | None = None,
    *,
    timeout_sec: float = _PROBE_TIMEOUT_SEC,
) -> frozenset[str] | None:
    """Return streamable camera logical names if camera_stream health responds."""
    base = (base_url or camera_stream_base_url()).rstrip("/")
    url = f"{base}/health"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status != 200:
                return None
            body = resp.read(65536).decode("utf-8", errors="replace")
        data = json.loads(body)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
        _log.debug("camera_stream probe failed (%s): %s", url, exc)
        return None
    if not isinstance(data, dict) or data.get("status") != "ok":
        return None
    raw = data.get("cameras")
    if not isinstance(raw, list):
        return None
    names = frozenset(str(x).strip() for x in raw if str(x).strip())
    return names if names else None


def resolve_camera_stream_relay(
    *,
    topics_path: Path | None,
    base_url: str | None = None,
) -> frozenset[str] | None:
    """OAP camera ids to pull from camera_stream when the server is up."""
    if _env_bool("OAP_CAMERA_STREAM_FORCE_GDK", False):
        return None
    if not _env_bool("OAP_CAMERA_STREAM_AUTO", True):
        return None
    available = probe_camera_stream(base_url)
    if not available:
        return None
    wanted = _gdk_poll_camera_ids()
    matched = frozenset(cid for cid in wanted if cid in available)
    if not matched:
        _log.warning(
            "camera_stream is up but none of %s are in %s",
            ",".join(sorted(wanted)),
            ",".join(sorted(available)),
        )
        return None
    return matched


class _MjpegFrameReader:
    """Blocking read of the latest JPEG from a multipart MJPEG HTTP stream."""

    def __init__(self, stream_url: str) -> None:
        self._url = stream_url
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def iter_jpegs(self):
        while not self._stop.is_set():
            try:
                req = urllib.request.Request(
                    self._url,
                    headers={"Accept": "multipart/x-mixed-replace"},
                )
                with urllib.request.urlopen(
                    req,
                    timeout=_STREAM_READ_TIMEOUT_SEC,
                ) as resp:
                    boundary = self._parse_boundary(
                        resp.headers.get("Content-Type", "")
                    )
                    if not boundary:
                        _log.warning("MJPEG missing boundary: %s", self._url)
                        time.sleep(_RECONNECT_DELAY_SEC)
                        continue
                    yield from self._read_multipart(resp, boundary)
            except Exception as exc:  # noqa: BLE001
                if not self._stop.is_set():
                    _log.debug("camera_stream read %s: %s", self._url, exc)
                time.sleep(_RECONNECT_DELAY_SEC)

    @staticmethod
    def _parse_boundary(content_type: str) -> bytes | None:
        match = re.search(r"boundary=([^;\s]+)", content_type, flags=re.I)
        if not match:
            return None
        token = match.group(1).strip().strip('"')
        return ("--" + token).encode("ascii")

    def _read_multipart(self, resp, boundary: bytes):
        buf = b""
        delim = boundary + b"\r\n"
        while not self._stop.is_set():
            chunk = resp.read(8192)
            if not chunk:
                break
            buf += chunk
            while True:
                start = buf.find(delim)
                if start < 0:
                    if len(buf) > 4_000_000:
                        buf = buf[-256:]
                    break
                buf = buf[start + len(delim) :]
                header_end = buf.find(b"\r\n\r\n")
                if header_end < 0:
                    buf = delim + buf
                    break
                headers = buf[:header_end].decode("latin-1", errors="replace")
                if "image/jpeg" not in headers.lower():
                    buf = buf[header_end + 4 :]
                    continue
                body_start = header_end + 4
                next_boundary = buf.find(boundary, body_start)
                if next_boundary < 0:
                    buf = delim + buf
                    break
                jpeg = buf[body_start:next_boundary].rstrip(b"\r\n")
                buf = buf[next_boundary:]
                if jpeg:
                    yield jpeg


class CameraStreamRelayService:
    """Frame hub fed from camera_stream HTTP MJPEG (no local GDK polling)."""

    def __init__(
        self,
        *,
        base_url: str,
        camera_ids: Iterable[str],
        topics_path: Path | None = None,
        apply_oap_pipeline: bool = False,
    ) -> None:
        ids = frozenset(camera_ids)
        specs = _specs_for_cameras(ids, topics_path=topics_path)
        if not specs:
            head = "head_color" if "head_color" in ids else next(iter(ids))
            specs = _specs_for_cameras(frozenset({head}), topics_path=topics_path)
        self._specs = specs
        self._spec_by_id = {spec.id: spec for spec in specs}
        self.head_camera_id = (
            "head_color"
            if "head_color" in self._spec_by_id
            else self._specs[0].id
        )
        self._base = base_url.rstrip("/")
        self._apply_pipeline = apply_oap_pipeline
        self.ready = False
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._latest: dict[str, FrameSnapshot] = {}
        self._frame_ids: dict[str, int] = {spec.id: 0 for spec in specs}
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def list_cameras(self) -> list[CameraSpec]:
        return list(self._specs)

    def default_camera_id(self) -> str:
        return self.head_camera_id

    def shutdown(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=3.0)
        self._threads.clear()
        self.ready = False

    def latest_frame(self, camera_id: str, *, wait: bool = True) -> FrameSnapshot | None:
        deadline = time.monotonic() + (2.0 if wait else 0.0)
        while True:
            with self._lock:
                snap = self._latest.get(camera_id)
                if snap is not None:
                    return snap
            if not wait or time.monotonic() >= deadline:
                return None
            time.sleep(0.02)

    def wait_for_frame(
        self,
        camera_id: str,
        *,
        after_frame_id: int,
        timeout: float = 0.5,
    ) -> FrameSnapshot | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                snap = self._latest.get(camera_id)
                if snap is not None and snap.frame_id > after_frame_id:
                    return snap
            self._stop.wait(0.01)
        return None

    def get_jpeg_with_id(self, camera_id: str) -> tuple[bytes | None, int]:
        snap = self.latest_frame(camera_id, wait=True)
        if snap is None:
            return None, 0
        return snap.jpeg, snap.frame_id

    def encode_jpeg(self, camera_id: str) -> bytes | None:
        jpeg, _ = self.get_jpeg_with_id(camera_id)
        return jpeg

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        for spec in self._specs:
            thread = threading.Thread(
                target=self._run_camera,
                args=(spec.id,),
                name=f"camera-stream-{spec.id}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        self.ready = True
        _log.info(
            "Camera stream relay started (%s, cameras=%s)",
            self._base,
            ",".join(spec.id for spec in self._specs),
        )

    def _publish(self, camera_id: str, jpeg: bytes, bgr: np.ndarray) -> None:
        self._frame_ids[camera_id] = self._frame_ids.get(camera_id, 0) + 1
        snap = FrameSnapshot(
            camera_id=camera_id,
            frame_id=self._frame_ids[camera_id],
            bgr=bgr,
            jpeg=jpeg,
            published_at=time.time(),
        )
        with self._lock:
            self._latest[camera_id] = snap
        self.last_error = None

    def _run_camera(self, camera_id: str) -> None:
        url = f"{self._base}/stream/{camera_id}"
        reader = _MjpegFrameReader(url)
        spec = self._spec_by_id.get(camera_id)
        try:
            for jpeg in reader.iter_jpegs():
                if self._stop.is_set():
                    break
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                if self._apply_pipeline and spec is not None:
                    bgr = _process_frame(spec, bgr)
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        bgr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                    )
                    if not ok:
                        continue
                    jpeg = encoded.tobytes()
                self._publish(camera_id, jpeg, bgr)
        finally:
            reader.stop()