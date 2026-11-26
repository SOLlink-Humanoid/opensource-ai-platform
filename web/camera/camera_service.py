"""Shared head-camera frames: GDK on the robot, synthetic frames in mock mode."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera_config import (
    CameraSpec,
    load_camera_pipeline,
    output_hw_for_pipeline,
    resolve_topics_path,
)
from .gdk_image_decode import decode_gdk_image
from .image_pipeline import resize_then_crop

_log = logging.getLogger(__name__)

JPEG_QUALITY = 80

_HEAD_CAMERAS: tuple[tuple[str, str, str], ...] = (
    ("head_color", "Head color", "kHeadColor"),
    ("hand_left_color", "Left hand", "kHandLeftColor"),
    ("hand_right_color", "Right hand", "kHandRightColor"),
)


@dataclass(frozen=True)
class FrameSnapshot:
    camera_id: str
    frame_id: int
    bgr: np.ndarray
    jpeg: bytes
    published_at: float


def _encode_jpeg(bgr: np.ndarray) -> bytes | None:
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ok:
        return None
    return encoded.tobytes()


def _specs_for_cameras(
    camera_ids: frozenset[str],
    *,
    topics_path: Path | None,
) -> list[CameraSpec]:
    specs: list[CameraSpec] = []
    for camera_id, label, attr in _HEAD_CAMERAS:
        if camera_id not in camera_ids:
            continue
        pipeline = load_camera_pipeline(topics_path, camera_id)
        output_hw = output_hw_for_pipeline(pipeline)
        specs.append(
            CameraSpec(
                id=camera_id,
                label=label,
                gdk_camera_id=attr,
                description=label,
                resize_hw=pipeline.resize_hw,
                crop=pipeline.crop,
                output_hw=output_hw,
            )
        )
    return specs


def _process_frame(spec: CameraSpec | None, raw: np.ndarray) -> np.ndarray:
    if spec is None:
        return raw
    processed = resize_then_crop(
        raw,
        resize_hw=spec.resize_hw,
        crop=spec.crop,
    )
    return raw if processed is None else processed


class _FrameHub:
    def __init__(self, specs: list[CameraSpec], head_camera_id: str) -> None:
        self._specs = specs
        self._spec_by_id = {spec.id: spec for spec in specs}
        self.head_camera_id = head_camera_id
        self.ready = False
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._latest: dict[str, FrameSnapshot] = {}
        self._frame_ids: dict[str, int] = {spec.id: 0 for spec in specs}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def list_cameras(self) -> list[CameraSpec]:
        return list(self._specs)

    def default_camera_id(self) -> str:
        return self.head_camera_id

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
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

    def _publish(self, camera_id: str, bgr: np.ndarray) -> None:
        jpeg = _encode_jpeg(bgr)
        if jpeg is None:
            return
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

    def _start_thread(self, name: str, target) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=target, name=name, daemon=True)
        self._thread.start()


class MockCameraService(_FrameHub):
    def __init__(self, *, topics_path: Path | None = None) -> None:
        pipeline = load_camera_pipeline(topics_path, "head_color")
        h, w = output_hw_for_pipeline(pipeline, default=(480, 640))
        spec = CameraSpec(
            id="mock_head",
            label="Mock head camera",
            gdk_camera_id="mock",
            description="Synthetic frames",
            resize_hw=pipeline.resize_hw,
            crop=pipeline.crop,
            output_hw=(h, w),
        )
        super().__init__([spec], spec.id)
        self._mock_spec = spec

    def start(self) -> None:
        self._start_thread("mock-camera", self._run)

    def _run(self) -> None:
        self.ready = True
        h, w = self._mock_spec.output_hw
        _log.info("Mock camera publishing %dx%d frames", w, h)
        while not self._stop.is_set():
            frame = np.zeros((max(h, 1), max(w, 1), 3), dtype=np.uint8)
            frame[:, :] = (32, 48, 32)
            cv2.putText(
                frame,
                "mock camera",
                (40, max(40, h // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            self._publish(self.head_camera_id, frame)
            self._stop.wait(0.1)


def _gdk_poll_camera_ids() -> frozenset[str]:
    """Cameras polled in the GDK worker (env overrides). Default: head only."""
    import os

    raw = os.environ.get("GDK_POLL_CAMERAS", "").strip()
    if not raw:
        return frozenset({"head_color"})
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


class GdkCameraService(_FrameHub):
    def __init__(self, *, topics_path: Path | None = None) -> None:
        allowed = _gdk_poll_camera_ids()
        specs = _specs_for_cameras(allowed, topics_path=topics_path)
        if not specs:
            specs = [
                CameraSpec(
                    id="head_color",
                    label="Head color",
                    gdk_camera_id="kHeadColor",
                    description="Head color",
                )
            ]
        super().__init__(specs, "head_color")
        self._attrs = {
            camera_id: attr
            for camera_id, _label, attr in _HEAD_CAMERAS
            if camera_id in allowed
        }
        topics = resolve_topics_path(topics_path)
        for spec in specs:
            _log.info(
                "Camera %s pipeline (topics=%s) resize=%s crop=%s → output %dx%d",
                spec.id,
                topics,
                spec.resize_hw,
                spec.crop,
                spec.output_hw[1],
                spec.output_hw[0],
            )

    def start(self) -> None:
        self._start_thread("gdk-camera", self._run)

    def _run(self) -> None:
        try:
            from web.runtime.gdk_session import call_gdk, get_camera
            import agibot_gdk

            camera = get_camera()
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            _log.error("GDK camera unavailable: %s", exc)
            return

        self.ready = True
        _log.info(
            "GDK camera poll started (head=%s, polling=%s)",
            self.head_camera_id,
            ",".join(self._attrs) or "(none)",
        )
        while not self._stop.is_set():
            for camera_id, attr in self._attrs.items():
                camera_type = getattr(agibot_gdk.CameraType, attr, None)
                if camera_type is None:
                    continue
                try:
                    image = call_gdk(lambda camera_type=camera_type: camera.get_latest_image(camera_type, 200.0))
                except Exception as exc:  # noqa: BLE001
                    self.last_error = f"{camera_id}: {exc}"
                    continue
                if image is None:
                    continue
                frame = decode_gdk_image(image, agibot_gdk)
                if frame is None:
                    continue
                spec = self._spec_by_id.get(camera_id)
                frame = _process_frame(spec, frame)
                self.last_error = None
                self._publish(camera_id, frame)
            self._stop.wait(0.05)


def build_camera_service(
    mock: bool,
    topics_path: Path | None = None,
    mapping_path: Path | None = None,
):
    del mapping_path
    if mock:
        return MockCameraService(topics_path=topics_path)
    from .camera_stream_relay import (
        camera_stream_base_url,
        CameraStreamRelayService,
        resolve_camera_stream_relay,
    )

    base_url = camera_stream_base_url()
    relay_ids = resolve_camera_stream_relay(topics_path=topics_path, base_url=base_url)
    if relay_ids:
        _log.info(
            "Using camera_stream relay at %s (cameras=%s); skipping GDK poll",
            base_url,
            ",".join(sorted(relay_ids)),
        )
        return CameraStreamRelayService(
            base_url=base_url,
            camera_ids=relay_ids,
            topics_path=topics_path,
            apply_oap_pipeline=False,
        )
    _log.info("camera_stream unavailable — using GDK camera poll")
    return GdkCameraService(topics_path=topics_path)
