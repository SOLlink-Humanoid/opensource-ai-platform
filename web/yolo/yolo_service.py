"""YOLO11 worker over the shared camera frame cache."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

_log = logging.getLogger(__name__)

_BOX_THICKNESS = 1
_LABEL_FONT_CANDIDATES: tuple[Path, ...] = (
    Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
)
_LABEL_FONT_SIZE = 10
_LABEL_STROKE_WIDTH = 1
_LABEL_OUTLINE_BGR = (255, 255, 255)
_DEFAULT_BGR = (0, 165, 255)  # orange

# BGR — matched as substrings in the class name (e.g. blue-box → blue).
_LABEL_COLOR_BGR: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("black", (0, 0, 0)),
    ("blue", (255, 120, 0)),
    ("green", (0, 200, 0)),
    ("red", (0, 0, 255)),
    ("yellow", (0, 255, 255)),
    ("orange", (0, 165, 255)),
    ("purple", (200, 0, 200)),
    ("violet", (200, 0, 200)),
    ("pink", (180, 105, 255)),
    ("white", (255, 255, 255)),
    ("gray", (160, 160, 160)),
    ("grey", (160, 160, 160)),
    ("cyan", (255, 255, 0)),
    ("magenta", (255, 0, 255)),
)


def _color_from_label(label: str) -> tuple[int, int, int]:
    lower = label.lower()
    for key, bgr in _LABEL_COLOR_BGR:
        if key in lower:
            return bgr
    return _DEFAULT_BGR


def _label_text_color(box_bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    if sum(box_bgr) < 80:
        return (255, 255, 255)
    return box_bgr


@lru_cache(maxsize=1)
def _label_font():
    from PIL import ImageFont

    for path in _LABEL_FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), _LABEL_FONT_SIZE)
    _log.warning("Label font not found; using default (%s)", _LABEL_FONT_CANDIDATES)
    return ImageFont.load_default()


def _paint_labels(
    canvas: np.ndarray,
    labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]],
) -> None:
    if not labels:
        return
    from PIL import Image, ImageDraw

    rgb = np.ascontiguousarray(canvas[..., ::-1])
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _label_font()
    for text, origin, color_bgr in labels:
        outline_bgr = _LABEL_OUTLINE_BGR if sum(color_bgr) < 700 else (0, 0, 0)
        fill = (color_bgr[2], color_bgr[1], color_bgr[0])
        stroke = (outline_bgr[2], outline_bgr[1], outline_bgr[0])
        draw.text(
            origin,
            text,
            font=font,
            fill=fill,
            stroke_width=_LABEL_STROKE_WIDTH,
            stroke_fill=stroke,
            anchor="ls",
        )
    canvas[:] = np.asarray(pil)[..., ::-1]


@dataclass
class ClassTarget:
    enabled: bool = True
    threshold: float = 0.7


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class YoloFrame:
    camera_id: str
    frame_id: int
    inference_ms: float
    detections: tuple[Detection, ...]
    jpeg: bytes


class YoloDetectionService:
    def __init__(
        self,
        camera_service,
        *,
        model_path: Path | None,
        camera_id: str,
        device: str | None,
        confidence: float,
        image_size: int,
        min_interval_sec: float = 0.2,
    ) -> None:
        self._camera = camera_service
        self.model_path = Path(model_path) if model_path is not None else None
        self._camera_id = camera_id
        self._device = device or "cpu"
        self.confidence = float(confidence)
        self._image_size = int(image_size)
        self._min_interval = max(0.05, float(min_interval_sec))
        self.ready = False
        self.last_error: str | None = None
        self._model = None
        self._names: dict[int, str] = {}
        self._targets: dict[int, ClassTarget] = {}
        self._lock = threading.Lock()
        self._latest: YoloFrame | None = None
        self._paused = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    def set_confidence(self, confidence: float) -> None:
        value = float(confidence)
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = value
        for target in self._targets.values():
            target.threshold = value

    def _sync_targets_with_names(self) -> None:
        default = float(self.confidence)
        for class_id, name in self._names.items():
            if class_id not in self._targets:
                self._targets[class_id] = ClassTarget(enabled=True, threshold=default)
        for class_id in list(self._targets):
            if class_id not in self._names:
                del self._targets[class_id]

    def targets_snapshot(self) -> list[dict[str, object]]:
        self._sync_targets_with_names()
        rows: list[dict[str, object]] = []
        for class_id in sorted(self._names):
            target = self._targets[class_id]
            rows.append(
                {
                    "class_id": class_id,
                    "name": self._names[class_id],
                    "enabled": target.enabled,
                    "threshold": float(target.threshold),
                }
            )
        return rows

    def apply_targets(self, updates: list[dict[str, object]]) -> None:
        self._sync_targets_with_names()
        for row in updates:
            class_id = int(row["class_id"])
            if class_id not in self._names:
                raise ValueError(f"Unknown class_id: {class_id}")
            target = self._targets[class_id]
            if "enabled" in row and row["enabled"] is not None:
                target.enabled = bool(row["enabled"])
            if "threshold" in row and row["threshold"] is not None:
                value = float(row["threshold"])
                if value < 0.0 or value > 1.0:
                    raise ValueError("threshold must be between 0 and 1")
                target.threshold = value

    def _predict_confidence(self) -> float:
        self._sync_targets_with_names()
        enabled = [
            float(target.threshold)
            for target in self._targets.values()
            if target.enabled
        ]
        if not enabled:
            return min(1.0, float(self.confidence))
        return min(enabled)

    def filter_detections(
        self,
        detections: list[Detection],
    ) -> tuple[Detection, ...]:
        self._sync_targets_with_names()
        kept: list[Detection] = []
        for det in detections:
            target = self._targets.get(det.class_id)
            if target is None or not target.enabled:
                continue
            if det.confidence + 1e-9 < float(target.threshold):
                continue
            kept.append(det)
        kept.sort(key=lambda d: (d.label, -d.confidence))
        return tuple(kept)

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def set_min_interval(self, interval_sec: float) -> None:
        self._min_interval = max(0.05, float(interval_sec))

    def latest(self, camera_id: str) -> YoloFrame | None:
        if camera_id != self._camera_id:
            return None
        with self._lock:
            return self._latest

    def start(self) -> None:
        if self.model_path is None or not self.model_path.is_file():
            self.last_error = f"YOLO weights not found: {self.model_path}"
            _log.warning(self.last_error)
            return
        from ultralytics import YOLO

        self._model = YOLO(str(self.model_path))
        names = getattr(self._model, "names", {}) or {}
        if isinstance(names, dict):
            self._names = {int(key): str(value) for key, value in names.items()}
        elif isinstance(names, (list, tuple)):
            self._names = {index: str(name) for index, name in enumerate(names)}
        self._sync_targets_with_names()
        if self._thread is not None:
            return
        self._stop.clear()
        self._paused.set()
        self._thread = threading.Thread(target=self._run, name="yolo", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.ready = False

    def _annotate(self, bgr: np.ndarray, detections: tuple[Detection, ...]) -> np.ndarray:
        canvas = bgr.copy()
        labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
        index_by_label: dict[str, int] = {}
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            box_color = _color_from_label(det.label)
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                box_color,
                _BOX_THICKNESS,
                cv2.LINE_AA,
            )
            index_by_label[det.label] = index_by_label.get(det.label, 0) + 1
            caption = (
                f"({index_by_label[det.label]}) {det.label} {det.confidence:.2f}"
            )
            text_color = _label_text_color(box_color)
            labels.append((caption, (x1, max(14, y1 - 4)), text_color))
        _paint_labels(canvas, labels)
        return canvas

    def _run(self) -> None:
        self.ready = True
        _log.info(
            "YOLO worker started (%s, device=%s, interval=%.2fs)",
            self.model_path,
            self._device,
            self._min_interval,
        )
        last_frame_id = 0
        last_infer_at = 0.0
        while not self._stop.is_set():
            if self._paused.is_set():
                self._stop.wait(0.05)
                continue
            now = time.monotonic()
            if now - last_infer_at < self._min_interval:
                self._stop.wait(0.02)
                continue
            snapshot = self._camera.latest_frame(self._camera_id, wait=False)
            if snapshot is None or snapshot.frame_id == last_frame_id:
                self._stop.wait(0.03)
                continue
            last_frame_id = snapshot.frame_id
            started = time.perf_counter()
            try:
                results = self._model.predict(
                    snapshot.bgr,
                    conf=self._predict_confidence(),
                    imgsz=self._image_size,
                    device=self._device,
                    verbose=False,
                )
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                _log.exception("YOLO inference failed")
                self._stop.wait(0.5)
                continue
            detections: list[Detection] = []
            if results:
                boxes = getattr(results[0], "boxes", None)
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls.item())
                        xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
                        detections.append(
                            Detection(
                                class_id=cls_id,
                                label=self._names.get(cls_id, str(cls_id)),
                                confidence=float(box.conf.item()),
                                xyxy=xyxy,  # type: ignore[arg-type]
                            )
                        )
            filtered = self.filter_detections(detections)
            annotated = self._annotate(snapshot.bgr, filtered)
            ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            frame = YoloFrame(
                camera_id=self._camera_id,
                frame_id=snapshot.frame_id,
                inference_ms=(time.perf_counter() - started) * 1000.0,
                detections=filtered,
                jpeg=encoded.tobytes(),
            )
            with self._lock:
                self._latest = frame
                self.last_error = None
            last_infer_at = time.monotonic()
