"""Camera metadata + data_logger-aligned crop from agibot_g2_topics.yaml."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .image_pipeline import Crop, resolve_crop, validate_crop

_log = logging.getLogger(__name__)

_DEFAULT_TOPICS = Path("/home/agi/data_studio/config/agibot_g2_topics.yaml")
_FALLBACK_TOPICS = Path("/home/agi/data_manager/config/agibot_g2_topics.yaml")


@dataclass(frozen=True)
class CameraSpec:
    id: str
    label: str
    gdk_camera_id: str = ""
    description: str = ""
    resize_hw: tuple[int, int] | None = None
    crop: Crop | None = None
    output_hw: tuple[int, int] = (400, 640)


@dataclass(frozen=True)
class CameraPipeline:
    resize_hw: tuple[int, int] | None
    crop: Crop | None


def resolve_topics_path(explicit: Path | None = None) -> Path:
    env = os.environ.get("HUMANOID_TOPICS_YAML", "").strip()
    if env:
        return Path(env)
    if explicit is not None and explicit.is_file():
        return explicit
    if _DEFAULT_TOPICS.is_file():
        return _DEFAULT_TOPICS
    if _FALLBACK_TOPICS.is_file():
        return _FALLBACK_TOPICS
    return explicit or _DEFAULT_TOPICS


def _resize_hw_from_topic(entry: dict) -> tuple[int, int] | None:
    shape = entry.get("resize_save_shape") or entry.get("resize_stream_shape") or entry.get("save_shape")
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        return None
    return int(shape[0]), int(shape[1])


def load_topic_entry(topics_path: Path, camera_id: str) -> dict | None:
    if not topics_path.is_file():
        return None
    with topics_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    for entry in data.get("topics", []):
        if isinstance(entry, dict) and entry.get("name") == camera_id:
            return entry
    return None


def load_camera_pipeline(topics_path: Path | None, camera_id: str) -> CameraPipeline:
    path = resolve_topics_path(topics_path)
    entry = load_topic_entry(path, camera_id)
    if entry is None:
        _log.warning("No topics entry for camera %s (%s)", camera_id, path)
        return CameraPipeline(None, None)
    resize_hw = _resize_hw_from_topic(entry)
    crop, err = resolve_crop(entry)
    if err:
        _log.warning("Camera %s crop config: %s", camera_id, err)
        crop = None
    elif crop is not None:
        validate_err = validate_crop(resize_hw, crop)
        if validate_err:
            _log.warning("Camera %s crop invalid: %s", camera_id, validate_err)
            crop = None
    return CameraPipeline(resize_hw, crop)


def output_hw_for_pipeline(
    pipeline: CameraPipeline,
    default: tuple[int, int] = (400, 640),
) -> tuple[int, int]:
    if pipeline.crop is not None:
        (_, _), (ch, cw) = pipeline.crop
        return ch, cw
    if pipeline.resize_hw is not None:
        return pipeline.resize_hw
    return default
