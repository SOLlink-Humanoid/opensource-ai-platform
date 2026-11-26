"""Read YOLO checkpoint metadata without running the live detection worker."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _normalize_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, dict):
        return {int(k): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {i: str(name) for i, name in enumerate(raw)}
    return {}


def yolo_class_names_from_config(
    model_path: Path | str | None,
    *,
    config_path=None,
) -> dict[int, str]:
    """Class names from ``config/yolo.json`` targets (no ``.pt`` load)."""
    from .yolo_targets_store import _read_file, _settings_file_candidates

    names: dict[int, str] = {}
    for candidate in _settings_file_candidates(model_path, config_path=config_path):
        if not candidate.is_file():
            continue
        data = _read_file(candidate)
        if not data:
            continue
        raw_targets = data.get("targets")
        if not isinstance(raw_targets, list):
            continue
        for row in raw_targets:
            if not isinstance(row, dict):
                continue
            if row.get("class_id") is None:
                continue
            label = str(row.get("name") or "").strip()
            if not label:
                continue
            try:
                names[int(row["class_id"])] = label
            except (TypeError, ValueError):
                continue
        if names:
            return names
    return names


def yolo_class_names(model_path: Path, *, allow_weight_load: bool = True) -> dict[int, str]:
    """Load class id → name map from an Ultralytics ``.pt`` file."""
    if not allow_weight_load:
        return yolo_class_names_from_config(model_path)
    path = model_path.expanduser()
    if not path.is_file():
        return {}
    try:
        from ultralytics import YOLO

        model = YOLO(str(path))
        return _normalize_names(getattr(model, "names", None))
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not read YOLO class names from %s: %s", path, exc)
        return {}
