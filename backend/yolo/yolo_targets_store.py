"""Persist YOLO class target toggles/thresholds next to the model weights."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from ..platform.config_file import default_config_path, resolve_config_path
from ..platform.model_config_dir import yolo_settings_file

_log = logging.getLogger(__name__)

DEFAULT_YOLO_CONFIDENCE = 0.7
_TARGETS_BASENAME = "yolo_targets.json"


def _model_dir(model_path: Path | str | None) -> Path | None:
    if model_path is None:
        return None
    path = Path(model_path).expanduser()
    try:
        path = path.resolve()
    except OSError:
        path = path.absolute()
    if path.is_file():
        return path.parent
    if path.is_dir():
        return path
    if path.parent.is_dir():
        return path.parent
    return None


def yolo_targets_path(
    model_path: Path | str | None = None,
    *,
    config_path: Path | None = None,
) -> Path | None:
    """Canonical write path: ``<platform>/config/yolo.json``."""
    del model_path, config_path
    return yolo_settings_file(create_dir=True)


def _model_dir_targets_path(model_path: Path | str | None) -> Path | None:
    model_dir = _model_dir(model_path)
    if model_dir is None:
        return None
    return model_dir / _TARGETS_BASENAME


def _legacy_yaml_path(json_path: Path) -> Path:
    return json_path.with_suffix(".yaml")


def _legacy_config_dir_paths(config_path: Path | None) -> list[Path]:
    cfg = config_path if config_path is not None else resolve_config_path()
    base = cfg.parent if cfg is not None else default_config_path().parent
    return [base / _TARGETS_BASENAME, base / "yolo_targets.yaml"]


def _settings_file_candidates(
    model_path: Path | str | None,
    *,
    config_path: Path | None = None,
) -> list[Path]:
    """Read order: model folder JSON/YAML, then legacy config-dir copies."""
    seen: set[Path] = set()
    ordered: list[Path] = []

    def add(path: Path | None) -> None:
        if path is None:
            return
        key = path.resolve() if path.exists() else path
        if key in seen:
            return
        seen.add(key)
        ordered.append(path)

    add(yolo_settings_file())
    legacy_model = _model_dir_targets_path(model_path)
    add(legacy_model)
    if legacy_model is not None:
        add(_legacy_yaml_path(legacy_model))

    for legacy in _legacy_config_dir_paths(config_path):
        add(legacy)
    return ordered


def _parse_targets_blob(data: dict[str, Any]) -> tuple[float | None, list[dict[str, Any]]]:
    default_conf: float | None = None
    raw_conf = data.get("default_confidence", data.get("confidence"))
    if raw_conf is not None:
        try:
            default_conf = float(raw_conf)
        except (TypeError, ValueError):
            pass

    raw_targets = data.get("targets")
    if not isinstance(raw_targets, list):
        return default_conf, []

    patches: list[dict[str, Any]] = []
    for row in raw_targets:
        if not isinstance(row, dict):
            continue
        if row.get("class_id") is None:
            continue
        try:
            class_id = int(row["class_id"])
        except (TypeError, ValueError):
            continue
        patch: dict[str, Any] = {"class_id": class_id}
        if "enabled" in row and row["enabled"] is not None:
            patch["enabled"] = bool(row["enabled"])
        threshold = row.get("threshold", row.get("confidence"))
        if threshold is not None:
            try:
                patch["threshold"] = float(threshold)
            except (TypeError, ValueError):
                pass
        patches.append(patch)
    return default_conf, patches


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YOLO targets file root must be a mapping: {path}")
    return data


def _read_file(path: Path) -> dict[str, Any] | None:
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = _load_yaml(path)
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not read %s: %s", path, exc)
        return None


def read_yolo_settings(
    model_path: Path | str | None,
    *,
    config_path: Path | None = None,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Load default confidence and per-class patches (model dir first, then legacy)."""
    for candidate in _settings_file_candidates(model_path, config_path=config_path):
        if not candidate.is_file():
            continue
        data = _read_file(candidate)
        if data is None:
            continue
        return _parse_targets_blob(data)
    return None, []


def migrate_yolo_settings_to_config(
    model_path: Path | str | None,
    *,
    config_path: Path | None = None,
) -> None:
    from ..platform.model_config_dir import migrate_file_if_missing, yolo_settings_file

    dest = yolo_settings_file(create_dir=True)
    sources: list[Path] = []
    model_file = _model_dir_targets_path(model_path)
    if model_file is not None:
        sources.append(model_file)
    sources.extend(_legacy_config_dir_paths(config_path))
    migrate_file_if_missing(dest, *sources)


def read_yolo_target_patches(
    model_path: Path | str | None,
    *,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return rows suitable for YoloDetectionService.apply_targets."""
    _default_conf, patches = read_yolo_settings(model_path, config_path=config_path)
    return patches


def write_yolo_targets(
    path: Path,
    *,
    model_path: str,
    default_confidence: float,
    targets: list[dict[str, Any]],
    tag_conditions: list[dict[str, Any]] | None = None,
) -> None:
    rows: list[dict[str, Any]] = []
    for row in targets:
        rows.append(
            {
                "class_id": int(row["class_id"]),
                "name": str(row.get("name", "")),
                "enabled": bool(row.get("enabled", True)),
                "threshold": float(row.get("threshold", default_confidence)),
            }
        )
    payload: dict[str, Any] = {
        "model_path": model_path,
        "default_confidence": float(default_confidence),
        "targets": rows,
    }
    if tag_conditions is not None:
        payload["tag_conditions"] = tag_conditions
    elif path.is_file():
        existing = _read_file(path)
        if existing and isinstance(existing.get("tag_conditions"), list):
            payload["tag_conditions"] = existing["tag_conditions"]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    text += "\n"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            prefix=".yolo_targets_",
            suffix=".tmp",
        ) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        tmp_path.replace(path)
    except Exception as exc:  # noqa: BLE001
        _log.warning("Could not write %s: %s", path, exc)
