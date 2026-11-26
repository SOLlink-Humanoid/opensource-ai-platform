"""Runtime settings read from the environment (filled by config.yaml)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _optional_path(key: str) -> Path | None:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


_OPENVLA_VERSION_RE = re.compile(r"^v(\d+)$", re.IGNORECASE)


def _openvla_version_sort_key(version_id: str) -> tuple[int, int | str]:
    match = _OPENVLA_VERSION_RE.match(version_id.strip())
    if match:
        return (0, int(match.group(1)))
    return (1, version_id)


@dataclass
class Settings:
    host: str
    port: int
    mock: bool
    phases_file: Path
    poses_file: Path
    openvla_root: Path | None
    openvla_preload_version: str
    openvla_device: str
    openvla_enabled: bool
    openvla_warmup: bool
    default_unnorm_key: str
    yolo_enabled: bool
    yolo_model_path: Path | None
    yolo_device: str | None
    yolo_confidence: float
    yolo_image_size: int
    yolo_interval_sec: float
    action_arm: str
    topics_yaml: Path | None
    mapping_yaml: Path | None
    motion_timeout_sec: float

    @classmethod
    def from_env(cls) -> Settings:
        platform_root = Path(__file__).resolve().parents[1]
        data_dir = platform_root / "data"
        mode = os.environ.get("OAP_MODE", "").strip().lower()
        mock = _env_bool("TASK_PLANNER_MOCK", mode == "mock")
        yolo_raw = os.environ.get("YOLO_MODEL_PATH", "").strip()
        openvla_raw = os.environ.get("OPENVLA_MODEL_ROOT", "").strip()
        host = os.environ.get("OAP_HOST") or os.environ.get("TASK_PLANNER_HOST") or "0.0.0.0"
        poses_file = _optional_path("TASK_PLANNER_POSES_FILE") or (data_dir / "poses.json")
        return cls(
            host=host,
            port=_env_int("OAP_PORT", _env_int("TASK_PLANNER_PORT", 8095)),
            mock=mock,
            phases_file=_optional_path("TASK_PLANNER_PHASES_FILE") or (data_dir / "phases.json"),
            poses_file=poses_file,
            openvla_root=Path(openvla_raw).expanduser() if openvla_raw else None,
            openvla_preload_version=os.environ.get("OPENVLA_PRELOAD", "").strip(),
            openvla_device=os.environ.get("OPENVLA_DEVICE", "cuda").strip() or "cuda",
            openvla_enabled=_env_bool("OPENVLA_ENABLED", True),
            openvla_warmup=_env_bool("OPENVLA_WARMUP", False),
            default_unnorm_key=os.environ.get("OPENVLA_UNNORM_KEY", "bridge_orig").strip()
            or "bridge_orig",
            yolo_enabled=_env_bool("YOLO_ENABLED", True),
            yolo_model_path=Path(yolo_raw).expanduser() if yolo_raw else None,
            yolo_device=os.environ.get("YOLO_DEVICE", "").strip() or None,
            yolo_confidence=_env_float("YOLO_CONFIDENCE", 0.7),
            yolo_image_size=_env_int("YOLO_IMAGE_SIZE", 640),
            yolo_interval_sec=_env_float("YOLO_INTERVAL_SEC", 0.2),
            action_arm=os.environ.get("TASK_PLANNER_ACTION_ARM", "left_arm").strip() or "left_arm",
            topics_yaml=_optional_path("TASK_PLANNER_TOPICS_YAML"),
            mapping_yaml=_optional_path("TASK_PLANNER_MAPPING_YAML"),
            motion_timeout_sec=_env_float("TASK_PLANNER_MOTION_TIMEOUT", 10.0),
        )

    def list_model_versions(self) -> list[dict[str, str]]:
        root = self.openvla_root
        if root is None or not root.is_dir():
            return []
        found: list[dict[str, str]] = []
        for child in (p for p in root.iterdir() if p.is_dir()):
            model_dir = child / "model"
            if (model_dir / "config.json").is_file():
                found.append({"id": child.name, "path": str(model_dir)})
        found.sort(key=lambda item: _openvla_version_sort_key(item["id"]))
        return found

    def latest_model_version(self) -> str | None:
        """Newest checkpoint under openvla_root (e.g. v2 over v1), ignoring config pin."""
        versions = self.list_model_versions()
        if not versions:
            return None
        return versions[-1]["id"]

    def model_dir(self, version: str) -> Path | None:
        for item in self.list_model_versions():
            if item["id"] == version:
                return Path(item["path"])
        return None
