"""Load config.yaml and map into process environment (env vars override file)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

VALID_MODES = frozenset({"robot", "mock"})


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config.yaml"


def resolve_config_path(explicit: str | Path | None = None) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    env = os.environ.get("OAP_CONFIG", "").strip()
    if env:
        path = Path(env).expanduser()
        return path if path.is_file() else None
    default = default_config_path()
    return default if default.is_file() else None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "config.yaml requires PyYAML (pip install pyyaml or use task_planner venv)"
        ) from exc
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _setdefault(key: str, value: str | None) -> None:
    if value is None or value == "":
        return
    os.environ.setdefault(key, str(value))


def _resolve_path(base: Path, raw: str | None) -> str | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def apply_config_file(path: Path | None = None) -> tuple[str | None, Path | None]:
    """Apply YAML settings with setdefault. Returns (mode, config_path)."""
    cfg_path = resolve_config_path(path)
    if cfg_path is None:
        return None, None

    try:
        data = _load_yaml(cfg_path)
    except RuntimeError as exc:
        _log.warning("%s", exc)
        return None, cfg_path
    config_dir = cfg_path.parent
    paths = data.get("paths") or {}
    if not isinstance(paths, dict):
        paths = {}

    vla_root_raw = paths.get("vla_root") or ".."
    vla_root = Path(vla_root_raw).expanduser()
    if not vla_root.is_absolute():
        vla_root = (config_dir / vla_root).resolve()
    _setdefault("VLA_ROOT", str(vla_root))

    yolo_rel = paths.get("yolo_model")
    openvla_rel = paths.get("openvla_root")
    _setdefault("YOLO_MODEL_PATH", _resolve_path(vla_root, yolo_rel))
    _setdefault("OPENVLA_MODEL_ROOT", _resolve_path(vla_root, openvla_rel))
    _setdefault("OPENVLA_PRELOAD", paths.get("openvla_version"))
    poses_resolved = _resolve_path(config_dir, paths.get("poses_file"))
    _setdefault("TASK_PLANNER_POSES_FILE", poses_resolved)
    _setdefault(
        "TASK_PLANNER_PHASES_FILE",
        _resolve_path(
            config_dir,
            paths.get("phases_file") or "config/openvla/phases.json",
        ),
    )
    _setdefault(
        "OAP_HAND_POSES_DIR",
        _resolve_path(
            config_dir,
            paths.get("hand_poses_dir") or "../poses/hand_poses",
        ),
    )

    mode = str(data.get("mode", "robot")).strip().lower()
    if mode not in VALID_MODES:
        raise ValueError(f"config mode must be one of {sorted(VALID_MODES)}: got {mode!r}")
    _setdefault("OAP_MODE", mode)
    if mode == "mock":
        _setdefault("TASK_PLANNER_MOCK", "1")

    server = data.get("server") or {}
    if isinstance(server, dict):
        _setdefault("OAP_HOST", server.get("host"))
        port = server.get("port")
        if port is not None:
            _setdefault("OAP_PORT", str(port))
            _setdefault("TASK_PLANNER_PORT", str(port))

    openvla = data.get("openvla") or {}
    if not isinstance(openvla, dict):
        openvla = {}
    _setdefault("OAP_OPENVLA_INSTRUCTION", openvla.get("instruction"))
    interval = openvla.get("interval_sec")
    if interval is not None:
        _setdefault("OAP_OPENVLA_INTERVAL_SEC", str(interval))
    if openvla.get("warmup") is not None:
        _setdefault("OPENVLA_WARMUP", "1" if openvla.get("warmup") else "0")
    if openvla.get("live") is not None:
        _setdefault("OAP_OPENVLA_LIVE", "1" if openvla.get("live") else "0")
    if openvla.get("enabled") is not None:
        _setdefault("OPENVLA_ENABLED", "1" if openvla.get("enabled") else "0")
    if mode == "robot":
        _setdefault("OPENVLA_DEVICE", openvla.get("device") or "cuda")
    else:
        _setdefault("OPENVLA_DEVICE", openvla.get("device") or "cpu")

    robot = data.get("robot") or {}
    if isinstance(robot, dict):
        _setdefault("TASK_PLANNER_TOPICS_YAML", robot.get("topics_yaml"))
        _setdefault("TASK_PLANNER_MAPPING_YAML", robot.get("mapping_yaml"))
        _setdefault("TASK_PLANNER_ACTION_ARM", robot.get("action_arm"))

    lerobot = data.get("lerobot") or {}
    if isinstance(lerobot, dict):
        _setdefault("LEROBOT_STATE_FILE", lerobot.get("state_file"))
        if lerobot.get("act_placeholder") is not None:
            _setdefault(
                "OAP_LEROBOT_PLACEHOLDER",
                "1" if lerobot.get("act_placeholder") else "0",
            )
        _setdefault(
            "OAP_LEROBOT_ROOT",
            _resolve_path(config_dir, lerobot.get("runtime_root", "lerobot_runtime")),
        )
        _setdefault(
            "OAP_LEROBOT_MODELS_DIR",
            _resolve_path(config_dir, lerobot.get("models_dir", "models/LeRobot")),
        )
        template = lerobot.get("policy_template")
        if template:
            _setdefault("OAP_LEROBOT_POLICY_TEMPLATE", str(template))

    yolo = data.get("yolo") or {}
    if not isinstance(yolo, dict):
        yolo = {}
    if yolo.get("enabled") is not None:
        _setdefault("YOLO_ENABLED", "1" if yolo.get("enabled") else "0")
    if yolo.get("confidence") is not None:
        _setdefault("YOLO_CONFIDENCE", str(yolo.get("confidence")))
    if yolo.get("image_size") is not None:
        _setdefault("YOLO_IMAGE_SIZE", str(yolo.get("image_size")))
    interval = yolo.get("interval_sec")
    if interval is not None:
        _setdefault("YOLO_INTERVAL_SEC", str(interval))
    if yolo.get("device"):
        _setdefault("YOLO_DEVICE", yolo.get("device"))
    elif mode == "robot":
        _setdefault("YOLO_DEVICE", "cuda")

    camera_stream = data.get("camera_stream") or {}
    if not isinstance(camera_stream, dict):
        camera_stream = {}
    if camera_stream.get("auto") is not None:
        _setdefault(
            "OAP_CAMERA_STREAM_AUTO",
            "1" if camera_stream.get("auto") else "0",
        )
    if camera_stream.get("base_url"):
        _setdefault("OAP_CAMERA_STREAM_URL", str(camera_stream["base_url"]))
    if camera_stream.get("force_gdk") is not None:
        _setdefault(
            "OAP_CAMERA_STREAM_FORCE_GDK",
            "1" if camera_stream.get("force_gdk") else "0",
        )

    studio = data.get("studio") or {}
    if not isinstance(studio, dict):
        studio = {}
    if studio.get("raw_preview_when_peers_up") is not None:
        _setdefault(
            "OAP_STUDIO_RAW_PREVIEW",
            "1" if studio.get("raw_preview_when_peers_up") else "0",
        )
    if studio.get("data_logger_url"):
        _setdefault("OAP_DATA_LOGGER_URL", str(studio["data_logger_url"]))

    if isinstance(robot, dict):
        if robot.get("openvla_device"):
            os.environ["OPENVLA_DEVICE"] = str(robot.get("openvla_device"))
        if robot.get("yolo_device"):
            os.environ["YOLO_DEVICE"] = str(robot.get("yolo_device"))

    _log.info("Loaded config %s (mode=%s)", cfg_path, mode)
    return mode, cfg_path
