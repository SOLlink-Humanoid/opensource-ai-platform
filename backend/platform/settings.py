"""Platform-specific settings (extends task_planner env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from web.config import Settings

from .inference_devices import default_openvla_device, default_yolo_device


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PlatformSettings:
    static_dir: Path
    openvla_live_instruction: str
    openvla_live_interval_sec: float
    openvla_live_enabled: bool
    lerobot_state_file: Path | None
    lerobot_act_placeholder: bool
    joint_stopped_epsilon: float
    openvla_version: str

    @classmethod
    def from_env(cls) -> PlatformSettings:
        root = Path(__file__).resolve().parents[2]
        lerobot = os.environ.get("LEROBOT_STATE_FILE", "").strip()
        return cls(
            static_dir=root / "frontend",
            openvla_live_instruction=os.environ.get(
                "OAP_OPENVLA_INSTRUCTION", "pick up the box"
            ).strip()
            or "pick up the box",
            openvla_live_interval_sec=_env_float("OAP_OPENVLA_INTERVAL_SEC", 2.0),
            openvla_live_enabled=_env_bool("OAP_OPENVLA_LIVE", False),
            lerobot_state_file=Path(lerobot).expanduser() if lerobot else None,
            lerobot_act_placeholder=_env_bool("OAP_LEROBOT_PLACEHOLDER", True),
            joint_stopped_epsilon=_env_float("OAP_JOINT_STOPPED_EPS", 0.002),
            openvla_version=os.environ.get("OPENVLA_PRELOAD", "").strip(),
        )


def apply_robot_gpu_settings(settings: Settings) -> Settings:
    """CUDA + bundled pydeps (task_planner/start_web.sh behaviour)."""
    settings = replace(settings)
    settings.mock = False
    settings.openvla_device = default_openvla_device("robot")
    settings.yolo_device = default_yolo_device("robot")
    return settings


def dev_profile_paths(settings: Settings, platform: PlatformSettings) -> dict[str, str]:
    yolo = str(settings.yolo_model_path) if settings.yolo_model_path else ""
    root = settings.openvla_root
    version = (
        settings.latest_model_version()
        or (settings.openvla_preload_version or platform.openvla_version or "").strip()
        or "v2"
    )
    model_dir = str(root / version / "model") if root else ""
    lerobot = (
        str(platform.lerobot_state_file)
        if platform.lerobot_state_file
        else "ACT policy (not loaded — placeholder)"
    )
    return {
        "yolo_model": yolo,
        "openvla_root": str(root) if root else "",
        "openvla_model_dir": model_dir,
        "openvla_version": version,
        "lerobot": lerobot,
    }
