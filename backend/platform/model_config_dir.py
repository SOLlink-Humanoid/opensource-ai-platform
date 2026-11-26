"""Central UI/model persistence under ``<platform>/config/``."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

_log = logging.getLogger(__name__)

_YOLO_SETTINGS = "yolo.json"
_OPENVLA_DIR = "openvla"
_OPENVLA_PHASES = "openvla/phases.json"
_OPENVLA_INSTRUCTIONS = "openvla/instruction_history.json"
_LEROBOT_DIR = "lerobot"
_LEROBOT_HAND_MAPPINGS = "lerobot/hand_mappings.json"
_LEROBOT_ACTIVE_POLICY = "lerobot/active_policy.json"
_PLANNER_DIR = "planner"
_TASK_POLICY_RULES = "planner/task_policy_rules.json"

# legacy flat names (pre subfolder layout)
_LEGACY_FLAT_INSTRUCTIONS = "openvla_instruction_history.json"
_LEGACY_FLAT_TASK_RULES = "task_policy_rules.json"


def platform_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_config_dir(*, create: bool = False) -> Path:
    raw = os.environ.get("OAP_MODEL_CONFIG_DIR", "").strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (platform_root() / path).resolve()
    else:
        path = platform_root() / "config"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def yolo_settings_file(*, create_dir: bool = False) -> Path:
    return model_config_dir(create=create_dir) / _YOLO_SETTINGS


def openvla_phases_file(*, create_dir: bool = False) -> Path:
    if create_dir:
        (model_config_dir(create=True) / _OPENVLA_DIR).mkdir(parents=True, exist_ok=True)
    return model_config_dir() / _OPENVLA_PHASES


def openvla_instruction_history_file(*, create_dir: bool = False) -> Path:
    if create_dir:
        (model_config_dir(create=True) / _OPENVLA_DIR).mkdir(parents=True, exist_ok=True)
    return model_config_dir() / _OPENVLA_INSTRUCTIONS


def lerobot_hand_mappings_file(*, create_dir: bool = False) -> Path:
    if create_dir:
        (model_config_dir(create=True) / _LEROBOT_DIR).mkdir(parents=True, exist_ok=True)
    return model_config_dir() / _LEROBOT_HAND_MAPPINGS


def lerobot_active_policy_file(*, create_dir: bool = False) -> Path:
    if create_dir:
        (model_config_dir(create=True) / _LEROBOT_DIR).mkdir(parents=True, exist_ok=True)
    return model_config_dir() / _LEROBOT_ACTIVE_POLICY


def task_policy_rules_file(*, create_dir: bool = False) -> Path:
    if create_dir:
        (model_config_dir(create=True) / _PLANNER_DIR).mkdir(parents=True, exist_ok=True)
    return model_config_dir() / _TASK_POLICY_RULES


def migrate_file_if_missing(dest: Path, *legacy_sources: Path) -> None:
    """Copy the first existing legacy file to dest when dest is missing."""
    if dest.is_file():
        return
    for source in legacy_sources:
        if not source.is_file():
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            _log.info("Migrated model config %s → %s", source, dest)
            return
        except OSError as exc:
            _log.warning("Could not migrate %s to %s: %s", source, dest, exc)


def bootstrap_model_config_layout(
    *,
    phases_legacy: Path | None = None,
    openvla_root: Path | None = None,
    task_rules_legacy: Path | None = None,
) -> None:
    """One-time copy from legacy locations into ``config/`` (no overwrite)."""
    root = platform_root()
    base = model_config_dir(create=True)

    migrate_file_if_missing(
        openvla_phases_file(),
        *(p for p in (
            phases_legacy,
            root / "data" / "phases.json",
            root.parent / "poses" / "phases.json",
        ) if p is not None),
    )

    migrate_file_if_missing(
        openvla_instruction_history_file(),
        base / _LEGACY_FLAT_INSTRUCTIONS,
        Path(openvla_root).expanduser() / "instruction_history.json"
        if openvla_root
        else root / "models" / "OpenVLA" / "instruction_history.json",
    )

    migrate_file_if_missing(
        lerobot_hand_mappings_file(),
        root / "data" / "lerobot_hand_mappings.json",
    )

    migrate_file_if_missing(
        task_policy_rules_file(),
        base / _LEGACY_FLAT_TASK_RULES,
        *(p for p in (task_rules_legacy,) if p is not None),
    )