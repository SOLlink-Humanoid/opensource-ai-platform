"""Offline OpenVLA: copy custom-code modules into each checkpoint and patch auto_map."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

_log = logging.getLogger(__name__)

CUSTOM_CODE_FILES: tuple[str, ...] = (
    "configuration_prismatic.py",
    "modeling_prismatic.py",
    "processing_prismatic.py",
)

AUTO_MAP_PATCHES: dict[str, dict[str, str]] = {
    "config.json": {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    },
    "preprocessor_config.json": {
        "AutoImageProcessor": "processing_prismatic.PrismaticImageProcessor",
        "AutoProcessor": "processing_prismatic.PrismaticProcessor",
    },
    "tokenizer_config.json": {
        "AutoProcessor": "processing_prismatic.PrismaticProcessor",
    },
}


def _platform_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _vendor_dir() -> Path:
    root = _platform_root()
    for candidate in (
        root.parent / "task_planner" / "scripts" / "vendor" / "openvla",
        root / "models" / "OpenVLA" / "v1" / "model",
    ):
        if candidate.is_dir() and all((candidate / name).is_file() for name in CUSTOM_CODE_FILES):
            return candidate
    return root / "models" / "OpenVLA" / "v1" / "model"


def _custom_code_source_dirs(model_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved in seen or not resolved.is_dir():
            return
        seen.add(resolved)
        candidates.append(resolved)

    add(_vendor_dir())
    cache = Path.home() / ".cache/huggingface/modules/transformers_modules/model"
    add(cache)

    model_root = model_dir.parent.parent
    if model_root.is_dir():
        for entry in sorted(model_root.iterdir()):
            if entry.is_dir():
                add(entry / "model")

    return candidates


def patch_model_configs(model_dir: Path) -> None:
    for filename, auto_map in AUTO_MAP_PATCHES.items():
        path = model_dir / filename
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("auto_map") == auto_map:
            continue
        data["auto_map"] = auto_map
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        _log.info("Patched %s for offline OpenVLA", path.name)


def ensure_custom_code(model_dir: Path) -> None:
    """Copy missing custom-code modules into model_dir and patch HF configs."""
    model_dir = model_dir.expanduser().resolve()
    missing = [name for name in CUSTOM_CODE_FILES if not (model_dir / name).is_file()]
    if missing:
        copied = False
        for source_dir in _custom_code_source_dirs(model_dir):
            if not all((source_dir / name).is_file() for name in missing):
                continue
            for name in missing:
                dest = model_dir / name
                shutil.copy2(source_dir / name, dest)
                _log.info("Installed OpenVLA custom code %s from %s", name, source_dir)
            copied = True
            break
        if not copied:
            still_missing = [
                name for name in CUSTOM_CODE_FILES if not (model_dir / name).is_file()
            ]
            raise FileNotFoundError(
                "OpenVLA custom code files are missing for offline inference: "
                f"{', '.join(still_missing)} in {model_dir}. "
                "Run: cd opensource-ai-platform && python3 scripts/patch_openvla_offline.py"
            )

    patch_model_configs(model_dir)


def ensure_model_root(model_root: Path | None) -> None:
    """Ensure custom code exists for every model version under model_root."""
    if model_root is None or not model_root.is_dir():
        return
    for entry in sorted(model_root.iterdir()):
        if not entry.is_dir():
            continue
        model_dir = entry / "model"
        if (model_dir / "config.json").is_file():
            ensure_custom_code(model_dir)
            continue
        if (entry / "config.json").is_file():
            ensure_custom_code(entry)
