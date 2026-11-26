"""Locate optional OpenVLA pydeps. The Thor venv is used when none are installed."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _has_torch(path: Path) -> bool:
    return (path / "torch").is_dir()


def _pydeps_dir() -> Path | None:
    raw = os.environ.get("OPENVLA_PYDEPS", "").strip()
    if raw:
        path = Path(raw).expanduser()
        if _has_torch(path):
            return path

    platform_root = Path(__file__).resolve().parents[2]
    marker_candidates = (
        platform_root.parent / "task_planner" / ".openvla_pydeps_path",
        platform_root.parent / "g2-controller" / "task_planner" / ".openvla_pydeps_path",
    )
    for marker in marker_candidates:
        if not marker.is_file():
            continue
        saved = Path(marker.read_text(encoding="utf-8").strip()).expanduser()
        if _has_torch(saved):
            return saved

    for candidate in (
        Path("/data/agi/openvla_pydeps"),
        platform_root.parent / "task_planner" / "pydeps",
        platform_root.parent / "g2-controller" / "task_planner" / "pydeps",
    ):
        if _has_torch(candidate):
            return candidate
    return None


def bootstrap_openvla_runtime() -> Path | None:
    pydeps = _pydeps_dir()
    if pydeps is None:
        return None
    path_str = str(pydeps.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    os.environ.setdefault("OPENVLA_PYDEPS", path_str)
    return pydeps


def openvla_deps_error() -> str | None:
    missing = [
        name
        for name in ("torch", "transformers", "PIL")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        return (
            "OpenVLA dependencies missing: "
            + ", ".join(missing)
            + ". Run start.sh after .venv-thor is installed."
        )
    return None
