"""Keep robot task_planner/pydeps off sys.path when using a local venv."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _use_venv_deps() -> bool:
    raw = os.environ.get("OAP_USE_VENV_DEPS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _robot_pydeps_roots() -> list[Path]:
    roots: list[Path] = []
    task_planner = Path(__file__).resolve().parents[3] / "g2-controller" / "task_planner"
    candidate = task_planner / "pydeps"
    if candidate.is_dir():
        roots.append(candidate.resolve())
    env = os.environ.get("OPENVLA_PYDEPS", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_dir():
            roots.append(path)
    return roots


def isolate_venv_from_robot_pydeps() -> None:
    if not _use_venv_deps():
        return

    os.environ.pop("OPENVLA_PYDEPS", None)
    blocked = {str(p) for p in _robot_pydeps_roots()}
    if blocked:
        sys.path[:] = [entry for entry in sys.path if entry not in blocked]

    current = os.environ.get("PYTHONPATH", "")
    if current:
        parts = [
            part
            for part in current.split(os.pathsep)
            if part and part not in blocked
        ]
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)
