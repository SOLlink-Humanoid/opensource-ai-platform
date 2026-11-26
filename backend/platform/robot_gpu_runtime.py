"""Robot GPU runtime — mirrors g2-controller/task_planner/start_web.sh pydeps setup."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_log = logging.getLogger(__name__)

_G2_BRIDGE_PYDEPS = Path("/home/agi/app/openvla-script/g2_bridge/server/pydeps")


def _use_venv_deps() -> bool:
    raw = os.environ.get("OAP_USE_VENV_DEPS", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _prepend_env_path(key: str, path: Path) -> None:
    if not path.is_dir():
        return
    path_str = str(path)
    current = os.environ.get(key, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if path_str in parts:
        return
    os.environ[key] = path_str + (os.pathsep + current if current else "")


def bootstrap_robot_gpu_runtime(task_planner_dir: Path | None = None) -> None:
    """Load robot OpenVLA pydeps + NVPL/CUDA libs (same intent as start_web.sh)."""
    if _use_venv_deps():
        _log.debug("Skipping robot pydeps (OAP_USE_VENV_DEPS)")
        return

    from web.openvla.openvla_deps import bootstrap_openvla_runtime

    pydeps = bootstrap_openvla_runtime()
    if pydeps is None:
        _log.warning(
            "Robot GPU pydeps not found; set OPENVLA_PYDEPS or run install_openvla.sh"
        )
        return

    planner = task_planner_dir or Path(__file__).resolve().parents[3] / "g2-controller" / "task_planner"
    extra = os.environ.get("G2_BRIDGE_PYDEPS", "").strip()
    g2_path = Path(extra).expanduser() if extra else _G2_BRIDGE_PYDEPS
    if g2_path.is_dir():
        g2_str = str(g2_path.resolve())
        if g2_str not in sys.path:
            sys.path.insert(0, g2_str)
        _prepend_env_path("PYTHONPATH", g2_path.resolve())

    # Redundant with openvla_deps.bootstrap but documents parity with start_web.sh.
    for lib_dir in (
        pydeps / "nvpl" / "lib",
        planner / "pydeps" / "nvpl" / "lib",
        Path("/usr/lib/aarch64-linux-gnu/libcudss/13"),
        Path("/usr/local/cuda-13.0/targets/sbsa-linux/lib"),
    ):
        _prepend_env_path("LD_LIBRARY_PATH", lib_dir)

    _log.info("Robot GPU runtime ready (pydeps=%s)", pydeps)
