"""python -m backend"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import uvicorn

_ROOT = Path(__file__).resolve().parents[1]


def _resolve_task_planner() -> Path:
    for candidate in (
        _ROOT.parent / "g2-controller" / "task_planner",
        _ROOT.parent / "task_planner",
    ):
        if (candidate / "web").is_dir():
            return candidate
    return _ROOT


_TASK_PLANNER = _resolve_task_planner()

# Config may set OAP_USE_VENV_DEPS; load before web imports mutate sys.path.
from backend.platform.config_file import apply_config_file  # noqa: E402

if __name__ == "__main__":
    apply_config_file(os.environ.get("OAP_CONFIG"))

from backend.platform.robot_gpu_runtime import bootstrap_robot_gpu_runtime  # noqa: E402

if (
    __name__ == "__main__"
    and os.environ.get("OAP_MODE", "").strip().lower() == "robot"
):
    bootstrap_robot_gpu_runtime(_TASK_PLANNER)
from backend.platform.venv_path import isolate_venv_from_robot_pydeps  # noqa: E402

isolate_venv_from_robot_pydeps()

for path in (_ROOT, _TASK_PLANNER):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

isolate_venv_from_robot_pydeps()

from web.config import Settings  # noqa: E402

from backend.app import create_app  # noqa: E402
from backend.platform.settings import PlatformSettings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Source AI Platform")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--config",
        help="YAML config path (default: OAP_CONFIG or ./config.yaml)",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s · %(message)s",
    )

    if args.config:
        os.environ["OAP_CONFIG"] = args.config
    apply_config_file(args.config)

    isolate_venv_from_robot_pydeps()
    if os.environ.get("OAP_MODE", "").strip().lower() == "robot":
        bootstrap_robot_gpu_runtime(_TASK_PLANNER)
    settings = Settings.from_env()
    platform = PlatformSettings.from_env()
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.mock:
        settings.mock = True

    host = settings.host
    port = settings.port
    if not args.host and (env_host := __import__("os").environ.get("OAP_HOST")):
        host = env_host
    if not args.port:
        env_port = __import__("os").environ.get("OAP_PORT")
        if env_port:
            try:
                port = int(env_port)
            except ValueError:
                pass

    uvicorn.run(
        create_app(settings, platform),
        host=host,
        port=port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
