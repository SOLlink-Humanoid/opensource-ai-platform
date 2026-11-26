"""Expose ``lerobot_runtime/vendor`` as the importable ``lerobot`` package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def setup_vendored_lerobot() -> Path:
    """Register vendored LeRobot under ``lerobot`` without a ``vendor/lerobot`` nesting."""
    runtime_root = Path(__file__).resolve().parents[2]
    vendor_root = runtime_root / "vendor"
    init_py = vendor_root / "__init__.py"
    if not init_py.is_file():
        raise ImportError(f"vendored lerobot missing: {init_py}")

    vendor_str = str(vendor_root.resolve())
    existing = sys.modules.get("lerobot")
    if existing is not None:
        existing_file = getattr(existing, "__file__", "") or ""
        if existing_file.startswith(vendor_str):
            return vendor_root

    spec = importlib.util.spec_from_file_location(
        "lerobot",
        init_py,
        submodule_search_locations=[vendor_str],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vendored lerobot from {init_py}")

    module = importlib.util.module_from_spec(spec)
    module.__path__ = [vendor_str]
    module.__package__ = "lerobot"
    sys.modules["lerobot"] = module
    spec.loader.exec_module(module)
    return vendor_root
