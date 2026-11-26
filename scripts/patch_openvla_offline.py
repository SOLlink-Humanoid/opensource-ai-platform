#!/usr/bin/env python3
"""Prepare OpenVLA checkpoints (v1, v2, …) for offline inference."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.config import Settings  # noqa: E402
from web.openvla.openvla_offline import ensure_custom_code, ensure_model_root  # noqa: E402


def main() -> None:
    if len(sys.argv) > 1:
        model_dir = Path(sys.argv[1]).expanduser().resolve()
        ensure_custom_code(model_dir)
        print(f"Ready: {model_dir}")
        return

    settings = Settings.from_env()
    root = settings.openvla_root
    if root is None or not root.is_dir():
        default = ROOT / "models" / "OpenVLA"
        ensure_model_root(default)
        print(f"Ready: {default}")
        return

    ensure_model_root(root)
    print(f"Ready: {root}")


if __name__ == "__main__":
    main()
