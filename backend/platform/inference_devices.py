"""Default OpenVLA/YOLO devices per platform mode."""

from __future__ import annotations

import os


def default_openvla_device(mode: str) -> str:
    explicit = os.environ.get("OPENVLA_DEVICE", "").strip()
    if explicit:
        return explicit
    if mode == "robot":
        return "cuda"
    return "cpu"


def default_yolo_device(mode: str) -> str | None:
    explicit = os.environ.get("YOLO_DEVICE", "").strip()
    if explicit:
        return explicit
    if mode == "robot":
        return default_openvla_device(mode)
    return None
