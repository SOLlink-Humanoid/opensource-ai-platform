#!/usr/bin/env python
"""Minimal dataset helpers for OAP ACT deployment (inference only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.utils.constants import OBS_STR

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items: list[tuple[str, Any]] = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    out: dict = {}
    for key, value in d.items():
        parts = key.split(sep)
        cursor = out
        for part in parts[:-1]:
            if part not in cursor:
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return out


def write_json(data: dict, fpath: Path) -> None:
    fpath = Path(fpath)
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)


def build_dataset_frame(
    ds_features: dict[str, dict],
    values: dict[str, Any],
    prefix: str,
) -> dict[str, np.ndarray]:
    frame: dict[str, np.ndarray] = {}
    for key, feature in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        if feature["dtype"] == "float32" and len(feature["shape"]) == 1:
            frame[key] = np.array(
                [values[name] for name in feature["names"]],
                dtype=np.float32,
            )
        elif feature["dtype"] in ("image", "video"):
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]
    return frame
