"""Load/save YOLO tag condition cards from yolo_targets.json."""

from __future__ import annotations

import secrets
from typing import Any

from .yolo_tag_rule_items import (
    clean_rule_items,
    legacy_to_rule_items,
    split_rule_items,
)
from .yolo_targets_store import (
    _parse_targets_blob,
    _read_file,
    _settings_file_candidates,
    read_yolo_settings,
    write_yolo_targets,
)


def _clean_tag_conditions(raw: list[Any]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        tag_name = str(item.get("tag_name") or "").strip()
        row_id = str(item.get("id") or f"tag-{index}-{secrets.token_hex(3)}")

        raw_items = item.get("rule_items")
        rule_items: list[dict[str, Any]] = []
        if isinstance(raw_items, list) and raw_items:
            rule_items = clean_rule_items(raw_items)

        class_contains: list[dict[str, str]] = []
        raw_class = item.get("class_contains")
        if isinstance(raw_class, list):
            for part in raw_class:
                if isinstance(part, dict):
                    text = str(part.get("text") or part.get("substring") or "").strip()
                else:
                    text = str(part or "").strip()
                if text:
                    class_contains.append({"text": text})

        regions: list[dict[str, float]] = []
        raw_regions = item.get("regions")
        if isinstance(raw_regions, list):
            for part in raw_regions:
                if not isinstance(part, dict):
                    continue
                try:
                    regions.append(
                        {
                            "x1": float(part["x1"]),
                            "y1": float(part["y1"]),
                            "x2": float(part["x2"]),
                            "y2": float(part["y2"]),
                            "min_overlap_pct": float(part.get("min_overlap_pct", 50)),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue

        size_rules: list[dict[str, Any]] = []
        raw_sizes = item.get("size_rules")
        if isinstance(raw_sizes, list):
            for part in raw_sizes:
                if not isinstance(part, dict):
                    continue
                rule: dict[str, Any] = {}
                for key in ("min_width", "max_width", "min_height", "max_height"):
                    if part.get(key) is not None and str(part.get(key)).strip() != "":
                        try:
                            rule[key] = float(part[key])
                        except (TypeError, ValueError):
                            pass
                orient = str(part.get("orientation") or "").strip().lower()
                if orient in {"horizontal", "vertical"}:
                    rule["orientation"] = orient
                if rule:
                    size_rules.append(rule)

        target_labels: list[str] = []
        if rule_items:
            class_contains, target_labels, regions, size_rules = split_rule_items(rule_items)
        elif class_contains or target_labels or regions or size_rules:
            rule_items = legacy_to_rule_items(
                class_contains,
                regions,
                size_rules,
                target_labels=target_labels,
            )

        cleaned.append(
            {
                "id": row_id,
                "enabled": bool(item.get("enabled", True)),
                "tag_name": tag_name,
                "rule_items": rule_items,
                "class_contains": class_contains,
                "target_labels": target_labels,
                "regions": regions,
                "size_rules": size_rules,
            }
        )
    return cleaned


def read_yolo_tag_conditions(
    model_path,
    *,
    config_path=None,
) -> list[dict[str, Any]]:
    """Prefer the first settings file with non-empty ``tag_conditions``."""
    fallback: list[dict[str, Any]] = []
    for candidate in _settings_file_candidates(model_path, config_path=config_path):
        if not candidate.is_file():
            continue
        data = _read_file(candidate)
        if data is None:
            continue
        raw = data.get("tag_conditions")
        if not isinstance(raw, list):
            continue
        cleaned = _clean_tag_conditions(raw)
        if cleaned:
            return cleaned
        if not fallback:
            fallback = cleaned
    return fallback


def read_full_yolo_settings(model_path, *, config_path=None):
    default_conf, patches = read_yolo_settings(model_path, config_path=config_path)
    tags = read_yolo_tag_conditions(model_path, config_path=config_path)
    return default_conf, patches, tags