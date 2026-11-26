"""Unified ordered tag rule items ↔ legacy class/region/size arrays."""

from __future__ import annotations

import secrets
from typing import Any


def _new_rule_id() -> str:
    return f"rule-{secrets.token_hex(4)}"


def split_rule_items(items: list[dict[str, Any]]) -> tuple[
    list[dict[str, str]],
    list[str],
    list[dict[str, float]],
    list[dict[str, Any]],
]:
    class_contains: list[dict[str, str]] = []
    target_labels: list[str] = []
    seen_targets: set[str] = set()
    regions: list[dict[str, float]] = []
    size_rules: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        if kind == "class":
            text = str(item.get("text") or "").strip()
            if text:
                class_contains.append({"text": text})
        elif kind == "target":
            raw_labels = item.get("labels")
            if isinstance(raw_labels, list):
                for part in raw_labels:
                    label = str(part or "").strip()
                    if label and label not in seen_targets:
                        seen_targets.add(label)
                        target_labels.append(label)
        elif kind == "region":
            try:
                regions.append(
                    {
                        "x1": float(item["x1"]),
                        "y1": float(item["y1"]),
                        "x2": float(item["x2"]),
                        "y2": float(item["y2"]),
                        "min_overlap_pct": float(item.get("min_overlap_pct", 50)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        elif kind == "orientation":
            orient = str(item.get("orientation") or "").strip().lower()
            if orient in {"horizontal", "vertical"}:
                size_rules.append({"orientation": orient})
        elif kind == "size":
            rule: dict[str, Any] = {}
            for key in ("min_width", "max_width", "min_height", "max_height"):
                if item.get(key) is not None and str(item.get(key)).strip() != "":
                    try:
                        rule[key] = float(item[key])
                    except (TypeError, ValueError):
                        pass
            orient = str(item.get("orientation") or "").strip().lower()
            if orient in {"horizontal", "vertical"}:
                rule["orientation"] = orient
            if rule:
                size_rules.append(rule)
    return class_contains, target_labels, regions, size_rules


def legacy_to_rule_items(
    class_contains: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    size_rules: list[dict[str, Any]],
    *,
    target_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    labels: list[str] = []
    if target_labels:
        for part in target_labels:
            label = str(part or "").strip()
            if label:
                labels.append(label)
    if labels:
        items.append({"id": _new_rule_id(), "kind": "target", "labels": labels})
    for part in class_contains:
        text = str(part.get("text") or part.get("substring") or "").strip()
        if text:
            items.append({"id": _new_rule_id(), "kind": "class", "text": text})
    for part in regions:
        if not isinstance(part, dict):
            continue
        try:
            items.append(
                {
                    "id": _new_rule_id(),
                    "kind": "region",
                    "x1": float(part["x1"]),
                    "y1": float(part["y1"]),
                    "x2": float(part["x2"]),
                    "y2": float(part["y2"]),
                    "min_overlap_pct": float(part.get("min_overlap_pct", 50)),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    for part in size_rules:
        if not isinstance(part, dict):
            continue
        dim_keys = ("min_width", "max_width", "min_height", "max_height")
        has_dim = any(part.get(key) is not None for key in dim_keys)
        orient = str(part.get("orientation") or "").strip().lower()
        if orient in {"horizontal", "vertical"}:
            items.append(
                {
                    "id": _new_rule_id(),
                    "kind": "orientation",
                    "orientation": orient,
                }
            )
        if has_dim:
            row: dict[str, Any] = {"id": _new_rule_id(), "kind": "size"}
            for key in dim_keys:
                if part.get(key) is not None:
                    try:
                        row[key] = float(part[key])
                    except (TypeError, ValueError):
                        pass
            items.append(row)
    return items


def clean_rule_items(raw: list[Any]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in {"class", "target", "region", "size", "orientation"}:
            continue
        row_id = str(item.get("id") or f"rule-{index}-{_new_rule_id()}")
        if kind == "class":
            text = str(item.get("text") or "").strip()
            cleaned.append({"id": row_id, "kind": "class", "text": text})
        elif kind == "target":
            labels: list[str] = []
            raw_labels = item.get("labels")
            if isinstance(raw_labels, list):
                for part in raw_labels:
                    label = str(part or "").strip()
                    if label:
                        labels.append(label)
            cleaned.append({"id": row_id, "kind": "target", "labels": labels})
        elif kind == "region":
            try:
                cleaned.append(
                    {
                        "id": row_id,
                        "kind": "region",
                        "x1": float(item["x1"]),
                        "y1": float(item["y1"]),
                        "x2": float(item["x2"]),
                        "y2": float(item["y2"]),
                        "min_overlap_pct": float(item.get("min_overlap_pct", 50)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        elif kind == "orientation":
            orient = str(item.get("orientation") or "horizontal").strip().lower()
            if orient not in {"horizontal", "vertical"}:
                orient = "horizontal"
            cleaned.append({"id": row_id, "kind": "orientation", "orientation": orient})
        elif kind == "size":
            row: dict[str, Any] = {"id": row_id, "kind": "size"}
            for key in ("min_width", "max_width", "min_height", "max_height"):
                if item.get(key) is not None and str(item.get(key)).strip() != "":
                    try:
                        row[key] = float(item[key])
                    except (TypeError, ValueError):
                        pass
            cleaned.append(row)
            orient = str(item.get("orientation") or "").strip().lower()
            if orient in {"horizontal", "vertical"}:
                cleaned.append(
                    {
                        "id": _new_rule_id(),
                        "kind": "orientation",
                        "orientation": orient,
                    }
                )
    return cleaned