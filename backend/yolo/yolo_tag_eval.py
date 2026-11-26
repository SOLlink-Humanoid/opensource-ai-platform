"""Evaluate YOLO tag conditions against detections."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _bbox_area(xyxy: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _matches_class_item(label: str, item: Mapping[str, Any]) -> bool:
    text = str(item.get("text") or "").strip()
    if not text:
        return True
    return text.lower() in str(label or "").lower()


def _matches_target_item(label: str, item: Mapping[str, Any]) -> bool:
    raw = item.get("labels")
    labels: list[str] = []
    if isinstance(raw, list):
        for part in raw:
            name = str(part or "").strip()
            if name:
                labels.append(name)
    if not labels:
        return True
    return str(label or "").strip() in labels


def _matches_region_item(xyxy: Sequence[float], item: Mapping[str, Any]) -> bool:
    try:
        rx1 = float(item["x1"])
        ry1 = float(item["y1"])
        rx2 = float(item["x2"])
        ry2 = float(item["y2"])
        pct = float(item.get("min_overlap_pct", 50))
    except (KeyError, TypeError, ValueError):
        return True
    area = _bbox_area(xyxy)
    if area <= 0:
        return False
    if rx2 < rx1:
        rx1, rx2 = rx2, rx1
    if ry2 < ry1:
        ry1, ry2 = ry2, ry1
    pct = min(100.0, max(0.0, pct))
    overlap = _intersection_area(xyxy, (rx1, ry1, rx2, ry2))
    return overlap / area * 100.0 + 1e-9 >= pct


def _matches_size_item(xyxy: Sequence[float], item: Mapping[str, Any]) -> bool:
    x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width <= 0 or height <= 0:
        return False
    has_constraint = False
    ok = True
    for key in ("min_width", "max_width", "min_height", "max_height"):
        raw = item.get(key)
        if raw is None or str(raw).strip() == "":
            continue
        has_constraint = True
        try:
            bound = float(raw)
        except (TypeError, ValueError):
            continue
        if key == "min_width" and width < bound:
            ok = False
        if key == "max_width" and width > bound:
            ok = False
        if key == "min_height" and height < bound:
            ok = False
        if key == "max_height" and height > bound:
            ok = False
    orient = str(item.get("orientation") or "").strip().lower()
    if orient in {"horizontal", "vertical"}:
        has_constraint = True
        if not _matches_orientation_item(xyxy, item):
            ok = False
    if not has_constraint:
        return True
    return ok


def _matches_orientation_item(xyxy: Sequence[float], item: Mapping[str, Any]) -> bool:
    x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width <= 0 or height <= 0:
        return False
    orient = str(item.get("orientation") or "").strip().lower()
    if orient not in {"horizontal", "vertical"}:
        return True
    if orient == "horizontal":
        return width > height
    return height > width


def _rule_item_matches(
    detection: Mapping[str, Any],
    item: Mapping[str, Any],
) -> bool:
    kind = str(item.get("kind") or "").strip().lower()
    label = str(detection.get("label") or "")
    xyxy = detection.get("xyxy")
    if kind == "class":
        return _matches_class_item(label, item)
    if kind == "target":
        return _matches_target_item(label, item)
    if kind in {"region", "size", "orientation"}:
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) < 4:
            return False
        if kind == "region":
            return _matches_region_item(xyxy, item)
        if kind == "orientation":
            return _matches_orientation_item(xyxy, item)
        return _matches_size_item(xyxy, item)
    return True


def _legacy_rule_items(condition: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    class_rows = condition.get("class_contains")
    if isinstance(class_rows, list):
        for row in class_rows:
            if isinstance(row, dict):
                text = str(row.get("text") or row.get("substring") or "").strip()
            else:
                text = str(row or "").strip()
            if text:
                items.append({"kind": "class", "text": text})
    raw_targets = condition.get("target_labels")
    target_labels: list[str] = []
    if isinstance(raw_targets, list):
        for part in raw_targets:
            name = str(part or "").strip()
            if name:
                target_labels.append(name)
    if target_labels:
        items.append({"kind": "target", "labels": target_labels})
    regions = condition.get("regions")
    if isinstance(regions, list):
        for region in regions:
            if isinstance(region, dict):
                items.append({"kind": "region", **region})
    size_rules = condition.get("size_rules")
    if isinstance(size_rules, list):
        for rule in size_rules:
            if isinstance(rule, dict):
                items.append({"kind": "size", **rule})
    return items


def _effective_rule_items(condition: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = condition.get("rule_items")
    if isinstance(raw, list) and raw:
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                out.append(dict(item))
        return out
    return _legacy_rule_items(condition)


def detection_matches_condition(
    detection: Mapping[str, Any],
    condition: Mapping[str, Any],
) -> bool:
    xyxy = detection.get("xyxy")
    if not isinstance(xyxy, (list, tuple)) or len(xyxy) < 4:
        return False
    rule_items = _effective_rule_items(condition)
    for item in rule_items:
        if not _rule_item_matches(detection, item):
            return False
    return True


def evaluate_yolo_tag_conditions(
    detections: Sequence[Mapping[str, Any]],
    conditions: Sequence[Mapping[str, Any]],
) -> list[str]:
    active: list[str] = []
    seen: set[str] = set()
    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        tag = str(cond.get("tag_name") or "").strip()
        if not tag or tag in seen:
            continue
        for det in detections:
            if detection_matches_condition(det, cond):
                active.append(tag)
                seen.add(tag)
                break
    return active