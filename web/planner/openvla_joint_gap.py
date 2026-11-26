"""Compare joint maps (e.g. LeRobot decode vs OpenVLA target) by body part."""

from __future__ import annotations

from typing import Any

from shared.joint_names import PART_ERROR_KEYS, joint_part_group


def part_errors_vs_openvla_target(
    predicted_joints: dict[str, float],
    openvla_target: dict[str, float],
    *,
    inferred_joints_by_part: dict[str, list[str]] | None = None,
) -> dict[str, float | None]:
    """Σ|openvla_target − predicted| per part; mirrors OpenVLA phase error semantics."""
    allowed_by_part: dict[str, set[str]] | None = None
    if inferred_joints_by_part:
        allowed_by_part = {}
        for part, names in inferred_joints_by_part.items():
            if not isinstance(names, list):
                continue
            allowed_by_part[str(part)] = {str(n) for n in names if str(n).strip()}

    totals = {key: 0.0 for key in PART_ERROR_KEYS}
    counted = {key: False for key in PART_ERROR_KEYS}
    for name, goal in openvla_target.items():
        if name not in predicted_joints:
            continue
        part = joint_part_group(name)
        if part is None:
            continue
        if allowed_by_part is not None:
            allowed = allowed_by_part.get(part)
            if allowed is not None and name not in allowed:
                continue
        totals[part] += abs(float(goal) - float(predicted_joints[name]))
        counted[part] = True
    if not any(counted.values()):
        return {part: None for part in PART_ERROR_KEYS}
    return {part: (totals[part] if counted[part] else None) for part in PART_ERROR_KEYS}


def error_sum(parts: dict[str, float | None]) -> float | None:
    numeric = [float(v) for v in parts.values() if v is not None]
    return sum(numeric) if numeric else None


def max_part_error(parts: dict[str, float | None]) -> float | None:
    numeric = [float(v) for v in parts.values() if v is not None]
    return max(numeric) if numeric else None