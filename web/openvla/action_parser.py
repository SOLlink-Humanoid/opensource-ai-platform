"""Map OpenVLA outputs onto G2 joint targets and phase rows."""

from __future__ import annotations

from typing import Any

from shared.action_layout import load_action_layout
from shared.joint_names import (
    ALL_MODEL_JOINTS,
    BODY_ARM_HAND_JOINTS,
    LEFT_ARM_JOINTS,
    PART_ERROR_KEYS,
    RIGHT_ARM_JOINTS,
    joint_part_group,
)

from web.planner.pose_catalog import PoseCatalog

# OpenVLA regression stores phase label / 10 in the last action dimension.
PHASE_TAIL_SCALE = 10.0


def scale_openvla_phase_tail(raw: float) -> float:
    """Model tail only — joints and UI phase strings must not use this."""
    return float(raw) * PHASE_TAIL_SCALE


def sanitize_g2_phase_text(phase_text: str | None) -> str | None:
    if phase_text is None:
        return None
    text = str(phase_text).strip()
    return text or None


def phase_text_from_action_tail(
    numbers: list[float],
    *,
    max_phase: int = 64,
) -> str | None:
    """Regression heads append phase as the last scalar in the action vector."""
    if not numbers:
        return None
    tail = scale_openvla_phase_tail(float(numbers[-1]))
    if abs(tail) >= max_phase:
        return None
    return str(int(round(tail)))


def _phase_index_from_text(phase_text: str | None, phase_count: int) -> int | None:
    text = sanitize_g2_phase_text(phase_text)
    if text is None:
        return None
    try:
        return _phase_index_from_integer_label(float(text), phase_count)
    except (TypeError, ValueError):
        return None


def _phase_index_from_integer_label(label: float, phase_count: int) -> int | None:
    """Map human / decoded phase number (1, 2, 3, …) to phases[] index."""
    rounded = int(round(float(label)))
    if abs(label) >= 64:
        return None
    if 1 <= rounded <= phase_count:
        return rounded - 1
    if 0 <= rounded < phase_count:
        return rounded
    return None


def _phase_index_from_model_scalar(raw_tail: float, phase_count: int) -> int | None:
    """Map raw model regression tail (label/10) to phases[] index."""
    scaled = scale_openvla_phase_tail(raw_tail)
    return _phase_index_from_integer_label(scaled, phase_count)


def _phase_label_from_integer_label(label: float) -> str | None:
    rounded = int(round(float(label)))
    if abs(label) >= 64:
        return None
    return str(rounded)


def _phase_label_from_model_scalar(raw_tail: float) -> str | None:
    return _phase_label_from_integer_label(scale_openvla_phase_tail(raw_tail))


def inferred_joints_by_part(
    joint_values: list[float | str],
    *,
    action_arm: str,
) -> dict[str, list[str]]:
    names = inferred_joint_names_from_values(joint_values, action_arm=action_arm)
    buckets: dict[str, list[str]] = {key: [] for key in PART_ERROR_KEYS}
    for name in sorted(names):
        part = joint_part_group(name)
        if part:
            buckets[part].append(name)
    return {part: joints for part, joints in buckets.items() if joints}


def inferred_joint_names_from_values(
    joint_values: list[float | str],
    *,
    action_arm: str,
) -> set[str]:
    """Joint names actually emitted by the model (not current-state fill-in)."""
    numbers: list[float] = []
    for value in joint_values:
        if isinstance(value, str):
            continue
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    layout = load_action_layout()
    n_body_arm_hand = len(BODY_ARM_HAND_JOINTS)
    n_full_body = len(ALL_MODEL_JOINTS)
    if layout.matches_openvla(len(numbers)):
        return set(layout.output_joint_names())
    if len(numbers) in (n_body_arm_hand, n_body_arm_hand + 1):
        return set(BODY_ARM_HAND_JOINTS)
    if len(numbers) >= n_full_body:
        return set(ALL_MODEL_JOINTS)
    if len(numbers) == 7:
        arm = RIGHT_ARM_JOINTS if action_arm == "right_arm" else LEFT_ARM_JOINTS
        return set(arm)
    return set()


def hand_joints_for_scalar(rows: Any, scalar: float) -> list[float] | None:
    """Nearest key in a hand-pose side list (0.1, 0.2, …)."""
    if not isinstance(rows, list):
        return None
    best: list[Any] | None = None
    best_dist: float | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        joints = row.get("joints")
        if not isinstance(joints, list) or not joints:
            continue
        try:
            key = float(row["key"])
        except (KeyError, TypeError, ValueError):
            continue
        dist = abs(key - float(scalar))
        if best is None or best_dist is None or dist < best_dist - 1e-9:
            best = joints
            best_dist = dist
    if best is None:
        return None
    try:
        return [float(value) for value in best]
    except (TypeError, ValueError):
        return None


def apply_hand_key_poses(
    target_map: dict[str, float],
    predicted_action: list[float | str] | None,
    hand_table: dict[str, Any] | None,
) -> dict[str, float]:
    """Replace open/close hand presets with joints from the profile hand-pose file."""
    if not isinstance(hand_table, dict) or not predicted_action:
        return target_map
    numbers: list[float] = []
    for value in predicted_action:
        if isinstance(value, str):
            continue
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    layout = load_action_layout()
    scalars = layout.gripper_scalars(numbers)
    if not scalars:
        return target_map
    updated = dict(target_map)
    changed = False
    for channel in layout.channels:
        if channel["kind"] != "gripper_scalar":
            continue
        scalar = scalars.get(channel["id"])
        if scalar is None:
            continue
        side = str(channel.get("side") or "")
        joints = hand_joints_for_scalar(hand_table.get(side), scalar)
        names = channel["joints"]
        if not joints or len(joints) != len(names):
            continue
        for name, value in zip(names, joints):
            updated[name] = value
        changed = True
    return updated if changed else target_map


def _legacy_arm_error(
    errors: dict[str, float | None],
) -> float | None:
    left = errors.get("left_arm")
    right = errors.get("right_arm")
    if left is None and right is None:
        return None
    return float(left or 0.0) + float(right or 0.0)


def parse_openvla_action(
    joint_values: list[float | str],
    current: dict[str, float],
    *,
    phase_text: str | None,
    phases: list[dict[str, Any]],
    action_arm: str,
) -> tuple[dict[str, float], int | None, str | None]:
    numbers: list[float] = []
    extra_text: str | None = None
    for value in joint_values:
        if isinstance(value, str):
            extra_text = value
            continue
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue

    target = {str(name): float(value) for name, value in current.items()}
    phase_from_tail: str | None = None
    phase_value: float | None = None
    layout = load_action_layout()
    n_body_arm_hand = len(BODY_ARM_HAND_JOINTS)
    n_full_body = len(ALL_MODEL_JOINTS)

    if layout.matches_openvla(len(numbers)):
        decoded, phase_value = layout.decode(numbers)
        target.update(decoded)
        if phase_value is not None:
            phase_from_tail = phase_text_from_action_tail([phase_value])
    elif len(numbers) in (n_body_arm_hand, n_body_arm_hand + 1):
        # 39-DOF (body 5 + left_arm 7 + right_arm 7 + left_hand 10 + right_hand 10) + optional phase
        for name, value in zip(BODY_ARM_HAND_JOINTS, numbers[:n_body_arm_hand]):
            target[name] = value
        if len(numbers) == n_body_arm_hand + 1:
            phase_from_tail = phase_text_from_action_tail(numbers)
    elif len(numbers) >= n_full_body:
        for name, value in zip(ALL_MODEL_JOINTS, numbers[:n_full_body]):
            target[name] = value
        if len(numbers) > n_full_body:
            phase_from_tail = phase_text_from_action_tail(numbers)
    elif len(numbers) == 7:
        arm = RIGHT_ARM_JOINTS if action_arm == "right_arm" else LEFT_ARM_JOINTS
        for name, value in zip(arm, numbers):
            target[name] = value

    phase_label: str | None = None
    phase_index: int | None = None
    if phase_value is not None:
        phase_label = _phase_label_from_model_scalar(phase_value)
        phase_index = _phase_index_from_model_scalar(phase_value, len(phases))
    elif phase_from_tail is not None and numbers:
        tail = float(numbers[-1])
        phase_label = _phase_label_from_model_scalar(tail)
        phase_index = _phase_index_from_model_scalar(tail, len(phases))
    else:
        text = sanitize_g2_phase_text(phase_text) or sanitize_g2_phase_text(extra_text)
        phase_label = text
        phase_index = _phase_index_from_text(text, len(phases))
        if phase_index is None and text is not None:
            try:
                phase_index = _phase_index_from_integer_label(
                    float(text), len(phases)
                )
            except (TypeError, ValueError):
                pass
    return target, phase_index, phase_label


def resolve_predicted_phase(
    phase_index: int | None,
    phases: list[dict[str, Any]],
    *,
    phase_text: str | None = None,
) -> dict[str, Any] | None:
    text = sanitize_g2_phase_text(phase_text)
    index = phase_index
    if index is None:
        index = _phase_index_from_text(text, len(phases))
    if index is None and not text:
        return None
    phase = phases[index] if index is not None and 0 <= index < len(phases) else None
    display_number = None
    if phase is not None:
        try:
            display_number = round(float(phase.get("number", (index or 0) + 1)), 1)
        except (TypeError, ValueError):
            display_number = (index + 1) if index is not None else None
    elif index is not None:
        display_number = index + 1
    return {
        "index": index,
        "phase_number": display_number,
        "phase_id": None if phase is None else phase.get("id"),
        "description": "" if phase is None else str(phase.get("description") or ""),
        "pose_name": "" if phase is None else str(phase.get("pose_name") or ""),
        "raw_text": text,
    }


def _match_from_phase(
    phase: dict[str, Any],
    index: int,
    pose_catalog: PoseCatalog,
    target_map: dict[str, float],
    *,
    inferred_joint_names: set[str] | None = None,
) -> dict[str, Any] | None:
    pose_name = str(phase.get("pose_name") or "")
    part_errors = pose_catalog.part_joint_errors_for_phase(
        phase,
        target_map,
        inferred_joint_names=inferred_joint_names,
    )
    if part_errors is None and not pose_name:
        return None
    errors = part_errors or {key: None for key in PART_ERROR_KEYS}
    numeric = [float(value) for value in errors.values() if value is not None]
    try:
        display_num = round(float(phase.get("number", index + 1)), 1)
    except (TypeError, ValueError):
        display_num = index + 1
    return {
        "phase_number": display_num,
        "phase_id": phase.get("id"),
        "description": str(phase.get("description") or ""),
        "pose_name": pose_name,
        "error_sum": sum(numeric) if numeric else None,
        "error_head": errors.get("head"),
        "error_body": errors.get("body"),
        "error_left_arm": errors.get("left_arm"),
        "error_right_arm": errors.get("right_arm"),
        "error_left_hand": errors.get("left_hand"),
        "error_right_hand": errors.get("right_hand"),
        "error_arms": _legacy_arm_error(errors),
    }


def compute_phase_pose_match(
    target_map: dict[str, float],
    phase_index: int | None,
    phases: list[dict[str, Any]],
    pose_catalog: PoseCatalog,
    *,
    predicted_action: list[float | str] | None = None,
    action_arm: str = "left_arm",
    hand_table: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if phase_index is None or phase_index < 0 or phase_index >= len(phases):
        return None
    inferred: set[str] | None = None
    if predicted_action:
        inferred = inferred_joint_names_from_values(predicted_action, action_arm=action_arm)
    joints = apply_hand_key_poses(target_map, predicted_action, hand_table)
    return _match_from_phase(
        phases[phase_index],
        phase_index,
        pose_catalog,
        joints,
        inferred_joint_names=inferred,
    )


def find_closest_phase_pose(
    target_map: dict[str, float],
    phases: list[dict[str, Any]],
    pose_catalog: PoseCatalog,
    *,
    predicted_action: list[float | str] | None = None,
    action_arm: str = "left_arm",
    hand_table: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    inferred: set[str] | None = None
    if predicted_action:
        inferred = inferred_joint_names_from_values(predicted_action, action_arm=action_arm)
    joints = apply_hand_key_poses(target_map, predicted_action, hand_table)
    best: dict[str, Any] | None = None
    best_error: float | None = None
    for index, phase in enumerate(phases):
        match = _match_from_phase(
            phase,
            index,
            pose_catalog,
            joints,
            inferred_joint_names=inferred,
        )
        if match is None or match.get("error_sum") is None:
            continue
        error = float(match["error_sum"])
        if best_error is None or error < best_error:
            best = match
            best_error = error
    return best
