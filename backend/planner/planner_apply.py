"""Rules for enabling「해당 결과로 자세 설정」(mirrors task_planner web UI)."""

from __future__ import annotations

from typing import Any

APPLY_POSE_MAX_GROUP_ERROR = 0.005


def _predicted_phase_number(
    predicted_phase: dict[str, Any] | None,
    *,
    phase_text: str | None = None,
) -> int | None:
    if not predicted_phase:
        return None
    index = predicted_phase.get("index")
    if index is not None and int(index) >= 0:
        return int(index)
    raw = predicted_phase.get("raw_text") or phase_text
    if raw is None or raw == "":
        return None
    text = str(raw).strip()
    if text.replace("-", "").isdigit():
        return int(text)
    return None


def group_errors_within_limit(
    match: dict[str, Any] | None,
    *,
    max_error: float = APPLY_POSE_MAX_GROUP_ERROR,
) -> bool:
    if not match:
        return False
    for key in ("error_body", "error_arms", "error_head"):
        value = match.get(key)
        if value is None or value != value or float(value) > max_error:
            return False
    return True


def apply_pose_block_reason(
    *,
    target_joints: dict[str, float] | None,
    predicted_phase: dict[str, Any] | None,
    closest_pose: dict[str, Any] | None,
    phase_text: str | None = None,
    max_error: float = APPLY_POSE_MAX_GROUP_ERROR,
) -> str:
    if not target_joints:
        return "추론 결과가 없습니다."
    predicted = _predicted_phase_number(predicted_phase, phase_text=phase_text)
    if predicted is None:
        return "Predicted Phase를 확인할 수 없습니다."
    if not closest_pose:
        return "Closest Phase Pose를 확인할 수 없습니다."
    closest_num = closest_pose.get("phase_number")
    if closest_num is None or int(predicted) != int(closest_num):
        return (
            f"Predicted Phase({predicted})와 "
            f"Closest Phase({closest_num})가 다릅니다."
        )
    if not group_errors_within_limit(closest_pose, max_error=max_error):
        return f"몸통/양팔/머리 오차가 {max_error} 이하여야 합니다."
    return ""


def can_apply_inference_pose(
    *,
    target_joints: dict[str, float] | None,
    predicted_phase: dict[str, Any] | None,
    closest_pose: dict[str, Any] | None,
    phase_text: str | None = None,
) -> tuple[bool, str]:
    reason = apply_pose_block_reason(
        target_joints=target_joints,
        predicted_phase=predicted_phase,
        closest_pose=closest_pose,
        phase_text=phase_text,
    )
    return (not reason, reason)
