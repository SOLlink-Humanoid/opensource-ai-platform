"""Phase list builders shared with task_planner web app."""

from __future__ import annotations

from typing import Any, Callable

from web.planner.pose_catalog import PoseCatalog
from web.planner.schemas import PhaseModel


def build_phase_models(
    phase_store,
    pose_catalog: PoseCatalog,
    current_joints: dict[str, float],
) -> list[PhaseModel]:
    models: list[PhaseModel] = []
    for index, phase in enumerate(phase_store.all()):
        pose_name = phase.get("pose_name") or ""
        error_sum = None
        error_body = None
        error_arms = None
        error_head = None
        if current_joints:
            grouped = pose_catalog.grouped_joint_errors_for_phase(phase, current_joints)
            if grouped is not None:
                error_body = grouped.get("body")
                error_arms = grouped.get("arms")
                error_head = grouped.get("head")
                parts = [v for v in grouped.values() if v is not None]
                error_sum = sum(parts) if parts else None
        raw_number = phase.get("number", index + 1)
        try:
            phase_number = round(float(raw_number), 1)
        except (TypeError, ValueError):
            phase_number = float(index + 1)
        models.append(
            PhaseModel(
                id=phase["id"],
                number=phase_number,
                description=phase.get("description") or "",
                pose_name=pose_name,
                error_sum=error_sum,
                error_body=error_body,
                error_arms=error_arms,
                error_head=error_head,
            )
        )
    return models


def enrich_openvla_matches(
    *,
    target_map: dict[str, float],
    phase_index: int | None,
    phase_list: list[dict[str, Any]],
    phase_text: str | None,
    pose_catalog: PoseCatalog,
    resolve_predicted_phase: Callable[..., dict[str, Any] | None],
    compute_phase_pose_match: Callable[..., dict[str, object] | None],
    find_closest_phase_pose: Callable[..., dict[str, object] | None],
    predicted_action: list[float | str] | None = None,
    action_arm: str = "left_arm",
    hand_table: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    predicted = resolve_predicted_phase(
        phase_index,
        phase_list,
        phase_text=phase_text,
    )
    match_index = (predicted or {}).get("index", phase_index)
    target_phase_pose = compute_phase_pose_match(
        target_map,
        match_index,
        phase_list,
        pose_catalog,
        predicted_action=predicted_action,
        action_arm=action_arm,
        hand_table=hand_table,
    )
    closest = find_closest_phase_pose(
        target_map,
        phase_list,
        pose_catalog,
        predicted_action=predicted_action,
        action_arm=action_arm,
        hand_table=hand_table,
    )
    pred_dict = dict(predicted) if predicted else None
    target_dict = dict(target_phase_pose) if target_phase_pose else None
    closest_dict = dict(closest) if closest else None
    return pred_dict, target_dict, closest_dict
