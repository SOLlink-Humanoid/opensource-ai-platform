"""On-demand OpenVLA inference from a shared camera frame."""

from __future__ import annotations

import time
from typing import Any, Callable

from web.openvla.action_parser import (
    apply_hand_key_poses,
    compute_phase_pose_match,
    find_closest_phase_pose,
    inferred_joints_by_part,
    parse_openvla_action,
    resolve_predicted_phase,
    sanitize_g2_phase_text,
)
from web.openvla.openvla_engine import OpenVLAEngine
from web.openvla.openvla_image import bgr_to_openvla_pil
from web.planner.pose_catalog import PoseCatalog

from ..planner.planner_board import enrich_openvla_matches
from .openvla_live import OpenVLALiveSnapshot, _joint_error_sum


def resolve_latest_model_version(settings) -> str:
    latest = settings.latest_model_version()
    if latest:
        return latest
    pinned = (getattr(settings, "openvla_preload_version", None) or "").strip()
    if pinned:
        return pinned
    return "v2"


def run_openvla_on_snapshot(
    engine: OpenVLAEngine,
    snapshot,
    *,
    model_version: str,
    unnorm_key: str,
    instruction: str,
    get_current_joints: Callable[[], dict[str, float]],
    joint_names: list[str],
    phases: Callable[[], list[dict[str, Any]]],
    action_arm: str,
    pose_catalog: PoseCatalog | None = None,
    hand_table: dict[str, Any] | None = None,
) -> OpenVLALiveSnapshot:
    pil_image = bgr_to_openvla_pil(snapshot.bgr)
    started = time.perf_counter()
    inference = engine.predict(
        pil_image,
        instruction,
        version=model_version,
        unnorm_key=unnorm_key,
    )
    infer_ms = (time.perf_counter() - started) * 1000.0

    current = get_current_joints()
    phase_list = phases()
    target_map, phase_index, phase_label = parse_openvla_action(
        inference.joint_values,
        current,
        phase_text=inference.phase_text,
        phases=phase_list,
        action_arm=action_arm,
    )
    display_phase = sanitize_g2_phase_text(phase_label) or sanitize_g2_phase_text(
        inference.phase_text
    )
    predicted_action: list[float | str] = list(inference.joint_values)
    target_map = apply_hand_key_poses(target_map, predicted_action, hand_table)
    pred_dict: dict[str, Any] | None = None
    target_dict: dict[str, Any] | None = None
    closest_dict: dict[str, Any] | None = None
    if pose_catalog is not None:
        pred_dict, target_dict, closest_dict = enrich_openvla_matches(
            target_map=target_map,
            phase_index=phase_index,
            phase_list=phase_list,
            phase_text=display_phase,
            pose_catalog=pose_catalog,
            resolve_predicted_phase=resolve_predicted_phase,
            compute_phase_pose_match=compute_phase_pose_match,
            find_closest_phase_pose=find_closest_phase_pose,
            predicted_action=predicted_action,
            action_arm=action_arm,
            hand_table=hand_table,
        )
    else:
        predicted = resolve_predicted_phase(
            phase_index,
            phase_list,
            phase_text=display_phase,
        )
        pred_dict = dict(predicted) if predicted else None

    error_sum = _joint_error_sum(target_map, current, joint_names)

    return OpenVLALiveSnapshot(
        camera_id=snapshot.camera_id,
        frame_id=snapshot.frame_id,
        completed_at=time.time(),
        inference_ms=infer_ms,
        command=instruction,
        model_version=model_version,
        phase_text=display_phase,
        phase_index=(pred_dict or {}).get("index") if pred_dict else phase_index,
        phase_description=(pred_dict or {}).get("description") or "",
        target_joints=target_map,
        joint_error_sum=error_sum,
        error_trend="unknown",
        predicted_action=predicted_action,
        predicted_phase=pred_dict,
        target_phase_pose=target_dict,
        closest_pose=closest_dict,
        inferred_joints_by_part=inferred_joints_by_part(
            predicted_action,
            action_arm=action_arm,
        ),
    )
