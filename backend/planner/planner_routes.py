"""Task Planner REST routes (poses, phases, apply motion)."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from fastapi import FastAPI, HTTPException

from ..lerobot import hand_pose_map
from web.planner.phase_store import DEFAULT_PROFILE_NAME, PhaseStore
from web.planner.pose_catalog import PoseCatalog
from web.planner.schemas import (
    ApplyCommandRequest,
    ApplyCommandResponse,
    HandPoseFileOption,
    PhasesResponse,
    PosesResponse,
    PoseFileGroup,
    PoseInfo,
    PhaseEntryInput,
    PhaseProfilePayload,
    SavePhasesRequest,
    SaveTaskPolicyRulesRequest,
    StopMotionRequest,
    StopMotionResponse,
    TaskPolicyRule,
    TaskPolicyRulesResponse,
    LerobotChunkCheckRequest,
    LerobotChunkCheckResponse,
)
from web.planner.task_policy_rules_store import TaskPolicyRulesStore
from web.planner.lerobot_chunk_analysis import analyze_lerobot_chunk_vs_openvla
from web.runtime.joint_motion import JointMotionController

from .planner_board import build_phase_models

log = logging.getLogger(__name__)


def register_planner_routes(
    app: FastAPI,
    *,
    settings: Any,
    state: dict[str, object],
    phase_store: PhaseStore,
    task_rules_store: TaskPolicyRulesStore,
    pose_catalog: PoseCatalog,
    current_joints: Callable[[], dict[str, float]],
    ros: Callable[[], Any],
) -> None:
    @app.get("/api/poses", response_model=PosesResponse)
    def list_poses() -> PosesResponse:
        groups = pose_catalog.list_file_groups()
        return PosesResponse(
            file=str(pose_catalog.location()),
            files=[
                PoseFileGroup(
                    file=str(group["file"]),
                    poses=[
                        PoseInfo(
                            name=str(row["name"]),
                            id=row.get("id"),
                            file=str(row.get("file") or group["file"]),
                        )
                        for row in group["poses"]
                    ],
                )
                for group in groups
            ],
            poses=[
                PoseInfo(
                    name=str(row["name"]),
                    id=row.get("id"),
                    file=str(row.get("file") or ""),
                )
                for row in pose_catalog.list_pose_infos()
            ],
        )

    def _phase_rows_from_input(items: list[PhaseEntryInput]) -> list[dict[str, object]]:
        import secrets

        rows: list[dict[str, object]] = []
        for item in items:
            phase_id = (item.id or "").strip() or secrets.token_hex(6)
            row: dict[str, object] = {
                "id": phase_id,
                "description": item.description.strip(),
                "pose_name": item.pose_name.strip(),
            }
            if item.number is not None:
                row["number"] = round(float(item.number), 1)
            pose_file = item.pose_file.strip()
            if pose_file:
                row["pose_file"] = pose_file
            rows.append(row)
        return rows

    def _phases_response() -> PhasesResponse:
        active = phase_store.active_profile()
        all_profiles = phase_store.all_profiles()
        profiles_payload: dict[str, PhaseProfilePayload] = {}
        for name, bucket in all_profiles.items():
            rows = bucket.get("phases") if isinstance(bucket, dict) else []
            tags = bucket.get("tags") if isinstance(bucket, dict) else []
            profiles_payload[name] = PhaseProfilePayload(
                phases=[
                    PhaseEntryInput(
                        id=str(row["id"]),
                        number=float(row["number"]) if row.get("number") is not None else None,
                        description=str(row.get("description") or ""),
                        pose_name=str(row.get("pose_name") or ""),
                        pose_file=str(row.get("pose_file") or ""),
                    )
                    for row in rows or []
                ],
                tags=[str(tag) for tag in (tags or [])],
                hand_pose_file=str(bucket.get("hand_pose_file") or "")
                if isinstance(bucket, dict)
                else "",
            )
        hand_files = [
            HandPoseFileOption(id=str(item["id"]), name=str(item["name"]))
            for item in hand_pose_map.list_pose_files()
        ]
        return PhasesResponse(
            file=str(phase_store.path),
            active_profile=active,
            profiles=profiles_payload,
            phases=build_phase_models(phase_store, pose_catalog, current_joints()),
            hand_pose_files=hand_files,
        )

    @app.get("/api/phases", response_model=PhasesResponse)
    def list_phases() -> PhasesResponse:
        return _phases_response()

    @app.put("/api/phases", response_model=PhasesResponse)
    def save_phases(request: SavePhasesRequest) -> PhasesResponse:
        try:
            if request.profiles:
                active = (request.active_profile or "").strip() or DEFAULT_PROFILE_NAME
                document: dict[str, dict[str, object]] = {}
                for name, bucket in request.profiles.items():
                    profile_name = str(name).strip()
                    if not profile_name:
                        continue
                    tags: list[str] = []
                    seen: set[str] = set()
                    for raw in bucket.tags or []:
                        text = str(raw).strip()
                        if not text or text in seen:
                            continue
                        seen.add(text)
                        tags.append(text)
                        if len(tags) >= 5:
                            break
                    hand_pose_file = hand_pose_map.normalize_pose_filename(
                        bucket.hand_pose_file
                    )
                    document[profile_name] = {
                        "phases": _phase_rows_from_input(list(bucket.phases or [])),
                        "tags": tags,
                        "hand_pose_file": hand_pose_file,
                    }
                phase_store.replace_document(active_profile=active, profiles=document)
            elif request.phases is not None:
                phase_store.replace(_phase_rows_from_input(list(request.phases)))
            else:
                raise HTTPException(
                    status_code=400,
                    detail="profiles or phases is required",
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return _phases_response()

    @app.post("/api/command/apply", response_model=ApplyCommandResponse)
    def apply_command(request: ApplyCommandRequest) -> ApplyCommandResponse:
        ros_rt = ros()
        motion_started = time.perf_counter()
        try:
            error_code, error_msg = ros_rt.apply_joint_targets(
                request.target_joints,
                speed=request.speed,
                timeout_sec=settings.motion_timeout_sec,
            )
        except ValueError as exc:
            log.warning("Motion rejected before ROS call: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Motion apply failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        motion_ms = (time.perf_counter() - motion_started) * 1000.0

        ok = error_code == 0
        if ok:
            message = "Pose applied"
        elif error_msg:
            message = f"Motion failed:\n{error_msg}"
        else:
            message = "Motion failed"
        return ApplyCommandResponse(
            ok=ok,
            message=message,
            motion_error_code=error_code,
            motion_ms=motion_ms,
        )

    def _joint_motion() -> JointMotionController:
        ctrl = state.get("joint_motion")
        if not isinstance(ctrl, JointMotionController):
            raise HTTPException(status_code=503, detail="Joint motion unavailable")
        return ctrl

    @app.post("/api/motion/move", response_model=ApplyCommandResponse)
    def motion_move(request: ApplyCommandRequest) -> ApplyCommandResponse:
        if not request.target_joints:
            raise HTTPException(status_code=400, detail="No joint targets")
        motion_started = time.perf_counter()
        ctrl = _joint_motion()
        try:
            ctrl.start_move(
                dict(request.target_joints),
                speed=float(request.speed),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Motion move failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finished = ctrl.wait(timeout=125.0)
        motion_ms = (time.perf_counter() - motion_started) * 1000.0
        if not finished:
            return ApplyCommandResponse(
                ok=False,
                message="이동 시간이 초과되었습니다",
                motion_error_code=1,
                motion_ms=motion_ms,
            )
        return ApplyCommandResponse(
            ok=True,
            message="",
            motion_error_code=0,
            motion_ms=motion_ms,
        )

    @app.post("/api/motion/stop", response_model=StopMotionResponse)
    def motion_stop(request: StopMotionRequest = StopMotionRequest()) -> StopMotionResponse:
        motion_started = time.perf_counter()
        try:
            error_code, error_msg = _joint_motion().stop(speed=float(request.speed))
        except ValueError as exc:
            log.warning("Motion stop rejected: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            log.exception("Motion stop failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        motion_ms = (time.perf_counter() - motion_started) * 1000.0
        ok = error_code == 0
        message = "관절 이동을 중지했습니다" if ok else (error_msg or "Stop failed")
        return StopMotionResponse(
            ok=ok,
            message=message,
            motion_error_code=error_code,
            motion_ms=motion_ms,
        )

    @app.get("/api/planner/task-policy-rules", response_model=TaskPolicyRulesResponse)
    def list_task_policy_rules() -> TaskPolicyRulesResponse:
        task_rules_store.ensure_file()
        raw = task_rules_store.read()
        rules = [TaskPolicyRule.model_validate(row) for row in raw]
        return TaskPolicyRulesResponse(file=str(task_rules_store.path), rules=rules)

    @app.put("/api/planner/task-policy-rules", response_model=TaskPolicyRulesResponse)
    def save_task_policy_rules(
        request: SaveTaskPolicyRulesRequest,
    ) -> TaskPolicyRulesResponse:
        rows = [rule.model_dump() for rule in request.rules]
        try:
            task_rules_store.replace(rows)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return TaskPolicyRulesResponse(file=str(task_rules_store.path), rules=request.rules)

    @app.post(
        "/api/planner/lerobot-chunk-check",
        response_model=LerobotChunkCheckResponse,
    )
    def lerobot_chunk_check(body: LerobotChunkCheckRequest) -> LerobotChunkCheckResponse:
        kwargs: dict[str, object] = {
            "action_chunk": body.action_chunk,
            "openvla_target_joints": dict(body.openvla_target_joints),
            "inferred_joints_by_part": dict(body.inferred_joints_by_part),
        }
        if body.part_abort_rad is not None:
            kwargs["part_abort_rad"] = float(body.part_abort_rad)
        try:
            result = analyze_lerobot_chunk_vs_openvla(**kwargs)  # type: ignore[arg-type]
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return LerobotChunkCheckResponse(**result)
