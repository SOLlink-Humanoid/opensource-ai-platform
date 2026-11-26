"""Planner REST models expected by backend.planner.planner_routes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PhaseModel(BaseModel):
    id: str
    number: float
    description: str = ""
    pose_name: str = ""
    error_sum: float | None = None
    error_body: float | None = None
    error_arms: float | None = None
    error_head: float | None = None


class PoseInfo(BaseModel):
    name: str
    id: str | None = None
    file: str = ""


class PoseFileGroup(BaseModel):
    file: str
    poses: list[PoseInfo]


class PosesResponse(BaseModel):
    file: str
    poses: list[PoseInfo]
    files: list[PoseFileGroup] = Field(default_factory=list)


class HandPoseFileOption(BaseModel):
    id: str
    name: str


class PhaseProfilePayload(BaseModel):
    phases: list["PhaseEntryInput"] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list, max_length=5)
    hand_pose_file: str = ""


class PhasesResponse(BaseModel):
    file: str
    active_profile: str = "기본"
    profiles: dict[str, PhaseProfilePayload] = Field(default_factory=dict)
    phases: list[PhaseModel]
    hand_pose_files: list[HandPoseFileOption] = Field(default_factory=list)


class PhaseEntryInput(BaseModel):
    id: str | None = None
    number: float | None = None
    description: str = ""
    pose_name: str = ""
    pose_file: str = ""


class SavePhasesRequest(BaseModel):
    active_profile: str | None = None
    profiles: dict[str, PhaseProfilePayload] | None = None
    phases: list[PhaseEntryInput] | None = None


class ApplyCommandRequest(BaseModel):
    target_joints: dict[str, float]
    speed: float = Field(0.05, ge=0.01, le=2.0)


class ApplyCommandResponse(BaseModel):
    ok: bool
    message: str
    motion_error_code: int = 0
    motion_ms: float | None = None


class StopMotionRequest(BaseModel):
    speed: float = Field(0.05, ge=0.01, le=2.0)


class StopMotionResponse(BaseModel):
    ok: bool
    message: str
    motion_error_code: int = 0
    motion_ms: float | None = None


class TaskPolicyRule(BaseModel):
    id: str
    openvla_profile: str = ""
    openvla_phase_id: str = ""
    yolo_label: str = ""
    yolo_min_count: int = Field(1, ge=0, le=99)
    policy_id: str = ""


class TaskPolicyRulesResponse(BaseModel):
    file: str
    rules: list[TaskPolicyRule] = Field(default_factory=list)


class SaveTaskPolicyRulesRequest(BaseModel):
    rules: list[TaskPolicyRule] = Field(default_factory=list)


class LerobotChunkCheckRequest(BaseModel):
    action_chunk: list[list[float]] = Field(default_factory=list)
    openvla_target_joints: dict[str, float] = Field(default_factory=dict)
    inferred_joints_by_part: dict[str, list[str]] = Field(default_factory=dict)
    part_abort_rad: float | None = Field(default=None, gt=0, le=3.0)


class LerobotChunkCheckResponse(BaseModel):
    ok: bool = True
    reject: bool = False
    reason: str = ""
    max_publish_steps: int = 0
    divergence_step: int | None = None
    steps_analyzed: int = 0
    initial_error_sum: float | None = None
    final_error_sum: float | None = None
    converging: bool | None = None
