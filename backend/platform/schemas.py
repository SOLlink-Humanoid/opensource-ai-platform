"""API models for the platform UI."""

from __future__ import annotations  # noqa: F404 — required for Pydantic on 3.9

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    ok: bool
    mode: str
    mock: bool
    version: str
    camera_ready: bool
    ros2_ready: bool
    yolo_ready: bool
    openvla_live_ready: bool
    lerobot_ready: bool
    head_camera_id: str | None = None
    openvla_device: str | None = None
    yolo_device: str | None = None
    startup_error: str | None = None
    studio_raw_preview: bool = False


class CameraInfo(BaseModel):
    id: str
    label: str


class CamerasResponse(BaseModel):
    cameras: list[CameraInfo]
    default_camera_id: str | None = None


class DetectionModel(BaseModel):
    class_id: int
    label: str
    confidence: float
    xyxy: list[int]


class YoloClassModel(BaseModel):
    class_id: int
    name: str


class YoloClassTargetModel(BaseModel):
    class_id: int
    name: str
    enabled: bool = True
    threshold: float = Field(0.7, ge=0.0, le=1.0)


class YoloTagClassFilterModel(BaseModel):
    text: str = ""


class YoloTagRegionModel(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    min_overlap_pct: float = Field(50.0, ge=0.0, le=100.0)


class YoloTagSizeRuleModel(BaseModel):
    min_width: float | None = None
    max_width: float | None = None
    min_height: float | None = None
    max_height: float | None = None
    orientation: str | None = None


class YoloTagRuleItemModel(BaseModel):
    id: str = ""
    kind: str
    text: str | None = None
    labels: list[str] = Field(default_factory=list)
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    min_overlap_pct: float | None = None
    min_width: float | None = None
    max_width: float | None = None
    min_height: float | None = None
    max_height: float | None = None
    orientation: str | None = None


class YoloTagConditionModel(BaseModel):
    id: str
    enabled: bool = True
    tag_name: str
    rule_items: list[YoloTagRuleItemModel] = Field(default_factory=list)
    class_contains: list[YoloTagClassFilterModel] = Field(default_factory=list)
    target_labels: list[str] = Field(default_factory=list)
    regions: list[YoloTagRegionModel] = Field(default_factory=list)
    size_rules: list[YoloTagSizeRuleModel] = Field(default_factory=list)


class YoloConfigResponse(BaseModel):
    model_path: str
    confidence: float
    classes: list[YoloClassModel] = Field(default_factory=list)
    targets: list[YoloClassTargetModel] = Field(default_factory=list)
    tag_conditions: list[YoloTagConditionModel] = Field(default_factory=list)
    ready: bool = False
    error: str | None = None


class YoloConfidenceRequest(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0)


class YoloTargetPatch(BaseModel):
    class_id: int
    enabled: bool | None = None
    threshold: float | None = Field(None, ge=0.0, le=1.0)


class YoloTargetsRequest(BaseModel):
    targets: list[YoloTargetPatch] = Field(default_factory=list)


class YoloTagConditionsRequest(BaseModel):
    tag_conditions: list[YoloTagConditionModel] = Field(default_factory=list)


class YoloWorkerRequest(BaseModel):
    enabled: bool = True
    interval_sec: float | None = Field(None, ge=0.05, le=10.0)


class YoloState(BaseModel):
    frame_id: int | None = None
    inference_ms: float | None = None
    detections: list[DetectionModel] = Field(default_factory=list)
    classes: list[YoloClassModel] = Field(default_factory=list)
    targets: list[YoloClassTargetModel] = Field(default_factory=list)
    tag_conditions: list[YoloTagConditionModel] = Field(default_factory=list)
    active_tags: list[str] = Field(default_factory=list)
    confidence_threshold: float | None = None
    ready: bool = False
    error: str | None = None


class PhaseMatchModel(BaseModel):
    phase_number: int
    phase_id: str | None = None
    description: str = ""
    pose_name: str = ""
    error_sum: float | None = None
    error_head: float | None = None
    error_body: float | None = None
    error_left_arm: float | None = None
    error_right_arm: float | None = None
    error_left_hand: float | None = None
    error_right_hand: float | None = None
    error_arms: float | None = None


class PredictedPhaseModel(BaseModel):
    index: int | None = None
    phase_id: str | None = None
    description: str = ""
    pose_name: str = ""
    raw_text: str | None = None


class OpenVLAState(BaseModel):
    frame_id: int | None = None
    inference_ms: float | None = None
    command: str = ""
    model_version: str = ""
    phase_text: str | None = None
    phase_index: int | None = None
    phase_description: str = ""
    joint_error_sum: float | None = None
    error_trend: str = "unknown"
    predicted_action: list[float] = Field(default_factory=list)
    target_joints: dict[str, float] = Field(default_factory=dict)
    inferred_joints_by_part: dict[str, list[str]] = Field(default_factory=dict)
    predicted_phase: PredictedPhaseModel | None = None
    target_phase_pose: PhaseMatchModel | None = None
    closest_pose: PhaseMatchModel | None = None
    can_apply_pose: bool = False
    apply_pose_block_reason: str = ""
    ready: bool = False
    error: str | None = None
    status: str = "idle"
    engine_ready: bool = False
    engine_status: str = "idle"
    engine_error: str | None = None


class OpenVLAInferRequest(BaseModel):
    instruction: str | None = None
    phase_profile: str | None = None


class OpenVLAProfileMetricsRequest(BaseModel):
    phase_profile: str
    phase_index: int | None = None
    phase_text: str | None = None
    target_joints: dict[str, float] = Field(default_factory=dict)
    predicted_action: list[float | str] = Field(default_factory=list)


class OpenVLAProfileMetricsResponse(BaseModel):
    predicted_phase: PredictedPhaseModel | None = None
    target_phase_pose: PhaseMatchModel | None = None
    target_joints: dict[str, float] = Field(default_factory=dict)


class OpenVLAInstructionEntry(BaseModel):
    id: str
    text: str
    last_used_at: float


class OpenVLAInstructionHistoryResponse(BaseModel):
    last_instruction: str = ""
    entries: list[OpenVLAInstructionEntry] = Field(default_factory=list)


class OpenVLAInferResponse(BaseModel):
    ok: bool
    message: str
    openvla: OpenVLAState


class OpenVLAModelsResponse(BaseModel):
    latest_version: str | None = None
    model_root: str
    versions: list[str] = Field(default_factory=list)


class LerobotState(BaseModel):
    timestamp: float | None = None
    action_chunk: list[list[float]] = Field(default_factory=list)
    source: str | None = None
    ready: bool = False
    note: str | None = None


class DevProfileResponse(BaseModel):
    mode: str
    config_file: str | None = None
    openvla_device: str | None = None
    yolo_device: str | None = None
    camera_source: str
    yolo_model: str
    openvla_root: str
    openvla_model_dir: str
    openvla_version: str
    openvla_loaded_version: str | None = None
    lerobot: str
    instruction: str = ""


class JointSample(BaseModel):
    name: str
    value: float


class JointsResponse(BaseModel):
    positions: dict[str, float]
    sample: list[JointSample]


class PlannerCheckModel(BaseModel):
    id: str
    label: str
    ok: bool | None = None
    detail: str = ""


class PlannerPhaseRow(BaseModel):
    id: str
    number: int
    description: str = ""
    pose_name: str = ""
    error_sum: float | None = None
    error_body: float | None = None
    error_arms: float | None = None
    error_head: float | None = None


class PlannerState(BaseModel):
    phase_description: str = ""
    phase_index: int | None = None
    joint_error_sum: float | None = None
    error_trend: str = "unknown"
    approaching_target: bool | None = None
    visible_labels: list[str] = Field(default_factory=list)
    checks: list[PlannerCheckModel] = Field(default_factory=list)
    phases: list[PlannerPhaseRow] = Field(default_factory=list)
    phases_file: str = ""


class LiveStateResponse(BaseModel):
    server_time: float
    camera_id: str | None = None
    frame_id: int | None = None
    studio_raw_preview: bool = False
    yolo: YoloState
    openvla: OpenVLAState
    lerobot: LerobotState
    planner: PlannerState
