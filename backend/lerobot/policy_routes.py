"""FastAPI routes for onboard LeRobot ACT (OAP lerobot_runtime + models/LeRobot)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import hand_pose_map, policy_run


class PolicyArmPostprocessOverrides(BaseModel):
    enabled: bool | None = None
    min_cutoff: float | None = Field(default=None, gt=0)
    beta: float | None = Field(default=None, ge=0)
    derivative_cutoff: float | None = Field(default=None, gt=0)


class PolicyRuntimeOverrides(BaseModel):
    model_config = ConfigDict(extra="allow")

    publish_hz: float | None = Field(default=None, gt=0)
    inference_hz: float | None = Field(default=None, gt=0)
    warming_up_time: float | None = Field(default=None, ge=0)
    min_run_time_before_complete: float | None = Field(default=None, ge=0)
    max_image_age_s: float | None = Field(default=None, gt=0)
    # -1 = always reinfer at inference_hz (no queue depth gate)
    reinfer_threshold: int | None = Field(default=None, ge=-1)
    motion_speed: float | None = Field(default=None, gt=0)
    arm_action_postprocess: PolicyArmPostprocessOverrides | None = None


class PolicyActionOverrides(BaseModel):
    gripper_threshold: float | None = Field(default=None, ge=0, le=1)


class HandMappingBody(BaseModel):
    policy: str = Field(..., min_length=1)
    file: str = ""


class PolicyRunBody(BaseModel):
    policy: str = Field(..., min_length=1)
    runtime: PolicyRuntimeOverrides | None = None
    action: PolicyActionOverrides | None = None
    timeout: float | None = None
    skip_warmup: bool = False


def _dump_overrides(model: BaseModel | None) -> dict[str, Any] | None:
    if model is None:
        return None
    raw = model.model_dump(exclude_none=True)
    if not raw:
        return None
    arm = raw.get("arm_action_postprocess")
    if isinstance(arm, dict) and not arm:
        raw.pop("arm_action_postprocess", None)
    return raw or None


def register_lerobot_policy_routes(router: APIRouter | None = None) -> APIRouter:
    r = router or APIRouter(prefix="/api/lerobot", tags=["lerobot-policy"])

    @r.get("/policies")
    def list_policies() -> dict:
        deployment = policy_run.deployment_config()
        try:
            items = policy_run.list_policies(deployment)
            status = policy_run.policy_status()
            return {
                "ok": True,
                "deployment": deployment,
                "policies": items,
                "status": status,
                "log_path": str(policy_run.LOG_FILE),
                "action_log_path": str(policy_run.LEROBOT_ACTION_LOG_FILE),
                "run_defaults": policy_run.run_defaults(),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @r.get("/hand-mappings")
    def hand_mappings() -> dict:
        return {
            "ok": True,
            "directory": str(hand_pose_map.hand_poses_dir()),
            "files": hand_pose_map.list_pose_files(),
            "selections": hand_pose_map.selections(),
        }

    @r.put("/hand-mappings")
    def save_hand_mapping(body: HandMappingBody) -> dict:
        try:
            selections = hand_pose_map.set_selection(body.policy, body.file)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "directory": str(hand_pose_map.hand_poses_dir()),
            "files": hand_pose_map.list_pose_files(),
            "selections": selections,
        }

    @r.get("/policies/run-defaults")
    def policy_run_defaults() -> dict:
        return {"ok": True, "defaults": policy_run.run_defaults()}

    @r.get("/policies/status")
    def policy_status() -> dict:
        return policy_run.policy_status()

    @r.get("/policies/log")
    def policy_log(lines: int = 2000) -> dict:
        return {"ok": True, "log": policy_run.policy_log(lines)}

    @r.get("/policies/action-log")
    def policy_action_log(lines: int = 500) -> dict:
        return policy_run.lerobot_action_log_tail(lines)

    @r.get("/policies/live-targets")
    def policy_live_targets() -> dict:
        return policy_run.read_live_policy_targets()

    @r.post("/policies/run")
    def policy_run_endpoint(body: PolicyRunBody) -> dict:
        try:
            return policy_run.start_policy(
                body.policy.strip(),
                runtime_overrides=_dump_overrides(body.runtime),
                action_overrides=_dump_overrides(body.action),
                skip_warmup=body.skip_warmup,
                timeout=body.timeout if body.timeout is not None else -1.0,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @r.post("/policies/stop")
    def policy_stop() -> dict:
        try:
            return policy_run.stop_policy()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return r
