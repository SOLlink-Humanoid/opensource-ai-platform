"""FastAPI app: shared camera hub + 4-tab live dashboard."""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, Response

def _resolve_task_planner_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    for candidate in (
        root.parent / "g2-controller" / "task_planner",
        root.parent / "task_planner",
    ):
        if (candidate / "web").is_dir():
            return candidate
    return root


_TASK_PLANNER_DIR = _resolve_task_planner_dir()
if _TASK_PLANNER_DIR.is_dir() and str(_TASK_PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_PLANNER_DIR))

from shared.joint_names import ALL_MODEL_JOINTS  # noqa: E402

from web.camera.camera_service import build_camera_service  # noqa: E402
from web.config import Settings  # noqa: E402
from web.openvla.openvla_deps import openvla_deps_error  # noqa: E402
from web.openvla.openvla_engine import OpenVLAEngine  # noqa: E402
from web.planner.phase_store import PhaseStore  # noqa: E402
from web.planner.pose_catalog import PoseCatalog  # noqa: E402
from web.runtime.ros_runtime import build_ros_runtime  # noqa: E402
from web.yolo.yolo_service import YoloDetectionService  # noqa: E402

from . import __version__
from .lerobot import hand_pose_map
from .lerobot.lerobot_bridge import LerobotBridge
from .lerobot.policy_routes import register_lerobot_policy_routes
from .openvla.instruction_history_store import (
    delete_instruction,
    instruction_history_path,
    load_history,
    record_instruction,
)
from .openvla.openvla_live import OpenVLALiveSnapshot, OpenVLALiveWorker
from .openvla.openvla_load import start_openvla_load
from .openvla.openvla_run import resolve_latest_model_version, run_openvla_on_snapshot
from web.openvla.action_parser import (  # noqa: E402
    apply_hand_key_poses,
    compute_phase_pose_match,
    resolve_predicted_phase,
)
from .planner.planner_apply import can_apply_inference_pose
from .planner.planner_routes import register_planner_routes
from .planner.planner_snapshot import build_planner_snapshot
from .platform.schemas import (
    CameraInfo,
    CamerasResponse,
    DevProfileResponse,
    DetectionModel,
    HealthResponse,
    JointSample,
    JointsResponse,
    LerobotState,
    LiveStateResponse,
    OpenVLAInferRequest,
    OpenVLAInferResponse,
    OpenVLAProfileMetricsRequest,
    OpenVLAProfileMetricsResponse,
    OpenVLAInstructionEntry,
    OpenVLAInstructionHistoryResponse,
    OpenVLAModelsResponse,
    OpenVLAState,
    PhaseMatchModel,
    PlannerCheckModel,
    PlannerState,
    PredictedPhaseModel,
    YoloClassModel,
    YoloClassTargetModel,
    YoloConfidenceRequest,
    YoloConfigResponse,
    YoloState,
    YoloTagConditionModel,
    YoloTagConditionsRequest,
    YoloTargetsRequest,
    YoloWorkerRequest,
)
from .platform.config_file import resolve_config_path
from .yolo.yolo_meta import yolo_class_names, yolo_class_names_from_config
from .yolo.yolo_tag_conditions_store import _clean_tag_conditions, read_yolo_tag_conditions
from .yolo.yolo_tag_eval import evaluate_yolo_tag_conditions
from .platform.model_config_dir import (
    bootstrap_model_config_layout,
    openvla_phases_file,
    task_policy_rules_file,
)
from .yolo.yolo_targets_store import (
    migrate_yolo_settings_to_config,
    read_yolo_settings,
    write_yolo_targets,
    yolo_targets_path,
)
from .platform.gpu_lock import MPS_INFERENCE_LOCK
from .platform.settings import (
    PlatformSettings,
    apply_robot_gpu_settings,
    dev_profile_paths,
)

log = logging.getLogger(__name__)

_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}

UI_TAB_SLUGS: tuple[str, ...] = ("yolo", "openvla", "lerobot", "planner", "dev")


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(_NO_CACHE)
        return response


def _openvla_engine_meta(
    settings: Settings,
    state: dict[str, object],
) -> dict[str, object]:
    load_status = str(state.get("openvla_load_status") or "idle")
    engine = state.get("openvla")
    load_error = state.get("openvla_load_error")
    if settings.mock:
        return {
            "engine_ready": True,
            "engine_status": "ready",
            "engine_error": None,
        }
    if not settings.openvla_enabled:
        return {
            "engine_ready": False,
            "engine_status": "disabled",
            "engine_error": "OpenVLA 비활성화됨 (config openvla.enabled)",
        }
    ready = (
        load_status == "ready"
        and isinstance(engine, OpenVLAEngine)
        and engine.ready
    )
    engine_error: str | None = None
    if load_status == "error":
        engine_error = str(load_error) if load_error else "OpenVLA load failed"
    elif isinstance(engine, OpenVLAEngine) and engine.last_error:
        engine_error = engine.last_error
    return {
        "engine_ready": ready,
        "engine_status": load_status,
        "engine_error": engine_error,
    }


def _snap_to_openvla_state(
    snap: OpenVLALiveSnapshot | None,
    *,
    platform: PlatformSettings,
    live_svc: object | None,
    engine_meta: dict[str, object],
    status: str = "ok",
) -> OpenVLAState:
    if snap is None:
        return OpenVLAState(
            command=platform.openvla_live_instruction,
            error=getattr(live_svc, "last_error", None) if live_svc else None,
            status="idle",
            engine_ready=bool(engine_meta.get("engine_ready")),
            engine_status=str(engine_meta.get("engine_status") or "idle"),
            engine_error=engine_meta.get("engine_error"),  # type: ignore[arg-type]
        )
    can_apply, block_reason = can_apply_inference_pose(
        target_joints=dict(snap.target_joints),
        predicted_phase=snap.predicted_phase,
        closest_pose=snap.closest_pose,
        phase_text=snap.phase_text,
    )
    return OpenVLAState(
        frame_id=snap.frame_id,
        inference_ms=snap.inference_ms,
        command=snap.command,
        model_version=snap.model_version,
        phase_text=snap.phase_text,
        phase_index=snap.phase_index,
        phase_description=snap.phase_description,
        joint_error_sum=snap.joint_error_sum,
        error_trend=snap.error_trend,
        predicted_action=list(snap.predicted_action),
        target_joints=dict(snap.target_joints),
        inferred_joints_by_part=dict(snap.inferred_joints_by_part or {}),
        predicted_phase=(
            PredictedPhaseModel(**snap.predicted_phase)
            if snap.predicted_phase
            else None
        ),
        target_phase_pose=(
            PhaseMatchModel(**snap.target_phase_pose)
            if snap.target_phase_pose
            else None
        ),
        closest_pose=(
            PhaseMatchModel(**snap.closest_pose) if snap.closest_pose else None
        ),
        can_apply_pose=can_apply,
        apply_pose_block_reason=block_reason,
        ready=True,
        status=status,
        engine_ready=bool(engine_meta.get("engine_ready")),
        engine_status=str(engine_meta.get("engine_status") or "ready"),
        engine_error=engine_meta.get("engine_error"),  # type: ignore[arg-type]
    )


def _runtime_mode(settings: Settings, platform: PlatformSettings) -> str:
    if settings.mock:
        return "mock"
    return "robot"


def create_app(
    settings: Settings | None = None,
    platform: PlatformSettings | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    platform = platform or PlatformSettings.from_env()
    runtime_mode = _runtime_mode(settings, platform)
    if runtime_mode == "robot":
        settings = apply_robot_gpu_settings(settings)
    bootstrap_model_config_layout(
        phases_legacy=settings.phases_file,
        openvla_root=settings.openvla_root,
        task_rules_legacy=settings.phases_file.parent / "task_policy_rules.json",
    )
    phase_store = PhaseStore(openvla_phases_file(create_dir=True))
    phase_store.ensure_file()
    from web.planner.task_policy_rules_store import TaskPolicyRulesStore

    task_rules_store = TaskPolicyRulesStore(task_policy_rules_file(create_dir=True))
    task_rules_store.ensure_file()
    pose_catalog = PoseCatalog(settings.poses_file)

    cfg_path = resolve_config_path()
    state: dict[str, object] = {
        "camera": None,
        "ros": None,
        "openvla": None,
        "yolo": None,
        "openvla_live": None,
        "openvla_manual": None,
        "lerobot": None,
        "startup_error": None,
        "oap_config_path": cfg_path,
        "yolo_tag_conditions": [],
    }

    def _yolo_targets_write_path() -> Path | None:
        yolo_svc = state.get("yolo")
        model_path = None
        if isinstance(yolo_svc, YoloDetectionService) and yolo_svc.model_path is not None:
            model_path = yolo_svc.model_path
        elif settings.yolo_model_path is not None:
            model_path = settings.yolo_model_path
        return yolo_targets_path(model_path)

    def _yolo_tag_conditions_snapshot() -> list[dict[str, object]]:
        raw = state.get("yolo_tag_conditions")
        if not isinstance(raw, list):
            return []
        return _clean_tag_conditions(raw)

    def _set_yolo_tag_conditions(rows: list[dict[str, object]]) -> None:
        state["yolo_tag_conditions"] = _clean_tag_conditions(rows)

    def _refresh_yolo_tag_conditions_from_disk() -> None:
        model_path = settings.yolo_model_path
        yolo_svc = state.get("yolo")
        if isinstance(yolo_svc, YoloDetectionService) and yolo_svc.model_path is not None:
            model_path = yolo_svc.model_path
        if model_path is None:
            return
        oap_cfg = state.get("oap_config_path")
        config_path = oap_cfg if isinstance(oap_cfg, Path) else None
        tags = read_yolo_tag_conditions(model_path, config_path=config_path)
        _set_yolo_tag_conditions(tags)

    def _persist_yolo_targets() -> None:
        from .yolo.yolo_targets_store import _read_file

        targets_path = _yolo_targets_write_path()
        if targets_path is None:
            return
        yolo_svc = state.get("yolo")
        if isinstance(yolo_svc, YoloDetectionService):
            model = yolo_svc.model_path
            write_yolo_targets(
                targets_path,
                model_path=str(model) if model is not None else "",
                default_confidence=yolo_svc.confidence,
                targets=yolo_svc.targets_snapshot(),
                tag_conditions=_yolo_tag_conditions_snapshot(),
            )
            return
        existing = _read_file(targets_path) if targets_path.is_file() else None
        payload = existing if isinstance(existing, dict) else {}
        raw_targets = payload.get("targets")
        targets = raw_targets if isinstance(raw_targets, list) else []
        try:
            default_conf = float(payload.get("default_confidence", settings.yolo_confidence))
        except (TypeError, ValueError):
            default_conf = float(settings.yolo_confidence)
        model_ref = str(payload.get("model_path") or settings.yolo_model_path or "")
        write_yolo_targets(
            targets_path,
            model_path=model_ref,
            default_confidence=default_conf,
            targets=targets,
            tag_conditions=_yolo_tag_conditions_snapshot(),
        )

    def _current_joints() -> dict[str, float]:
        ros_rt = state.get("ros")
        if ros_rt is None:
            return {}
        if not getattr(ros_rt, "ready", False):
            cached = getattr(ros_rt, "last_positions", None)
            return dict(cached) if cached else {}
        try:
            return ros_rt.get_positions(timeout_sec=1.0)  # type: ignore[union-attr]
        except Exception:
            cached = getattr(ros_rt, "last_positions", None)
            return dict(cached) if cached else {}

    def _instruction_history_file() -> Path:
        return instruction_history_path(settings.openvla_root)

    def _instruction_history_response() -> OpenVLAInstructionHistoryResponse:
        data = load_history(_instruction_history_file())
        return OpenVLAInstructionHistoryResponse(
            last_instruction=str(data.get("last_instruction") or ""),
            entries=[
                OpenVLAInstructionEntry(
                    id=str(row["id"]),
                    text=str(row["text"]),
                    last_used_at=float(row["last_used_at"]),
                )
                for row in data.get("entries", [])
            ],
        )

    def _head_camera_id() -> str | None:
        cam = state.get("camera")
        if cam is None:
            return None
        return getattr(cam, "head_camera_id", None) or getattr(
            cam, "default_camera_id", None
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        camera = build_camera_service(
            settings.mock,
            topics_path=settings.topics_yaml,
            mapping_path=settings.mapping_yaml,
        )
        from web.camera.studio_preview import resolve_studio_raw_preview

        studio_raw_preview = resolve_studio_raw_preview(camera)
        state["studio_raw_preview"] = studio_raw_preview
        ros = build_ros_runtime(settings.mock)

        openvla = OpenVLAEngine(
            settings.openvla_root if settings.openvla_enabled else None,
            default_unnorm_key=settings.default_unnorm_key,
            mock=settings.mock,
            device_preference=settings.openvla_device,
        )
        state["openvla"] = openvla
        if settings.openvla_enabled:
            start_openvla_load(
                engine=openvla,
                settings=settings,
                state=state,
                resolve_version=resolve_latest_model_version,
                deps_error=openvla_deps_error,
                inference_lock=MPS_INFERENCE_LOCK,
                warmup=settings.openvla_warmup,
                unnorm_key=settings.default_unnorm_key,
            )
        else:
            state["openvla_load_status"] = "disabled"
            state["openvla_load_error"] = None
        lerobot_synthetic = settings.mock
        default_lerobot_state = (
            Path("/tmp/oap_lerobot_live.json")
            if not platform.lerobot_act_placeholder
            else None
        )
        lerobot_state_path = platform.lerobot_state_file or default_lerobot_state
        lerobot = LerobotBridge(
            state_file=lerobot_state_path,
            mock=lerobot_synthetic or platform.lerobot_act_placeholder,
            synthetic_mock=lerobot_synthetic,
        )

        try:
            if hasattr(camera, "start"):
                camera.start()  # type: ignore[union-attr]
            ros.start()  # type: ignore[union-attr]
            if lerobot_state_path or lerobot_synthetic:
                lerobot.start()
            state["camera"] = camera
            state["ros"] = ros
            from web.runtime.joint_motion import JointMotionController

            state["joint_motion"] = JointMotionController(lambda: state["ros"])
            state["lerobot"] = lerobot

            head_id = _head_camera_id()
            oap_cfg = state.get("oap_config_path")
            config_path = oap_cfg if isinstance(oap_cfg, Path) else None
            yolo_model_path = settings.yolo_model_path
            if yolo_model_path is not None:
                migrate_yolo_settings_to_config(
                    yolo_model_path,
                    config_path=config_path,
                )
                _refresh_yolo_tag_conditions_from_disk()
            if settings.yolo_enabled and head_id and not studio_raw_preview:
                yolo = YoloDetectionService(
                    camera,
                    model_path=settings.yolo_model_path,
                    camera_id=head_id,
                    device=settings.yolo_device,
                    confidence=settings.yolo_confidence,
                    image_size=settings.yolo_image_size,
                    min_interval_sec=settings.yolo_interval_sec,
                )
                try:
                    yolo.start()
                    state["yolo"] = yolo
                    if yolo_model_path is not None:
                        saved_confidence, patches = read_yolo_settings(
                            yolo_model_path,
                            config_path=config_path,
                        )
                        try:
                            if saved_confidence is not None:
                                yolo.set_confidence(saved_confidence)
                            if patches:
                                yolo.apply_targets(patches)
                        except ValueError as exc:
                            log.warning(
                                "Ignored invalid YOLO targets for model %s: %s",
                                yolo_model_path,
                                exc,
                            )
                        else:
                            if patches or saved_confidence is not None:
                                _persist_yolo_targets()
                except Exception as exc:  # noqa: BLE001
                    log.warning("YOLO disabled: %s", exc)

            model_version = resolve_latest_model_version(settings)
            if settings.openvla_enabled and platform.openvla_live_enabled and head_id:
                use_real_openvla = not settings.mock
                if use_real_openvla:
                    deps_err = openvla_deps_error()
                    if deps_err:
                        log.warning("OpenVLA live disabled: %s", deps_err.splitlines()[0])
                worker = OpenVLALiveWorker(
                    camera,
                    openvla,
                    camera_id=head_id,
                    model_version=model_version,
                    unnorm_key=settings.default_unnorm_key,
                    instruction=platform.openvla_live_instruction,
                    min_interval_sec=platform.openvla_live_interval_sec,
                    get_current_joints=_current_joints,
                    joint_names=list(ALL_MODEL_JOINTS),
                    phases=phase_store.all,
                    action_arm=settings.action_arm,
                    joint_stopped_epsilon=platform.joint_stopped_epsilon,
                )
                worker.start()
                state["openvla_live"] = worker
        except Exception as exc:  # noqa: BLE001
            state["startup_error"] = str(exc)
            log.exception("Platform partial startup failure")

        try:
            yield
        finally:
            for key in ("openvla_live", "yolo", "lerobot", "camera", "ros"):
                svc = state.get(key)
                if svc is not None and hasattr(svc, "shutdown"):
                    svc.shutdown()  # type: ignore[union-attr]

    app = FastAPI(title="Open Source AI Platform", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    static_dir = platform.static_dir
    if static_dir.is_dir():
        app.mount("/assets", NoCacheStaticFiles(directory=static_dir), name="assets")

        def _brand_icon_file(filename: str, media_type: str) -> FileResponse:
            path = static_dir / filename
            if not path.is_file():
                raise HTTPException(status_code=404)
            return FileResponse(path, media_type=media_type, headers=_NO_CACHE)

        @app.get("/favicon.ico", include_in_schema=False)
        def favicon_ico() -> FileResponse:
            return _brand_icon_file("favicon.ico", "image/x-icon")

        @app.get("/favicon.svg", include_in_schema=False)
        def favicon_svg() -> FileResponse:
            return _brand_icon_file("favicon.svg", "image/svg+xml")

        @app.get("/apple-touch-icon.png", include_in_schema=False)
        def apple_touch_icon() -> FileResponse:
            return _brand_icon_file("apple-touch-icon.png", "image/png")

    def camera():
        svc = state.get("camera")
        if svc is None:
            raise HTTPException(
                status_code=503,
                detail=state.get("startup_error") or "Camera unavailable",
            )
        return svc

    def _camera_ids() -> set[str]:
        svc = state.get("camera")
        if svc is None or not hasattr(svc, "list_cameras"):
            return set()
        return {spec.id for spec in svc.list_cameras()}

    async def _mjpeg(jpeg_supplier, poll_sec: float = 0.05) -> AsyncIterator[bytes]:
        import asyncio

        last_frame_id = -1
        while True:
            jpeg, frame_id = jpeg_supplier()
            if jpeg and frame_id != last_frame_id:
                last_frame_id = frame_id
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(poll_sec)

    def _raw_jpeg(camera_id: str) -> tuple[bytes | None, int]:
        svc = camera()
        if hasattr(svc, "get_jpeg_with_id"):
            jpeg, frame_id = svc.get_jpeg_with_id(camera_id)  # type: ignore[union-attr]
            return jpeg, frame_id
        jpeg = svc.encode_jpeg(camera_id)  # type: ignore[union-attr]
        return jpeg, 0

    def _yolo_jpeg(camera_id: str) -> tuple[bytes | None, int]:
        if state.get("studio_raw_preview"):
            return _raw_jpeg(camera_id)
        detector = state.get("yolo")
        if isinstance(detector, YoloDetectionService):
            detected = detector.latest(camera_id)
            if detected is not None:
                return detected.jpeg, detected.frame_id
        return _raw_jpeg(camera_id)

    def _index_html() -> HTMLResponse:
        index_path = static_dir / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="frontend/index.html missing")
        return HTMLResponse(index_path.read_text(encoding="utf-8"), headers=_NO_CACHE)

    @app.get("/", response_class=RedirectResponse)
    def index_root() -> RedirectResponse:
        return RedirectResponse(url="/yolo", status_code=302)

    for _slug in UI_TAB_SLUGS:

        @app.get(f"/{_slug}", response_class=HTMLResponse, name=f"ui-{_slug}")
        def _ui_tab_page() -> HTMLResponse:
            return _index_html()

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        cam = state.get("camera")
        ros_rt = state.get("ros")
        yolo = state.get("yolo")
        live = state.get("openvla_live")
        lerobot = state.get("lerobot")
        return HealthResponse(
            ok=cam is not None,
            mode=_runtime_mode(settings, platform),
            mock=settings.mock,
            version=__version__,
            camera_ready=bool(getattr(cam, "ready", False)),
            ros2_ready=bool(getattr(ros_rt, "ready", False)),
            yolo_ready=isinstance(yolo, YoloDetectionService) and yolo.ready,
            openvla_live_ready=isinstance(live, OpenVLALiveWorker),
            lerobot_ready=lerobot is not None,
            head_camera_id=_head_camera_id(),
            openvla_device=getattr(state.get("openvla"), "device", None)
            or settings.openvla_device,
            yolo_device=settings.yolo_device,
            startup_error=state.get("startup_error"),  # type: ignore[arg-type]
            studio_raw_preview=bool(state.get("studio_raw_preview")),
        )

    @app.get("/api/cameras", response_model=CamerasResponse)
    def list_cameras() -> CamerasResponse:
        svc = camera()
        cameras = [
            CameraInfo(id=spec.id, label=spec.label)
            for spec in svc.list_cameras()  # type: ignore[union-attr]
        ]
        return CamerasResponse(
            cameras=cameras,
            default_camera_id=_head_camera_id(),
        )

    @app.get("/api/stream/raw/{camera_id}")
    async def stream_raw(camera_id: str) -> StreamingResponse:
        if camera_id not in _camera_ids():
            raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")
        return StreamingResponse(
            _mjpeg(lambda: _raw_jpeg(camera_id)),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/stream/yolo/{camera_id}")
    async def stream_yolo(camera_id: str) -> StreamingResponse:
        if camera_id not in _camera_ids():
            raise HTTPException(status_code=404, detail=f"Unknown camera: {camera_id}")
        return StreamingResponse(
            _mjpeg(lambda: _yolo_jpeg(camera_id)),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    def _yolo_class_models(model_path: Path | None = None) -> list[YoloClassModel]:
        path = model_path or settings.yolo_model_path
        if path is None:
            return []
        yolo_svc = state.get("yolo")
        if isinstance(yolo_svc, YoloDetectionService):
            names = yolo_svc.class_names()
        elif settings.yolo_enabled:
            names = yolo_class_names(path, allow_weight_load=True)
        else:
            oap_cfg = state.get("oap_config_path")
            config_path = oap_cfg if isinstance(oap_cfg, Path) else None
            names = yolo_class_names_from_config(path, config_path=config_path)
            if not names:
                names = {}
        return [
            YoloClassModel(class_id=class_id, name=name)
            for class_id, name in sorted(names.items())
        ]

    def _yolo_target_models(yolo_svc: YoloDetectionService | None) -> list[YoloClassTargetModel]:
        if not isinstance(yolo_svc, YoloDetectionService):
            return []
        return [YoloClassTargetModel(**row) for row in yolo_svc.targets_snapshot()]

    def _yolo_config_response(
        *,
        refresh_tag_conditions_from_disk: bool = True,
    ) -> YoloConfigResponse:
        if refresh_tag_conditions_from_disk:
            _refresh_yolo_tag_conditions_from_disk()
        yolo_svc = state.get("yolo")
        model_path = settings.yolo_model_path
        if isinstance(yolo_svc, YoloDetectionService):
            model_path = yolo_svc.model_path
        confidence = float(settings.yolo_confidence)
        ready = False
        error: str | None = None
        if isinstance(yolo_svc, YoloDetectionService):
            confidence = yolo_svc.confidence
            ready = yolo_svc.ready
            error = yolo_svc.last_error
        elif yolo_svc is None:
            error = "YOLO service unavailable"
        tag_rows = _yolo_tag_conditions_snapshot()
        return YoloConfigResponse(
            model_path=str(model_path) if model_path else "",
            confidence=confidence,
            classes=_yolo_class_models(model_path),
            targets=_yolo_target_models(yolo_svc if isinstance(yolo_svc, YoloDetectionService) else None),
            tag_conditions=[YoloTagConditionModel(**row) for row in tag_rows],
            ready=ready,
            error=error,
        )

    @app.get("/api/yolo/config", response_model=YoloConfigResponse)
    def yolo_config() -> YoloConfigResponse:
        return _yolo_config_response()

    @app.put("/api/yolo/confidence", response_model=YoloConfigResponse)
    def yolo_set_confidence(body: YoloConfidenceRequest) -> YoloConfigResponse:
        yolo_svc = state.get("yolo")
        if not isinstance(yolo_svc, YoloDetectionService):
            raise HTTPException(status_code=503, detail="YOLO service unavailable")
        try:
            yolo_svc.set_confidence(body.confidence)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _persist_yolo_targets()
        return _yolo_config_response()

    @app.put("/api/yolo/targets", response_model=YoloConfigResponse)
    def yolo_set_targets(body: YoloTargetsRequest) -> YoloConfigResponse:
        yolo_svc = state.get("yolo")
        if not isinstance(yolo_svc, YoloDetectionService):
            raise HTTPException(status_code=503, detail="YOLO service unavailable")
        try:
            yolo_svc.apply_targets([row.model_dump() for row in body.targets])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _persist_yolo_targets()
        return _yolo_config_response()

    @app.put("/api/yolo/tag-conditions", response_model=YoloConfigResponse)
    def yolo_set_tag_conditions(body: YoloTagConditionsRequest) -> YoloConfigResponse:
        _set_yolo_tag_conditions([row.model_dump() for row in body.tag_conditions])
        _persist_yolo_targets()
        return _yolo_config_response(refresh_tag_conditions_from_disk=False)

    @app.put("/api/yolo/worker")
    def yolo_set_worker(body: YoloWorkerRequest) -> dict[str, object]:
        yolo_svc = state.get("yolo")
        if not isinstance(yolo_svc, YoloDetectionService):
            raise HTTPException(status_code=503, detail="YOLO service unavailable")
        if body.interval_sec is not None:
            yolo_svc.set_min_interval(body.interval_sec)
        if body.enabled:
            yolo_svc.resume()
        else:
            yolo_svc.pause()
        return {"enabled": body.enabled, "interval_sec": yolo_svc._min_interval}

    @app.get("/api/live/state", response_model=LiveStateResponse)
    def live_state() -> LiveStateResponse:
        head_id = _head_camera_id()
        frame_id: int | None = None
        if head_id:
            cam = state.get("camera")
            if cam is not None and hasattr(cam, "latest_frame"):
                snap = cam.latest_frame(head_id, wait=False)  # type: ignore[union-attr]
                if snap is not None:
                    frame_id = snap.frame_id

        yolo_svc = state.get("yolo")
        yolo_frame = None
        if isinstance(yolo_svc, YoloDetectionService) and head_id:
            yolo_frame = yolo_svc.latest(head_id)
        tag_rows = _yolo_tag_conditions_snapshot()
        tag_models = [YoloTagConditionModel(**row) for row in tag_rows]
        det_dicts = [
            {
                "class_id": d.class_id,
                "label": d.label,
                "confidence": d.confidence,
                "xyxy": list(d.xyxy),
            }
            for d in (yolo_frame.detections if yolo_frame else ())
        ]
        active_tags = evaluate_yolo_tag_conditions(det_dicts, tag_rows)
        yolo_state = YoloState(
            frame_id=yolo_frame.frame_id if yolo_frame else None,
            inference_ms=yolo_frame.inference_ms if yolo_frame else None,
            detections=[DetectionModel(**row) for row in det_dicts],
            classes=_yolo_class_models(
                yolo_svc.model_path
                if isinstance(yolo_svc, YoloDetectionService)
                else settings.yolo_model_path
            ),
            targets=_yolo_target_models(yolo_svc if isinstance(yolo_svc, YoloDetectionService) else None),
            tag_conditions=tag_models,
            active_tags=active_tags,
            confidence_threshold=(
                yolo_svc.confidence if isinstance(yolo_svc, YoloDetectionService) else None
            ),
            ready=isinstance(yolo_svc, YoloDetectionService) and yolo_svc.ready,
            error=getattr(yolo_svc, "last_error", None) if yolo_svc else None,
        )

        live_svc = state.get("openvla_live")
        manual_snap = state.get("openvla_manual")
        live_snap = live_svc.latest() if isinstance(live_svc, OpenVLALiveWorker) else None
        active_snap = (
            manual_snap
            if isinstance(manual_snap, OpenVLALiveSnapshot)
            else live_snap
        )
        openvla_state = _snap_to_openvla_state(
            active_snap if isinstance(active_snap, OpenVLALiveSnapshot) else None,
            platform=platform,
            live_svc=live_svc,
            engine_meta=_openvla_engine_meta(settings, state),
            status="ok" if active_snap else "idle",
        )

        lerobot_svc = state.get("lerobot")
        lerobot_snap = lerobot_svc.latest() if lerobot_svc is not None else None
        lerobot_note = None
        if platform.lerobot_act_placeholder and not lerobot_snap:
            lerobot_note = (
                "LeRobot ACT policy is not configured yet. "
                "Set LEROBOT_STATE_FILE when the trained model is available."
            )
        lerobot_state = LerobotState(
            timestamp=lerobot_snap.timestamp if lerobot_snap else None,
            action_chunk=[list(row) for row in lerobot_snap.action_chunk]
            if lerobot_snap
            else [],
            source=lerobot_snap.source if lerobot_snap else None,
            ready=lerobot_snap is not None,
            note=lerobot_note,
        )

        planner = build_planner_snapshot(
            openvla_phase_description=openvla_state.phase_description,
            openvla_phase_index=openvla_state.phase_index,
            joint_error_sum=openvla_state.joint_error_sum,
            error_trend=openvla_state.error_trend,
            detections=yolo_frame.detections if yolo_frame else None,
        )
        planner_state = PlannerState(
            phase_description=planner.phase_description,
            phase_index=planner.phase_index,
            joint_error_sum=planner.joint_error_sum,
            error_trend=planner.error_trend,
            approaching_target=planner.approaching_target,
            visible_labels=list(planner.visible_labels),
            checks=[
                PlannerCheckModel(
                    id=c.id,
                    label=c.label,
                    ok=c.ok,
                    detail=c.detail,
                )
                for c in planner.checks
            ],
            phases_file=str(phase_store.path),
        )

        return LiveStateResponse(
            server_time=time.time(),
            camera_id=head_id,
            frame_id=frame_id,
            studio_raw_preview=bool(state.get("studio_raw_preview")),
            yolo=yolo_state,
            openvla=openvla_state,
            lerobot=lerobot_state,
            planner=planner_state,
        )

    def _openvla_engine() -> OpenVLAEngine:
        engine = state.get("openvla")
        if not isinstance(engine, OpenVLAEngine):
            raise HTTPException(status_code=503, detail="OpenVLA engine unavailable")
        return engine

    @app.get("/api/openvla/models", response_model=OpenVLAModelsResponse)
    def openvla_models() -> OpenVLAModelsResponse:
        versions = [m["id"] for m in settings.list_model_versions()]
        latest = resolve_latest_model_version(settings)
        return OpenVLAModelsResponse(
            latest_version=latest,
            model_root=str(settings.openvla_root),
            versions=versions,
        )

    @app.get(
        "/api/openvla/instruction-history",
        response_model=OpenVLAInstructionHistoryResponse,
    )
    def openvla_instruction_history() -> OpenVLAInstructionHistoryResponse:
        return _instruction_history_response()

    @app.delete(
        "/api/openvla/instruction-history/{entry_id}",
        response_model=OpenVLAInstructionHistoryResponse,
    )
    def openvla_instruction_history_delete(
        entry_id: str,
    ) -> OpenVLAInstructionHistoryResponse:
        delete_instruction(_instruction_history_file(), entry_id)
        return _instruction_history_response()

    def _openvla_hand_table(profile: str) -> dict | None:
        return hand_pose_map.table_for_file(phase_store.hand_pose_file_for(profile))

    def _resolve_openvla_phase_profile(name: str | None) -> str:
        requested = (name or "").strip()
        if not requested:
            return phase_store.active_profile()
        names = phase_store.profile_names()
        if requested not in names:
            raise HTTPException(status_code=400, detail=f"Unknown phase profile: {requested}")
        return requested

    @app.post("/api/openvla/profile-metrics", response_model=OpenVLAProfileMetricsResponse)
    def openvla_profile_metrics(
        body: OpenVLAProfileMetricsRequest,
    ) -> OpenVLAProfileMetricsResponse:
        profile = _resolve_openvla_phase_profile(body.phase_profile)
        phase_list = phase_store.phases_for(profile)
        if not phase_list:
            raise HTTPException(status_code=400, detail=f"Profile has no phases: {profile}")
        predicted = resolve_predicted_phase(
            body.phase_index,
            phase_list,
            phase_text=body.phase_text,
        )
        phase_index = (predicted or {}).get("index", body.phase_index)
        hand_table = _openvla_hand_table(profile)
        joints = apply_hand_key_poses(
            dict(body.target_joints),
            list(body.predicted_action),
            hand_table,
        )
        target_match = compute_phase_pose_match(
            joints,
            phase_index,
            phase_list,
            pose_catalog,
            predicted_action=list(body.predicted_action),
            action_arm=settings.action_arm,
            hand_table=hand_table,
        )
        return OpenVLAProfileMetricsResponse(
            predicted_phase=(
                PredictedPhaseModel(**predicted) if predicted else None
            ),
            target_phase_pose=(
                PhaseMatchModel(**target_match) if target_match else None
            ),
            target_joints=joints,
        )

    @app.post("/api/openvla/infer", response_model=OpenVLAInferResponse)
    def openvla_infer(body: OpenVLAInferRequest) -> OpenVLAInferResponse:
        head_id = _head_camera_id()
        if not head_id:
            raise HTTPException(status_code=503, detail="No camera configured")

        use_real = not settings.mock
        if use_real:
            if not settings.openvla_enabled:
                raise HTTPException(
                    status_code=503,
                    detail="OpenVLA 비활성화됨 (config openvla.enabled)",
                )
            deps_err = openvla_deps_error()
            if deps_err:
                raise HTTPException(status_code=503, detail=deps_err)
            load_status = str(state.get("openvla_load_status") or "idle")
            if load_status == "disabled":
                raise HTTPException(
                    status_code=503,
                    detail="OpenVLA 비활성화됨 (config openvla.enabled)",
                )
            if load_status == "loading":
                raise HTTPException(status_code=503, detail="openVLA model is still loading")
            if load_status == "error":
                err = state.get("openvla_load_error") or "OpenVLA load failed"
                raise HTTPException(status_code=503, detail=str(err))

        cam_svc = camera()
        snapshot = cam_svc.latest_frame(head_id, wait=True)  # type: ignore[union-attr]
        if snapshot is None:
            raise HTTPException(status_code=503, detail="No camera frame available")

        model_version = resolve_latest_model_version(settings)
        versions = {m["id"] for m in settings.list_model_versions()}
        if model_version not in versions and use_real:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_version!r} not under {settings.openvla_root}",
            )

        instruction = (body.instruction or platform.openvla_live_instruction).strip()
        if not instruction:
            raise HTTPException(status_code=400, detail="Instruction is empty")
        phase_profile = _resolve_openvla_phase_profile(body.phase_profile)

        engine = _openvla_engine()
        if use_real and not engine.ready:
            raise HTTPException(status_code=503, detail="OpenVLA engine is not ready")
        yolo_svc = state.get("yolo")
        if isinstance(yolo_svc, YoloDetectionService) and hasattr(yolo_svc, "pause"):
            yolo_svc.pause()
        try:
            with MPS_INFERENCE_LOCK:
                snap = run_openvla_on_snapshot(
                    engine,
                    snapshot,
                    model_version=model_version,
                    unnorm_key=settings.default_unnorm_key,
                    instruction=instruction,
                    get_current_joints=_current_joints,
                    joint_names=list(ALL_MODEL_JOINTS),
                    phases=lambda: phase_store.phases_for(phase_profile),
                    action_arm=settings.action_arm,
                    pose_catalog=pose_catalog,
                    hand_table=_openvla_hand_table(phase_profile),
                )
        except Exception as exc:
            log.exception("OpenVLA manual inference failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        state["openvla_manual"] = snap
        try:
            record_instruction(_instruction_history_file(), instruction)
        except OSError as exc:
            log.warning("Could not save instruction history: %s", exc)
        ov = _snap_to_openvla_state(
            snap,
            platform=platform,
            live_svc=None,
            engine_meta=_openvla_engine_meta(settings, state),
            status="ok",
        )
        return OpenVLAInferResponse(
            ok=True,
            message=f"Inference complete ({model_version})",
            openvla=ov,
        )

    @app.get("/api/dev/profile", response_model=DevProfileResponse)
    def dev_profile() -> DevProfileResponse:
        paths = dev_profile_paths(settings, platform)
        camera_source = "mock frames" if settings.mock else "robot GDK"
        cfg = resolve_config_path()
        engine = state.get("openvla")
        loaded_version: str | None = None
        if isinstance(engine, OpenVLAEngine):
            loaded_version = engine.loaded_version
        return DevProfileResponse(
            mode=_runtime_mode(settings, platform),
            config_file=str(cfg) if cfg else None,
            openvla_device=getattr(engine, "device", None) or settings.openvla_device,
            yolo_device=settings.yolo_device,
            camera_source=camera_source,
            yolo_model=paths["yolo_model"],
            openvla_root=paths["openvla_root"],
            openvla_model_dir=paths["openvla_model_dir"],
            openvla_version=paths["openvla_version"],
            openvla_loaded_version=loaded_version,
            lerobot=paths["lerobot"],
            instruction=platform.openvla_live_instruction,
        )

    @app.get("/api/joints", response_model=JointsResponse)
    def joints() -> JointsResponse:
        current = _current_joints()
        sample_names = list(ALL_MODEL_JOINTS[:8])
        sample = [
            JointSample(name=name, value=float(current.get(name, 0.0)))
            for name in sample_names
        ]
        return JointsResponse(positions=current, sample=sample)

    def _ros_runtime():
        ros_rt = state.get("ros")
        if ros_rt is None:
            raise HTTPException(status_code=503, detail="Robot runtime unavailable")
        return ros_rt

    app.include_router(register_lerobot_policy_routes())

    register_planner_routes(
        app,
        settings=settings,
        state=state,
        phase_store=phase_store,
        task_rules_store=task_rules_store,
        pose_catalog=pose_catalog,
        current_joints=_current_joints,
        ros=_ros_runtime,
    )

    return app
