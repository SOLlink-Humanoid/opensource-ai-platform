"""Background OpenVLA inference over the shared head-camera frame cache."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from web.openvla.action_parser import (
    inferred_joints_by_part,
    parse_openvla_action,
    resolve_predicted_phase,
    sanitize_g2_phase_text,
)
from web.openvla.openvla_engine import OpenVLAEngine
from web.openvla.openvla_image import bgr_to_openvla_pil

_log = logging.getLogger(__name__)

ErrorTrend = Literal["decreasing", "increasing", "stopped", "unknown"]


@dataclass(frozen=True)
class OpenVLALiveSnapshot:
    camera_id: str
    frame_id: int
    completed_at: float
    inference_ms: float
    command: str
    model_version: str
    phase_text: str | None
    phase_index: int | None
    phase_description: str
    target_joints: dict[str, float]
    joint_error_sum: float
    error_trend: ErrorTrend
    predicted_action: list[float | str]
    predicted_phase: dict[str, Any] | None = None
    target_phase_pose: dict[str, Any] | None = None
    closest_pose: dict[str, Any] | None = None
    inferred_joints_by_part: dict[str, list[str]] | None = None


def _joint_error_sum(
    target: dict[str, float],
    current: dict[str, float],
    joint_names: list[str],
) -> float:
    total = 0.0
    for name in joint_names:
        if name not in target or name not in current:
            continue
        total += abs(float(target[name]) - float(current[name]))
    return total


class OpenVLALiveWorker:
    """Run OpenVLA on the newest frame after each inference (no queue)."""

    def __init__(
        self,
        camera_service: Any,
        engine: OpenVLAEngine,
        *,
        camera_id: str,
        model_version: str,
        unnorm_key: str,
        instruction: str,
        min_interval_sec: float,
        get_current_joints: Callable[[], dict[str, float]],
        joint_names: list[str],
        phases: Callable[[], list[dict[str, Any]]],
        action_arm: str,
        joint_stopped_epsilon: float,
    ) -> None:
        self._camera = camera_service
        self._engine = engine
        self._camera_id = camera_id
        self._model_version = model_version
        self._unnorm_key = unnorm_key
        self._instruction = instruction
        self._min_interval = max(0.05, float(min_interval_sec))
        self._get_joints = get_current_joints
        self._joint_names = joint_names
        self._phases = phases
        self._action_arm = action_arm
        self._stopped_eps = joint_stopped_epsilon
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: OpenVLALiveSnapshot | None = None
        self._last_error: str | None = None
        self._prev_error_sum: float | None = None
        self._prev_joints: dict[str, float] | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="openvla-live",
            daemon=True,
        )
        self._thread.start()
        _log.info(
            "OpenVLA live worker started (camera=%s, model=%s)",
            self._camera_id,
            self._model_version,
        )

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=8.0)
            self._thread = None

    def latest(self) -> OpenVLALiveSnapshot | None:
        with self._lock:
            return self._latest

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _next_frame(self, after_frame_id: int):
        wait_for_frame = getattr(self._camera, "wait_for_frame", None)
        if callable(wait_for_frame):
            return wait_for_frame(
                self._camera_id,
                after_frame_id=after_frame_id,
                timeout=0.5,
            )
        snapshot = self._camera.latest_frame(self._camera_id, wait=False)
        if snapshot is not None and snapshot.frame_id > after_frame_id:
            return snapshot
        self._stop.wait(0.02)
        return None

    def _error_trend(
        self,
        current: dict[str, float],
        error_sum: float,
    ) -> ErrorTrend:
        stopped = False
        if self._prev_joints is not None and current:
            max_delta = 0.0
            for name, value in current.items():
                prev = self._prev_joints.get(name)
                if prev is None:
                    continue
                max_delta = max(max_delta, abs(value - prev))
            stopped = max_delta < self._stopped_eps
        self._prev_joints = dict(current) if current else self._prev_joints

        if stopped:
            return "stopped"
        if self._prev_error_sum is None:
            return "unknown"
        if error_sum < self._prev_error_sum - 1e-6:
            return "decreasing"
        if error_sum > self._prev_error_sum + 1e-6:
            return "increasing"
        return "unknown"

    def _infer_once(self, snapshot) -> OpenVLALiveSnapshot:
        pil_image = bgr_to_openvla_pil(snapshot.bgr)
        started = time.perf_counter()
        inference = self._engine.predict(
            pil_image,
            self._instruction,
            version=self._model_version,
            unnorm_key=self._unnorm_key,
        )
        infer_ms = (time.perf_counter() - started) * 1000.0

        current = self._get_joints()
        phases = self._phases()
        target_map, phase_index, phase_label = parse_openvla_action(
            inference.joint_values,
            current,
            phase_text=inference.phase_text,
            phases=phases,
            action_arm=self._action_arm,
        )
        display_phase = sanitize_g2_phase_text(phase_label) or sanitize_g2_phase_text(
            inference.phase_text
        )
        predicted = resolve_predicted_phase(
            phase_index,
            phases,
            phase_text=display_phase,
        )
        predicted_action: list[float | str] = list(inference.joint_values)

        error_sum = _joint_error_sum(target_map, current, self._joint_names)
        trend = self._error_trend(current, error_sum)
        self._prev_error_sum = error_sum

        return OpenVLALiveSnapshot(
            camera_id=snapshot.camera_id,
            frame_id=snapshot.frame_id,
            completed_at=time.time(),
            inference_ms=infer_ms,
            command=self._instruction,
            model_version=self._model_version,
            phase_text=display_phase,
            phase_index=predicted.get("index") if predicted else phase_index,
            phase_description=(predicted or {}).get("description") or "",
            target_joints=target_map,
            joint_error_sum=error_sum,
            error_trend=trend,
            predicted_action=predicted_action,
            inferred_joints_by_part=inferred_joints_by_part(
                predicted_action,
                action_arm=self._action_arm,
            ),
        )

    def _run(self) -> None:
        last_frame_id = 0
        last_infer_at = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_infer_at < self._min_interval:
                self._stop.wait(0.02)
                continue
            snapshot = self._next_frame(last_frame_id)
            if snapshot is None:
                continue
            last_frame_id = snapshot.frame_id
            try:
                live = self._infer_once(snapshot)
                with self._lock:
                    self._latest = live
                    self._last_error = None
                last_infer_at = time.monotonic()
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                _log.exception("OpenVLA live inference failed")
                self._stop.wait(0.25)
