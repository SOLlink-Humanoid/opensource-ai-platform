"""Background joint servo toward OpenVLA / planner targets with cancel."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

_log = logging.getLogger(__name__)

_ARRIVE_EPS_RAD = 0.008
_SERVO_HZ = 12.0


def _max_joint_delta(
    current: dict[str, float],
    target: dict[str, float],
) -> float:
    delta = 0.0
    for name, goal in target.items():
        cur = current.get(name)
        if cur is None:
            continue
        delta = max(delta, abs(float(goal) - float(cur)))
    return delta


class JointMotionController:
    """Repeatedly sends joint targets until arrival or stop."""

    def __init__(self, get_runtime: Callable[[], Any]) -> None:
        self._get_runtime = get_runtime
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def moving(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start_move(self, target_joints: dict[str, float], *, speed: float) -> None:
        if not target_joints:
            raise ValueError("No joint targets")
        if speed <= 0:
            raise ValueError("Speed must be positive")
        with self._lock:
            self._halt_locked(wait=False)
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                args=(dict(target_joints), float(speed)),
                name="oap-joint-motion",
                daemon=True,
            )
            self._thread.start()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the current move thread exits. True when it has finished."""
        with self._lock:
            thread = self._thread
        if thread is None or not thread.is_alive():
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def stop(self, *, speed: float = 0.05) -> tuple[int, str]:
        with self._lock:
            self._halt_locked(wait=True)
            runtime = self._get_runtime()
            try:
                current = runtime.get_positions(timeout_sec=2.0)
            except Exception as exc:  # noqa: BLE001
                return 1, str(exc)
            return runtime.apply_joint_targets(
                current,
                speed=speed,
                timeout_sec=5.0,
            )

    def _halt_locked(self, *, wait: bool) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and wait:
            thread.join(timeout=4.0)
        self._thread = None

    def _run(self, targets: dict[str, float], speed: float) -> None:
        runtime = self._get_runtime()
        period = 1.0 / _SERVO_HZ
        deadline = time.monotonic() + 120.0
        while not self._stop.is_set() and time.monotonic() < deadline:
            try:
                current = runtime.get_positions(timeout_sec=1.0)
            except Exception as exc:  # noqa: BLE001
                _log.warning("Joint motion stopped: read failed (%s)", exc)
                break
            if _max_joint_delta(current, targets) <= _ARRIVE_EPS_RAD:
                break
            try:
                runtime.apply_joint_targets(
                    targets,
                    speed=speed,
                    timeout_sec=2.0,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning("Joint motion stopped: command failed (%s)", exc)
                break
            if self._stop.wait(period):
                break
