"""Joint feedback and motion. Mock mode stays in-process; robot mode uses GDK."""

from __future__ import annotations

import logging
import time

from shared.joint_names import ALL_MODEL_JOINTS

_log = logging.getLogger(__name__)


class MockRosRuntime:
    def __init__(self) -> None:
        self._positions: dict[str, float] = {name: 0.0 for name in ALL_MODEL_JOINTS}
        self.last_positions: dict[str, float] = dict(self._positions)
        self.ready = False
        self.last_error: str | None = None
        self._topic = "mock://joints"

    def start(self) -> None:
        self.ready = True
        self.last_positions = dict(self._positions)

    def shutdown(self) -> None:
        self.ready = False

    def get_positions(self, timeout_sec: float = 1.0) -> dict[str, float]:
        del timeout_sec
        self.last_positions = dict(self._positions)
        return dict(self._positions)

    def apply_joint_targets(
        self,
        target_joints: dict[str, float],
        *,
        speed: float,
        timeout_sec: float,
    ) -> tuple[int, str]:
        del speed, timeout_sec
        for name, value in target_joints.items():
            self._positions[str(name)] = float(value)
        self.last_positions = dict(self._positions)
        return 0, ""

    def stop_all_motion(
        self,
        *,
        speed: float,
        timeout_sec: float,
    ) -> tuple[int, str]:
        del timeout_sec
        return self.apply_joint_targets(
            dict(self._positions),
            speed=speed,
            timeout_sec=5.0,
        )


class GdkRosRuntime:
    """Read joint state and send joint targets through agibot_gdk.Robot."""

    def __init__(self) -> None:
        self._positions: dict[str, float] = {}
        self.last_positions: dict[str, float] = {}
        self.ready = False
        self.last_error: str | None = None

    def start(self) -> None:
        try:
            self.get_positions(timeout_sec=2.0)
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            _log.error("GDK joint state unavailable: %s", exc)
            return
        self.ready = True
        _log.info("GDK joint runtime ready (%d joints)", len(self._positions))

    def shutdown(self) -> None:
        self.ready = False

    def get_positions(self, timeout_sec: float = 1.0) -> dict[str, float]:
        del timeout_sec
        from .gdk_session import call_gdk, get_robot

        robot = get_robot()
        states = call_gdk(robot.get_joint_states)
        positions: dict[str, float] = {}
        for state in states.get("states") or []:
            name = state.get("name")
            if not name:
                continue
            value = state.get("position", state.get("motor_position", 0.0))
            try:
                positions[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
        if not positions:
            raise RuntimeError("GDK returned no joint positions")
        self._positions = positions
        self.last_positions = dict(positions)
        self.last_error = None
        return dict(positions)

    def apply_joint_targets(
        self,
        target_joints: dict[str, float],
        *,
        speed: float,
        timeout_sec: float,
    ) -> tuple[int, str]:
        del timeout_sec
        if not target_joints:
            raise ValueError("No joint targets")
        from .gdk_session import call_gdk, ensure_gdk, get_robot

        gdk = ensure_gdk()
        robot = get_robot()
        names = [str(name) for name in target_joints]
        positions = [float(target_joints[name]) for name in names]
        vel = max(0.01, float(speed))
        velocities = [vel] * len(names)
        try:
            current = self.get_positions(timeout_sec=0.5)
        except Exception:  # noqa: BLE001
            current = {}
        max_delta = 0.0
        for name, goal in zip(names, positions):
            cur = current.get(name)
            if cur is not None:
                max_delta = max(max_delta, abs(goal - float(cur)))
        life_time = max(0.35, min(8.0, max_delta / vel * 1.3 + 0.2))
        request = gdk.JointControlReq()
        request.life_time = float(life_time)
        request.joint_names = names
        request.joint_positions = positions
        request.joint_velocities = velocities
        started = time.perf_counter()
        call_gdk(lambda: robot.joint_control_request(request))
        _log.info(
            "Sent %d joint targets in %.1f ms",
            len(names),
            (time.perf_counter() - started) * 1000.0,
        )
        return 0, ""

    def stop_all_motion(
        self,
        *,
        speed: float,
        timeout_sec: float,
    ) -> tuple[int, str]:
        del timeout_sec
        current = self.get_positions(timeout_sec=1.0)
        return self.apply_joint_targets(current, speed=speed, timeout_sec=5.0)


def build_ros_runtime(mock: bool):
    if mock:
        return MockRosRuntime()
    return GdkRosRuntime()
