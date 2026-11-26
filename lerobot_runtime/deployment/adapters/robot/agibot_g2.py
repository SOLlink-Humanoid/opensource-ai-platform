#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agibot G2 GDK adapter.

G2 publishes no ROS topics. Joint states, camera frames, and servo commands all
go through the vendor SDK (agibot_gdk) over the robot's local shared memory, so
a policy using this adapter must run onboard after sourcing
/home/agi/app/env.sh.

Camera frames are not owned by this adapter's polling loop; the generator in
main/general_policy_action_generator_agibot_g2.py drives them and calls the
read_camera_* helpers here so that every GDK call stays in one file.
"""

import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .base import RobotComponent, RobotIOAdapter, component_key, topic_enabled


def _payload_empty(data: Any) -> bool:
    """GDK Image.data is a numpy array; `if not data` raises a truth-value error."""
    if data is None:
        return True
    try:
        return len(data) == 0
    except TypeError:
        return False


class AgibotG2Adapter(RobotIOAdapter):
    """Agibot G2 GDK state polling and joint servo adapter."""

    adapter_name = "agibot_g2"

    # Some older GDK builds omitted joint names. Keep their documented order
    # only as a fallback; current builds are parsed by name.
    _STATE_SLICES: Tuple[Tuple[RobotComponent, int, int], ...] = (
        (RobotComponent.WAIST, 0, 5),
        (RobotComponent.HEAD, 5, 8),
        (RobotComponent.LEFT_ARM, 8, 15),
        (RobotComponent.RIGHT_ARM, 15, 22),
    )
    _BODY_JOINT_COUNT = 22
    _POSITION_KEYS = ("motor_position", "position", "pos", "q")
    _ERROR_LOG_PERIOD_S = 5.0

    def __init__(self, owner):
        super().__init__(owner)
        gdk_cfg = owner.robot_config.get("gdk", {})
        self._state_poll_hz = float(gdk_cfg.get("state_poll_hz", 50.0))
        self._command_hz = float(gdk_cfg.get("command_hz", 100.0))
        self._control_period = float(gdk_cfg.get("control_period", 0.020))
        self._settle_sec = float(gdk_cfg.get("settle_sec", 2.0))
        self._camera_warmup_sec = float(gdk_cfg.get("camera_warmup_sec", 2.0))
        self._camera_timeout_ms = float(gdk_cfg.get("camera_timeout_ms", 1000.0))
        self._limit_margin = float(gdk_cfg.get("joint_limit_margin_rad", 0.03))
        self._state_stale_timeout = float(gdk_cfg.get("state_stale_timeout_sec", 0.1))
        self._target_timeout = float(gdk_cfg.get("target_timeout_sec", 0.4))
        self._direct_targets = bool(gdk_cfg.get("direct_targets", False))
        self._arm_lead_base = float(gdk_cfg.get("arm_lead_rad", 0.04))
        self._waist_lead_base = float(gdk_cfg.get("waist_lead_rad", 0.04))
        self._head_velocity_base = float(gdk_cfg.get("head_velocity_rad_s", 3.0))
        self._finger_velocity_base = float(gdk_cfg.get("finger_velocity_rad_s", 8.0))
        self._arm_lead = self._arm_lead_base
        self._waist_lead = self._waist_lead_base
        self._head_velocity = self._head_velocity_base
        self._finger_velocity = self._finger_velocity_base
        self._enable_low_latency = bool(gdk_cfg.get("enable_low_latency", False))

        self._gdk = None
        self._robot = None
        self._camera = None
        self._camera_types: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._target_lock = threading.Lock()
        self._stop = threading.Event()
        self._state_thread: Optional[threading.Thread] = None
        self._command_thread: Optional[threading.Thread] = None
        self._limits = self._load_limits()
        self._latest_targets: Dict[RobotComponent, Tuple[np.ndarray, float]] = {}
        self._timeout_holds: Dict[RobotComponent, np.ndarray] = {}
        self._last_state_received_at = 0.0
        self._component_state_received_at: Dict[RobotComponent, float] = {
            component: 0.0
            for component, _, _ in self._STATE_SLICES
        }
        self._command_fault: Optional[str] = None
        self._ordered_state_fallback_warned = False
        self._poll_hands = topic_enabled(
            owner.robot_config["topics"]["states"].get("hand")
        )
        self._hand_commands_enabled = self._poll_hands
        self._hand_sent: Dict[RobotComponent, np.ndarray] = {}
        self._last_error_log: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_message_types(self):
        try:
            import agibot_gdk  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "agibot_gdk is required for the agibotG2 adapter. "
                "Source /home/agi/app/env.sh on the robot before running."
            ) from exc

        result = agibot_gdk.gdk_init()
        if result != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError(f"gdk_init failed: {result}")
        self._gdk = agibot_gdk
        self._robot = agibot_gdk.Robot()
        time.sleep(self._settle_sec)
        self.owner.get_logger().info(
            f"GDK ready: {getattr(agibot_gdk, '__file__', '?')} "
            f"direct_targets={self._direct_targets}"
        )

    def apply_motion_speed(self, arm_lead_rad: float) -> None:
        """Set GDK arm slew (rad per 100Hz tick); distance-independent cap in _follow_*."""
        lead = float(arm_lead_rad)
        if not np.isfinite(lead):
            lead = self._arm_lead_base
        lead = max(0.002, min(0.05, lead))
        ratio = lead / self._arm_lead_base if self._arm_lead_base > 1e-9 else 1.0
        self._arm_lead = lead
        self._waist_lead = self._waist_lead_base * ratio
        self._head_velocity = self._head_velocity_base * ratio
        self._finger_velocity = self._finger_velocity_base * ratio
        self.owner.get_logger().info(
            f"GDK arm_lead={self._arm_lead:.3f} rad/tick "
            f"(100Hz, uniform toward target) "
            f"finger_vel={self._finger_velocity:.2f} rad/s"
        )

    def setup_ros_interfaces(self):
        self._validate_joint_counts()
        self.owner.joint_state.setdefault(RobotComponent.WAIST, None)
        self._stop.clear()
        self._state_thread = threading.Thread(
            target=self._state_loop,
            daemon=True,
            name="agibot_g2_state_poll",
        )
        self._command_thread = threading.Thread(
            target=self._command_loop,
            daemon=True,
            name="agibot_g2_servo",
        )
        self._state_thread.start()
        self._command_thread.start()
        self._subscribe_gripper_states()
        self.owner.get_logger().info(
            f"Polling GDK joint states at {self._state_poll_hz:.0f}Hz "
            f"(hands={'on' if self._poll_hands else 'off'})"
        )
        self.owner.get_logger().info(
            f"GDK joint_servo_control loop at {self._command_hz:.0f}Hz "
            f"(period={self._control_period:.3f}s, low_latency={self._enable_low_latency})"
        )

    def _subscribe_gripper_states(self):
        """Gripper open/close bits are ROS Int32 topics, not a GDK call."""
        states = self.owner.robot_config["topics"]["states"]
        sub_qos = self.owner.make_qos(self.owner.robot_config.get("qos", {}).get("subscriber"))
        for gripper in (RobotComponent.LEFT_GRIPPER, RobotComponent.RIGHT_GRIPPER):
            cfg = self.owner.gripper_config(gripper)
            state_key = cfg.get("state_topic_key")
            if cfg.get("state_source") != "scalar_topic" or not topic_enabled(states.get(state_key)):
                continue
            topic = states[state_key]["topic"]
            self.owner.ros_subscriptions.append(
                self.owner.create_subscription(
                    self.owner.Int32Msg,
                    topic,
                    self.owner.make_scalar_gripper_callback(gripper),
                    sub_qos,
                )
            )
            self.owner.get_logger().info(
                f"Subscribed {gripper.name}: {topic} (std_msgs/Int32)"
            )

    def close(self):
        self._stop.set()
        threads = (self._state_thread, self._command_thread)
        self._state_thread = None
        self._command_thread = None
        for thread in threads:
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=1.0)
        with self._lock:
            if self._camera is not None:
                try:
                    self._camera.close_camera()
                except Exception as exc:
                    self.owner.get_logger().warning(f"close_camera failed: {exc}")
                self._camera = None
            if self._gdk is not None:
                try:
                    self._gdk.gdk_release()
                except Exception as exc:
                    self.owner.get_logger().warning(f"gdk_release failed: {exc}")
            self._gdk = None
            self._robot = None

    def _load_limits(self) -> Dict[RobotComponent, Optional[np.ndarray]]:
        limits: Dict[RobotComponent, Optional[np.ndarray]] = {}
        for component in (
            RobotComponent.HEAD,
            RobotComponent.WAIST,
            RobotComponent.LEFT_ARM,
            RobotComponent.RIGHT_ARM,
            RobotComponent.LEFT_HAND,
            RobotComponent.RIGHT_HAND,
        ):
            raw = self.owner.robot_config["joints"][component_key(component)].get("limits")
            limits[component] = np.asarray(raw, dtype=np.float64) if raw else None
        return limits

    def _validate_joint_counts(self):
        robot_config = self.owner.robot_config
        for component, start, end in self._STATE_SLICES:
            key = component_key(component)
            joint_cfg = robot_config["joints"].get(key)
            if joint_cfg is None:
                raise ValueError(f"robot config missing joints.{key}")
            configured = len(joint_cfg.get("order", joint_cfg.get("ids", [])))
            if configured != end - start:
                raise ValueError(
                    f"agibotG2 joints.{key} defines {configured} joints but "
                    f"get_joint_states() supplies {end - start}"
                )

    # ------------------------------------------------------------------
    # State polling
    # ------------------------------------------------------------------

    def _state_loop(self):
        period = 1.0 / max(1.0, self._state_poll_hz)
        last_completed_at = 0.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._poll_joint_states()
                if self._poll_hands:
                    self._poll_end_state()
            except Exception as exc:
                self._log_throttled("state_poll", f"GDK state poll failed: {exc}")
            completed_at = time.monotonic()
            if (
                last_completed_at > 0
                and completed_at - last_completed_at > self._state_stale_timeout
            ):
                self._log_throttled(
                    "state_poll_gap",
                    "GDK state poll completion gap "
                    f"{completed_at - last_completed_at:.3f}s > "
                    f"{self._state_stale_timeout:.3f}s",
                )
            last_completed_at = completed_at
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                self._stop.wait(remaining)

    def _poll_joint_states(self):
        with self._lock:
            if self._robot is None:
                return
            message = self._robot.get_joint_states()
        samples = self._joint_samples(message)
        ordered = self._ordered_joint_positions(message)
        if not samples and (ordered is None or len(ordered) < self._BODY_JOINT_COUNT):
            self._log_throttled(
                "joint_states",
                "GDK get_joint_states() returned "
                f"{0 if ordered is None else len(ordered)} usable joints, "
                f"expected at least {self._BODY_JOINT_COUNT}",
            )
            return
        joint_state = self.owner.joint_state
        received_at = time.monotonic()
        complete = True
        for component, start, end in self._STATE_SLICES:
            names = self.owner.robot_config["joints"][component_key(component)]["order"]
            if all(name in samples for name in names):
                values = np.asarray([samples[name] for name in names], dtype=np.float64)
            elif not samples and ordered is not None and len(ordered) >= end:
                values = np.asarray(ordered[start:end], dtype=np.float64)
                if not self._ordered_state_fallback_warned:
                    self._ordered_state_fallback_warned = True
                    self.owner.get_logger().warning(
                        "GDK joint states have incomplete names; using legacy positional order"
                    )
            else:
                complete = False
                self._log_throttled(
                    f"joint_names:{component.name}",
                    f"GDK joint states missing names for {component.name}: "
                    f"{[name for name in names if name not in samples]}",
                )
                continue
            joint_state[component] = values
            self._component_state_received_at[component] = received_at
        if complete:
            self._last_state_received_at = received_at

    def stale_state_components(
        self,
        components: Sequence[RobotComponent],
        now: Optional[float] = None,
    ) -> Dict[RobotComponent, float]:
        """Return controlled components whose last measured state is too old."""
        checked_at = time.monotonic() if now is None else float(now)
        stale: Dict[RobotComponent, float] = {}
        for component in components:
            if component not in self._component_state_received_at:
                continue
            received_at = self._component_state_received_at[component]
            age = float("inf") if received_at <= 0 else checked_at - received_at
            if age > self._state_stale_timeout:
                stale[component] = age
        return stale

    @staticmethod
    def _format_stale_components(stale: Dict[RobotComponent, float]) -> str:
        return ", ".join(
            f"{component.name}={'never' if not np.isfinite(age) else f'{age:.3f}s'}"
            for component, age in stale.items()
        )

    def _poll_end_state(self):
        with self._lock:
            if self._robot is None:
                return
            message = self._robot.get_end_state()
        if message is None:
            return
        joints = self.owner.robot_config["joints"]
        for block_key, component in (
            ("left_end_state", RobotComponent.LEFT_HAND),
            ("right_end_state", RobotComponent.RIGHT_HAND),
        ):
            block = (
                message.get(block_key)
                if isinstance(message, dict)
                else getattr(message, block_key, None)
            )
            order = joints[component_key(component)]["order"]
            values = self._end_state_positions(block, order)
            if values is not None:
                self.owner.joint_state[component] = values

    @classmethod
    def _joint_samples(cls, message: Any) -> Dict[str, float]:
        if isinstance(message, dict):
            states = message.get("states") or message.get("joint_states")
        else:
            states = getattr(message, "states", None)
        if not states:
            return {}
        samples: Dict[str, float] = {}
        for state in states:
            name = state.get("name") if isinstance(state, dict) else getattr(state, "name", None)
            position = cls._pick_position(state)
            if name is not None and position is not None:
                samples[str(name).strip()] = position
        return samples

    @classmethod
    def _ordered_joint_positions(cls, message: Any) -> Optional[np.ndarray]:
        if isinstance(message, dict):
            states = message.get("states") or message.get("joint_states")
        else:
            states = getattr(message, "states", None)
        if not states:
            return None
        values = []
        for state in states:
            position = cls._pick_position(state)
            if position is None:
                return None
            values.append(position)
        return np.asarray(values, dtype=np.float64)

    @classmethod
    def _end_state_positions(
        cls, block: Any, order: Sequence[str]
    ) -> Optional[np.ndarray]:
        if block is None:
            return None
        if isinstance(block, dict):
            names = block.get("names") or []
            items = block.get("end_states") or block.get("states") or []
        else:
            names = getattr(block, "names", None) or []
            items = getattr(block, "end_states", None) or getattr(block, "states", None) or []
        if not items:
            return None
        # end_states entries carry an id, not a name; names is a parallel list.
        by_name: Dict[str, float] = {}
        for index, item in enumerate(items):
            position = cls._pick_position(item)
            if position is None or index >= len(names):
                continue
            name = str(names[index]).strip()
            if name:
                by_name[name] = position
        try:
            return np.asarray([by_name[name] for name in order], dtype=np.float64)
        except KeyError:
            return None

    @classmethod
    def _pick_position(cls, state: Any) -> Optional[float]:
        for key in cls._POSITION_KEYS:
            value = state.get(key) if isinstance(state, dict) else getattr(state, key, None)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def publish_targets(
        self,
        targets: Dict[RobotComponent, np.ndarray],
        velocity: Optional[float] = None,
    ):
        del velocity
        if self._command_fault is not None:
            raise RuntimeError(
                f"agibotG2 servo fault is latched; restart required: {self._command_fault}"
            )
        now = time.monotonic()
        updates = {}
        for component in (
            RobotComponent.HEAD,
            RobotComponent.WAIST,
            RobotComponent.LEFT_ARM,
            RobotComponent.RIGHT_ARM,
            RobotComponent.LEFT_HAND,
            RobotComponent.RIGHT_HAND,
        ):
            values = self._servo_vector(targets, component)
            if values is not None:
                updates[component] = (np.asarray(values, dtype=np.float64), now)
        if not updates:
            return
        with self._target_lock:
            self._latest_targets.update(updates)
            for component in updates:
                self._timeout_holds.pop(component, None)

    def _servo_vector(
        self,
        targets: Dict[RobotComponent, np.ndarray],
        component: RobotComponent,
    ) -> Optional[List[float]]:
        values = targets.get(component)
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"agibotG2 {component.name} target contains non-finite values")
        array = self._clip_to_limits(component, array)
        return [float(value) for value in array]

    def _command_loop(self):
        period = 1.0 / max(1.0, self._command_hz)
        deadline = time.monotonic()
        while not self._stop.is_set():
            deadline += period
            try:
                self._command_tick()
            except Exception as exc:
                self._latch_command_fault(str(exc))
            wait = deadline - time.monotonic()
            if wait > 0:
                self._stop.wait(wait)
            else:
                deadline = time.monotonic()

    def _motion_lead_for_error(self, base_lead: float, max_joint_error: float) -> float:
        """When the move is smaller than motion_speed (rad/tick), slew at half speed."""
        base = float(base_lead)
        if not np.isfinite(base) or base <= 0.0:
            return base
        err = float(max_joint_error)
        if not np.isfinite(err) or err <= 1e-12:
            return base
        if err < base:
            return base * 0.5
        return base

    def _keepalive_stale_target(
        self,
        component: RobotComponent,
        target: np.ndarray,
        received_at: float,
        now: float,
    ) -> float:
        """After target_timeout, re-stamp the same goal (policy resend equivalent)."""
        if now - received_at <= self._target_timeout:
            return received_at
        stamp = now
        with self._target_lock:
            self._latest_targets[component] = (
                np.asarray(target, dtype=np.float64).copy(),
                stamp,
            )
            self._timeout_holds.pop(component, None)
        return stamp

    def _command_tick(self):
        if self._command_fault is not None:
            return
        with self._target_lock:
            targets = {
                component: (values.copy(), received_at)
                for component, (values, received_at) in self._latest_targets.items()
            }
        if not targets:
            return

        controlled_components = tuple(targets)
        stale = self.stale_state_components(controlled_components)
        if stale:
            try:
                self._poll_joint_states()
            except Exception as exc:
                raise RuntimeError(
                    "stale measured state; synchronous refresh failed: "
                    f"{self._format_stale_components(stale)}; error={exc}"
                ) from exc
            stale = self.stale_state_components(controlled_components)
            if stale:
                raise RuntimeError(
                    "stale measured state "
                    f"{self._format_stale_components(stale)} > "
                    f"{self._state_stale_timeout:.3f}s"
                )

        now = time.monotonic()
        hands = (RobotComponent.LEFT_HAND, RobotComponent.RIGHT_HAND)
        body_targets = {
            component: item
            for component, item in targets.items()
            if component not in hands
        }
        measured = self._measured_components()
        candidates: Dict[RobotComponent, np.ndarray] = {}
        fresh: Dict[RobotComponent, np.ndarray] = {}
        for component, (target, received_at) in body_targets.items():
            actual = measured.get(component)
            if actual is None:
                continue
            received_at = self._keepalive_stale_target(
                component, target, received_at, now
            )
            fresh[component] = target

        arms = (RobotComponent.LEFT_ARM, RobotComponent.RIGHT_ARM)
        if all(component in fresh and component in measured for component in arms):
            arm_candidates = self._follow_coupled(
                arms,
                fresh,
                measured,
                self._arm_lead,
            )
            candidates.update(arm_candidates)
            fresh.pop(RobotComponent.LEFT_ARM)
            fresh.pop(RobotComponent.RIGHT_ARM)

        for component, target in fresh.items():
            actual = measured[component]
            if component in arms:
                step = self._arm_lead
            elif component == RobotComponent.WAIST:
                step = self._waist_lead
            else:
                step = self._head_velocity / max(1.0, self._command_hz)
            candidates[component] = self._follow_one(component, actual, target, step)

        hand_candidates = self._hand_candidates(targets, now)
        candidates.update(hand_candidates)

        if not candidates:
            return
        self._send_candidates(candidates)

    def _measured_components(self) -> Dict[RobotComponent, np.ndarray]:
        measured = {}
        for component in (
            RobotComponent.HEAD,
            RobotComponent.WAIST,
            RobotComponent.LEFT_ARM,
            RobotComponent.RIGHT_ARM,
        ):
            values = self.owner.joint_state.get(component)
            if values is not None:
                measured[component] = np.asarray(values, dtype=np.float64).copy()
        return measured

    def _follow_one(
        self,
        component: RobotComponent,
        actual: np.ndarray,
        target: np.ndarray,
        max_lead: float,
    ) -> np.ndarray:
        if self._direct_targets:
            return self._clip_to_limits(
                component, np.asarray(target, dtype=np.float64)
            )
        delta = target - actual
        max_abs = float(np.max(np.abs(delta))) if len(delta) else 0.0
        lead = self._motion_lead_for_error(max_lead, max_abs)
        scale = 1.0 if max_abs <= lead or max_abs <= 1e-12 else lead / max_abs
        return self._clip_to_limits(component, actual + delta * scale)

    def _follow_coupled(
        self,
        components: Sequence[RobotComponent],
        targets: Dict[RobotComponent, np.ndarray],
        measured: Dict[RobotComponent, np.ndarray],
        max_lead: float,
    ) -> Dict[RobotComponent, np.ndarray]:
        if self._direct_targets:
            return {
                component: self._clip_to_limits(
                    component,
                    np.asarray(targets[component], dtype=np.float64),
                )
                for component in components
            }
        max_abs = max(
            float(np.max(np.abs(targets[component] - measured[component])))
            for component in components
        )
        lead = self._motion_lead_for_error(max_lead, max_abs)
        scale = 1.0 if max_abs <= lead or max_abs <= 1e-12 else lead / max_abs
        return {
            component: self._clip_to_limits(
                component,
                measured[component] + (targets[component] - measured[component]) * scale,
            )
            for component in components
        }

    def _clip_to_limits(
        self,
        component: RobotComponent,
        values: np.ndarray,
    ) -> np.ndarray:
        limits = self._limits.get(component)
        if limits is None or len(limits) != len(values):
            return values
        names = self.owner.robot_config["joints"][component_key(component)]["order"]
        lower = limits[:, 0].copy()
        upper = limits[:, 1].copy()
        for index, name in enumerate(names):
            # The G2 waist home sits exactly on these two boundaries.
            if name != "idx02_body_joint2":
                lower[index] += self._limit_margin
            if name != "idx01_body_joint1":
                upper[index] -= self._limit_margin
        return np.clip(values, np.minimum(lower, upper), np.maximum(lower, upper))

    @staticmethod
    def _as_list(values: Optional[np.ndarray]) -> Optional[List[float]]:
        if values is None:
            return None
        return [float(value) for value in values]

    def _hand_key_pose_transition_sec(self) -> Optional[float]:
        policy = getattr(self.owner, "policy_config", None) or {}
        if not isinstance(policy, dict) or not policy.get("hand_key_poses"):
            return None
        try:
            sec = float(policy.get("hand_pose_transition_sec", 1.0))
        except (TypeError, ValueError):
            sec = 1.0
        if sec <= 0:
            return None
        return sec

    def _hand_slew_step(self, delta: np.ndarray) -> float:
        default = max(0.005, self._finger_velocity / max(1.0, self._command_hz))
        transition = self._hand_key_pose_transition_sec()
        if transition is None:
            return default
        if len(delta) == 0:
            return default
        max_delta = float(np.max(np.abs(delta)))
        if max_delta <= 1e-9:
            return default
        keyed = max_delta / (transition * max(1.0, self._command_hz))
        return max(0.005, keyed)

    def _hand_candidates(
        self,
        targets: Dict[RobotComponent, Tuple[np.ndarray, float]],
        now: float,
    ) -> Dict[RobotComponent, np.ndarray]:
        """Slew each finger from the last command, not from the measured lead."""
        if not self._hand_commands_enabled:
            return {}
        candidates: Dict[RobotComponent, np.ndarray] = {}
        for component in (RobotComponent.LEFT_HAND, RobotComponent.RIGHT_HAND):
            item = targets.get(component)
            if item is None:
                continue
            target, received_at = item
            received_at = self._keepalive_stale_target(
                component, target, received_at, now
            )
            if self._direct_targets:
                candidates[component] = self._clip_to_limits(
                    component, np.asarray(target, dtype=np.float64)
                )
                continue
            previous = self._hand_sent.get(component)
            if previous is None:
                measured = self.owner.joint_state.get(component)
                if measured is None:
                    continue
                previous = np.asarray(measured, dtype=np.float64).copy()
            delta = np.asarray(target, dtype=np.float64) - previous
            step = self._hand_slew_step(delta)
            delta = np.clip(delta, -step, step)
            candidates[component] = self._clip_to_limits(component, previous + delta)
        return candidates

    def _send_candidates(self, candidates: Dict[RobotComponent, np.ndarray]) -> None:
        hands = (RobotComponent.LEFT_HAND, RobotComponent.RIGHT_HAND)
        include_hands = any(component in candidates for component in hands)
        try:
            self._send_servo_frame(candidates, include_hands=include_hands)
        except RuntimeError as exc:
            if not include_hands:
                raise
            self._hand_commands_enabled = False
            self._hand_sent.clear()
            self.owner.get_logger().error(
                "Hand servo rejected; continuing body servo without hands: "
                f"{exc}"
            )
            self._send_servo_frame(candidates, include_hands=False)
            return
        if include_hands:
            for component in hands:
                values = candidates.get(component)
                if values is not None:
                    self._hand_sent[component] = values.copy()

    def _send_servo_frame(
        self,
        candidates: Dict[RobotComponent, np.ndarray],
        include_hands: bool,
    ):
        head = self._as_list(candidates.get(RobotComponent.HEAD))
        groups = [
            (RobotComponent.WAIST, candidates.get(RobotComponent.WAIST)),
            (RobotComponent.LEFT_ARM, candidates.get(RobotComponent.LEFT_ARM)),
            (RobotComponent.RIGHT_ARM, candidates.get(RobotComponent.RIGHT_ARM)),
        ]
        if include_hands:
            groups.extend(
                (
                    (RobotComponent.LEFT_HAND, candidates.get(RobotComponent.LEFT_HAND)),
                    (RobotComponent.RIGHT_HAND, candidates.get(RobotComponent.RIGHT_HAND)),
                )
            )
        body_or_hands = any(values is not None for _, values in groups)
        if self._robot is None:
            raise RuntimeError("GDK is not initialized")
        head_fn = getattr(self._robot, "move_head_joint_servo", None)
        joint_fn = getattr(self._robot, "joint_servo_control", None)
        if head is not None and head_fn is None:
            raise RuntimeError("move_head_joint_servo is missing; GDK 3.3+ is required")
        if body_or_hands and joint_fn is None:
            raise RuntimeError("joint_servo_control is missing; GDK 3.3+ is required")

        period = self._control_period
        # Do not call set_control_mode before servoing. CTRL_JOINT_POSITION makes
        # the servo calls succeed while the arms stay still.
        with self._lock:
            if head is not None:
                self._check_result("move_head_joint_servo", head_fn(head, period, False))
            names: List[str] = []
            positions: List[float] = []
            for component, values in groups:
                if values is None:
                    continue
                order = self.owner.robot_config["joints"][component_key(component)]["order"]
                names.extend(str(name).strip() for name in order)
                positions.extend(float(value) for value in values)
            if names:
                request = self._gdk.JointServoControlReq()
                request.control_period = period
                request.joint_names = names
                request.joint_positions = positions
                result = joint_fn(
                    request,
                    enable_low_latency=self._enable_low_latency,
                )
                code = int(result) if result is not None else 0
                if code != 0:
                    raise RuntimeError(f"joint_servo_control failed result={code}")

    @staticmethod
    def _check_result(call: str, result: Any):
        if result is None or int(result) != 0:
            raise RuntimeError(f"{call} failed result={result}")

    def _latch_command_fault(self, reason: str):
        if self._command_fault is not None:
            return
        self._command_fault = str(reason)
        self.owner.get_logger().error(
            f"GDK servo fault latched; commands stopped until restart: {reason}"
        )

    # ------------------------------------------------------------------
    # Cameras (driven by the agibot_g2 policy generator)
    # ------------------------------------------------------------------

    def prepare_cameras(self):
        """Open the shared GDK camera handle and wait for the streams to attach."""
        if self._gdk is None:
            raise RuntimeError("load_message_types() must run before camera access")
        if self._camera is not None:
            return
        self._camera = self._gdk.Camera()
        time.sleep(self._camera_warmup_sec)

    def read_camera_bgr(self, camera_id: str) -> Optional[np.ndarray]:
        image = self._latest_image(camera_id)
        return None if image is None else self._decode_bgr(image)

    def read_camera_depth_u16(self, camera_id: str) -> Optional[np.ndarray]:
        image = self._latest_image(camera_id)
        if image is None:
            return None
        data = getattr(image, "data", None)
        if _payload_empty(data):
            return None
        return np.frombuffer(data, np.uint16).reshape(
            (int(image.height), int(image.width))
        )

    def camera_info(self, camera_id: str) -> Optional[Dict[str, Any]]:
        if self._camera is None:
            return None
        camera_type = self._camera_type(camera_id)
        return {
            "shape": list(self._camera.get_image_shape(camera_type)),
            "fps": float(self._camera.get_image_fps(camera_type)),
        }

    def _camera_type(self, camera_id: str):
        cached = self._camera_types.get(camera_id)
        if cached is not None:
            return cached
        camera_type_cls = getattr(self._gdk, "CameraType", None)
        if camera_type_cls is None:
            raise RuntimeError("agibot_gdk.CameraType is missing")
        resolved = getattr(camera_type_cls, camera_id, None)
        if resolved is None:
            available = [name for name in dir(camera_type_cls) if name.startswith("k")]
            raise ValueError(f"CameraType '{camera_id}' not found. available={available}")
        self._camera_types[camera_id] = resolved
        return resolved

    def _latest_image(self, camera_id: str):
        camera = self._camera
        if camera is None:
            raise RuntimeError("prepare_cameras() must run before reading frames")
        # get_latest_image requires the timeout argument; without it no frame arrives.
        return camera.get_latest_image(self._camera_type(camera_id), self._camera_timeout_ms)

    def _decode_bgr(self, image: Any) -> Optional[np.ndarray]:
        data = getattr(image, "data", None)
        if _payload_empty(data):
            return None
        encoding_cls = self._gdk.Encoding
        color_cls = self._gdk.ColorFormat
        encoding = image.encoding
        if encoding in (encoding_cls.JPEG, encoding_cls.PNG):
            return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if encoding != encoding_cls.UNCOMPRESSED:
            self._log_throttled("encoding", f"Unsupported GDK encoding: {encoding}")
            return None

        color_format = image.color_format
        height = int(image.height)
        width = int(image.width)
        if color_format == color_cls.RGB:
            frame = np.frombuffer(data, np.uint8).reshape((height, width, 3))
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if color_format == color_cls.BGR:
            return np.frombuffer(data, np.uint8).reshape((height, width, 3))
        if color_format == color_cls.GRAY8:
            frame = np.frombuffer(data, np.uint8).reshape((height, width))
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if color_format == color_cls.GRAY16:
            frame = np.frombuffer(data, np.uint16).reshape((height, width))
            return cv2.cvtColor((frame / 256).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self._log_throttled("color_format", f"Unsupported GDK color_format: {color_format}")
        return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_throttled(self, key: str, message: str):
        now = time.monotonic()
        last = self._last_error_log.get(key, 0.0)
        if now - last < self._ERROR_LOG_PERIOD_S:
            return
        self._last_error_log[key] = now
        self.owner.get_logger().warning(message)
