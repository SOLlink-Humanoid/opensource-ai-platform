#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General upper-body policy action generator.

This module keeps policy execution robot-agnostic. Robot-specific ROS topics,
message types, and joint order live in a robot JSON file. Model path,
observation layout, action layout, virtual gripper presets, reset pose, and
completion conditions live in a policy JSON file. Policy ``grippers`` overlay
matching keys in the robot JSON. ``external_data`` is policy-only.
"""

import argparse
import json
import os
import re
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

_DEPLOYMENT_DIR = Path(__file__).resolve().parent.parent
if str(_DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOYMENT_DIR))

from lerobot_action_log import (
    DEFAULT_LEROBOT_ACTION_LOG,
    WAIST_YAW_INDEX_IN_BODY,
    LerobotActionLog,
)

from adapters.robot import (
    CAMERA_COMPONENTS,
    CANONICAL_GRIPPER_KEYS,
    COMPONENT_TO_KEY,
    GRIPPER_TO_HAND,
    SUPPORTED_ADAPTERS,
    RobotComponent,
    component_from_name as _component_from_name,
    component_list as _component_list,
    component_names as _component_names,
    create_robot_adapter,
    has_component as _has,
    joint_count as _joint_count,
    resolve_external_data_config as _resolve_external_data_config,
    resolve_gripper_config as _resolve_gripper_config,
    topic_enabled as _topic_enabled,
    validate_robot_schema as _validate_robot_schema,
)

from depth_colormap import depth_to_turbo_rgb_from_config, unpack_camera_config

try:
    import rclpy
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
except ImportError as exc:  # Allows --dry-run in non-ROS environments.
    rclpy = None
    MultiThreadedExecutor = None
    MutuallyExclusiveCallbackGroup = None
    QoSProfile = None
    ReliabilityPolicy = None
    HistoryPolicy = None
    DurabilityPolicy = None
    _ROS_IMPORT_ERROR = exc

    class Node:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "ROS2 Python packages are required to instantiate "
                "GeneralPolicyActionGenerator. Use --dry-run for config checks."
            ) from _ROS_IMPORT_ERROR


def _oap_lerobot_state_path() -> Path:
    import os

    return Path(
        os.environ.get("OAP_LEROBOT_STATE_FILE", "/tmp/oap_lerobot_live.json")
    )


def _merge_oap_lerobot_state_file(patch: Dict[str, Any]) -> None:
    path = _oap_lerobot_state_path()
    try:
        data: Dict[str, Any] = {}
        if path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = raw
            except (json.JSONDecodeError, OSError):
                data = {}
        for key, value in patch.items():
            if value is None:
                continue
            data[key] = value
        data["timestamp"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError):
        pass


def _platform_data_dir() -> Path:
    for key in ("OAP_PLATFORM_ROOT", "OAP_VLA_ROOT"):
        raw = (os.environ.get(key) or "").strip()
        if raw:
            return Path(raw) / "data"
    return Path(__file__).resolve().parents[3] / "data"


def _right_gripper_chunk_trace_path() -> Path:
    return _platform_data_dir() / "right_gripper_raw_inference_trace.txt"


def _write_oap_lerobot_state(actions_list: List[List[float]]) -> None:
    """Expose latest action chunk for OAP UI (Task Planner chunk vs OpenVLA check)."""
    try:
        rows = [[float(x) for x in row] for row in actions_list]
        _merge_oap_lerobot_state_file({"action_chunk": rows})
    except (TypeError, ValueError):
        pass


class CompleteCondition(Enum):
    """Policy execution completion conditions."""

    TIME_BASED = 0
    JOINT_REACHED = 6
    CONDITION_1 = 1
    CONDITION_2 = 2
    CONDITION_3 = 3
    CONDITION_4 = 4
    CONDITION_5 = 5


@dataclass
class JointReachedCondition:
    """Position threshold plus settle-time completion condition."""

    target: Dict[str, Any]
    threshold: float = 0.05
    settle_time: float = 0.3


@dataclass(frozen=True)
class ArmActionPostprocessConfig:
    """Configuration for per-joint arm action filtering before publication."""

    enabled: bool = True
    min_cutoff: float = 1.0
    beta: float = 0.01
    derivative_cutoff: float = 1.0


@dataclass(frozen=True)
class ArmLiveGuardConfig:
    """Clamp arm targets so commands do not run far ahead of measured joint pose."""

    enabled: bool = True
    max_ahead_rad: float = 0.055
    max_behind_rad: float = 0.1


@dataclass(frozen=True)
class PublishConsumeConfig:
    """How the action queue is drained into robot commands."""

    mode: str = "on_settle"
    # Strict settle (chunk splice defer, diagnostics).
    waypoint_settle_rad: float = 0.02
    # Pop next queue item when max joint error to last command is below this
    # (before full stop — larger than waypoint_settle_rad).
    waypoint_approach_rad: float = 0.05
    publish_poll_s: float = 0.015
    max_waypoint_pop_hz: float = 25.0
    # Optional time-based pop if still not within approach_rad (0 = disabled).
    waypoint_dwell_timeout_s: float = 0.0
    # Re-send last waypoint without popping while waiting for approach (GDK target TTL).
    hold_republish_while_waiting: bool = True
    hold_republish_period_s: float = 0.2


@dataclass(frozen=True)
class PublishTargetBlendConfig:
    """Exponential blend between successive published action vectors (same Hz cap)."""

    enabled: bool = True
    alpha: float = 0.4


@dataclass(frozen=True)
class HandKeySelectionConfig:
    """Hand-key pose selection with confident bands and gray zones."""

    enabled: bool = True
    key_confidence_margin: float = 0.03
    scalar_ema_alpha: float = 0.22
    hold_publish_ticks_on_chunk_append: int = 10
    downgrade_to_min_key_max_scalar: float = 0.11
    # When gripper scalar is in a between-key gray band, waypoint approach/settle
    # uses this per-hand error instead of |live-cmd| (avoids blocking on slow hand keys).
    waypoint_gripper_gray_err_rad: float = 0.01
    # If |scalar - active_key| >= this, snap to nearest key (bypass hold / gray / downgrade).
    force_nearest_key_min_scalar_delta: float = 0.15


@dataclass(frozen=True)
class WaistChunkPrefixFixConfig:
    """Legacy waist-only chunk fix (used when action_chunk_prefix_fix is disabled)."""

    enabled: bool = False
    reference: str = "live"
    ramp_steps: int = 5
    only_when_chunk_increase: bool = True
    epsilon_rad: float = 0.0005


@dataclass(frozen=True)
class ActionChunkPrefixFixConfig:
    """Align a newly inferred chunk to the current command / live state."""

    enabled: bool = True
    on_append_only: bool = True
    reference: str = "last_command"
    max_joint_shift_rad: float = 1.5
    min_shift_rad: float = 0.002
    blend_steps: int = 0


@dataclass(frozen=True)
class ChunkQueueSpliceConfig:
    """Replace the action queue on re-inference; drop reverse prefixes in new chunks."""

    enabled: bool = True
    on_append_only: bool = True
    reverse_projection_rad: float = 0.02
    min_forward_norm: float = 0.005
    min_actions_after_trim: int = 1
    defer_until_settle: bool = True
    defer_finish_rad: float = 0.025
    gripper_weight: float = 0.15
    hand_weight: float = 0.25


class OneEuroFilter:
    """Stateful One Euro filter applied independently to each value dimension."""

    def __init__(
        self,
        sample_hz: float,
        min_cutoff: float,
        beta: float,
        derivative_cutoff: float,
    ):
        self.sample_hz = float(sample_hz)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self._previous_raw: Optional[np.ndarray] = None
        self._filtered: Optional[np.ndarray] = None
        self._filtered_derivative: Optional[np.ndarray] = None
        self._last_timestamp: Optional[float] = None

    def reset(self) -> None:
        self._previous_raw = None
        self._filtered = None
        self._filtered_derivative = None
        self._last_timestamp = None

    @staticmethod
    def _alpha(sample_period: float, cutoff: np.ndarray) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / sample_period)

    def filter(
        self,
        values: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> np.ndarray:
        current = np.asarray(values, dtype=np.float64)
        if current.ndim != 1:
            raise ValueError(
                f"One Euro arm action must be one-dimensional, got {current.shape}"
            )
        now = time.monotonic() if timestamp is None else float(timestamp)
        if self._filtered is None:
            self._previous_raw = current.copy()
            self._filtered = current.copy()
            self._filtered_derivative = np.zeros_like(current)
            self._last_timestamp = now
            return current.copy()
        if current.shape != self._filtered.shape:
            raise ValueError(
                "One Euro arm action shape changed from "
                f"{self._filtered.shape} to {current.shape}"
            )
        assert self._previous_raw is not None
        assert self._filtered_derivative is not None

        sample_period = (
            now - self._last_timestamp
            if self._last_timestamp is not None
            else 1.0 / self.sample_hz
        )
        if not np.isfinite(sample_period) or sample_period <= 0:
            sample_period = 1.0 / self.sample_hz

        derivative = (current - self._previous_raw) / sample_period
        derivative_alpha = self._alpha(
            sample_period,
            np.full_like(current, self.derivative_cutoff),
        )
        self._filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self._filtered_derivative
        )
        cutoff = self.min_cutoff + self.beta * np.abs(
            self._filtered_derivative
        )
        value_alpha = self._alpha(sample_period, cutoff)
        self._filtered = (
            value_alpha * current
            + (1.0 - value_alpha) * self._filtered
        )
        self._previous_raw = current.copy()
        self._last_timestamp = now
        return self._filtered.copy()


class StaleCameraFrameError(RuntimeError):
    """Raised when get_obs() would otherwise reuse a frozen camera frame."""


def collect_stale_camera_ages(
    image_specs: List["ImageSpec"],
    images: Dict[RobotComponent, Optional[np.ndarray]],
    received_at: Dict[RobotComponent, Optional[float]],
    now: float,
    max_age_s: float,
) -> List[Tuple[str, float]]:
    """Return (component name, age_s) for stored frames older than max_age_s.

    ``max_age_s <= 0`` disables the check. A stored image with no receive
    timestamp is treated as infinitely old. Missing images are skipped;
    callers still handle those as not-ready.
    """
    if max_age_s <= 0:
        return []
    stale: List[Tuple[str, float]] = []
    for spec in image_specs:
        if images.get(spec.component) is None:
            continue
        timestamp = received_at.get(spec.component)
        age = float("inf") if timestamp is None else now - timestamp
        if age > max_age_s:
            stale.append((spec.component.name, age))
    return stale


def _format_stale_camera_detail(stale: List[Tuple[str, float]]) -> str:
    parts = []
    for name, age in stale:
        if age == float("inf"):
            parts.append(f"{name} age=unknown")
        else:
            parts.append(f"{name} age={age:.3f}s")
    return ", ".join(parts)


def inference_latency_action_skip(
    inference_start_tick: int,
    inference_end_tick: int,
    available_actions: int,
    latency_compensation_ratio: float = 0.0,
) -> int:
    """Return the result prefix elapsed while inference was running."""
    if not 0 <= inference_start_tick <= inference_end_tick:
        raise ValueError(
            "Action publish ticks must be monotonic: "
            f"start={inference_start_tick}, "
            f"end={inference_end_tick}"
        )
    if available_actions < 0:
        raise ValueError("available_actions must be >= 0")
    if not 0.0 <= latency_compensation_ratio <= 1.0:
        raise ValueError("latency_compensation_ratio must be between 0 and 1")
    elapsed_during_inference = min(
        inference_end_tick - inference_start_tick,
        available_actions,
    )
    return int(
        elapsed_during_inference * latency_compensation_ratio + 0.5
    )


def active_inference_latency_compensation_ratio(
    configured_ratio: float,
    compensation_number: int,
    completed_inferences: int,
) -> float:
    """Return the ratio for the next inference in the startup window."""
    if compensation_number < -1:
        raise ValueError("compensation_number must be -1 or >= 0")
    if completed_inferences < 0:
        raise ValueError("completed_inferences must be >= 0")
    return (
        configured_ratio
        if compensation_number == -1
        or completed_inferences < compensation_number
        else 0.0
    )


@dataclass
class DepthPreprocessSpec:
    """Fixed metric-depth mapping used when the training dataset was created."""

    algorithm: str
    depth_unit: str
    invalid_value: float
    vmin: float
    vmax: float
    input_scale: float = 1.0


@dataclass
class ImageSpec:
    """Camera observation spec from a policy JSON.

    Shape fields follow data collection YAML: [H, W] = [height, width],
    or [H, W, C] with the channel ignored. crop_start_xy is [x, y].

    Processing order is original → resize_shape → crop → optional
    target_shape. The final size is crop_shape if cropping is set,
    otherwise resize_shape. cv2.resize is skipped when the image is
    already that size.

    swap_rb swaps the R and B channels last, for models trained on images
    the data logger saved with R/B swapped (rgb8 sensor_msgs/Image written
    by cv2.imwrite as if BGR).
    """

    component: RobotComponent
    obs_key: str
    resize_shape: Optional[Tuple[int, int]] = None  # (H, W) before crop
    crop_start_xy: Optional[Tuple[int, int]] = None  # (x, y)
    crop_shape: Optional[Tuple[int, int]] = None  # (H, W)
    target_shape: Optional[Tuple[int, int]] = None  # (H, W) after crop
    depth_preprocess: Optional[DepthPreprocessSpec] = None
    swap_rb: bool = False

    @property
    def output_shape(self) -> Optional[Tuple[int, int]]:
        """Final observation size (H, W)."""
        if self.target_shape is not None:
            return self.target_shape
        if self.crop_shape is not None:
            return self.crop_shape
        return self.resize_shape

    @property
    def output_size_wh(self) -> Optional[Tuple[int, int]]:
        """Final observation size (W, H) for OpenCV resize."""
        shape = self.output_shape
        if shape is None:
            return None
        height, width = shape
        return (width, height)


@dataclass
class LoadedConfigs:
    robot: Dict[str, Any]
    policy: Dict[str, Any]
    robot_path: Path
    policy_path: Path
    state_order: List[RobotComponent]
    action_order: List[RobotComponent]
    image_specs: List[ImageSpec]


def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    json_path = Path(path).expanduser().resolve()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{json_path} must contain a JSON object")
    return data


def _resolve_path(path_value: str, config_path: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    config_relative = (config_path.parent / path).resolve()
    if config_relative.exists():
        return config_relative
    return (Path.cwd() / path).resolve()


def _as_int_pair(value: Any, field_name: str) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be [a, b], got {value!r}")
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be integers, got {value!r}") from exc


def _parse_shape_hw(value: Any, field_name: str) -> Tuple[int, int]:
    """Parse [H, W] or [H, W, C]. Channel is ignored when present."""
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
        raise ValueError(f"{field_name} must be [H, W] or [H, W, C], got {value!r}")
    try:
        height, width = int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be integers, got {value!r}") from exc
    if height <= 0 or width <= 0:
        raise ValueError(f"{field_name} must be positive [H, W], got {[height, width]}")
    return (height, width)


def _parse_optional_shape_hw(value: Any, field_name: str) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    return _parse_shape_hw(value, field_name)


def _reject_removed_image_fields(item: dict) -> None:
    removed = {
        "target_size": "resize_shape (and optional target_shape) as [H, W]",
        "resize_size": "resize_shape as [H, W]",
    }
    for old_name, replacement in removed.items():
        if old_name in item:
            raise ValueError(
                f"observation.images no longer accepts {old_name}. "
                f"Use {replacement}. All *_shape fields follow data collection: "
                "[H, W] or [H, W, C]"
            )


def _parse_crop(
    item: dict,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Parse crop_start_xy [x, y] and crop_shape [H, W]. Both or neither."""
    raw_start = item.get("crop_start_xy")
    raw_shape = item.get("crop_shape")
    if raw_start is None and raw_shape is None:
        return None, None
    label = item.get("obs_key", item.get("component", "image"))
    if raw_start is None or raw_shape is None:
        missing = "crop_start_xy" if raw_start is None else "crop_shape"
        raise ValueError(
            f"observation.images {label}: crop config is incomplete — {missing} "
            "is missing. crop_start_xy and crop_shape must both be set"
        )
    start_xy = _as_int_pair(raw_start, "crop_start_xy")
    crop_hw = _parse_shape_hw(raw_shape, "crop_shape")
    if start_xy[0] < 0 or start_xy[1] < 0:
        raise ValueError(f"crop_start_xy must be >= 0, got {list(start_xy)}")
    return start_xy, crop_hw


def _validate_crop_fits_resize(spec: ImageSpec) -> None:
    """Reject crop regions that would miss the pre-crop resize canvas."""
    if spec.crop_start_xy is None or spec.crop_shape is None:
        return
    if spec.resize_shape is None:
        return
    height, width = spec.resize_shape
    x, y = spec.crop_start_xy
    crop_h, crop_w = spec.crop_shape
    if y + crop_h > height or x + crop_w > width:
        raise ValueError(
            f"observation.images.{spec.obs_key}: crop exceeds resize_shape "
            f"[{height}, {width}] — start (x={x}, y={y}) needs "
            f"{x + crop_w}px width and {y + crop_h}px height"
        )


def _parse_depth_preprocess(
    item: dict,
    component: RobotComponent,
) -> Optional[DepthPreprocessSpec]:
    raw = item.get("depth_preprocess")
    if raw is None:
        return None
    if component != RobotComponent.CAMERA_HEAD_DEPTH:
        raise ValueError(
            "depth_preprocess is only valid for CAMERA_HEAD_DEPTH"
        )
    if not isinstance(raw, dict):
        raise ValueError("depth_preprocess must be an object")

    algorithm = str(raw.get("algorithm", "")).strip().lower()
    if algorithm != "clip_normalize_turbo_rgb":
        raise ValueError(
            "depth_preprocess.algorithm must be 'clip_normalize_turbo_rgb'"
        )
    depth_unit = str(raw.get("depth_unit", "")).strip().lower()
    if depth_unit != "mm":
        raise ValueError("depth_preprocess.depth_unit must be 'mm'")
    try:
        invalid_value = float(raw.get("invalid_value", 0))
        vmin = float(raw["vmin"])
        vmax = float(raw["vmax"])
        input_scale = float(raw.get("input_scale", 1.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "depth_preprocess requires numeric vmin, vmax, and invalid_value"
        ) from exc
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("depth_preprocess requires finite vmax > vmin")
    if not np.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("depth_preprocess.input_scale must be a positive finite number")
    return DepthPreprocessSpec(
        algorithm=algorithm,
        depth_unit=depth_unit,
        invalid_value=invalid_value,
        vmin=vmin,
        vmax=vmax,
        input_scale=input_scale,
    )


def _resize_camera_image(
    image: np.ndarray,
    shape_hw: Tuple[int, int],
    is_depth: bool,
) -> np.ndarray:
    height, width = shape_hw
    if image.shape[0] == height and image.shape[1] == width:
        return image
    interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def _crop_camera_image(
    image: np.ndarray,
    start_xy: Tuple[int, int],
    crop_hw: Tuple[int, int],
) -> np.ndarray:
    x, y = start_xy
    crop_h, crop_w = crop_hw
    height, width = image.shape[:2]
    if y + crop_h > height or x + crop_w > width:
        raise ValueError(
            f"crop exceeds image ({width}x{height}) — start (x={x}, y={y}) "
            f"needs {x + crop_w}px width and {y + crop_h}px height"
        )
    return np.ascontiguousarray(image[y : y + crop_h, x : x + crop_w])


def preprocess_camera_image(
    image: np.ndarray,
    spec: ImageSpec,
    is_depth: bool = False,
) -> np.ndarray:
    """Match data collection: original → resize_shape → crop → target_shape."""
    if spec.resize_shape is not None:
        image = _resize_camera_image(image, spec.resize_shape, is_depth)
    if spec.crop_start_xy is not None and spec.crop_shape is not None:
        image = _crop_camera_image(image, spec.crop_start_xy, spec.crop_shape)
    if spec.target_shape is not None:
        image = _resize_camera_image(image, spec.target_shape, is_depth)
    if spec.depth_preprocess is not None:
        image = _depth_to_turbo_rgb(image, spec.depth_preprocess)
    if spec.swap_rb:
        image = np.ascontiguousarray(image[:, :, ::-1])
    return image


def _depth_to_turbo_rgb(
    depth: np.ndarray,
    config: DepthPreprocessSpec,
) -> np.ndarray:
    """Apply the same fixed-range Turbo RGB mapping used for training JPEGs."""
    return depth_to_turbo_rgb_from_config(depth, config)


def _describe_image_preprocess(spec: ImageSpec) -> str:
    parts = []
    if spec.resize_shape is not None:
        height, width = spec.resize_shape
        parts.append(f"resize=[{height}, {width}]")
    if spec.crop_start_xy is not None and spec.crop_shape is not None:
        x, y = spec.crop_start_xy
        crop_h, crop_w = spec.crop_shape
        parts.append(f"crop=start({x},{y}) [{crop_h}, {crop_w}]")
    if spec.target_shape is not None:
        height, width = spec.target_shape
        parts.append(f"target=[{height}, {width}]")
    if spec.depth_preprocess is not None:
        depth = spec.depth_preprocess
        parts.append(
            f"depth={depth.algorithm}[{depth.vmin:g},{depth.vmax:g}]{depth.depth_unit}"
            + (f"x{depth.input_scale:g}" if depth.input_scale != 1.0 else "")
        )
    if spec.swap_rb:
        parts.append("swap_rb")
    return " → ".join(parts) if parts else "native"


def _normalize_image_specs(raw_images: Any) -> List[ImageSpec]:
    specs: List[ImageSpec] = []
    if isinstance(raw_images, dict):
        # {"camera_head": "CAMERA_HEAD", "camera_wrist": "CAMERA_HAND_RIGHT"}
        for obs_key, component_name in raw_images.items():
            specs.append(ImageSpec(_component_from_name(component_name), str(obs_key)))
        return specs

    if isinstance(raw_images, list):
        for item in raw_images:
            if not isinstance(item, dict):
                raise ValueError("observation.images list entries must be objects")
            _reject_removed_image_fields(item)
            component = _component_from_name(item["component"])
            obs_key = str(item.get("obs_key", item.get("key", COMPONENT_TO_KEY[component])))
            crop_start_xy, crop_shape = _parse_crop(item)
            spec = ImageSpec(
                component,
                obs_key,
                _parse_optional_shape_hw(item.get("resize_shape"), "resize_shape"),
                crop_start_xy,
                crop_shape,
                _parse_optional_shape_hw(item.get("target_shape"), "target_shape"),
                _parse_depth_preprocess(item, component),
                bool(item.get("swap_rb", False)),
            )
            _validate_crop_fits_resize(spec)
            specs.append(spec)
        return specs

    raise ValueError("observation.images must be a list or object")


def _validate_gripper_threshold_config(policy: Dict[str, Any]) -> None:
    action = policy.get("action") or {}
    kind = str(action.get("gripper_threshold_type", "single")).lower()
    if kind not in ("single", "dual"):
        raise ValueError(
            "action.gripper_threshold_type must be 'single' or 'dual'"
        )
    if kind != "dual":
        return
    if "gripper_threshold_close" not in action or "gripper_threshold_open" not in action:
        raise ValueError(
            "action.gripper_threshold_type='dual' requires "
            "gripper_threshold_close and gripper_threshold_open"
        )
    close_th = float(action["gripper_threshold_close"])
    open_th = float(action["gripper_threshold_open"])
    if not 0.0 <= open_th <= 1.0 or not 0.0 <= close_th <= 1.0:
        raise ValueError(
            "gripper_threshold_open and gripper_threshold_close must be in [0, 1]"
        )
    if close_th < open_th:
        raise ValueError(
            "gripper_threshold_close must be >= gripper_threshold_open"
        )


def _validate_no_hand_gripper_conflict(action_order: List[RobotComponent]) -> None:
    action_set = set(action_order)
    for gripper, hand in GRIPPER_TO_HAND.items():
        if gripper in action_set and hand in action_set:
            raise ValueError(
                f"action.order cannot include both {hand.name} and {gripper.name}; "
                "use either direct hand joints or virtual gripper control for that side"
            )


def _validate_used_gripper(
    robot: Dict[str, Any],
    policy: Dict[str, Any],
    gripper: RobotComponent,
    in_action: bool,
) -> None:
    cfg = _resolve_gripper_config(robot, gripper, policy)
    if not cfg:
        raise ValueError(
            f"policy uses {gripper.name} but grippers.{gripper.name} is missing. "
            "Add it to the policy JSON."
        )
    if not in_action:
        return
    mode = str(cfg.get("command_mode", "hand_preset"))
    if mode == "hand_preset":
        expected = _joint_count(robot, GRIPPER_TO_HAND[gripper])
        for key in ("open_preset", "close_preset"):
            if key not in cfg:
                raise ValueError(
                    f"grippers.{gripper.name} missing {key} for command_mode=hand_preset"
                )
            preset = cfg[key]
            if not isinstance(preset, (list, tuple)) or len(preset) != expected:
                raise ValueError(
                    f"grippers.{gripper.name}.{key} must be length {expected}"
                )
        return
    if mode == "gripper_preset":
        for key in ("open_value", "close_value"):
            if key not in cfg:
                raise ValueError(
                    f"grippers.{gripper.name} missing {key} for command_mode=gripper_preset"
                )
        return
    if mode == "scalar_topic":
        if not cfg.get("command_topic_key"):
            raise ValueError(
                f"grippers.{gripper.name} scalar_topic command missing command_topic_key"
            )
        return
    raise ValueError(
        f"Unsupported {gripper.name} command_mode: {mode}. "
        "Expected 'hand_preset', 'gripper_preset', or 'scalar_topic'."
    )


def _validate_component_value_length(
    robot_config: Dict[str, Any],
    component: RobotComponent,
    value: Any,
    context: str,
) -> None:
    if value is None:
        return
    size = _joint_count(robot_config, component)
    if component in (
        RobotComponent.LEFT_GRIPPER,
        RobotComponent.RIGHT_GRIPPER,
        RobotComponent.EXTERNAL_DATA,
    ):
        arr = np.asarray(value, dtype=object)
        if arr.ndim == 0:
            return
        if len(arr) == 1:
            return
        raise ValueError(f"{context}.{component.name} must be a scalar or length-1 value")

    arr = np.asarray(value, dtype=object)
    if arr.ndim == 0:
        raise ValueError(f"{context}.{component.name} must be length {size}, got scalar")
    if len(arr) != size:
        raise ValueError(f"{context}.{component.name} must be length {size}, got {len(arr)}")


@dataclass(frozen=True)
class DummyMotionConfig:
    """Idle-arm dither applied to arms that are not in action.order."""

    enabled: bool = False
    amplitude_deg: float = 1.5
    period_sec: float = 4.0


_DEG_TO_RAD = float(np.pi / 180.0)


def dummy_arm_motion(
    base: np.ndarray,
    elapsed_sec: float,
    amplitude_deg: float = 1.5,
    period_sec: float = 4.0,
    phase_seed: float = 0.0,
    amp_scale: Optional[np.ndarray] = None,
    freq_scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a small sine/cosine offset around a held arm pose."""
    values = np.asarray(base, dtype=np.float64).reshape(-1)
    n = values.size
    amplitude = float(amplitude_deg) * _DEG_TO_RAD
    period = max(float(period_sec), 1e-3)
    omega = 2.0 * np.pi / period
    if amp_scale is None:
        amp_scale = np.ones(n, dtype=np.float64)
    else:
        amp_scale = np.asarray(amp_scale, dtype=np.float64).reshape(-1)
    if freq_scale is None:
        freq_scale = np.ones(n, dtype=np.float64)
    else:
        freq_scale = np.asarray(freq_scale, dtype=np.float64).reshape(-1)
    joints = np.arange(n, dtype=np.float64)
    phases = phase_seed + joints * (2.0 * np.pi / 5.0)
    omega_j = omega * freq_scale
    wave = np.sin(omega_j * elapsed_sec + phases)
    cosine_mix = np.cos(omega_j * elapsed_sec * 0.5 + phases * 0.7)
    return values + amplitude * amp_scale * (0.7 * wave + 0.3 * cosine_mix)


def _parse_dummy_motion_config(runtime_config: Dict[str, Any]) -> DummyMotionConfig:
    raw = runtime_config.get("dummy_motion", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("runtime.dummy_motion must be an object")
    enabled = bool(raw.get("enabled", False))
    try:
        amplitude_deg = float(raw.get("amplitude_deg", 1.5))
        period_sec = float(raw.get("period_sec", 4.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "runtime.dummy_motion amplitude_deg and period_sec must be numbers"
        ) from exc
    if not np.isfinite(amplitude_deg) or amplitude_deg < 0:
        raise ValueError("runtime.dummy_motion.amplitude_deg must be >= 0")
    if not np.isfinite(period_sec) or period_sec <= 0:
        raise ValueError("runtime.dummy_motion.period_sec must be > 0")
    return DummyMotionConfig(
        enabled=enabled,
        amplitude_deg=amplitude_deg,
        period_sec=period_sec,
    )


def _parse_inference_hz(runtime_config: Dict[str, Any]) -> float:
    """Return the inference loop rate. Missing means 30 Hz."""
    value = runtime_config.get("inference_hz", 30.0)
    if isinstance(value, bool):
        raise ValueError("runtime.inference_hz must be a positive number")
    try:
        hz = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime.inference_hz must be a positive number"
        ) from exc
    if not np.isfinite(hz) or hz <= 0.0:
        raise ValueError("runtime.inference_hz must be a positive number")
    return hz


def _parse_publish_consume_config(runtime_config: Dict[str, Any]) -> PublishConsumeConfig:
    raw = runtime_config.get("publish_consume", {})
    if raw is None:
        raw = {}
    if raw is False:
        return PublishConsumeConfig(mode="periodic")
    if not isinstance(raw, dict):
        raise ValueError("runtime.publish_consume must be an object or false")
    mode = str(raw.get("mode", "on_settle")).strip().lower()
    if mode not in ("on_settle", "periodic"):
        raise ValueError(
            "runtime.publish_consume.mode must be 'on_settle' or 'periodic'"
        )

    def positive_float(key: str, default: float) -> float:
        try:
            value = float(raw.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"runtime.publish_consume.{key} must be a number") from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"runtime.publish_consume.{key} must be positive")
        return value

    def non_negative_float(key: str, default: float) -> float:
        try:
            value = float(raw.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"runtime.publish_consume.{key} must be a number") from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"runtime.publish_consume.{key} must be non-negative")
        return value

    settle_rad = positive_float("waypoint_settle_rad", 0.02)
    approach_rad = positive_float("waypoint_approach_rad", 0.05)
    if approach_rad < settle_rad:
        raise ValueError(
            "runtime.publish_consume.waypoint_approach_rad must be "
            ">= waypoint_settle_rad"
        )
    hold_republish = raw.get("hold_republish_while_waiting", True)
    if not isinstance(hold_republish, bool):
        raise ValueError(
            "runtime.publish_consume.hold_republish_while_waiting must be a boolean"
        )
    return PublishConsumeConfig(
        mode=mode,
        waypoint_settle_rad=settle_rad,
        waypoint_approach_rad=approach_rad,
        publish_poll_s=positive_float("publish_poll_s", 0.015),
        max_waypoint_pop_hz=positive_float("max_waypoint_pop_hz", 25.0),
        waypoint_dwell_timeout_s=non_negative_float(
            "waypoint_dwell_timeout_s", 0.0
        ),
        hold_republish_while_waiting=hold_republish,
        hold_republish_period_s=positive_float("hold_republish_period_s", 0.2),
    )


def _parse_motion_speed(runtime_config: Dict[str, Any]) -> float:
    """UI motion_speed = GDK arm_lead_rad (rad per command tick, 100Hz)."""
    try:
        value = float(runtime_config.get("motion_speed", 0.012))
    except (TypeError, ValueError):
        return 0.012
    if not np.isfinite(value):
        return 0.012
    return float(max(0.002, min(0.05, value)))


def _parse_arm_live_guard_config(runtime_config: Dict[str, Any]) -> ArmLiveGuardConfig:
    raw = runtime_config.get("arm_live_guard", {})
    if raw is None:
        raw = {}
    if raw is False:
        return ArmLiveGuardConfig(enabled=False)
    if not isinstance(raw, dict):
        raise ValueError("runtime.arm_live_guard must be an object or false")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("runtime.arm_live_guard.enabled must be a boolean")

    def non_negative(key: str, default: float) -> float:
        try:
            value = float(raw.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"runtime.arm_live_guard.{key} must be a number") from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"runtime.arm_live_guard.{key} must be non-negative")
        return value

    return ArmLiveGuardConfig(
        enabled=enabled,
        max_ahead_rad=non_negative("max_ahead_rad", 0.055),
        max_behind_rad=non_negative("max_behind_rad", 0.1),
    )


def _parse_arm_action_postprocess_config(
    runtime_config: Dict[str, Any],
) -> ArmActionPostprocessConfig:
    raw = runtime_config.get("arm_action_postprocess", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("runtime.arm_action_postprocess must be an object")

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("runtime.arm_action_postprocess.enabled must be a boolean")

    def positive_float(key: str, default: float) -> float:
        try:
            value = float(raw.get(key, default))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"runtime.arm_action_postprocess.{key} must be a positive number"
            ) from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"runtime.arm_action_postprocess.{key} must be a positive number"
            )
        return value

    try:
        beta = float(raw.get("beta", 0.01))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "runtime.arm_action_postprocess.beta must be a non-negative number"
        ) from exc
    if not np.isfinite(beta) or beta < 0:
        raise ValueError(
            "runtime.arm_action_postprocess.beta must be a non-negative number"
        )

    return ArmActionPostprocessConfig(
        enabled=enabled,
        min_cutoff=positive_float("min_cutoff", 1.0),
        beta=beta,
        derivative_cutoff=positive_float("derivative_cutoff", 1.0),
    )


def _parse_publish_target_blend_config(
    runtime_config: Dict[str, Any],
) -> PublishTargetBlendConfig:
    raw = runtime_config.get("publish_target_blend", {})
    if raw is None:
        raw = {}
    if raw is False:
        return PublishTargetBlendConfig(enabled=False)
    if not isinstance(raw, dict):
        raise ValueError("runtime.publish_target_blend must be an object or false")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("runtime.publish_target_blend.enabled must be a boolean")
    try:
        alpha = float(raw.get("alpha", 0.4))
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime.publish_target_blend.alpha must be a number") from exc
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
        raise ValueError("runtime.publish_target_blend.alpha must be in (0, 1]")
    return PublishTargetBlendConfig(enabled=enabled, alpha=alpha)


def _parse_hand_key_selection_config(
    runtime_config: Dict[str, Any],
) -> HandKeySelectionConfig:
    raw = runtime_config.get("hand_key_selection", {})
    if raw is None:
        raw = {}
    if raw is False:
        return HandKeySelectionConfig(enabled=False)
    if not isinstance(raw, dict):
        raise ValueError("runtime.hand_key_selection must be an object or false")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("runtime.hand_key_selection.enabled must be a boolean")

    def positive_float(key: str, default: float) -> float:
        try:
            value = float(raw.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"runtime.hand_key_selection.{key} must be a number") from exc
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"runtime.hand_key_selection.{key} must be non-negative")
        return value

    margin = positive_float("key_confidence_margin", 0.03)
    if "key_confidence_margin" not in raw and "midpoint_hysteresis" in raw:
        margin = positive_float("midpoint_hysteresis", 0.03)

    def non_negative_int(key: str, default: int) -> int:
        try:
            value = int(raw.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"runtime.hand_key_selection.{key} must be an integer"
            ) from exc
        if value < 0:
            raise ValueError(f"runtime.hand_key_selection.{key} must be non-negative")
        return value

    return HandKeySelectionConfig(
        enabled=enabled,
        key_confidence_margin=margin,
        scalar_ema_alpha=positive_float("scalar_ema_alpha", 0.22),
        hold_publish_ticks_on_chunk_append=non_negative_int(
            "hold_publish_ticks_on_chunk_append", 10
        ),
        downgrade_to_min_key_max_scalar=positive_float(
            "downgrade_to_min_key_max_scalar", 0.11
        ),
        waypoint_gripper_gray_err_rad=positive_float(
            "waypoint_gripper_gray_err_rad", 0.01
        ),
        force_nearest_key_min_scalar_delta=positive_float(
            "force_nearest_key_min_scalar_delta", 0.15
        ),
    )


def _parse_waist_chunk_prefix_fix_config(
    runtime_config: Dict[str, Any],
) -> WaistChunkPrefixFixConfig:
    raw = runtime_config.get("waist_chunk_prefix_fix", {})
    if raw is None:
        raw = {}
    if raw is False:
        return WaistChunkPrefixFixConfig(enabled=False)
    if not isinstance(raw, dict):
        raise ValueError("runtime.waist_chunk_prefix_fix must be an object or false")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("runtime.waist_chunk_prefix_fix.enabled must be a boolean")
    reference = str(raw.get("reference", "live")).strip().lower()
    if reference not in ("live", "obs", "max"):
        raise ValueError(
            "runtime.waist_chunk_prefix_fix.reference must be live, obs, or max"
        )
    try:
        ramp_steps = int(raw.get("ramp_steps", 5))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime.waist_chunk_prefix_fix.ramp_steps must be a positive integer"
        ) from exc
    if ramp_steps < 1:
        raise ValueError(
            "runtime.waist_chunk_prefix_fix.ramp_steps must be a positive integer"
        )
    only_inc = raw.get("only_when_chunk_increase", True)
    if not isinstance(only_inc, bool):
        raise ValueError(
            "runtime.waist_chunk_prefix_fix.only_when_chunk_increase must be boolean"
        )
    try:
        epsilon = float(raw.get("epsilon_rad", 0.0005))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime.waist_chunk_prefix_fix.epsilon_rad must be a number"
        ) from exc
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError(
            "runtime.waist_chunk_prefix_fix.epsilon_rad must be non-negative"
        )
    return WaistChunkPrefixFixConfig(
        enabled=enabled,
        reference=reference,
        ramp_steps=ramp_steps,
        only_when_chunk_increase=only_inc,
        epsilon_rad=epsilon,
    )


def _parse_action_chunk_prefix_fix_config(
    runtime_config: Dict[str, Any],
) -> ActionChunkPrefixFixConfig:
    raw = runtime_config.get("action_chunk_prefix_fix", {})
    if raw is None:
        raw = {}
    if raw is False:
        return ActionChunkPrefixFixConfig(enabled=False)
    if not isinstance(raw, dict):
        raise ValueError("runtime.action_chunk_prefix_fix must be an object or false")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("runtime.action_chunk_prefix_fix.enabled must be a boolean")
    on_append = raw.get("on_append_only", True)
    if not isinstance(on_append, bool):
        raise ValueError(
            "runtime.action_chunk_prefix_fix.on_append_only must be a boolean"
        )
    reference = str(raw.get("reference", "last_command")).strip().lower()
    if reference not in ("last_command", "live_state", "max_live_command"):
        raise ValueError(
            "runtime.action_chunk_prefix_fix.reference must be "
            "last_command, live_state, or max_live_command"
        )

    def non_negative_float(key: str, default: float) -> float:
        try:
            value = float(raw.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"runtime.action_chunk_prefix_fix.{key} must be a number"
            ) from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"runtime.action_chunk_prefix_fix.{key} must be non-negative"
            )
        return value

    try:
        blend_steps = int(raw.get("blend_steps", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime.action_chunk_prefix_fix.blend_steps must be a non-negative integer"
        ) from exc
    if blend_steps < 0:
        raise ValueError(
            "runtime.action_chunk_prefix_fix.blend_steps must be a non-negative integer"
        )
    return ActionChunkPrefixFixConfig(
        enabled=enabled,
        on_append_only=on_append,
        reference=reference,
        max_joint_shift_rad=non_negative_float("max_joint_shift_rad", 1.5),
        min_shift_rad=non_negative_float("min_shift_rad", 0.002),
        blend_steps=blend_steps,
    )


def _parse_action_chunk_execute_steps(
    runtime_config: Dict[str, Any],
) -> Optional[int]:
    """Max waypoint steps taken from each ACT chunk (None = full chunk)."""
    raw = runtime_config.get("action_chunk_execute_steps")
    if raw is None:
        return None
    try:
        steps = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime.action_chunk_execute_steps must be a positive integer"
        ) from exc
    if steps < 1:
        raise ValueError("runtime.action_chunk_execute_steps must be >= 1")
    return steps


def _parse_action_chunk_skip_first_steps(runtime_config: Dict[str, Any]) -> int:
    """Drop the first N chunk rows before enqueue (after prefix_fix / execute_steps)."""
    raw = runtime_config.get("action_chunk_skip_first_steps", 0)
    try:
        steps = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime.action_chunk_skip_first_steps must be a non-negative integer"
        ) from exc
    if steps < 0:
        raise ValueError("runtime.action_chunk_skip_first_steps must be >= 0")
    return steps


def _parse_chunk_queue_splice_config(
    runtime_config: Dict[str, Any],
) -> ChunkQueueSpliceConfig:
    raw = runtime_config.get("chunk_queue_splice", {})
    if raw is None:
        raw = {}
    if raw is False:
        return ChunkQueueSpliceConfig(enabled=False)
    if not isinstance(raw, dict):
        raise ValueError("runtime.chunk_queue_splice must be an object or false")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("runtime.chunk_queue_splice.enabled must be a boolean")
    on_append = raw.get("on_append_only", True)
    if not isinstance(on_append, bool):
        raise ValueError(
            "runtime.chunk_queue_splice.on_append_only must be a boolean"
        )

    def positive_float(key: str, default: float) -> float:
        try:
            value = float(raw.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"runtime.chunk_queue_splice.{key} must be a number"
            ) from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                f"runtime.chunk_queue_splice.{key} must be a positive number"
            )
        return value

    def non_negative_float(key: str, default: float) -> float:
        try:
            value = float(raw.get(key, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"runtime.chunk_queue_splice.{key} must be a number"
            ) from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"runtime.chunk_queue_splice.{key} must be non-negative"
            )
        return value

    try:
        min_actions = int(raw.get("min_actions_after_trim", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime.chunk_queue_splice.min_actions_after_trim must be an integer"
        ) from exc
    if min_actions < 1:
        raise ValueError(
            "runtime.chunk_queue_splice.min_actions_after_trim must be >= 1"
        )
    defer_settle = raw.get("defer_until_settle", True)
    if not isinstance(defer_settle, bool):
        raise ValueError(
            "runtime.chunk_queue_splice.defer_until_settle must be a boolean"
        )
    return ChunkQueueSpliceConfig(
        enabled=enabled,
        on_append_only=on_append,
        reverse_projection_rad=non_negative_float("reverse_projection_rad", 0.02),
        min_forward_norm=positive_float("min_forward_norm", 0.005),
        min_actions_after_trim=min_actions,
        defer_until_settle=defer_settle,
        defer_finish_rad=non_negative_float("defer_finish_rad", 0.025),
        gripper_weight=non_negative_float("gripper_weight", 0.15),
        hand_weight=non_negative_float("hand_weight", 0.25),
    )


def load_and_validate_configs(
    robot_config_path: Union[str, Path],
    policy_config_path: Union[str, Path],
) -> LoadedConfigs:
    """Load and validate robot and policy JSON files."""

    robot_path = Path(robot_config_path).expanduser().resolve()
    policy_path = Path(policy_config_path).expanduser().resolve()
    robot = _load_json(robot_path)
    policy = _load_json(policy_path)
    if robot.get("adapter") == "agibot_g2":
        platform_root = Path(__file__).resolve().parents[3]
        if str(platform_root) not in sys.path:
            sys.path.insert(0, str(platform_root))
        from shared.action_layout import apply_to_lerobot_policy

        policy = apply_to_lerobot_policy(policy)

    _validate_robot_schema(robot)
    if robot["adapter"] not in SUPPORTED_ADAPTERS:
        raise ValueError(f"Unsupported robot adapter: {robot['adapter']}")

    for key in ["model_path", "observation", "action"]:
        if key not in policy:
            raise ValueError(f"policy config missing required key: {key}")
    if "state_order" not in policy["observation"]:
        raise ValueError("policy config missing observation.state_order")
    if "images" not in policy["observation"]:
        raise ValueError("policy config missing observation.images")
    if "order" not in policy["action"]:
        raise ValueError("policy config missing action.order")
    inference_mode = str(
        policy.get("runtime", {}).get("inference_mode", "chunk")
    ).lower()
    if inference_mode not in ("chunk", "temporal_ensemble_batch"):
        raise ValueError(
            "runtime.inference_mode must be 'chunk' or "
            "'temporal_ensemble_batch'"
        )
    arm_postprocess_config = _parse_arm_action_postprocess_config(
        policy.get("runtime", {})
    )
    _parse_inference_hz(policy.get("runtime", {}))
    latency_compensation_value = policy.get("runtime", {}).get(
        "inference_latency_chunk_compensation_ratio",
        0.0,
    )
    try:
        latency_compensation_ratio = float(latency_compensation_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime.inference_latency_chunk_compensation_ratio "
            "must be a number between 0 and 1"
        ) from exc
    if not 0.0 <= latency_compensation_ratio <= 1.0:
        raise ValueError(
            "runtime.inference_latency_chunk_compensation_ratio "
            "must be between 0 and 1"
        )
    latency_compensation_number_value = policy.get("runtime", {}).get(
        "inference_latency_chunk_compensation_number",
        -1,
    )
    try:
        latency_compensation_number_float = float(
            latency_compensation_number_value
        )
        latency_compensation_number = int(
            latency_compensation_number_float
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "runtime.inference_latency_chunk_compensation_number "
            "must be -1 or a non-negative integer"
        ) from exc
    if (
        isinstance(latency_compensation_number_value, bool)
        or latency_compensation_number < -1
        or latency_compensation_number_float != latency_compensation_number
    ):
        raise ValueError(
            "runtime.inference_latency_chunk_compensation_number "
            "must be -1 or a non-negative integer"
        )

    state_order = _component_list(policy["observation"]["state_order"])
    action_order = _component_list(policy["action"]["order"])
    image_specs = _normalize_image_specs(policy["observation"]["images"])

    for component in action_order:
        if _has(CAMERA_COMPONENTS, component):
            raise ValueError(f"Camera component cannot be in action.order: {component.name}")
    for component in state_order:
        if _has(CAMERA_COMPONENTS, component):
            raise ValueError(
                f"Camera component cannot be in observation.state_order: {component.name}; "
                "put cameras under observation.images"
            )
    _validate_no_hand_gripper_conflict(action_order)
    _validate_gripper_threshold_config(policy)

    for component in state_order + action_order:
        _joint_count(robot, component)
    if arm_postprocess_config.enabled:
        for component in (
            RobotComponent.LEFT_ARM,
            RobotComponent.RIGHT_ARM,
        ):
            if component in action_order and _joint_count(robot, component) != 7:
                raise ValueError(
                    "runtime.arm_action_postprocess supports 7-joint arms only; "
                    f"{component.name} has {_joint_count(robot, component)} joints"
                )

    if RobotComponent.EXTERNAL_DATA in state_order:
        external_state = _resolve_external_data_config(robot, policy)
        if not _topic_enabled(external_state):
            raise ValueError(
                "observation.state_order includes EXTERNAL_DATA but "
                "policy external_data.topic is missing"
            )

    camera_topics = robot.get("topics", {}).get("cameras", {})
    for spec in image_specs:
        if not _has(CAMERA_COMPONENTS, spec.component):
            raise ValueError(f"observation image component is not a camera: {spec.component.name}")
        if spec.component.name not in camera_topics:
            raise ValueError(f"robot config missing camera topic for {spec.component.name}")
        if not camera_topics[spec.component.name].get("enabled", True):
            raise ValueError(
                f"policy requires disabled camera topic: {spec.component.name}"
            )

    policy_grippers = policy.get("grippers")
    if policy_grippers is not None:
        if not isinstance(policy_grippers, dict):
            raise ValueError("policy grippers must be an object")
        for key, cfg in policy_grippers.items():
            if key not in CANONICAL_GRIPPER_KEYS:
                raise ValueError(f"policy grippers has unknown key: {key}")
            if not isinstance(cfg, dict):
                raise ValueError(f"policy grippers.{key} must be an object")

    initial_pose = policy.get("initial_pose") or {}
    if not isinstance(initial_pose, dict):
        raise ValueError("policy initial_pose must be an object")
    initial_pose_components = [_component_from_name(key) for key in initial_pose]
    used_grippers = {
        gripper
        for gripper in (RobotComponent.LEFT_GRIPPER, RobotComponent.RIGHT_GRIPPER)
        if gripper in state_order
        or gripper in action_order
        or gripper in initial_pose_components
    }
    for gripper in used_grippers:
        _validate_used_gripper(
            robot,
            policy,
            gripper,
            in_action=gripper in action_order or gripper in initial_pose_components,
        )
    for key, value in initial_pose.items():
        component = _component_from_name(key)
        if _has(CAMERA_COMPONENTS, component):
            raise ValueError(f"initial_pose cannot include camera component: {component.name}")
        _validate_component_value_length(robot, component, value, "initial_pose")

    for idx, condition in enumerate(policy.get("complete_conditions", [])):
        if condition.get("type") != "JOINT_REACHED":
            continue
        target = condition.get("target", {})
        if not isinstance(target, dict):
            raise ValueError(f"complete_conditions[{idx}].target must be an object")
        for key, value in target.items():
            component = _component_from_name(key)
            _validate_component_value_length(
                robot, component, value, f"complete_conditions[{idx}].target"
            )

    return LoadedConfigs(
        robot=robot,
        policy=policy,
        robot_path=robot_path,
        policy_path=policy_path,
        state_order=state_order,
        action_order=action_order,
        image_specs=image_specs,
    )


def describe_mapping(configs: LoadedConfigs) -> str:
    """Return a dry-run summary of vector mapping and robot topics."""

    lines = []
    robot = configs.robot
    policy = configs.policy
    lines.append(f"Robot: {robot['robot_name']} ({robot['adapter']})")
    lines.append(f"Robot config: {configs.robot_path}")
    lines.append(f"Policy config: {configs.policy_path}")
    lines.append(f"Model path: {policy['model_path']}")
    lines.append(
        "Inference mode: "
        f"{policy.get('runtime', {}).get('inference_mode', 'chunk')}"
    )
    lines.append("")
    lines.append("Observation state vector:")
    idx = 0
    for component in configs.state_order:
        size = _joint_count(robot, component)
        lines.append(f"  {idx:02d}:{idx + size:02d}  {component.name} ({size})")
        idx += size
    lines.append(f"  total: {idx}")
    lines.append("")
    lines.append("Observation images:")
    for spec in configs.image_specs:
        topic = robot["topics"]["cameras"][spec.component.name]["topic"]
        todo = robot["topics"]["cameras"][spec.component.name].get("todo")
        suffix = f"  TODO: {todo}" if todo else ""
        preprocess = _describe_image_preprocess(spec)
        lines.append(
            f"  {spec.obs_key} <- {spec.component.name} ({topic}) {preprocess}{suffix}"
        )
    lines.append("")
    lines.append("Action vector:")
    idx = 0
    for component in configs.action_order:
        size = _joint_count(robot, component)
        lines.append(f"  {idx:02d}:{idx + size:02d}  {component.name} ({size})")
        idx += size
    lines.append(f"  total: {idx}")
    return "\n".join(lines)


class GeneralPolicyActionGenerator(Node):
    """
    Robot-agnostic upper-body policy executor.

    The public API mirrors policy_action_generator.py:
    start, asyncStart, stop, getTaskStatus, getProcessDone,
    setCompleteCondition, and setCompleteConditions.
    """

    _global_execution_lock = threading.Lock()
    _global_executing_instance = None

    def __init__(
        self,
        robot_config_path: Union[str, Path],
        policy_config_path: Union[str, Path],
        node_name_suffix: Optional[str] = None,
        async_model_load: bool = False,
    ):
        startup_started = time.perf_counter()
        startup_timings: List[Tuple[str, float]] = []
        phase_started = startup_started

        def record_startup_phase(name: str) -> None:
            nonlocal phase_started
            now = time.perf_counter()
            startup_timings.append((name, (now - phase_started) * 1000.0))
            phase_started = now

        if rclpy is None:
            raise RuntimeError("rclpy is required for runtime execution") from _ROS_IMPORT_ERROR

        base = "general_policy_action_generator"
        if node_name_suffix:
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", str(node_name_suffix))[:32]
            node_name = f"{base}_{safe}"
        else:
            node_name = base
        super().__init__(node_name)
        record_startup_phase("node")

        self.configs = load_and_validate_configs(robot_config_path, policy_config_path)
        self.robot_config = self.configs.robot
        self.policy_config = self.configs.policy
        self.adapter_name = self.robot_config["adapter"]
        self.state_order = self.configs.state_order
        self.action_order = self.configs.action_order
        self.image_specs = self.configs.image_specs

        runtime_cfg = self.policy_config.get("runtime", {})
        control_cfg = self.robot_config.get("control", {})
        self.publish_period = 1.0 / float(runtime_cfg.get("publish_hz", control_cfg.get("publish_hz", 30.0)))
        self._arm_action_postprocess_config = (
            _parse_arm_action_postprocess_config(runtime_cfg)
        )
        self._arm_live_guard = _parse_arm_live_guard_config(runtime_cfg)
        self._publish_consume = _parse_publish_consume_config(runtime_cfg)
        self._last_waypoint_pop_mono: Optional[float] = None
        self._last_hold_republish_mono: Optional[float] = None
        self._publish_target_blend = _parse_publish_target_blend_config(runtime_cfg)
        self._publish_target_ema_state: Optional[np.ndarray] = None
        self._hand_key_selection = _parse_hand_key_selection_config(runtime_cfg)
        self._gripper_scalar_ema: Dict[RobotComponent, float] = {}
        self._hand_key_hold_until_publish_tick: Dict[str, int] = {}
        self._waist_chunk_prefix_fix = _parse_waist_chunk_prefix_fix_config(
            runtime_cfg
        )
        self._action_chunk_prefix_fix = _parse_action_chunk_prefix_fix_config(
            runtime_cfg
        )
        self._chunk_queue_splice = _parse_chunk_queue_splice_config(runtime_cfg)
        self._chunk_splice_weights = self._build_chunk_splice_weights()
        self._dummy_motion_config = _parse_dummy_motion_config(runtime_cfg)
        self._dummy_motion_components: List[RobotComponent] = []
        if self._dummy_motion_config.enabled:
            self._dummy_motion_components = [
                component
                for component in (RobotComponent.LEFT_ARM, RobotComponent.RIGHT_ARM)
                if component not in self.action_order
            ]
        self._dummy_motion_bases: Dict[RobotComponent, np.ndarray] = {}
        self._dummy_motion_scales: Dict[RobotComponent, Tuple[np.ndarray, np.ndarray]] = {}
        self._dummy_motion_t0: Optional[float] = None
        self._dummy_motion_skip_warned = False
        self._dummy_motion_applied_logged = False
        self._dummy_motion_blocked: set = set()
        self._arm_action_filters: Dict[
            RobotComponent, OneEuroFilter
        ] = {}
        if self._arm_action_postprocess_config.enabled:
            for component in (
                RobotComponent.LEFT_ARM,
                RobotComponent.RIGHT_ARM,
            ):
                if component in self.action_order:
                    self._arm_action_filters[component] = OneEuroFilter(
                        sample_hz=1.0 / self.publish_period,
                        min_cutoff=self._arm_action_postprocess_config.min_cutoff,
                        beta=self._arm_action_postprocess_config.beta,
                        derivative_cutoff=(
                            self._arm_action_postprocess_config.derivative_cutoff
                        ),
                    )
        self._inference_hz = _parse_inference_hz(runtime_cfg)
        self._inference_period = 1.0 / self._inference_hz
        self.inference_mode = str(
            runtime_cfg.get("inference_mode", "chunk")
        ).lower()
        if self.inference_mode not in ("chunk", "temporal_ensemble_batch"):
            raise ValueError(
                "runtime.inference_mode must be 'chunk' or "
                "'temporal_ensemble_batch'"
            )
        self._inference_latency_chunk_compensation_ratio = float(
            runtime_cfg.get(
                "inference_latency_chunk_compensation_ratio",
                0.0,
            )
        )
        self._inference_latency_chunk_compensation_number = int(
            runtime_cfg.get(
                "inference_latency_chunk_compensation_number",
                -1,
            )
        )
        try:
            self._reinfer_threshold = int(runtime_cfg.get("reinfer_threshold", -1))
        except (TypeError, ValueError):
            self._reinfer_threshold = -1
        self._action_chunk_execute_steps = _parse_action_chunk_execute_steps(
            runtime_cfg
        )
        self._action_chunk_skip_first_steps = _parse_action_chunk_skip_first_steps(
            runtime_cfg
        )
        self._right_gripper_03_range_seen = False
        self._right_gripper_export_lock = threading.Lock()
        clear_raw = runtime_cfg.get("clear_action_queue_on_reinfer", False)
        if not isinstance(clear_raw, bool):
            raise ValueError(
                "runtime.clear_action_queue_on_reinfer must be a boolean"
            )
        self._clear_action_queue_on_reinfer = clear_raw
        self._profile_inference = bool(runtime_cfg.get("profile_inference", False))
        self._cudnn_benchmark = bool(runtime_cfg.get("cudnn_benchmark", False))
        self._allow_tf32 = bool(runtime_cfg.get("allow_tf32", False))
        self._fp16_autocast = bool(runtime_cfg.get("fp16_autocast", False))
        self._profile_camera_callbacks = bool(
            runtime_cfg.get("profile_camera_callbacks", False)
        )
        self._save_inference_images = bool(
            runtime_cfg.get("save_inference_images", False)
        )
        self._save_all_inference_images = bool(
            runtime_cfg.get("save_all_inference_images", False)
        )
        self._stream_inference_images = bool(
            runtime_cfg.get("stream_inference_images", False)
        )
        self._stream_inference_images_port = int(
            runtime_cfg.get("stream_inference_images_port", 8004)
        )
        self._stream_inference_images_hz = float(
            runtime_cfg.get("stream_inference_images_hz", 10.0)
        )
        self._stream_inference_images_quality = int(
            runtime_cfg.get("stream_inference_images_quality", 70)
        )
        self._inference_image_stream = None
        self._inference_images_saved = False
        self._inference_image_seq = 0
        self._inference_image_run_id = time.strftime("%Y%m%d_%H%M%S")
        raw_policy_name = str(
            self.policy_config.get("name") or self.configs.policy_path.stem
        )
        safe_policy_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_policy_name).strip("._")
        self._policy_image_stem = (safe_policy_name or "policy")[:80]
        # Always under bwr/deployment/logs, independent of policy subdirectory depth.
        self._inference_image_log_dir = (
            Path(__file__).resolve().parent.parent / "logs" / "inference_images"
        )
        self._temporal_ensemble_device = str(
            runtime_cfg.get("temporal_ensemble_device", "model")
        ).lower()
        if self._temporal_ensemble_device not in {"model", "cpu"}:
            raise ValueError(
                "runtime.temporal_ensemble_device must be 'model' or 'cpu'"
            )
        self.warming_up_time = float(runtime_cfg.get("warming_up_time", 3.0))
        self._min_run_time_before_complete = float(runtime_cfg.get("min_run_time_before_complete", 5.0))
        self._state_wait_timeout = float(runtime_cfg.get("state_wait_timeout", 5.0))
        self._max_image_age_s = float(runtime_cfg.get("max_image_age_s", 1.0))
        if self._max_image_age_s < 0:
            raise ValueError("runtime.max_image_age_s must be >= 0 (0 disables the check)")
        raw_action_log = runtime_cfg.get(
            "lerobot_action_log",
            runtime_cfg.get("waist_debug_trace", True),
        )
        if raw_action_log is False:
            self._action_log = LerobotActionLog(enabled=False)
        elif isinstance(raw_action_log, dict):
            path = raw_action_log.get("path")
            self._action_log = LerobotActionLog(
                enabled=bool(raw_action_log.get("enabled", True)),
                path=path if path else DEFAULT_LEROBOT_ACTION_LOG,
                publish_sample_every=int(
                    raw_action_log.get("publish_sample_every", 10)
                ),
            )
        else:
            self._action_log = LerobotActionLog(enabled=True)
        record_startup_phase("configs")

        self._load_common_message_types()
        self.robot_io = create_robot_adapter(self)
        self.robot_io.load_message_types()
        self._apply_motion_speed_from_policy()
        record_startup_phase("message_types")

        from cv_bridge import CvBridge

        self.bridge = CvBridge()
        record_startup_phase("cv_bridge")

        # Latest state caches.
        self.joint_state: Dict[RobotComponent, Optional[np.ndarray]] = {
            RobotComponent.HEAD: None,
            RobotComponent.LEFT_ARM: None,
            RobotComponent.RIGHT_ARM: None,
            RobotComponent.LEFT_HAND: None,
            RobotComponent.RIGHT_HAND: None,
        }
        self.gripper_state: Dict[RobotComponent, float] = {
            RobotComponent.LEFT_GRIPPER: 0.0,
            RobotComponent.RIGHT_GRIPPER: 0.0,
        }
        self._gripper_command_bit: Dict[RobotComponent, int] = {
            RobotComponent.LEFT_GRIPPER: 0,
            RobotComponent.RIGHT_GRIPPER: 0,
        }
        self.aux_scalar_state: Dict[RobotComponent, Optional[float]] = {
            RobotComponent.EXTERNAL_DATA: None,
        }
        self.images: Dict[RobotComponent, Optional[np.ndarray]] = {
            RobotComponent.CAMERA_HEAD: None,
            RobotComponent.CAMERA_HAND_LEFT: None,
            RobotComponent.CAMERA_HAND_RIGHT: None,
            RobotComponent.CAMERA_HEAD_DEPTH: None,
            RobotComponent.CAMERA_HAND_LEFT_2: None,
            RobotComponent.CAMERA_HAND_RIGHT_2: None,
        }
        self._image_received_at: Dict[RobotComponent, Optional[float]] = {
            RobotComponent.CAMERA_HEAD: None,
            RobotComponent.CAMERA_HAND_LEFT: None,
            RobotComponent.CAMERA_HAND_RIGHT: None,
            RobotComponent.CAMERA_HEAD_DEPTH: None,
            RobotComponent.CAMERA_HAND_LEFT_2: None,
            RobotComponent.CAMERA_HAND_RIGHT_2: None,
        }
        self._camera_callback_seen = set()
        self._camera_ready_logged = set()
        self._camera_callback_groups = []
        self._camera_timing_lock = threading.Lock()
        self._camera_timing = {
            spec.component: {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
            for spec in self.image_specs
        }
        self._camera_timing_last_reset = time.perf_counter()

        # ROS entities.
        self.command_publishers: Dict[str, Any] = {}
        self.ros_subscriptions = []
        self.robot_io.setup_ros_interfaces()
        self._setup_aux_scalar_interfaces()
        self._setup_camera_interfaces()
        record_startup_phase("ros_interfaces")

        self.action_policy = None
        self._model_load_error: Optional[BaseException] = None
        self._model_load_event = threading.Event()
        self._model_load_thread: Optional[threading.Thread] = None
        self._async_model_load = bool(async_model_load)
        if self._async_model_load:
            self.get_logger().info("Starting policy model load in background")
            self._model_load_thread = threading.Thread(
                target=self._model_load_worker,
                daemon=True,
                name=f"{self.get_name()}_model_loader",
            )
            self._model_load_thread.start()
            record_startup_phase("policy_load_started")
        else:
            self._load_policy_model()
            self._model_load_event.set()
            record_startup_phase("policy_agent")

        self.action_queue = deque()
        self.action_queue_lock = threading.Lock()
        self._publish_cycle_lock = threading.Lock()
        self._pending_chunk_splice: Optional[List[np.ndarray]] = None
        self.last_action = None
        self._actions_consumed_since_inference = 0
        self._published_action_tick = 0
        self.inference_running = False
        self.publish_running = False

        self._execution_running = False
        self._inference_thread = None
        self._policy_inference_thread = None
        self._publish_thread = None
        self._stop_requested = False
        self._failed = False
        self._failure_message = ""
        self._current_task_status = ""
        self._process_done = True
        self._state_lock = threading.Lock()
        self._start_time = 0.0
        self._inference_start_time: Optional[float] = None
        self._complete_check_eligible_time: Optional[float] = None
        self._inference_count = 0
        self._condition_progress = 0.0
        self._joint_reached_settle_starts: Dict[int, Optional[float]] = {}
        self._active_complete_conditions = self._parse_complete_conditions_from_policy()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

        total_startup_ms = (time.perf_counter() - startup_started) * 1000.0
        self.get_logger().info(
            "Startup timing | "
            + " ".join(f"{name}={elapsed_ms:.1f}ms" for name, elapsed_ms in startup_timings)
            + f" total={total_startup_ms:.1f}ms"
        )
        self.get_logger().info("General policy action generator initialized")
        self.get_logger().info(f"  Robot: {self.robot_config['robot_name']} ({self.adapter_name})")
        self.get_logger().info(f"  Observation: {_component_names(self.state_order)}")
        self.get_logger().info(f"  Action: {_component_names(self.action_order)}")
        if self._dummy_motion_config.enabled:
            if self._dummy_motion_components:
                self.get_logger().info(
                    "  Dummy motion: "
                    f"{_component_names(self._dummy_motion_components)} "
                    f"amp={self._dummy_motion_config.amplitude_deg:g}deg "
                    f"period={self._dummy_motion_config.period_sec:g}s"
                )
            else:
                self.get_logger().info(
                    "  Dummy motion enabled, but no unused LEFT_ARM/RIGHT_ARM"
                )
        self.get_logger().info(f"  Inference mode: {self.inference_mode}")
        for spec in self.image_specs:
            self.get_logger().info(
                f"  Image {spec.obs_key}: {_describe_image_preprocess(spec)}"
            )
        if self._max_image_age_s > 0:
            self.get_logger().info(
                f"  max_image_age_s: {self._max_image_age_s:.3f} "
                "(stale camera frames fail inference)"
            )
        else:
            self.get_logger().warning(
                "  max_image_age_s: 0 (stale camera frame check disabled)"
            )
        if self._save_inference_images or self._save_all_inference_images:
            mode = "all frames" if self._save_all_inference_images else "first frame"
            self.get_logger().info(
                f"  save_inference_images: {self._inference_image_log_dir} "
                f"policy={self._policy_image_stem} mode={mode}"
            )
        if self._stream_inference_images:
            self._start_inference_image_stream()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _inference_stream_cameras(self) -> List[Dict[str, Any]]:
        cameras = []
        for spec in self.image_specs:
            cameras.append(
                {
                    "name": spec.obs_key,
                    "label": spec.obs_key,
                    "component": spec.component.name,
                    "ros_topic": spec.obs_key,
                    "preprocess": _describe_image_preprocess(spec),
                    "publish_hz": self._stream_inference_images_hz,
                }
            )
        return cameras

    def _inference_stream_frame(self, obs_key: str) -> Optional[np.ndarray]:
        for spec in self.image_specs:
            if spec.obs_key == obs_key:
                image = self.images.get(spec.component)
                return None if image is None else image
        return None

    def _start_inference_image_stream(self) -> None:
        from inference_image_stream import InferenceImageStreamServer

        cameras = self._inference_stream_cameras()
        if not cameras:
            self.get_logger().warning(
                "stream_inference_images is enabled but this policy has no cameras"
            )
            return
        try:
            server = InferenceImageStreamServer(
                cameras=cameras,
                get_frame=self._inference_stream_frame,
                policy_name=self._policy_image_stem,
                port=self._stream_inference_images_port,
                hz=self._stream_inference_images_hz,
                quality=self._stream_inference_images_quality,
            )
            server.start()
        except OSError as exc:
            self.get_logger().error(
                f"Failed to start inference image stream on port "
                f"{self._stream_inference_images_port}: {exc}"
            )
            return
        self._inference_image_stream = server
        self.get_logger().info(
            f"  stream_inference_images: http://0.0.0.0:"
            f"{self._stream_inference_images_port}/ "
            f"cameras={[spec.obs_key for spec in self.image_specs]} "
            f"hz={self._stream_inference_images_hz:.1f}"
        )

    def _load_policy_model(self) -> None:
        from action_policy import PolicyAgent

        model_path = _resolve_path(
            str(self.policy_config["model_path"]),
            self.configs.policy_path,
        )
        if not model_path.exists():
            raise FileNotFoundError(f"Policy model not found: {model_path}")

        self.get_logger().info(f"Loading policy from: {model_path}")
        action_policy = PolicyAgent(
            str(model_path),
            camera_configs=[
                (
                    spec.obs_key,
                    f"observation.images.{spec.obs_key}",
                    spec.output_size_wh,
                    spec.component == RobotComponent.CAMERA_HEAD_DEPTH,
                    spec.depth_preprocess,
                )
                for spec in self.image_specs
            ],
            profile_inference=self._profile_inference,
            temporal_ensemble_device=self._temporal_ensemble_device,
            cudnn_benchmark=self._cudnn_benchmark,
            allow_tf32=self._allow_tf32,
            fp16_autocast=self._fp16_autocast,
        )
        unused_trailing_action_dim = self._observation_only_external_data_dim()
        action_policy.validate_deployment_layout(
            expected_state_dim=sum(
                _joint_count(self.robot_config, component)
                for component in self.state_order
            ),
            expected_action_dim=sum(
                _joint_count(self.robot_config, component)
                for component in self.action_order
            ),
            unused_trailing_action_dim=unused_trailing_action_dim,
            expected_images={
                f"observation.images.{spec.obs_key}": (
                    3,
                    spec.output_shape[0],
                    spec.output_shape[1],
                )
                if spec.output_shape is not None
                else None
                for spec in self.image_specs
            },
        )
        self._unused_trailing_action_dim = unused_trailing_action_dim
        self.action_policy = action_policy

    def _model_load_worker(self) -> None:
        started = time.perf_counter()
        try:
            self._load_policy_model()
            self.get_logger().info(
                f"Policy model ready in {(time.perf_counter() - started):.2f}s"
            )
        except BaseException as exc:
            self._model_load_error = exc
            self.get_logger().error(f"Policy model loading failed: {exc}")
        finally:
            self._model_load_event.set()

    def _wait_for_model_ready(self) -> None:
        while not self._model_load_event.wait(timeout=0.1):
            if self._stop_requested:
                raise RuntimeError("Stopped while waiting for policy model")
        if self._model_load_error is not None:
            raise RuntimeError(
                f"Policy model loading failed: {self._model_load_error}"
            ) from self._model_load_error
        if self.action_policy is None:
            raise RuntimeError("Policy model loader completed without a model")

    def _load_common_message_types(self):
        from sensor_msgs.msg import Image, JointState
        from std_msgs.msg import Header, Int32

        self.ImageMsg = Image
        self.JointStateMsg = JointState
        self.HeaderMsg = Header
        self.Int32Msg = Int32

    def _external_data_msg_type(self, message_type: Optional[str]):
        name = str(message_type or "std_msgs/Int32").replace("/msg/", "/")
        if name in ("std_msgs/Int32", "Int32"):
            return self.Int32Msg
        if name in ("std_msgs/Float32", "Float32"):
            from std_msgs.msg import Float32

            return Float32
        if name in ("std_msgs/Bool", "Bool"):
            from std_msgs.msg import Bool

            return Bool
        raise ValueError(
            f"Unsupported EXTERNAL_DATA message_type: {message_type}. "
            "Use std_msgs/Int32, std_msgs/Float32, or std_msgs/Bool."
        )

    def _setup_aux_scalar_interfaces(self):
        if RobotComponent.EXTERNAL_DATA not in self.state_order:
            return
        external_cfg = _resolve_external_data_config(self.robot_config, self.policy_config)
        qos = (
            self.make_qos(self.robot_config.get("qos", {}).get("subscriber"))
            if self.adapter_name == "walker_s2"
            else 10
        )
        msg_type = self._external_data_msg_type(external_cfg.get("message_type"))
        self.ros_subscriptions.append(
            self.create_subscription(
                msg_type,
                external_cfg["topic"],
                self.make_scalar_aux_callback(RobotComponent.EXTERNAL_DATA),
                qos,
            )
        )
        self.get_logger().info(
            f"Subscribed EXTERNAL_DATA: {external_cfg['topic']} "
            f"({external_cfg.get('message_type')})"
        )

    def _setup_camera_interfaces(self):
        cameras = self.robot_config["topics"]["cameras"]
        is_walker_s2 = self.adapter_name == "walker_s2"
        camera_qos = (
            self.make_qos(
                self.robot_config.get("qos", {}).get(
                    "camera_subscriber",
                    self.robot_config.get("qos", {}).get("subscriber"),
                )
            )
            if is_walker_s2
            else 10
        )
        for spec in self.image_specs:
            camera_cfg = cameras[spec.component.name]
            topic = camera_cfg["topic"]
            message_type = str(camera_cfg["message_type"])
            if message_type in ("sensor_msgs/Image", "sensor_msgs/msg/Image"):
                msg_type = self.ImageMsg
                callback = self._make_camera_callback(spec)
            elif message_type in ("shm_msgs/Image2m", "shm_msgs/msg/Image2m"):
                try:
                    from shm_msgs.msg import Image2m
                except ImportError as exc:
                    raise RuntimeError(
                        "shm_msgs is required for an Image2m camera. "
                        "Source the Walker S2 ROS2 workspace before running."
                    ) from exc
                msg_type = Image2m
                callback = self._make_image2m_camera_callback(spec)
            else:
                raise ValueError(
                    f"Unsupported camera message type for {spec.component.name}: "
                    f"{message_type}"
                )
            subscription_kwargs = {}
            if is_walker_s2:
                callback_group = MutuallyExclusiveCallbackGroup()
                self._camera_callback_groups.append(callback_group)
                subscription_kwargs["callback_group"] = callback_group
            self.ros_subscriptions.append(
                self.create_subscription(
                    msg_type,
                    topic,
                    callback,
                    camera_qos,
                    **subscription_kwargs,
                )
            )
            self.get_logger().info(
                f"Subscribed image {spec.component.name}: {topic} ({message_type})"
            )

    def make_qos(self, cfg: Optional[Dict[str, Any]]):
        if not cfg:
            return 10
        reliability = str(cfg.get("reliability", "RELIABLE")).upper()
        durability = str(cfg.get("durability", "VOLATILE")).upper()
        history = str(cfg.get("history", "KEEP_LAST")).upper()
        depth = int(cfg.get("depth", 10))
        return QoSProfile(
            reliability=getattr(ReliabilityPolicy, reliability),
            durability=getattr(DurabilityPolicy, durability),
            history=getattr(HistoryPolicy, history),
            depth=depth,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _make_camera_callback(self, spec: ImageSpec):
        def _callback(msg):
            callback_start = time.perf_counter()
            try:
                self._log_camera_callback_once(spec.component, msg)
                desired_encoding = (
                    "rgb8" if self.adapter_name == "walker_s2" else "bgr8"
                )
                image = self.bridge.imgmsg_to_cv2(
                    msg, desired_encoding=desired_encoding
                )
                self._store_camera_image(spec, image)
            except Exception as exc:
                self.get_logger().error(f"Error converting {spec.component.name}: {exc}")
            finally:
                self._record_camera_callback_timing(spec.component, callback_start)

        return _callback

    def _make_image2m_camera_callback(self, spec: ImageSpec):
        def _callback(msg):
            callback_start = time.perf_counter()
            try:
                self._log_camera_callback_once(spec.component, msg)
                height = int(msg.height)
                step = int(msg.step)
                data_size = height * step
                if height <= 0 or int(msg.width) <= 0 or step <= 0:
                    raise ValueError(
                        f"invalid Image2m dimensions: {msg.width}x{msg.height}, step={step}"
                    )
                if len(msg.data) < data_size:
                    raise ValueError(
                        f"Image2m data is short: need {data_size}, got {len(msg.data)}"
                    )

                encoding = self._decode_image2m_encoding(msg.encoding)
                image = self._image2m_to_image(
                    msg=msg,
                    encoding=encoding,
                    height=height,
                    step=step,
                    data_size=data_size,
                    is_depth=spec.component == RobotComponent.CAMERA_HEAD_DEPTH,
                )
                self._store_camera_image(spec, image)
            except Exception as exc:
                self.get_logger().error(f"Error converting {spec.component.name}: {exc}")
            finally:
                self._record_camera_callback_timing(spec.component, callback_start)

        return _callback

    def _image2m_to_image(
        self,
        msg: Any,
        encoding: str,
        height: int,
        step: int,
        data_size: int,
        is_depth: bool,
    ) -> np.ndarray:
        """Decode color Image2m directly, avoiding an intermediate ROS message copy."""
        width = int(msg.width)
        if not is_depth and encoding in {"rgb8", "bgr8"} and step >= width * 3:
            try:
                raw = np.frombuffer(msg.data, dtype=np.uint8, count=data_size)
            except (TypeError, ValueError):
                raw = np.asarray(msg.data, dtype=np.uint8).reshape(-1)[:data_size]
            if raw.size != data_size:
                raise ValueError(
                    f"Image2m data size changed during decode: expected {data_size}, got {raw.size}"
                )
            image = raw.reshape(height, step)[:, : width * 3].reshape(height, width, 3)
            if encoding == "bgr8":
                image = image[:, :, ::-1]
            return np.ascontiguousarray(image)

        standard_msg = self.ImageMsg()
        standard_msg.height = height
        standard_msg.width = width
        standard_msg.encoding = str(encoding)
        standard_msg.is_bigendian = int(msg.is_bigendian)
        standard_msg.step = step
        standard_msg.data = bytes(msg.data[:data_size])
        desired_encoding = "passthrough" if is_depth else "rgb8"
        return self.bridge.imgmsg_to_cv2(standard_msg, desired_encoding=desired_encoding)

    def _record_camera_callback_timing(
        self,
        component: RobotComponent,
        callback_start: float,
    ) -> None:
        elapsed_ms = (time.perf_counter() - callback_start) * 1000.0
        with self._camera_timing_lock:
            timing = self._camera_timing.get(component)
            if timing is None:
                return
            timing["count"] += 1
            timing["total_ms"] += elapsed_ms
            timing["max_ms"] = max(timing["max_ms"], elapsed_ms)

    def _consume_camera_callback_metrics(self) -> List[str]:
        """Return and reset callback metrics collected since the prior inference."""
        now = time.perf_counter()
        with self._camera_timing_lock:
            elapsed_s = max(now - self._camera_timing_last_reset, 1e-9)
            metrics = []
            for component, timing in self._camera_timing.items():
                count = timing["count"]
                if count:
                    metrics.append(
                        f"{component.name} count={count} rate={count / elapsed_s:.1f}Hz "
                        f"avg={timing['total_ms'] / count:.1f}ms max={timing['max_ms']:.1f}ms"
                    )
                timing["count"] = 0
                timing["total_ms"] = 0.0
                timing["max_ms"] = 0.0
            self._camera_timing_last_reset = now
        return metrics

    def _log_camera_callback_once(self, component: RobotComponent, msg: Any):
        if self.adapter_name != "walker_s2":
            return
        if component in self._camera_callback_seen:
            return
        self._camera_callback_seen.add(component)
        encoding = getattr(msg, "encoding", "")
        if not isinstance(encoding, str):
            try:
                encoding = self._decode_image2m_encoding(encoding)
            except Exception:
                encoding = "<decode pending>"
        self.get_logger().info(
            f"First camera callback {component.name}: "
            f"{msg.width}x{msg.height} encoding={encoding}"
        )

    @staticmethod
    def _decode_image2m_encoding(encoding_field: Any) -> str:
        """Decode shm_msgs/String fixed uint8 storage into a ROS encoding."""
        raw = getattr(encoding_field, "data", encoding_field)
        if isinstance(raw, str):
            decoded = raw
        else:
            values = np.asarray(raw, dtype=np.uint8).reshape(-1)
            declared_size = getattr(encoding_field, "size", None)
            if isinstance(declared_size, (int, np.integer)) and 0 < declared_size <= len(values):
                values = values[: int(declared_size)]
            raw_bytes = bytes(values.tolist()).split(b"\x00", 1)[0]
            decoded = raw_bytes.decode("ascii", errors="ignore").strip()

        # Some UBTECH shm_msgs/String builds expose an extra trailing byte
        # after the encoding text (for example "bgr81"). Prefer a known ROS
        # encoding prefix rather than passing the storage artifact to cv_bridge.
        known_encodings = (
            "16UC1",
            "16SC1",
            "32FC1",
            "bgr8",
            "rgb8",
            "bgra8",
            "rgba8",
            "mono8",
            "mono16",
        )
        for known in known_encodings:
            if decoded.startswith(known):
                return known
        raise ValueError(f"Unsupported Image2m encoding bytes: {decoded!r}")

    def _store_camera_image(self, spec: ImageSpec, image: np.ndarray):
        image = preprocess_camera_image(
            image,
            spec,
            is_depth=spec.component == RobotComponent.CAMERA_HEAD_DEPTH,
        )
        expected = spec.output_shape
        if expected is not None:
            actual_hw = (int(image.shape[0]), int(image.shape[1]))
            if actual_hw != expected:
                raise ValueError(
                    f"{spec.component.name} preprocess produced {actual_hw[1]}x{actual_hw[0]} "
                    f"but expected {expected[1]}x{expected[0]} "
                    f"({_describe_image_preprocess(spec)})"
                )
        self.images[spec.component] = image
        self._image_received_at[spec.component] = time.monotonic()
        if (
            self.adapter_name == "walker_s2"
            and spec.component not in self._camera_ready_logged
        ):
            self._camera_ready_logged.add(spec.component)
            self.get_logger().info(
                f"Camera ready {spec.component.name}: shape={image.shape} "
                f"dtype={image.dtype} ({_describe_image_preprocess(spec)})"
            )

    def make_scalar_gripper_callback(self, component: RobotComponent):
        def _callback(msg):
            self.gripper_state[component] = float(msg.data)

        return _callback

    def make_scalar_aux_callback(self, component: RobotComponent):
        def _callback(msg):
            self.aux_scalar_state[component] = float(msg.data)

        return _callback

    def make_hand_joint_state_callback(self, component: RobotComponent):
        def _callback(msg):
            self.joint_state[component] = np.array(list(msg.position), dtype=np.float64)

        return _callback

    # ------------------------------------------------------------------
    # Observation and action mapping
    # ------------------------------------------------------------------

    def gripper_config(self, component: RobotComponent) -> Dict[str, Any]:
        return _resolve_gripper_config(self.robot_config, component, self.policy_config)

    def _waist_yaw_vector_index(self, order: List[RobotComponent]) -> Optional[int]:
        if RobotComponent.WAIST not in order:
            return None
        offset = 0
        for component in order:
            if component == RobotComponent.WAIST:
                count = _joint_count(self.robot_config, component)
                if count <= WAIST_YAW_INDEX_IN_BODY:
                    return None
                return offset + WAIST_YAW_INDEX_IN_BODY
            offset += _joint_count(self.robot_config, component)
        return None

    def _live_waist_yaw_rad(self) -> Optional[float]:
        waist = self.joint_state.get(RobotComponent.WAIST)
        if waist is None or len(waist) <= WAIST_YAW_INDEX_IN_BODY:
            return None
        return float(waist[WAIST_YAW_INDEX_IN_BODY])

    def _obs_waist_yaw_rad(self, obs: Dict[str, Any]) -> Optional[float]:
        idx = self._waist_yaw_vector_index(self.state_order)
        joints = obs.get("arm_gripper_joints")
        if idx is None or joints is None:
            return None
        try:
            return float(joints[idx])
        except (TypeError, ValueError, IndexError):
            return None

    def _action_waist_yaw_rad(self, action: np.ndarray) -> Optional[float]:
        idx = self._waist_yaw_vector_index(self.action_order)
        if idx is None or action is None:
            return None
        try:
            return float(action[idx])
        except (TypeError, ValueError, IndexError):
            return None

    def _gripper_action_indices(self) -> Dict[str, int]:
        indices: Dict[str, int] = {}
        offset = 0
        for component in self.action_order:
            size = _joint_count(self.robot_config, component)
            if component == RobotComponent.LEFT_GRIPPER:
                indices["left"] = offset
            elif component == RobotComponent.RIGHT_GRIPPER:
                indices["right"] = offset
            offset += size
        return indices

    def _schedule_hand_key_hold_after_chunk_append(
        self, *, remaining_before: int
    ) -> None:
        cfg = self._hand_key_selection
        hold = int(cfg.hold_publish_ticks_on_chunk_append)
        if hold <= 0 or not self.policy_config.get("hand_key_poses"):
            return
        with self.action_queue_lock:
            base_tick = int(self._published_action_tick)
        until_tick = base_tick + int(remaining_before) + hold
        for side in ("left", "right"):
            self._hand_key_hold_until_publish_tick[side] = until_tick
        self.get_logger().info(
            f"Hand key hold until publish_tick={until_tick} "
            f"(queue_tail={remaining_before} hold={hold})"
        )

    def _waist_prefix_reference_rad(self, obs: Dict[str, Any]) -> Optional[float]:
        live = self._live_waist_yaw_rad()
        obs_q = self._obs_waist_yaw_rad(obs)
        mode = self._waist_chunk_prefix_fix.reference
        if mode == "live":
            return live if live is not None else obs_q
        if mode == "obs":
            return obs_q if obs_q is not None else live
        if live is not None and obs_q is not None:
            return max(live, obs_q)
        return live if live is not None else obs_q

    def _apply_waist_chunk_prefix_fix(
        self,
        actions_list: List[np.ndarray],
        obs: Dict[str, Any],
        yaw_index: int,
    ) -> Optional[Dict[str, float]]:
        cfg = self._waist_chunk_prefix_fix
        if not cfg.enabled or not actions_list:
            return None
        ref = self._waist_prefix_reference_rad(obs)
        if ref is None:
            return None
        try:
            first_before = float(actions_list[0][yaw_index])
            last_before = float(actions_list[-1][yaw_index])
        except (TypeError, ValueError, IndexError):
            return None
        eps = cfg.epsilon_rad
        if cfg.only_when_chunk_increase and last_before <= first_before + eps:
            return None
        if first_before >= ref - eps:
            return None

        shift = ref - first_before
        for index, action in enumerate(actions_list):
            row = np.asarray(action, dtype=np.float64).copy()
            row[yaw_index] = float(row[yaw_index]) + shift
            actions_list[index] = row

        ramp = min(cfg.ramp_steps, len(actions_list))
        for index in range(ramp):
            row = np.asarray(actions_list[index], dtype=np.float64).copy()
            if float(row[yaw_index]) < ref - eps:
                row[yaw_index] = ref
            actions_list[index] = row

        first_after = float(actions_list[0][yaw_index])
        return {
            "ref_rad": ref,
            "shift_rad": shift,
            "first_before_rad": first_before,
            "first_after_rad": first_after,
        }

    def _live_policy_state_vector(self) -> Optional[np.ndarray]:
        try:
            return np.asarray(self.get_current_state(), dtype=np.float64)
        except Exception:
            return None

    def _action_prefix_reference_vector(
        self, obs: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        cfg = self._action_chunk_prefix_fix
        live = self._live_policy_state_vector()
        with self.action_queue_lock:
            last_cmd = (
                None
                if self.last_action is None
                else np.asarray(self.last_action, dtype=np.float64)
            )
        if cfg.reference == "live_state":
            return live
        if cfg.reference == "last_command":
            if last_cmd is not None:
                return last_cmd
            return live
        # max_live_command: per-dimension max(live, last) when both exist
        if live is None:
            return last_cmd
        if last_cmd is None:
            return live
        if live.shape != last_cmd.shape:
            return last_cmd
        return np.maximum(live, last_cmd)

    def _apply_action_chunk_prefix_fix(
        self,
        actions_list: List[np.ndarray],
        obs: Dict[str, Any],
        *,
        on_append: bool,
        yaw_index: int,
    ) -> Optional[Dict[str, float]]:
        cfg = self._action_chunk_prefix_fix
        if not cfg.enabled or not actions_list:
            return None
        if cfg.on_append_only and not on_append:
            return None
        ref = self._action_prefix_reference_vector(obs)
        if ref is None:
            return None
        first = np.asarray(actions_list[0], dtype=np.float64)
        if first.shape != ref.shape:
            return None
        delta = ref - first
        shift_norm = float(np.linalg.norm(delta))
        if shift_norm < cfg.min_shift_rad:
            return None
        if cfg.max_joint_shift_rad > 0:
            delta = np.clip(
                delta,
                -cfg.max_joint_shift_rad,
                cfg.max_joint_shift_rad,
            )
        first_before = first.copy()
        for index, action in enumerate(actions_list):
            row = np.asarray(action, dtype=np.float64) + delta
            actions_list[index] = row
        if cfg.blend_steps > 0:
            steps = min(cfg.blend_steps, len(actions_list))
            for index in range(steps):
                alpha = float(index + 1) / float(steps)
                blended = ref + alpha * (
                    np.asarray(actions_list[index], dtype=np.float64) - ref
                )
                actions_list[index] = blended
        first_after = np.asarray(actions_list[0], dtype=np.float64)
        out: Dict[str, float] = {
            "shift_norm_rad": shift_norm,
            "on_append": float(on_append),
        }
        if 0 <= yaw_index < len(first_before):
            out["first_q5_before_rad"] = float(first_before[yaw_index])
            out["first_q5_after_rad"] = float(first_after[yaw_index])
            out["ref_q5_rad"] = float(ref[yaw_index])
        grip_idx = self._gripper_action_indices()
        for side, idx in grip_idx.items():
            if 0 <= idx < len(first_before):
                out[f"grip_{side}_before"] = float(first_before[idx])
                out[f"grip_{side}_after"] = float(first_after[idx])
                if idx < len(ref):
                    out[f"grip_{side}_ref"] = float(ref[idx])
        return out

    def _build_chunk_splice_weights(self) -> np.ndarray:
        cfg = self._chunk_queue_splice
        parts: List[float] = []
        for component in self.action_order:
            if component in (
                RobotComponent.LEFT_GRIPPER,
                RobotComponent.RIGHT_GRIPPER,
            ):
                weight = cfg.gripper_weight
            elif component in (RobotComponent.LEFT_HAND, RobotComponent.RIGHT_HAND):
                weight = cfg.hand_weight
            else:
                weight = 1.0
            size = _joint_count(self.robot_config, component)
            parts.extend([float(weight)] * int(size))
        return np.asarray(parts, dtype=np.float64)

    def _chunk_splice_anchor_vector(self) -> Optional[np.ndarray]:
        with self.action_queue_lock:
            if self.last_action is not None:
                return np.asarray(self.last_action, dtype=np.float64).reshape(-1)
        live = self._live_policy_state_vector()
        return None if live is None else live.reshape(-1)

    def _chunk_splice_forward_direction(
        self,
        anchor: np.ndarray,
        old_queue_head: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        cfg = self._chunk_queue_splice
        forward: Optional[np.ndarray] = None
        if old_queue_head is not None and old_queue_head.shape == anchor.shape:
            forward = np.asarray(old_queue_head, dtype=np.float64) - anchor
        if forward is None or float(np.linalg.norm(forward)) < cfg.min_forward_norm:
            live = self._live_policy_state_vector()
            if live is not None and live.reshape(-1).shape == anchor.shape:
                forward = live.reshape(-1) - anchor
        if forward is None:
            return None
        weighted = forward * self._chunk_splice_weights
        if float(np.linalg.norm(weighted)) < cfg.min_forward_norm:
            return None
        return forward

    def _trim_chunk_reverse_prefix(
        self,
        actions: List[np.ndarray],
        anchor: np.ndarray,
        forward: np.ndarray,
    ) -> Tuple[List[np.ndarray], Dict[str, float]]:
        cfg = self._chunk_queue_splice
        if not actions:
            return actions, {"reverse_trimmed": 0.0, "reverse_remain": 0.0}
        weights = self._chunk_splice_weights
        if weights.shape != anchor.shape:
            return actions, {
                "reverse_trimmed": 0.0,
                "reverse_remain": float(len(actions)),
            }
        fw = forward * weights
        fn = float(np.linalg.norm(fw))
        if fn < cfg.min_forward_norm:
            return actions, {
                "reverse_trimmed": 0.0,
                "reverse_remain": float(len(actions)),
            }
        unit = fw / fn

        trim = 0
        for index, row in enumerate(actions):
            delta = (np.asarray(row, dtype=np.float64).reshape(-1) - anchor) * weights
            projection = float(np.dot(delta, unit))
            if projection >= -float(cfg.reverse_projection_rad):
                break
            trim = index + 1

        if trim >= len(actions):
            keep = min(len(actions), max(1, int(cfg.min_actions_after_trim)))
            trim = max(0, len(actions) - keep)

        trimmed = actions[trim:]
        meta = {
            "reverse_trimmed": float(trim),
            "reverse_remain": float(len(trimmed)),
        }
        if trim > 0 and trimmed:
            meta["first_proj_rad"] = float(
                np.dot(
                    (np.asarray(trimmed[0], dtype=np.float64).reshape(-1) - anchor)
                    * weights,
                    unit,
                )
            )
        return trimmed, meta

    def _should_defer_chunk_splice(self, anchor: np.ndarray) -> bool:
        cfg = self._chunk_queue_splice
        if not cfg.defer_until_settle or self.last_action is None:
            return False
        if self._waypoint_settled():
            return False
        live = self._live_policy_state_vector()
        if live is None or live.reshape(-1).shape != anchor.shape:
            return False
        err = float(np.max(np.abs(live.reshape(-1) - anchor)))
        return err <= float(cfg.defer_finish_rad)

    def _try_flush_pending_chunk_splice(self) -> bool:
        if self._pending_chunk_splice is None:
            return False
        cfg = self._chunk_queue_splice
        if cfg.defer_until_settle and not self._waypoint_settled():
            return False
        with self.action_queue_lock:
            if self._pending_chunk_splice is None:
                return False
            self.action_queue.clear()
            self.action_queue.extend(self._pending_chunk_splice)
            self._pending_chunk_splice = None
        self.get_logger().info("Chunk queue splice: flushed pending chunk")
        return True

    def _component_state(self, component: RobotComponent) -> Optional[np.ndarray]:
        if component in (RobotComponent.LEFT_GRIPPER, RobotComponent.RIGHT_GRIPPER):
            return np.array([self._virtual_gripper_value(component)], dtype=np.float64)
        if component == RobotComponent.EXTERNAL_DATA:
            value = self.aux_scalar_state.get(component)
            if value is None:
                return None
            return np.array([value], dtype=np.float64)
        if component in self.joint_state:
            state = self.joint_state[component]
            return None if state is None else np.array(state, dtype=np.float64)
        raise ValueError(f"Component has no state vector: {component.name}")

    def _default_gripper_observation_scalar(self, component: RobotComponent) -> float:
        """Fallback gripper dim for policy state (before first publish)."""
        side = self._hand_key_side(component)
        if side and self.policy_config.get("hand_key_poses"):
            pairs = self._hand_key_rows(side)
            if pairs:
                return float(pairs[0][0])
        cfg = self.gripper_config(component)
        return float(cfg.get("open_value", 0.0))

    def _gripper_scalar_from_last_command(self, component: RobotComponent) -> Optional[float]:
        with self.action_queue_lock:
            last = self.last_action
        if last is None:
            return None
        try:
            sliced = self._slice_action(np.asarray(last, dtype=np.float64).reshape(-1))
        except (TypeError, ValueError):
            return None
        values = sliced.get(component)
        if values is None:
            return None
        return float(np.asarray(values, dtype=np.float64).reshape(-1)[0])

    def _nearest_hand_key_from_live_joints(
        self, component: RobotComponent
    ) -> Optional[float]:
        """Map measured hand joints to the closest hand-pose file key (training scale)."""
        side = self._hand_key_side(component)
        if side is None:
            return None
        pairs = self._hand_key_rows(side)
        if not pairs:
            return None
        hand = GRIPPER_TO_HAND.get(component)
        if hand is None:
            return None
        live = self.joint_state.get(hand)
        if live is None:
            return None
        live = np.asarray(live, dtype=np.float64).reshape(-1)
        best_key: Optional[float] = None
        best_dist: Optional[float] = None
        for key, row in pairs:
            joints = np.asarray(row.get("joints"), dtype=np.float64).reshape(-1)
            if joints.size == 0 or joints.shape != live.shape:
                continue
            dist = float(np.linalg.norm(live - joints))
            if best_dist is None or dist < best_dist - 1e-12:
                best_dist = dist
                best_key = float(key)
        return best_key

    def _virtual_gripper_value(self, component: RobotComponent) -> float:
        if self.policy_config.get("hand_key_poses"):
            mapped = self._nearest_hand_key_from_live_joints(component)
            if mapped is not None:
                return mapped
            return self._default_gripper_observation_scalar(component)
        cfg = self.gripper_config(component)
        source = cfg.get("state_source", "last_command")
        if source == "last_command":
            from_cmd = self._gripper_scalar_from_last_command(component)
            if from_cmd is not None:
                return from_cmd
            return self._default_gripper_observation_scalar(component)
        if source == "scalar_topic":
            return float(self.gripper_state.get(component, 0.0))
        if source == "hand_average":
            hand = GRIPPER_TO_HAND[component]
            hand_state = self.joint_state.get(hand)
            if hand_state is None:
                return float(self.gripper_state.get(component, 0.0))
            open_preset = np.asarray(cfg.get("open_preset", np.zeros_like(hand_state)), dtype=np.float64)
            close_preset = np.asarray(cfg.get("close_preset", np.ones_like(hand_state)), dtype=np.float64)
            if len(open_preset) != len(hand_state) or len(close_preset) != len(hand_state):
                return float(self.gripper_state.get(component, 0.0))
            d_open = float(np.linalg.norm(hand_state - open_preset))
            d_close = float(np.linalg.norm(hand_state - close_preset))
            denom = d_open + d_close
            if denom <= 1e-9:
                return 0.0
            return float(np.clip(d_open / denom, 0.0, 1.0))
        return float(self.gripper_state.get(component, 0.0))

    def get_current_state(self) -> np.ndarray:
        parts = []
        for component in self.state_order:
            state = self._component_state(component)
            if state is None:
                parts.append(np.zeros(_joint_count(self.robot_config, component), dtype=np.float64))
            else:
                parts.append(state)
        return np.concatenate(parts) if parts else np.array([], dtype=np.float64)

    def get_obs(self) -> Optional[Dict[str, Any]]:
        obs = {"images": {}, "arm_gripper_joints": None}
        for spec in self.image_specs:
            image = self.images.get(spec.component)
            if image is None:
                return None
            obs["images"][spec.obs_key] = image

        stale = collect_stale_camera_ages(
            self.image_specs,
            self.images,
            self._image_received_at,
            time.monotonic(),
            self._max_image_age_s,
        )
        if stale:
            raise StaleCameraFrameError(
                "Stale camera frames, refusing to infer: "
                f"{_format_stale_camera_detail(stale)} "
                f"(max_image_age_s={self._max_image_age_s})"
            )

        state_parts = []
        for component in self.state_order:
            state = self._component_state(component)
            if state is None:
                return None
            state_parts.append(state)
        obs["arm_gripper_joints"] = (
            np.concatenate(state_parts) if state_parts else np.array([], dtype=np.float64)
        )
        return obs

    def _save_inference_images_if_needed(self, obs: Dict[str, Any]) -> None:
        """Persist camera frames fed to policy inference."""
        if self._save_all_inference_images:
            self._write_inference_images(obs, sequential=True)
            self._inference_images_saved = True
            return
        if not self._save_inference_images or self._inference_images_saved:
            return
        self._inference_images_saved = True
        self._write_inference_images(obs, sequential=False)

    def _write_inference_images(self, obs: Dict[str, Any], sequential: bool) -> None:
        try:
            self._inference_image_log_dir.mkdir(parents=True, exist_ok=True)
            images = obs.get("images", {})
            camera_by_key = {
                camera_key: (target_size, is_depth, depth_preprocess)
                for camera_key, _obs_key, target_size, is_depth, depth_preprocess in (
                    unpack_camera_config(config)
                    for config in self.action_policy.camera_configs
                )
            }
            seq = None
            if sequential:
                self._inference_image_seq += 1
                seq = self._inference_image_seq
            log_paths = not sequential or seq == 1

            for spec in self.image_specs:
                image = images.get(spec.obs_key)
                if not isinstance(image, np.ndarray):
                    self.get_logger().warning(
                        f"Inference image not saved {spec.obs_key}: missing or not a NumPy array"
                    )
                    continue

                expected = spec.output_shape
                if expected is not None:
                    actual_hw = (int(image.shape[0]), int(image.shape[1]))
                    if actual_hw != expected:
                        self.get_logger().error(
                            f"Inference image {spec.obs_key} has shape {actual_hw[1]}x{actual_hw[0]} "
                            f"but preprocess expected {expected[1]}x{expected[0]} "
                            f"({_describe_image_preprocess(spec)}). "
                            "Saving aborted for this camera — check resize/crop config."
                        )
                        continue

                # Match PolicyAgent.prepare_inference_obs image path exactly.
                target_size, is_depth, depth_preprocess = camera_by_key.get(
                    spec.obs_key, (None, False, None)
                )
                image_to_write = self.action_policy._prepare_image(
                    image, target_size, is_depth, depth_preprocess
                )
                if image_to_write.ndim == 3 and image_to_write.shape[2] == 3:
                    image_to_write = cv2.cvtColor(image_to_write, cv2.COLOR_RGB2BGR)

                height, width = int(image_to_write.shape[0]), int(image_to_write.shape[1])
                if seq is None:
                    filename = (
                        f"{self._policy_image_stem}_{self._inference_image_run_id}_"
                        f"{spec.obs_key}_{width}x{height}.png"
                    )
                else:
                    filename = (
                        f"{self._policy_image_stem}_{self._inference_image_run_id}_"
                        f"{seq:06d}_{spec.obs_key}_{width}x{height}.png"
                    )
                output_path = self._inference_image_log_dir / filename
                if not cv2.imwrite(str(output_path), image_to_write):
                    self.get_logger().error(
                        f"Failed to save inference image {spec.obs_key}: {output_path}"
                    )
                    continue
                if log_paths:
                    self.get_logger().info(
                        f"Inference image saved {spec.obs_key}: {output_path} "
                        f"shape={image_to_write.shape} "
                        f"preprocess={_describe_image_preprocess(spec)}"
                    )
        except Exception as exc:
            self.get_logger().error(f"Failed to save inference images: {exc}")

    def _missing_observations(self) -> List[str]:
        missing = [
            f"image:{spec.component.name}"
            for spec in self.image_specs
            if self.images.get(spec.component) is None
        ]
        for component in self.state_order:
            if self._component_state(component) is None:
                missing.append(f"state:{component.name}")
        return missing

    def _observation_only_external_data_dim(self) -> int:
        if (
            RobotComponent.EXTERNAL_DATA in self.state_order
            and RobotComponent.EXTERNAL_DATA not in self.action_order
        ):
            return _joint_count(self.robot_config, RobotComponent.EXTERNAL_DATA)
        return 0

    def publish_action(self, action: np.ndarray):
        targets = self._slice_action(action)
        targets = self._apply_dummy_motion(targets)
        targets = self._apply_arm_live_guard(targets)
        targets = self._postprocess_action_targets(targets)
        self._log_gripper_pair(targets)
        self._publish_targets(targets)
        self._write_oap_lerobot_live_targets(action, targets)

    def _targets_to_policy_components(
        self, targets: Dict[RobotComponent, np.ndarray]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for component, values in targets.items():
            key = component.name
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
            if arr.size == 1:
                out[key] = round(float(arr[0]), 5)
            else:
                out[key] = [round(float(v), 5) for v in arr]
        return out

    def _vector_to_policy_components(self, action: np.ndarray) -> Dict[str, Any]:
        sliced = self._slice_action(np.asarray(action, dtype=np.float64).reshape(-1))
        return self._targets_to_policy_components(sliced)

    def _write_oap_lerobot_live_targets(
        self,
        published: np.ndarray,
        command_targets: Optional[Dict[RobotComponent, np.ndarray]] = None,
    ) -> None:
        try:
            command = np.asarray(published, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return
        with self.action_queue_lock:
            queue_remaining = len(self.action_queue)
            queue_head = (
                None
                if not self.action_queue
                else np.asarray(self.action_queue[0], dtype=np.float64).reshape(-1)
            )
            last_cmd = (
                None
                if self.last_action is None
                else np.asarray(self.last_action, dtype=np.float64).reshape(-1)
            )
            publish_tick = int(self._published_action_tick)
        live = self._live_policy_state_vector()
        approach_err = self._waypoint_command_error()
        payload: Dict[str, Any] = {
            "policy": self.policy_config.get("name"),
            "queue_remaining": queue_remaining,
            "publish_tick": publish_tick,
            "inference_count": int(self._inference_count),
            "published": (
                self._targets_to_policy_components(command_targets)
                if command_targets is not None
                else self._vector_to_policy_components(command)
            ),
            "waypoint_approach_err_rad": (
                None if approach_err is None else round(float(approach_err), 5)
            ),
        }
        if last_cmd is not None and last_cmd.shape == command.shape:
            payload["last_command"] = self._vector_to_policy_components(last_cmd)
        if queue_head is not None:
            payload["queue_head"] = self._vector_to_policy_components(queue_head)
        if live is not None and live.reshape(-1).shape == command.shape:
            payload["live"] = self._vector_to_policy_components(live.reshape(-1))
        payload["gripper_topic"] = {
            RobotComponent.LEFT_GRIPPER.name: round(
                float(self.gripper_state.get(RobotComponent.LEFT_GRIPPER, 0.0)), 5
            ),
            RobotComponent.RIGHT_GRIPPER.name: round(
                float(self.gripper_state.get(RobotComponent.RIGHT_GRIPPER, 0.0)), 5
            ),
        }
        payload["gripper_observation"] = {
            RobotComponent.LEFT_GRIPPER.name: round(
                float(self._virtual_gripper_value(RobotComponent.LEFT_GRIPPER)), 5
            ),
            RobotComponent.RIGHT_GRIPPER.name: round(
                float(self._virtual_gripper_value(RobotComponent.RIGHT_GRIPPER)), 5
            ),
        }
        _merge_oap_lerobot_state_file(payload)

    def _apply_arm_live_guard(
        self, targets: Dict[RobotComponent, np.ndarray]
    ) -> Dict[RobotComponent, np.ndarray]:
        cfg = self._arm_live_guard
        if not cfg.enabled:
            return targets
        processed = dict(targets)
        ahead = float(cfg.max_ahead_rad)
        behind = float(cfg.max_behind_rad)
        for component in (RobotComponent.LEFT_ARM, RobotComponent.RIGHT_ARM):
            command = targets.get(component)
            live = self._component_state(component)
            if command is None or live is None:
                continue
            cmd = np.asarray(command, dtype=np.float64).reshape(-1)
            state = np.asarray(live, dtype=np.float64).reshape(-1)
            if cmd.shape != state.shape:
                continue
            delta = np.clip(cmd - state, -behind, ahead)
            processed[component] = state + delta
        return processed

    def _postprocess_action_targets(
        self,
        targets: Dict[RobotComponent, np.ndarray],
    ) -> Dict[RobotComponent, np.ndarray]:
        """Filter each arm joint immediately before sending robot targets."""
        if not self._arm_action_postprocess_config.enabled:
            return targets

        processed = dict(targets)
        timestamp = time.monotonic()
        for component, action_filter in self._arm_action_filters.items():
            values = targets.get(component)
            if values is not None:
                processed[component] = action_filter.filter(
                    values,
                    timestamp=timestamp,
                )
        return processed

    def _reset_arm_action_postprocess(self) -> None:
        for action_filter in self._arm_action_filters.values():
            action_filter.reset()
        self._publish_target_ema_state = None
        self._gripper_scalar_ema = {}
        self._hand_key_logged = {}
        self._hand_key_hold_until_publish_tick = {}

    def _blend_publish_action(self, action: np.ndarray) -> np.ndarray:
        cfg = self._publish_target_blend
        vector = np.asarray(action, dtype=np.float64).reshape(-1)
        if not cfg.enabled:
            return vector
        prev = self._publish_target_ema_state
        if prev is None or prev.shape != vector.shape:
            self._publish_target_ema_state = vector.copy()
            return vector
        alpha = float(cfg.alpha)
        blended = alpha * vector + (1.0 - alpha) * prev
        self._publish_target_ema_state = blended.copy()
        return blended

    def _reset_dummy_motion(self) -> None:
        self._dummy_motion_bases = {}
        self._dummy_motion_scales = {}
        self._dummy_motion_t0 = None
        self._dummy_motion_skip_warned = False
        self._dummy_motion_applied_logged = False
        self._dummy_motion_blocked: set = set()

    @staticmethod
    def _is_uninitialized_joint_pose(values: Optional[np.ndarray]) -> bool:
        if values is None:
            return True
        pose = np.asarray(values, dtype=np.float64).reshape(-1)
        if pose.size == 0 or not np.all(np.isfinite(pose)):
            return True
        return bool(np.max(np.abs(pose)) < 1e-3)

    def _dummy_joint_randomizers(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng()
        amp_scale = rng.uniform(0.9, 1.1, size=size)
        freq_scale = rng.uniform(0.75, 1.25, size=size)
        return amp_scale, freq_scale

    def _store_dummy_motion_base(self, component: RobotComponent, live: np.ndarray) -> np.ndarray:
        base = np.asarray(live, dtype=np.float64).reshape(-1).copy()
        self._dummy_motion_bases[component] = base
        self._dummy_motion_scales[component] = self._dummy_joint_randomizers(base.size)
        return base

    def _capture_dummy_motion_bases(self) -> None:
        self._dummy_motion_bases = {}
        self._dummy_motion_scales = {}
        self._dummy_motion_t0 = time.monotonic()
        self._dummy_motion_blocked = set()
        blocked = []
        for component in self._dummy_motion_components:
            live = self.joint_state.get(component)
            if self._is_uninitialized_joint_pose(live):
                self._dummy_motion_blocked.add(component)
                blocked.append(component.name)
                continue
            self._store_dummy_motion_base(component, live)
        if blocked:
            self.get_logger().warning(
                "Dummy motion blocked, current joints missing or zero: "
                + ", ".join(blocked)
            )
        if self._dummy_motion_bases:
            self.get_logger().info(
                "Dummy motion base captured from current joints: "
                + ", ".join(comp.name for comp in self._dummy_motion_bases)
            )

    def _dummy_motion_base(self, component: RobotComponent) -> Optional[np.ndarray]:
        if component in self._dummy_motion_blocked:
            return None
        cached = self._dummy_motion_bases.get(component)
        if cached is not None:
            return cached
        live = self.joint_state.get(component)
        if self._is_uninitialized_joint_pose(live):
            self._dummy_motion_blocked.add(component)
            return None
        return self._store_dummy_motion_base(component, live)

    def _apply_dummy_motion(
        self,
        targets: Dict[RobotComponent, np.ndarray],
    ) -> Dict[RobotComponent, np.ndarray]:
        if not self._dummy_motion_config.enabled or not self._dummy_motion_components:
            return targets
        if self._dummy_motion_t0 is None:
            self._dummy_motion_t0 = time.monotonic()
        elapsed = time.monotonic() - self._dummy_motion_t0
        processed = dict(targets)
        applied = []
        skipped = []
        for index, component in enumerate(self._dummy_motion_components):
            if component in processed:
                continue
            base = self._dummy_motion_base(component)
            if base is None:
                skipped.append(component.name)
                continue
            amp_scale, freq_scale = self._dummy_motion_scales.get(
                component, (None, None)
            )
            processed[component] = dummy_arm_motion(
                base,
                elapsed,
                amplitude_deg=self._dummy_motion_config.amplitude_deg,
                period_sec=self._dummy_motion_config.period_sec,
                phase_seed=index * 1.7,
                amp_scale=amp_scale,
                freq_scale=freq_scale,
            )
            applied.append(component.name)
        if skipped and not self._dummy_motion_skip_warned:
            self._dummy_motion_skip_warned = True
            self.get_logger().warning(
                "Dummy motion skipped, missing or zero joints for: "
                + ", ".join(skipped)
            )
        if applied and not self._dummy_motion_applied_logged:
            self._dummy_motion_applied_logged = True
            self.get_logger().info(
                "Dummy motion publishing: "
                + ", ".join(applied)
                + f" amp={self._dummy_motion_config.amplitude_deg:g}deg"
            )
        return processed

    def _gripper_threshold_type(self) -> str:
        return str(
            self.policy_config.get("action", {}).get("gripper_threshold_type", "single")
        ).lower()

    def _single_gripper_threshold(self, component: RobotComponent) -> float:
        policy_grip = (self.policy_config.get("grippers") or {}).get(component.name) or {}
        if isinstance(policy_grip, dict) and "threshold" in policy_grip:
            return float(policy_grip["threshold"])
        action_threshold = self.policy_config.get("action", {}).get("gripper_threshold")
        if action_threshold is not None:
            return float(action_threshold)
        return 0.6

    def _decide_gripper_closed(self, component: RobotComponent, scalar: float) -> int:
        previous = int(self._gripper_command_bit.get(component, 0))
        value = float(scalar)
        if self._gripper_threshold_type() == "dual":
            action = self.policy_config.get("action") or {}
            close_th = float(action["gripper_threshold_close"])
            open_th = float(action["gripper_threshold_open"])
            if value >= close_th:
                return 1
            if value <= open_th:
                return 0
            return previous
        return 1 if value >= self._single_gripper_threshold(component) else 0

    def _gripper_closed(self, component: RobotComponent, scalar: float) -> int:
        return self._decide_gripper_closed(component, scalar)

    def _right_gripper_scalar_index(self) -> Optional[int]:
        if RobotComponent.RIGHT_GRIPPER not in self.action_order:
            return None
        indices = self._gripper_action_indices()
        right = indices.get("right")
        return int(right) if right is not None else None

    def _slice_raw_inference_for_gripper_export(
        self, actions_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """ACT output only (no prefix_fix / splice / hand keys); same skip as queue."""
        rows = [np.asarray(row, dtype=np.float64).copy() for row in actions_list]
        if self._action_chunk_execute_steps is not None:
            rows = rows[: int(self._action_chunk_execute_steps)]
        skip_first = int(self._action_chunk_skip_first_steps)
        if skip_first > 0:
            rows = rows[skip_first:]
        return rows

    def _init_right_gripper_chunk_trace(self) -> None:
        """Create trace file at session start (append after first 0.3-range chunk)."""
        self._right_gripper_03_range_seen = False
        path = _right_gripper_chunk_trace_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            path.write_text(
                "# RAW ACT RIGHT_GRIPPER per queue step (one line = one step)\n"
                "# Source: policy inference only — NOT prefix_fix, NOT hand keys, NOT publish\n"
                f"# session_start={stamp} policy={self._policy_image_stem}\n"
                f"# absolute_path={path.resolve()}\n"
                "# Data below starts at the first chunk with any raw step >= 0.27 (0.3 entry)\n",
                encoding="utf-8",
            )
            self.get_logger().info(
                f"Right gripper raw inference trace: {path.resolve()}"
            )
        except OSError as exc:
            self.get_logger().warning(
                f"Could not init right gripper trace at {path}: {exc}"
            )

    def _export_right_gripper_chunk_trace(self, work_list: List[np.ndarray]) -> None:
        """Append raw-inference right-gripper scalars (one line per step) after 0.3 entry."""
        idx = self._right_gripper_scalar_index()
        if idx is None or not work_list:
            return
        scalars: List[float] = []
        for row in work_list:
            arr = np.asarray(row, dtype=np.float64).reshape(-1)
            if idx >= arr.size:
                return
            scalars.append(float(arr[idx]))
        in_03 = any(v >= 0.3 - 0.03 for v in scalars)
        path = _right_gripper_chunk_trace_path()
        if not self._right_gripper_03_range_seen:
            if not in_03:
                return
            self._right_gripper_03_range_seen = True
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write("# --- raw 0.3 entry; logging all following chunks ---\n")
            except OSError:
                return
        with self._right_gripper_export_lock:
            with self.action_queue_lock:
                inf_n = int(self._inference_count) + 1
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(f"\n# inference={inf_n} steps={len(scalars)}\n")
                    for value in scalars:
                        fh.write(f"{value:.8f}\n")
            except OSError:
                pass

    def _log_gripper_pair(self, targets: Dict[RobotComponent, np.ndarray]) -> None:
        left = targets.get(RobotComponent.LEFT_GRIPPER)
        right = targets.get(RobotComponent.RIGHT_GRIPPER)
        if left is None and right is None:
            return

        def action_txt(values: Optional[np.ndarray]) -> str:
            if values is None:
                return "-"
            return f"{round(float(values[0]), 3):.3f}"

        def bit_txt(component: RobotComponent, values: Optional[np.ndarray]) -> str:
            if values is None:
                return "-"
            return str(self._gripper_closed(component, float(values[0])))

        self.get_logger().info(
            "LEFT : RIGHT --- "
            f"{action_txt(left)} : {action_txt(right)} ---- "
            f"{bit_txt(RobotComponent.LEFT_GRIPPER, left)} : "
            f"{bit_txt(RobotComponent.RIGHT_GRIPPER, right)}"
        )

    def _slice_action(self, action: np.ndarray) -> Dict[RobotComponent, np.ndarray]:
        expected_size = sum(
            _joint_count(self.robot_config, component)
            for component in self.action_order
        )
        unused = getattr(self, "_unused_trailing_action_dim", 0)
        if unused and len(action) == expected_size + unused:
            action = np.asarray(action[:expected_size], dtype=np.float64)
        if len(action) != expected_size:
            raise ValueError(
                f"Action vector length mismatch: expected {expected_size}, got {len(action)}"
            )
        targets: Dict[RobotComponent, np.ndarray] = {}
        idx = 0
        for component in self.action_order:
            size = _joint_count(self.robot_config, component)
            targets[component] = np.array(action[idx : idx + size], dtype=np.float64)
            idx += size
        return targets

    def _publish_targets(self, targets: Dict[RobotComponent, np.ndarray]):
        hand_targets: Dict[RobotComponent, np.ndarray] = {}
        direct_targets: Dict[RobotComponent, np.ndarray] = {}

        for component, values in targets.items():
            if component == RobotComponent.EXTERNAL_DATA:
                # External Int32 publisher owns this signal; do not command it.
                continue
            if component in (RobotComponent.LEFT_GRIPPER, RobotComponent.RIGHT_GRIPPER):
                hand_component, preset = self._resolve_gripper_action(component, float(values[0]))
                if hand_component is None:
                    continue
                hand_targets[hand_component] = preset
            else:
                direct_targets[component] = values

        for hand_component, preset in hand_targets.items():
            direct_targets[hand_component] = preset

        self.robot_io.publish_targets(direct_targets)

    def _hand_key_side(self, component: RobotComponent) -> Optional[str]:
        if component == RobotComponent.LEFT_GRIPPER:
            return "left"
        if component == RobotComponent.RIGHT_GRIPPER:
            return "right"
        return None

    def _hand_key_rows(self, side: str) -> List[Tuple[float, dict]]:
        table = self.policy_config.get("hand_key_poses")
        if not isinstance(table, dict):
            return []
        rows = table.get(side)
        if not isinstance(rows, list):
            return []
        parsed: List[Tuple[float, dict]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            joints = row.get("joints")
            if not isinstance(joints, list) or not joints:
                continue
            try:
                key = float(row["key"])
            except (KeyError, TypeError, ValueError):
                continue
            parsed.append((key, row))
        parsed.sort(key=lambda item: item[0])
        return parsed

    @staticmethod
    def _nearest_hand_key_row(
        scalar: float, pairs: List[Tuple[float, dict]]
    ) -> Tuple[float, dict]:
        key, row = min(pairs, key=lambda item: abs(item[0] - scalar))
        return float(key), row

    def _hand_key_hold_active(self, side: str) -> bool:
        until = self._hand_key_hold_until_publish_tick.get(side)
        if until is None:
            return False
        with self.action_queue_lock:
            tick = int(self._published_action_tick)
        return tick < until

    def _pick_hand_key_gray_zone(
        self,
        *,
        pairs: List[Tuple[float, dict]],
        scalar: float,
        active_key: Optional[float],
        side: str,
    ) -> Optional[Tuple[float, dict]]:
        if not pairs:
            return None
        cfg = self._hand_key_selection
        scalar_f = float(scalar)
        keys = [key for key, _ in pairs]
        row_by_key = {float(key): row for key, row in pairs}

        if self._scalar_escapes_active_hand_key(scalar_f, active_key):
            key_f, row = self._nearest_hand_key_row(scalar_f, pairs)
            return key_f, row

        if not cfg.enabled:
            return self._nearest_hand_key_row(scalar_f, pairs)

        margin = float(cfg.key_confidence_margin)
        min_key = float(min(keys))
        downgrade_max = float(cfg.downgrade_to_min_key_max_scalar)

        if active_key is not None and float(active_key) > min_key + 1e-6:
            if (
                scalar_f <= downgrade_max + 1e-9
                and abs(scalar_f - min_key) <= margin + 1e-9
            ):
                min_row = row_by_key.get(min_key)
                if min_row is not None:
                    return min_key, min_row

        if (
            cfg.enabled
            and active_key is not None
            and self._hand_key_hold_active(side)
        ):
            active_f = float(active_key)
            held = row_by_key.get(active_f)
            if held is not None:
                return active_f, held

        for key, row in pairs:
            if abs(scalar_f - float(key)) <= margin + 1e-9:
                key_f = float(key)
                if (
                    active_key is not None
                    and key_f < float(active_key) - 1e-6
                    and abs(key_f - min_key) <= 1e-6
                    and scalar_f > downgrade_max + 1e-9
                ):
                    active_f = float(active_key)
                    held = row_by_key.get(active_f)
                    if held is not None:
                        return active_f, held
                    continue
                return key_f, row

        def key_matches(candidate: float, target: float) -> bool:
            return abs(float(candidate) - float(target)) <= 1e-6

        for index in range(len(keys) - 1):
            lo = float(keys[index])
            hi = float(keys[index + 1])
            gray_lo = lo + margin
            gray_hi = hi - margin
            if gray_lo + 1e-9 >= gray_hi:
                continue
            if gray_lo < scalar_f < gray_hi - 1e-9:
                if active_key is not None:
                    active_f = float(active_key)
                    if key_matches(active_f, lo) or key_matches(active_f, hi):
                        held = row_by_key.get(active_f)
                        if held is not None:
                            return active_f, held
                return self._nearest_hand_key_row(scalar_f, pairs)

        if scalar_f < float(keys[0]) - margin - 1e-9:
            return self._nearest_hand_key_row(scalar_f, pairs)
        if scalar_f > float(keys[-1]) + margin + 1e-9:
            return self._nearest_hand_key_row(scalar_f, pairs)

        return self._nearest_hand_key_row(scalar_f, pairs)

    def _smooth_gripper_scalar(self, component: RobotComponent, scalar: float) -> float:
        cfg = self._hand_key_selection
        value = float(scalar)
        if not cfg.enabled or not self.policy_config.get("hand_key_poses"):
            return value
        alpha = float(cfg.scalar_ema_alpha)
        if alpha <= 0.0 or alpha > 1.0:
            return value
        prev = self._gripper_scalar_ema.get(component)
        if prev is None or not np.isfinite(prev):
            smoothed = value
        else:
            smoothed = alpha * value + (1.0 - alpha) * float(prev)
        self._gripper_scalar_ema[component] = smoothed
        return smoothed

    def _mapped_hand_joints(
        self, component: RobotComponent, scalar: float
    ) -> Optional[np.ndarray]:
        """Snap a policy hand key (0.1, 0.2, …) to the configured pose file."""
        side = self._hand_key_side(component)
        if side is None:
            return None
        logged = getattr(self, "_hand_key_logged", None)
        if not isinstance(logged, dict):
            logged = {}
            self._hand_key_logged = logged
        active_key: Optional[float] = None
        if side in logged:
            try:
                active_key = float(logged[side])
            except (TypeError, ValueError):
                active_key = None
        pairs = self._hand_key_rows(side)
        picked = self._pick_hand_key_gray_zone(
            pairs=pairs,
            scalar=scalar,
            active_key=active_key,
            side=side,
        )
        if picked is None:
            return None
        chosen, row = picked[0], picked[1]
        if logged.get(side) != chosen:
            logged[side] = chosen
            description = str(row.get("description") or "").strip()
            label = f" {description}" if description else ""
            self.get_logger().info(
                f"{component.name} key {float(scalar):.3f} -> {chosen:.1f}{label}"
            )
        return np.asarray(row["joints"], dtype=np.float64)

    def _resolve_gripper_action(
        self, component: RobotComponent, scalar: float
    ) -> Tuple[Optional[RobotComponent], np.ndarray]:
        scalar = self._smooth_gripper_scalar(component, scalar)
        cfg = self.gripper_config(component)
        closed = bool(self._decide_gripper_closed(component, scalar))
        self._gripper_command_bit[component] = 1 if closed else 0

        mapped = self._mapped_hand_joints(component, scalar)
        if mapped is not None:
            return GRIPPER_TO_HAND[component], mapped

        if not self.policy_config.get("hand_key_poses"):
            return None, np.array([], dtype=np.float64)

        mode = cfg.get("command_mode", "hand_preset")
        if mode == "scalar_topic":
            topic_key = cfg.get("command_topic_key")
            if not topic_key:
                raise ValueError(f"{component.name} scalar_topic command missing command_topic_key")
            msg = self.Int32Msg()
            msg.data = int(cfg.get("close_value", 1) if closed else cfg.get("open_value", 0))
            pub = self.command_publishers.get(topic_key)
            if pub is None:
                raise ValueError(f"No publisher for gripper topic key: {topic_key}")
            pub.publish(msg)
            return None, np.array([], dtype=np.float64)

        if mode == "gripper_preset":
            value_key = "close_value" if closed else "open_value"
            return component, np.asarray([cfg[value_key]], dtype=np.float64)

        if mode != "hand_preset":
            raise ValueError(
                f"Unsupported {component.name} command_mode: {mode}. "
                "Expected 'hand_preset', 'gripper_preset', or 'scalar_topic'."
            )

        hand_component = GRIPPER_TO_HAND[component]
        preset_key = "close_preset" if closed else "open_preset"
        preset = np.asarray(cfg[preset_key], dtype=np.float64)
        return hand_component, preset

    # ------------------------------------------------------------------
    # Reset and conditions
    # ------------------------------------------------------------------

    def _apply_motion_speed_from_policy(self) -> None:
        speed = _parse_motion_speed(self.policy_config.get("runtime", {}))
        apply = getattr(self.robot_io, "apply_motion_speed", None)
        if callable(apply):
            apply(speed)

    def _scalar_escapes_active_hand_key(
        self, scalar: float, active_key: Optional[float]
    ) -> bool:
        if active_key is None:
            return False
        cfg = self._hand_key_selection
        delta = float(cfg.force_nearest_key_min_scalar_delta)
        if delta <= 0.0 or not cfg.enabled:
            return False
        return abs(float(scalar) - float(active_key)) >= delta - 1e-9

    def _hand_scalar_in_key_gray_zone(self, side: str, scalar: float) -> bool:
        """True when scalar lies strictly between adjacent hand keys (outside confident margins)."""
        if not self.policy_config.get("hand_key_poses"):
            return False
        pairs = self._hand_key_rows(side)
        if len(pairs) < 2:
            return False
        margin = float(self._hand_key_selection.key_confidence_margin)
        scalar_f = float(scalar)
        keys = [float(key) for key, _ in pairs]
        for index in range(len(keys) - 1):
            lo = keys[index]
            hi = keys[index + 1]
            gray_lo = lo + margin
            gray_hi = hi - margin
            if gray_lo + 1e-9 >= gray_hi:
                continue
            if gray_lo < scalar_f < gray_hi - 1e-9:
                return True
        return False

    def _waypoint_gripper_dimension_error(
        self, side: str, cmd_scalar: float, live_scalar: float
    ) -> float:
        raw = float(abs(live_scalar - cmd_scalar))
        cfg = self._hand_key_selection
        if not cfg.enabled:
            return raw
        if self._hand_scalar_in_key_gray_zone(
            side, cmd_scalar
        ) or self._hand_scalar_in_key_gray_zone(side, live_scalar):
            return float(cfg.waypoint_gripper_gray_err_rad)
        return raw

    def _waypoint_command_error(self) -> Optional[float]:
        if self.last_action is None:
            return None
        live = self._live_policy_state_vector()
        if live is None:
            return None
        cmd = np.asarray(self.last_action, dtype=np.float64).reshape(-1)
        if live.shape != cmd.shape:
            return None
        diff = np.abs(live - cmd)
        if self._hand_key_selection.enabled and self.policy_config.get(
            "hand_key_poses"
        ):
            for side, idx in self._gripper_action_indices().items():
                if 0 <= idx < diff.size:
                    diff[idx] = self._waypoint_gripper_dimension_error(
                        side, float(cmd[idx]), float(live[idx])
                    )
        return float(np.max(diff))

    def _waypoint_settled(self) -> bool:
        """Strict settle (used for splice defer / pending flush)."""
        err = self._waypoint_command_error()
        if err is None:
            return self.last_action is None
        return err <= float(self._publish_consume.waypoint_settle_rad)

    def _waypoint_approach_ready(self) -> bool:
        """Loose approach — pop next waypoint before full stop."""
        if self.last_action is None:
            return True
        err = self._waypoint_command_error()
        if err is None:
            return False
        return err <= float(self._publish_consume.waypoint_approach_rad)

    def _waypoint_dwell_elapsed(self) -> bool:
        """True if the current command was held longer than dwell timeout."""
        cfg = self._publish_consume
        if cfg.waypoint_dwell_timeout_s <= 0.0 or self.last_action is None:
            return False
        prev = self._last_waypoint_pop_mono
        if prev is None:
            return False
        return (
            time.monotonic() - prev
        ) >= float(cfg.waypoint_dwell_timeout_s) - 1e-9

    def _ready_for_waypoint_pop(self) -> bool:
        return self._waypoint_approach_ready() or self._waypoint_dwell_elapsed()

    def _can_pop_waypoint_now(self) -> bool:
        cfg = self._publish_consume
        if cfg.max_waypoint_pop_hz <= 0:
            return True
        min_gap = 1.0 / float(cfg.max_waypoint_pop_hz)
        prev = self._last_waypoint_pop_mono
        if prev is None:
            return True
        return (time.monotonic() - prev) >= min_gap - 1e-9

    def _can_hold_republish_now(self) -> bool:
        cfg = self._publish_consume
        if not cfg.hold_republish_while_waiting or cfg.mode != "on_settle":
            return False
        period = float(cfg.hold_republish_period_s)
        if period <= 0.0:
            return False
        prev = self._last_hold_republish_mono
        if prev is None:
            return True
        return (time.monotonic() - prev) >= period - 1e-9

    def _run_hold_republish_cycle(self) -> Optional[str]:
        """Re-send the current waypoint (no queue pop) to refresh GDK target timestamps."""
        publish_error = None
        with self._publish_cycle_lock:
            with self.action_queue_lock:
                action = self.last_action
            if action is None or self._stop_requested:
                return None
            try:
                if not isinstance(action, np.ndarray):
                    action = np.asarray(action, dtype=np.float64)
                action = self._blend_publish_action(action)
                self.publish_action(action)
                self._last_hold_republish_mono = time.monotonic()
            except Exception as exc:
                import traceback

                publish_error = (
                    f"Action hold-republish failed: {exc}\n"
                    f"{traceback.format_exc()}"
                )
        return publish_error

    def _run_one_publish_cycle(self) -> Optional[str]:
        """Pop (if allowed), publish one action. Returns error text or None."""
        publish_error = None
        with self._publish_cycle_lock:
            with self.action_queue_lock:
                warming_up = (
                    self._inference_start_time is not None
                    and time.time() - self._inference_start_time
                    < self.warming_up_time
                )
                action = None
                if self.action_queue and not warming_up:
                    action = self.action_queue.popleft()
                    self._actions_consumed_since_inference += 1
                    self.last_action = (
                        action.copy() if hasattr(action, "copy") else action
                    )
                    self._last_waypoint_pop_mono = time.monotonic()
                elif not warming_up:
                    action = self.last_action

            if action is not None and not self._stop_requested:
                try:
                    if not isinstance(action, np.ndarray):
                        action = np.asarray(action, dtype=np.float64)
                    action = self._blend_publish_action(action)
                    self.publish_action(action)
                    cmd_yaw = self._action_waist_yaw_rad(action)
                    with self.action_queue_lock:
                        remaining = len(self.action_queue)
                        tick = self._published_action_tick + 1
                    if cmd_yaw is not None:
                        self._action_log.note_publish(
                            tick=tick,
                            cmd_idx05=cmd_yaw,
                            live_idx05=self._live_waist_yaw_rad(),
                            queue_remaining=remaining,
                        )
                    with self.action_queue_lock:
                        self._published_action_tick += 1
                    self._last_hold_republish_mono = time.monotonic()
                except Exception as exc:
                    import traceback

                    publish_error = (
                        f"Action publish failed: {exc}\n"
                        f"{traceback.format_exc()}"
                    )
        return publish_error

    def warm_up(self):
        self.get_logger().info("Warming up, waiting for sensor data...")
        if self.adapter_name != "walker_s2":
            deadline = time.time() + float(
                self.policy_config.get("runtime", {}).get("warmup_sleep", 1.0)
            )
            while time.time() < deadline and not self._stop_requested:
                time.sleep(min(0.05, max(0.0, deadline - time.time())))
            self.get_logger().info("Warm-up completed")
            return

        deadline = time.time() + self._state_wait_timeout
        last_log_time = 0.0
        missing: List[str] = []
        while time.time() < deadline and not self._stop_requested:
            missing = self._missing_observations()
            if not missing:
                self.get_logger().info("Warm-up completed")
                return
            now = time.time()
            if now - last_log_time >= 1.0:
                self.get_logger().warning(
                    f"Waiting for observations: {', '.join(missing)}"
                )
                last_log_time = now
            time.sleep(0.05)
        if self._stop_requested:
            return
        raise TimeoutError(
            f"Observations were not ready within {self._state_wait_timeout}s: "
            f"{', '.join(missing)}"
        )

    def _parse_complete_conditions_from_policy(self) -> List[Tuple[CompleteCondition, Any]]:
        raw = self.policy_config.get("complete_conditions", [])
        if not raw:
            duration = float(self.policy_config.get("duration", -1.0))
            return [(CompleteCondition.TIME_BASED, duration)]
        result = []
        for item in raw:
            condition_type = item.get("type", "TIME_BASED")
            condition = CompleteCondition[condition_type]
            if condition == CompleteCondition.TIME_BASED:
                result.append((condition, float(item.get("duration", -1.0))))
            elif condition == CompleteCondition.JOINT_REACHED:
                result.append(
                    (
                        condition,
                        JointReachedCondition(
                            target=item.get("target", {}),
                            threshold=float(item.get("threshold", 0.05)),
                            settle_time=float(item.get("settle_time", 0.3)),
                        ),
                    )
                )
            else:
                result.append((condition, item.get("param")))
        return result

    def setCompleteCondition(self, condition: CompleteCondition, param: Any = None):
        self.setCompleteConditions([(condition, param)])

    def setCompleteConditions(
        self,
        conditions: List[Union[Tuple[CompleteCondition, Any], CompleteCondition]],
    ):
        normalized = []
        for item in conditions:
            if isinstance(item, tuple):
                normalized.append(item)
            else:
                normalized.append((item, None))
        with self._state_lock:
            self._active_complete_conditions = normalized
            self._joint_reached_settle_starts = {}
        self.get_logger().info(
            f"Complete conditions: {[condition.name for condition, _ in normalized]}"
        )

    def _check_joint_reached(self, cond: JointReachedCondition) -> bool:
        parts = []
        for key, target in cond.target.items():
            if target is None:
                continue
            component = _component_from_name(key)
            current = self._component_state(component)
            if current is None:
                return False
            target_arr = np.asarray(target, dtype=object)
            if target_arr.ndim == 0:
                errors = np.array([abs(float(current[0]) - float(target_arr.item()))])
            else:
                if len(current) != len(target_arr):
                    return False
                errors = []
                for cur, tgt in zip(current, target_arr):
                    if tgt is None:
                        continue
                    errors.append(abs(float(cur) - float(tgt)))
                errors = np.asarray(errors, dtype=np.float64)
            if len(errors) and np.any(errors > cond.threshold):
                parts.append(False)
            else:
                parts.append(True)

        if not parts or not all(parts):
            self._joint_reached_settle_starts[id(cond)] = None
            with self._state_lock:
                self._condition_progress = 0.0
            return False

        now = time.time()
        if self._joint_reached_settle_starts.get(id(cond)) is None:
            self._joint_reached_settle_starts[id(cond)] = now
        elapsed = now - float(self._joint_reached_settle_starts[id(cond)])
        with self._state_lock:
            self._condition_progress = min(1.0, elapsed / cond.settle_time) if cond.settle_time > 0 else 1.0
        return elapsed >= cond.settle_time

    def _check_one_condition(self, condition: CompleteCondition, param: Any) -> bool:
        if condition == CompleteCondition.TIME_BASED:
            duration = float(param) if param is not None else 0.0
            if duration <= 0:
                return False
            elapsed = time.time() - self._start_time
            with self._state_lock:
                self._condition_progress = min(1.0, elapsed / duration)
            return elapsed >= duration
        if condition == CompleteCondition.JOINT_REACHED and isinstance(param, JointReachedCondition):
            return self._check_joint_reached(param)
        return False

    def _check_complete_condition(self) -> bool:
        eligible = self._complete_check_eligible_time
        if eligible is not None and time.time() < eligible:
            return False
        for condition, param in self._active_complete_conditions:
            if self._check_one_condition(condition, param):
                return True
        return False

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------

    def start(
        self,
        timeout: float = -1.0,
        skip_warmup: bool = False,
    ) -> bool:
        if not GeneralPolicyActionGenerator._global_execution_lock.acquire(blocking=False):
            self.get_logger().warning("Cannot start: another policy is running")
            return False
        try:
            GeneralPolicyActionGenerator._global_executing_instance = self.get_name()
            if not self._async_start_internal(skip_warmup=skip_warmup):
                return False
            if self._inference_thread:
                self._inference_thread.join(timeout=timeout if timeout > 0 else None)
                if timeout > 0 and self._inference_thread.is_alive():
                    self.get_logger().warning(f"Execution timeout ({timeout}s)")
                    self.stop()
                    return False
            return (
                self.getProcessDone()
                and not self._stop_requested
                and not self._failed
            )
        finally:
            GeneralPolicyActionGenerator._global_executing_instance = None
            if GeneralPolicyActionGenerator._global_execution_lock.locked():
                GeneralPolicyActionGenerator._global_execution_lock.release()

    def asyncStart(self, skip_warmup: bool = False) -> bool:
        if not GeneralPolicyActionGenerator._global_execution_lock.acquire(blocking=False):
            self.get_logger().warning("Cannot start: another policy is running")
            return False
        with self._state_lock:
            if self._execution_running:
                GeneralPolicyActionGenerator._global_execution_lock.release()
                return False
        GeneralPolicyActionGenerator._global_executing_instance = self.get_name()
        started = self._async_start_internal(skip_warmup=skip_warmup)
        if not started:
            GeneralPolicyActionGenerator._global_executing_instance = None
            if GeneralPolicyActionGenerator._global_execution_lock.locked():
                GeneralPolicyActionGenerator._global_execution_lock.release()
        return started

    def _async_start_internal(self, skip_warmup: bool = False) -> bool:
        with self._state_lock:
            if self._stop_requested:
                self._process_done = True
                self._current_task_status = "Stopped before start"
                return False
            self._stop_requested = False
            self._failed = False
            self._failure_message = ""
            self._process_done = False
            self._start_time = time.time()
            self._current_task_status = "Initializing"
            self._skip_warmup = skip_warmup
            self._joint_reached_settle_starts = {}
            self._complete_check_eligible_time = None
            self._inference_count = 0
            self._condition_progress = 0.0
            self._inference_images_saved = False
            self._inference_image_seq = 0
            self._inference_image_run_id = time.strftime("%Y%m%d_%H%M%S")
            self._reset_arm_action_postprocess()
            self._reset_dummy_motion()
        with self.action_queue_lock:
            self.action_queue.clear()
            self.last_action = None
            self._pending_chunk_splice = None
            self._actions_consumed_since_inference = 0
            self._published_action_tick = 0
        self._inference_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self._inference_thread.start()
        return True

    def _mark_failed(self, message: str):
        self.get_logger().error(message)
        with self.action_queue_lock:
            self.action_queue.clear()
            self.last_action = None
            self._pending_chunk_splice = None
        self.inference_running = False
        self.publish_running = False
        with self._state_lock:
            self._failed = True
            self._failure_message = message
            self._stop_requested = True
            self._current_task_status = "Failed"

    def stop(self, wait: bool = True):
        self.get_logger().info("Stop requested")
        with self._state_lock:
            self._stop_requested = True
            self._current_task_status = "Stopping"
        with self.action_queue_lock:
            self.action_queue.clear()
            self.last_action = None
            self._pending_chunk_splice = None
        self.inference_running = False
        self.publish_running = False
        streamer = getattr(self, "_inference_image_stream", None)
        if streamer is not None:
            try:
                streamer.stop()
            except Exception:
                pass
            self._inference_image_stream = None
        if wait:
            current_thread = threading.current_thread()
            execution_thread = self._inference_thread
            if (
                execution_thread
                and execution_thread is not current_thread
                and execution_thread.is_alive()
            ):
                execution_thread.join()

    def getTaskStatus(self) -> str:
        with self._state_lock:
            return self._current_task_status

    def getProcessDone(self) -> bool:
        with self._state_lock:
            return self._process_done

    def getInferenceInfo(self) -> dict:
        with self._state_lock:
            return {
                "inference_count": self._inference_count,
                "action_queue_remaining": len(self.action_queue),
                "condition_progress": round(self._condition_progress, 3),
                "failed": self._failed,
                "failure_message": self._failure_message,
            }

    def _signal_handler(self, sig, frame):
        self.get_logger().info(f"Signal {sig} received, stopping execution")
        # Signal handlers run on the main thread. Only request shutdown here;
        # the execution thread waits for an in-flight GPU inference to finish
        # before ROS/PyTorch resources are destroyed.
        self.stop(wait=False)

    # ------------------------------------------------------------------
    # Execution loop
    # ------------------------------------------------------------------

    def _execution_loop(self):
        execution_started = time.perf_counter()
        warmup_ms = 0.0
        try:
            with self._state_lock:
                self._execution_running = True
                self._process_done = False
                skip_warmup = getattr(self, "_skip_warmup", False)

            if not skip_warmup:
                with self._state_lock:
                    self._current_task_status = "Warming up"
                phase_started = time.perf_counter()
                self.warm_up()
                warmup_ms = (time.perf_counter() - phase_started) * 1000.0
            if self._stop_requested:
                return

            if self._dummy_motion_config.enabled:
                self._capture_dummy_motion_bases()

            with self._state_lock:
                self._current_task_status = "Waiting for model"
            model_wait_started = time.perf_counter()
            self._wait_for_model_ready()
            model_wait_ms = (time.perf_counter() - model_wait_started) * 1000.0
            if self._stop_requested:
                return
            if hasattr(self.action_policy, "reset") and callable(self.action_policy.reset):
                self.action_policy.reset()

            with self._state_lock:
                self._current_task_status = "Running"
                self._complete_check_eligible_time = time.time() + self._min_run_time_before_complete

            self._inference_start_time = time.time()
            self._action_log.begin_session(self._policy_image_stem)
            self._init_right_gripper_chunk_trace()
            self.inference_running = True
            self.publish_running = True
            inference_thread = threading.Thread(target=self._inference_thread_impl, daemon=True)
            publish_thread = threading.Thread(target=self._publish_thread_impl, daemon=True)
            self._policy_inference_thread = inference_thread
            self._publish_thread = publish_thread
            inference_thread.start()
            publish_thread.start()
            self.get_logger().info(
                "Execution startup timing | "
                f"warm_up={warmup_ms:.1f}ms "
                f"model_wait={model_wait_ms:.1f}ms "
                f"total_to_threads={(time.perf_counter() - execution_started) * 1000.0:.1f}ms"
            )

            while not self._stop_requested:
                if self._check_complete_condition():
                    self.get_logger().info("Complete condition met")
                    break
                time.sleep(0.1)

            self.inference_running = False
            self.publish_running = False
            # PyTorch/CUDA inference cannot be cancelled safely mid-call.
            # Wait for it to return before destroying the node or interpreter.
            inference_thread.join()
            publish_thread.join()
        except Exception as exc:
            import traceback

            self._mark_failed(
                f"Error in execution loop: {exc}\n{traceback.format_exc()}"
            )
        finally:
            with self._state_lock:
                self._execution_running = False
                self._process_done = True
                if self._failed:
                    self._current_task_status = "Failed"
                elif self._stop_requested:
                    self._current_task_status = "Stopped"
                else:
                    self._current_task_status = "Completed"
                end_reason = self._current_task_status
            self._action_log.end_session(str(end_reason).lower())
            self.inference_running = False
            self.publish_running = False
            if GeneralPolicyActionGenerator._global_execution_lock.locked():
                GeneralPolicyActionGenerator._global_executing_instance = None
                GeneralPolicyActionGenerator._global_execution_lock.release()

    def _inference_thread_impl(self):
        queue_gated = (
            self._reinfer_threshold >= 0 and self.inference_mode == "chunk"
        )
        reinfer_note = (
            "queue gate off (inference_hz paces each cycle)"
            if self._reinfer_threshold < 0
            else (
                f"reinfer when queue<={self._reinfer_threshold} "
                "(queue depth only, not inference_hz floor)"
            )
        )
        exec_note = (
            f"execute_steps={self._action_chunk_execute_steps}"
            if self._action_chunk_execute_steps is not None
            else "execute_steps=full_chunk"
        )
        skip_note = (
            f"skip_first={self._action_chunk_skip_first_steps}"
            if self._action_chunk_skip_first_steps > 0
            else "skip_first=0"
        )
        clear_note = (
            "clear_queue_on_reinfer"
            if self._clear_action_queue_on_reinfer
            else "append_on_reinfer"
        )
        self.get_logger().info(
            f"Inference thread started | inference_hz={self._inference_hz:.1f} "
            f"| {reinfer_note} | {exec_note} | {skip_note} | {clear_note}"
        )
        try:
            while self.inference_running and not self._stop_requested:
                self._try_flush_pending_chunk_splice()
                with self.action_queue_lock:
                    remaining_actions = len(self.action_queue)
                remaining_before = remaining_actions
                if (
                    queue_gated
                    and remaining_actions > self._reinfer_threshold
                ):
                    time.sleep(min(0.01, self.publish_period))
                    continue

                with self.action_queue_lock:
                    inference_start_tick = self._published_action_tick

                cycle_start = time.perf_counter()
                obs_start = time.time()
                obs = self.get_obs()
                obs_duration = time.time() - obs_start
                if obs is None:
                    self.get_logger().warning("Inference: observations not ready")
                    time.sleep(0.1)
                    continue

                self._save_inference_images_if_needed(obs)
                inference_start = time.time()
                with self.action_queue_lock:
                    n_consumed = self._actions_consumed_since_inference
                    self._actions_consumed_since_inference = 0
                if self.inference_mode == "temporal_ensemble_batch":
                    actions_list = (
                        self.action_policy.inference_temporal_ensemble_batch(
                            obs,
                            n_consumed=n_consumed,
                        )
                    )
                else:
                    actions_list = self.action_policy.inference_batch(obs)
                inference_duration = time.time() - inference_start
                if not actions_list:
                    raise RuntimeError("ACT inference returned an empty action chunk")
                yaw_index = self._waist_yaw_vector_index(self.action_order)
                if yaw_index is None:
                    yaw_index = WAIST_YAW_INDEX_IN_BODY
                on_append = remaining_before > 0
                splice_cfg = self._chunk_queue_splice
                use_queue_splice = (
                    self.inference_mode == "chunk"
                    and splice_cfg.enabled
                    and (not splice_cfg.on_append_only or on_append)
                )
                work_list = [
                    np.asarray(row, dtype=np.float64).copy() for row in actions_list
                ]
                queue_splice_meta: Optional[Dict[str, float]] = None
                splice_deferred = False
                old_queue_head: Optional[np.ndarray] = None
                if use_queue_splice and on_append:
                    with self.action_queue_lock:
                        if self.action_queue:
                            old_queue_head = np.asarray(
                                self.action_queue[0], dtype=np.float64
                            ).reshape(-1)
                    anchor = self._chunk_splice_anchor_vector()
                    if anchor is not None:
                        forward = self._chunk_splice_forward_direction(
                            anchor, old_queue_head
                        )
                        if forward is not None:
                            work_list, queue_splice_meta = (
                                self._trim_chunk_reverse_prefix(
                                    work_list, anchor, forward
                                )
                            )
                        else:
                            queue_splice_meta = {
                                "reverse_trimmed": 0.0,
                                "reverse_remain": float(len(work_list)),
                            }
                        splice_deferred = self._should_defer_chunk_splice(anchor)
                splice_fix = self._apply_action_chunk_prefix_fix(
                    work_list,
                    obs,
                    on_append=on_append,
                    yaw_index=yaw_index,
                )
                prefix_fix = None
                if not self._action_chunk_prefix_fix.enabled:
                    prefix_fix = self._apply_waist_chunk_prefix_fix(
                        work_list, obs, yaw_index
                    )
                if self._action_chunk_execute_steps is not None:
                    work_list = work_list[: int(self._action_chunk_execute_steps)]
                skip_first = int(self._action_chunk_skip_first_steps)
                if skip_first > 0:
                    if len(work_list) <= skip_first:
                        raise RuntimeError(
                            "action_chunk_skip_first_steps="
                            f"{skip_first} leaves no actions "
                            f"(chunk len={len(work_list)})"
                        )
                    work_list = work_list[skip_first:]
                raw_gripper_export = self._slice_raw_inference_for_gripper_export(
                    actions_list
                )
                self._export_right_gripper_chunk_trace(raw_gripper_export)
                queue_cleared = False
                stale_prefix = 0
                active_latency_compensation_ratio = 0.0
                with self._publish_cycle_lock:
                    with self.action_queue_lock:
                        if self.inference_mode == "temporal_ensemble_batch":
                            active_latency_compensation_ratio = (
                                active_inference_latency_compensation_ratio(
                                    self._inference_latency_chunk_compensation_ratio,
                                    self._inference_latency_chunk_compensation_number,
                                    self._inference_count,
                                )
                            )
                            stale_prefix = inference_latency_action_skip(
                                inference_start_tick,
                                self._published_action_tick,
                                len(actions_list),
                                active_latency_compensation_ratio,
                            )
                            self.action_queue.clear()
                            queue_cleared = True
                            self._pending_chunk_splice = None
                            self.action_queue.extend(actions_list[stale_prefix:])
                        elif on_append and self._clear_action_queue_on_reinfer:
                            self.action_queue.clear()
                            queue_cleared = True
                            self._pending_chunk_splice = None
                            self.action_queue.extend(work_list)
                        elif use_queue_splice and on_append:
                            self.action_queue.clear()
                            queue_cleared = True
                            if splice_deferred:
                                self._pending_chunk_splice = work_list
                            else:
                                self._pending_chunk_splice = None
                                self.action_queue.extend(work_list)
                        else:
                            stale_prefix = 0
                            active_latency_compensation_ratio = 0.0
                            self.action_queue.extend(work_list)
                with self._state_lock:
                    self._inference_count += 1
                    inference_serial = self._inference_count
                if on_append and self._hand_key_selection.enabled:
                    self._schedule_hand_key_hold_after_chunk_append(
                        remaining_before=remaining_before
                    )
                if splice_fix is not None:
                    self._action_log.note_chunk_splice(
                        inference_count=inference_serial,
                        reference=self._action_chunk_prefix_fix.reference,
                        **splice_fix,
                    )
                elif prefix_fix is not None:
                    self._action_log.note_waist_prefix_fix(
                        inference_count=inference_serial,
                        **prefix_fix,
                    )
                if queue_splice_meta is not None:
                    self._action_log.note_queue_splice(
                        inference_count=inference_serial,
                        queue_cleared=queue_cleared,
                        deferred=int(splice_deferred),
                        queue_before=remaining_before,
                        **queue_splice_meta,
                    )
                _write_oap_lerobot_state(work_list)
                chunk_flipped = self._action_log.note_inference(
                    inference_count=inference_serial,
                    obs_idx05=self._obs_waist_yaw_rad(obs),
                    live_idx05=self._live_waist_yaw_rad(),
                    n_consumed=n_consumed,
                    remaining_actions_before=remaining_before,
                    queue_cleared=queue_cleared,
                    stale_skipped=stale_prefix,
                    actions_list=work_list,
                    inference_mode=self.inference_mode,
                    inference_ms=inference_duration * 1000.0,
                    action_vector_index=yaw_index,
                )
                if chunk_flipped:
                    self.get_logger().warn(
                        "WAIST idx05 chunk direction flipped "
                        f"(inference={inference_serial}, see lerobot_action.log)"
                    )
                self.get_logger().info(
                    f"Inference | obs={obs_duration*1000:.1f}ms "
                    f"inference={inference_duration*1000:.1f}ms "
                    f"actions={len(actions_list)} consumed={n_consumed} "
                    f"latency_skipped={stale_prefix} "
                    f"latency_compensation="
                    f"{active_latency_compensation_ratio:.2f} "
                    f"compensation_number="
                    f"{self._inference_latency_chunk_compensation_number} "
                    f"mode={self.inference_mode}"
                )
                if self._profile_camera_callbacks:
                    camera_metrics = self._consume_camera_callback_metrics()
                    if camera_metrics:
                        self.get_logger().info(
                            "Camera callbacks | " + " | ".join(camera_metrics)
                        )
                if self._profile_inference:
                    timings = self.action_policy.last_inference_timings_ms
                    self.get_logger().info(
                        "Inference stages | "
                        f"prepare={timings.get('prepare', 0.0):.1f}ms "
                        f"preprocess={timings.get('preprocess', 0.0):.1f}ms "
                        f"model={timings.get('model', 0.0):.1f}ms "
                        f"model_gpu={timings.get('model_gpu', 0.0):.1f}ms "
                        f"ensemble_transfer="
                        f"{timings.get('ensemble_transfer', 0.0):.1f}ms "
                        f"ensemble={timings.get('ensemble', 0.0):.1f}ms "
                        f"postprocess={timings.get('postprocess', 0.0):.1f}ms "
                        f"transfer={timings.get('transfer', 0.0):.1f}ms "
                        f"numpy={timings.get('numpy', 0.0):.1f}ms "
                        f"total={timings.get('total', 0.0):.1f}ms"
                    )
                    model_parts = [
                        (
                            f"backbone_camera_{index}="
                            f"{timings.get(f'backbone_camera_{index}', 0.0):.1f}ms"
                        )
                        for index in range(len(self.image_specs))
                    ]
                    self.get_logger().info(
                        "Model forward stages | "
                        + " ".join(model_parts)
                        + " "
                        + f"state_tokens="
                        f"{timings.get('state_tokens', 0.0):.1f}ms "
                        + f"image_projection="
                        f"{timings.get('image_projection', 0.0):.1f}ms "
                        + f"token_stack={timings.get('token_stack', 0.0):.1f}ms "
                        + f"transformer_encoder="
                        f"{timings.get('transformer_encoder', 0.0):.1f}ms "
                        + f"transformer_decoder="
                        f"{timings.get('transformer_decoder', 0.0):.1f}ms "
                        + f"action_head={timings.get('action_head', 0.0):.1f}ms "
                        + f"forward_total="
                        f"{timings.get('model_forward_total', 0.0):.1f}ms"
                    )
                if not queue_gated:
                    remaining = self._inference_period - (
                        time.perf_counter() - cycle_start
                    )
                    if remaining > 0.0:
                        time.sleep(remaining)
        except Exception as exc:
            import traceback

            self._mark_failed(
                f"Inference failed: {exc}\n{traceback.format_exc()}"
            )
        self.get_logger().info("Inference thread stopped")

    def _publish_thread_impl(self):
        consume = self._publish_consume
        self.get_logger().info(
            "Publish thread started | "
            f"consume_mode={consume.mode} "
            f"approach_rad={consume.waypoint_approach_rad:g} "
            f"settle_rad={consume.waypoint_settle_rad:g} "
            f"dwell_timeout_s={consume.waypoint_dwell_timeout_s:g} "
            f"pop_hz_cap={consume.max_waypoint_pop_hz:g} "
            f"hold_republish={int(consume.hold_republish_while_waiting)} "
            f"hold_republish_s={consume.hold_republish_period_s:g}"
        )
        while self.publish_running and not self._stop_requested:
            publish_error = None
            if consume.mode == "on_settle":
                with self.action_queue_lock:
                    queue_len = len(self.action_queue)
                    warming_up = (
                        self._inference_start_time is not None
                        and time.time() - self._inference_start_time
                        < self.warming_up_time
                    )
                ready = self._ready_for_waypoint_pop()
                should_pop = (
                    queue_len > 0
                    and not warming_up
                    and ready
                    and self._can_pop_waypoint_now()
                )
                if should_pop:
                    if (
                        queue_len > 0
                        and not self._waypoint_approach_ready()
                        and self._waypoint_dwell_elapsed()
                    ):
                        self.get_logger().debug(
                            "Waypoint pop: dwell timeout "
                            f"(>{consume.waypoint_dwell_timeout_s:g}s, not at approach)"
                        )
                    publish_error = self._run_one_publish_cycle()
                else:
                    if (
                        not warming_up
                        and self.last_action is not None
                        and self._can_hold_republish_now()
                    ):
                        publish_error = self._run_hold_republish_cycle()
                    if publish_error is None:
                        time.sleep(float(consume.publish_poll_s))
            else:
                publish_start = time.time()
                publish_error = self._run_one_publish_cycle()
                sleep_time = max(
                    0.0, self.publish_period - (time.time() - publish_start)
                )
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self._try_flush_pending_chunk_splice()
            if publish_error is not None:
                self._mark_failed(publish_error)
                break
        self.get_logger().info("Publish thread stopped")

    def destroy_node(self):
        execution_thread = self._inference_thread
        if (
            self._execution_running
            or (
                execution_thread is not None
                and execution_thread.is_alive()
            )
        ):
            self.stop(wait=True)
        streamer = getattr(self, "_inference_image_stream", None)
        if streamer is not None:
            streamer.stop()
            self._inference_image_stream = None
        model_load_thread = getattr(self, "_model_load_thread", None)
        if (
            model_load_thread is not None
            and model_load_thread is not threading.current_thread()
            and model_load_thread.is_alive()
        ):
            model_load_thread.join()
        self.action_policy = None
        if hasattr(self, "robot_io"):
            self.robot_io.close()
        return super().destroy_node()


def _main():
    parser = argparse.ArgumentParser(description="General upper-body policy action generator")
    parser.add_argument("--robot", required=True, help="Robot I/O JSON config")
    parser.add_argument("--policy", required=True, help="Policy/model JSON config")
    parser.add_argument("--timeout", type=float, default=-1.0, help="Synchronous execution timeout")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate JSON and print vector mapping")
    args = parser.parse_args()

    configs = load_and_validate_configs(args.robot, args.policy)
    if args.dry_run:
        print(describe_mapping(configs))
        return 0

    if rclpy is None:
        raise RuntimeError("rclpy is required unless --dry-run is used") from _ROS_IMPORT_ERROR

    rclpy.init()
    generator = None
    executor = None
    try:
        generator = GeneralPolicyActionGenerator(args.robot, args.policy)
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(generator)
        executor_thread = threading.Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        success = generator.start(
            timeout=args.timeout,
            skip_warmup=args.skip_warmup,
        )
        return 0 if success else 1
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=1.0)
        if generator is not None:
            generator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(_main())
