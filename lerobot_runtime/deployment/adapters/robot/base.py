#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared robot adapter types, schema, and base class."""

from enum import IntFlag, auto
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


class RobotComponent(IntFlag):
    """Upper-body component flags shared by all robot adapters."""

    NONE = 0
    HEAD = auto()
    LEFT_ARM = auto()
    RIGHT_ARM = auto()
    LEFT_HAND = auto()
    RIGHT_HAND = auto()
    LEFT_GRIPPER = auto()
    RIGHT_GRIPPER = auto()
    CAMERA_HEAD = auto()
    CAMERA_HAND_LEFT = auto()
    CAMERA_HAND_RIGHT = auto()
    CAMERA_HEAD_DEPTH = auto()
    EXTERNAL_DATA = auto()
    WAIST = auto()
    CAMERA_HAND_LEFT_2 = auto()
    CAMERA_HAND_RIGHT_2 = auto()


CAMERA_COMPONENTS = (
    RobotComponent.CAMERA_HEAD
    | RobotComponent.CAMERA_HAND_LEFT
    | RobotComponent.CAMERA_HAND_RIGHT
    | RobotComponent.CAMERA_HEAD_DEPTH
    | RobotComponent.CAMERA_HAND_LEFT_2
    | RobotComponent.CAMERA_HAND_RIGHT_2
)

GRIPPER_TO_HAND = {
    RobotComponent.LEFT_GRIPPER: RobotComponent.LEFT_HAND,
    RobotComponent.RIGHT_GRIPPER: RobotComponent.RIGHT_HAND,
}

COMPONENT_TO_KEY = {
    RobotComponent.HEAD: "head",
    RobotComponent.LEFT_ARM: "left_arm",
    RobotComponent.RIGHT_ARM: "right_arm",
    RobotComponent.LEFT_HAND: "left_hand",
    RobotComponent.RIGHT_HAND: "right_hand",
    RobotComponent.LEFT_GRIPPER: "left_gripper",
    RobotComponent.RIGHT_GRIPPER: "right_gripper",
    RobotComponent.CAMERA_HEAD: "camera_head",
    RobotComponent.CAMERA_HAND_LEFT: "camera_hand_left",
    RobotComponent.CAMERA_HAND_RIGHT: "camera_hand_right",
    RobotComponent.CAMERA_HEAD_DEPTH: "camera_head_depth",
    RobotComponent.CAMERA_HAND_LEFT_2: "camera_hand_left_2",
    RobotComponent.CAMERA_HAND_RIGHT_2: "camera_hand_right_2",
    RobotComponent.EXTERNAL_DATA: "external_data",
    RobotComponent.WAIST: "waist",
}

KEY_TO_COMPONENT = {
    "HEAD": RobotComponent.HEAD,
    "head": RobotComponent.HEAD,
    "LEFT_ARM": RobotComponent.LEFT_ARM,
    "left_arm": RobotComponent.LEFT_ARM,
    "RIGHT_ARM": RobotComponent.RIGHT_ARM,
    "right_arm": RobotComponent.RIGHT_ARM,
    "LEFT_HAND": RobotComponent.LEFT_HAND,
    "left_hand": RobotComponent.LEFT_HAND,
    "RIGHT_HAND": RobotComponent.RIGHT_HAND,
    "right_hand": RobotComponent.RIGHT_HAND,
    "LEFT_GRIPPER": RobotComponent.LEFT_GRIPPER,
    "left_gripper": RobotComponent.LEFT_GRIPPER,
    "RIGHT_GRIPPER": RobotComponent.RIGHT_GRIPPER,
    "right_gripper": RobotComponent.RIGHT_GRIPPER,
    "CAMERA_HEAD": RobotComponent.CAMERA_HEAD,
    "camera_head": RobotComponent.CAMERA_HEAD,
    "CAMERA_HAND_LEFT": RobotComponent.CAMERA_HAND_LEFT,
    "camera_hand_left": RobotComponent.CAMERA_HAND_LEFT,
    "CAMERA_HAND_RIGHT": RobotComponent.CAMERA_HAND_RIGHT,
    "camera_hand_right": RobotComponent.CAMERA_HAND_RIGHT,
    "CAMERA_HEAD_DEPTH": RobotComponent.CAMERA_HEAD_DEPTH,
    "camera_head_depth": RobotComponent.CAMERA_HEAD_DEPTH,
    "CAMERA_HAND_LEFT_2": RobotComponent.CAMERA_HAND_LEFT_2,
    "camera_hand_left_2": RobotComponent.CAMERA_HAND_LEFT_2,
    "CAMERA_HAND_RIGHT_2": RobotComponent.CAMERA_HAND_RIGHT_2,
    "camera_hand_right_2": RobotComponent.CAMERA_HAND_RIGHT_2,
    "EXTERNAL_DATA": RobotComponent.EXTERNAL_DATA,
    "external_data": RobotComponent.EXTERNAL_DATA,
    "WAIST": RobotComponent.WAIST,
    "waist": RobotComponent.WAIST,
}

CANONICAL_STATE_TOPIC_KEYS = [
    "head",
    "arm",
    "hand",
    "left_hand",
    "right_hand",
    "left_gripper",
    "right_gripper",
]

CANONICAL_COMMAND_TOPIC_KEYS = [
    "head",
    "arm",
    "hand",
    "left_hand",
    "right_hand",
    "left_gripper",
    "right_gripper",
]

CANONICAL_JOINT_KEYS = [
    "head",
    "arm",
    "left_arm",
    "right_arm",
    "hand",
    "left_hand",
    "right_hand",
]

CANONICAL_GRIPPER_KEYS = ["LEFT_GRIPPER", "RIGHT_GRIPPER"]


def has_component(flags: RobotComponent, component: RobotComponent) -> bool:
    return (flags & component) == component


def component_from_name(name: str) -> RobotComponent:
    if name in KEY_TO_COMPONENT:
        return KEY_TO_COMPONENT[name]
    upper = str(name).upper()
    if upper in KEY_TO_COMPONENT:
        return KEY_TO_COMPONENT[upper]
    raise ValueError(f"Unknown component: {name}")


def component_key(component: RobotComponent) -> str:
    return COMPONENT_TO_KEY[component]


def component_list(names: Iterable[str]) -> List[RobotComponent]:
    return [component_from_name(name) for name in names]


def component_names(components: Iterable[RobotComponent]) -> List[str]:
    return [component.name for component in components]


def joint_count(robot_config: Dict[str, Any], component: RobotComponent) -> int:
    if component in (
        RobotComponent.LEFT_GRIPPER,
        RobotComponent.RIGHT_GRIPPER,
        RobotComponent.EXTERNAL_DATA,
    ):
        return 1
    if component in (
        RobotComponent.CAMERA_HEAD,
        RobotComponent.CAMERA_HAND_LEFT,
        RobotComponent.CAMERA_HAND_RIGHT,
        RobotComponent.CAMERA_HEAD_DEPTH,
        RobotComponent.CAMERA_HAND_LEFT_2,
        RobotComponent.CAMERA_HAND_RIGHT_2,
    ):
        return 0
    key = component_key(component)
    joints = robot_config.get("joints", {})
    if key not in joints:
        raise ValueError(f"robot config missing joints.{key}")
    joint_cfg = joints[key]
    if "order" in joint_cfg:
        return len(joint_cfg["order"])
    if "ids" in joint_cfg:
        return len(joint_cfg["ids"])
    raise ValueError(f"robot config joints.{key} must define order or ids")


def topic_enabled(entry: Optional[Dict[str, Any]]) -> bool:
    return bool(entry and entry.get("enabled", True) and entry.get("topic"))


def validate_topic_entry(section: str, key: str, entry: Any) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"robot config topics.{section}.{key} must be an object")
    for required in ["enabled", "topic", "message_type"]:
        if required not in entry:
            raise ValueError(f"robot config topics.{section}.{key} missing key: {required}")
    if entry["enabled"] and not entry["topic"]:
        raise ValueError(f"robot config topics.{section}.{key}.topic is required when enabled=true")
    if entry["enabled"] and not entry["message_type"]:
        raise ValueError(
            f"robot config topics.{section}.{key}.message_type is required when enabled=true"
        )


def resolve_gripper_config(
    robot: Dict[str, Any],
    component: RobotComponent,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return gripper settings, with policy values overlaying robot defaults."""
    name = component.name
    robot_cfg = robot.get("grippers", {}).get(name, {})
    policy_cfg = (policy or {}).get("grippers", {}).get(name, {})
    if not isinstance(robot_cfg, dict):
        robot_cfg = {}
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}
    return {**robot_cfg, **policy_cfg}


def resolve_external_data_config(
    robot: Dict[str, Any],
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return EXTERNAL_DATA topic settings from the policy.

    Policy may set top-level ``external_data`` or ``observation.external_data``.
    This is not a canonical robot topic. Robot JSON is only a leftover overlay.
    """
    robot_cfg = robot.get("topics", {}).get("states", {}).get("external_data", {})
    policy = policy or {}
    observation = policy.get("observation") if isinstance(policy.get("observation"), dict) else {}
    policy_cfg = policy.get("external_data")
    if not isinstance(policy_cfg, dict):
        policy_cfg = observation.get("external_data") if isinstance(observation, dict) else {}
    if not isinstance(robot_cfg, dict):
        robot_cfg = {}
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}
    return {**robot_cfg, **{k: v for k, v in policy_cfg.items() if v is not None}}


def validate_robot_schema(robot: Dict[str, Any]) -> None:
    required_top_keys = [
        "robot_name",
        "adapter",
        "control",
        "qos",
        "pd_params",
        "topics",
        "joints",
    ]
    for key in required_top_keys:
        if key not in robot:
            raise ValueError(f"robot config missing required key: {key}")

    topics = robot["topics"]
    for section in ["states", "commands", "cameras"]:
        if section not in topics:
            raise ValueError(f"robot config missing topics.{section}")

    for key in CANONICAL_STATE_TOPIC_KEYS:
        if key not in topics["states"]:
            raise ValueError(f"robot config missing topics.states.{key}")
        validate_topic_entry("states", key, topics["states"][key])
    for key in CANONICAL_COMMAND_TOPIC_KEYS:
        if key not in topics["commands"]:
            raise ValueError(f"robot config missing topics.commands.{key}")
        validate_topic_entry("commands", key, topics["commands"][key])
    for key in ["CAMERA_HEAD", "CAMERA_HAND_LEFT", "CAMERA_HAND_RIGHT"]:
        if key not in topics["cameras"]:
            raise ValueError(f"robot config missing topics.cameras.{key}")
        validate_topic_entry("cameras", key, topics["cameras"][key])
    if "CAMERA_HEAD_DEPTH" in topics["cameras"]:
        validate_topic_entry(
            "cameras", "CAMERA_HEAD_DEPTH", topics["cameras"]["CAMERA_HEAD_DEPTH"]
        )
    for extra_camera in ["CAMERA_HAND_LEFT_2", "CAMERA_HAND_RIGHT_2"]:
        if extra_camera in topics["cameras"]:
            validate_topic_entry(
                "cameras", extra_camera, topics["cameras"][extra_camera]
            )

    for key in CANONICAL_JOINT_KEYS:
        if key not in robot["joints"]:
            raise ValueError(f"robot config missing joints.{key}")
    if "grippers" in robot:
        if not isinstance(robot["grippers"], dict):
            raise ValueError("robot config grippers must be an object")
        for key in CANONICAL_GRIPPER_KEYS:
            if key not in robot["grippers"]:
                raise ValueError(f"robot config missing grippers.{key}")

    if robot["adapter"] == "walker_s2":
        transport = robot.get("action_transport")
        if not isinstance(transport, dict):
            raise ValueError("walker_s2 robot config missing action_transport")
        if transport.get("type") != "tcp_json":
            raise ValueError("walker_s2 action_transport.type must be tcp_json")
        if not str(transport.get("host", "")).strip():
            raise ValueError("walker_s2 action_transport.host is required")
        port = int(transport.get("port", 0))
        if port < 1 or port > 65535:
            raise ValueError("walker_s2 action_transport.port must be between 1 and 65535")
        if "waist" in topics["states"]:
            validate_topic_entry("states", "waist", topics["states"]["waist"])
        if "waist" in topics.get("commands", {}):
            validate_topic_entry("commands", "waist", topics["commands"]["waist"])
        if "waist" in robot["joints"]:
            waist_cfg = robot["joints"]["waist"]
            if not isinstance(waist_cfg, dict):
                raise ValueError("robot config joints.waist must be an object")
            if "order" not in waist_cfg or "ids" not in waist_cfg:
                raise ValueError("robot config joints.waist must define ids and order")
            if len(waist_cfg["ids"]) != len(waist_cfg["order"]):
                raise ValueError("robot config joints.waist ids/order length mismatch")
        elif topic_enabled(topics["states"].get("waist")):
            raise ValueError(
                "walker_s2 topics.states.waist is enabled but joints.waist is missing"
            )


class RobotIOAdapter:
    """Base class for robot-specific ROS I/O."""

    adapter_name = ""

    def __init__(self, owner):
        self.owner = owner

    def load_message_types(self):
        raise NotImplementedError

    def setup_ros_interfaces(self):
        raise NotImplementedError

    def publish_targets(
        self,
        targets: Dict[RobotComponent, np.ndarray],
        velocity: Optional[float] = None,
    ):
        raise NotImplementedError

    def trajectory_hand_state(self, component: RobotComponent, msg: Any) -> np.ndarray:
        """Return hand feedback in the adapter's trajectory target units."""
        return np.asarray(list(msg.position), dtype=np.float64)

    def close(self):
        """Release adapter-owned resources."""


