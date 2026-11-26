"""Shared LeRobot / OpenVLA action vector layout.

The JSON file is the source of truth for how a model vector lines up with
body, arm, and simplified gripper channels. OpenVLA appends one phase scalar.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

LAYOUT_PATH = Path(__file__).with_name("model_action_layout.json")


class ActionLayout:
    def __init__(self, data: dict[str, Any], path: Path) -> None:
        channels = data.get("channels")
        if not isinstance(channels, list) or not channels:
            raise ValueError(f"{path} channels is empty")
        trailing = data.get("openvla_trailing") or []
        if not isinstance(trailing, list):
            raise ValueError(f"{path} openvla_trailing must be a list")
        self.path = path
        self.channels = [self._channel(row, path) for row in channels]
        self.trailing = [self._trailing(row, path) for row in trailing]
        try:
            self.gripper_threshold = float(data.get("gripper_threshold", 0.5))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} gripper_threshold must be a number") from exc
        self.vector_dim = sum(channel["size"] for channel in self.channels)
        self.phase_dim = sum(row["size"] for row in self.trailing)
        self.openvla_dim = self.vector_dim + self.phase_dim
        self.summary = " + ".join(
            f"{channel['id']}({channel['size']})" for channel in self.channels
        )

    @staticmethod
    def _channel(row: Any, path: Path) -> dict[str, Any]:
        if not isinstance(row, dict):
            raise ValueError(f"{path} channel must be an object")
        kind = str(row.get("kind") or "")
        channel_id = str(row.get("id") or "").strip()
        if not channel_id:
            raise ValueError(f"{path} channel id is empty")
        if kind == "joints":
            joints = _name_list(row.get("joints"), path, channel_id)
            return {"id": channel_id, "kind": kind, "size": len(joints), "joints": joints}
        if kind == "gripper_scalar":
            joints = _name_list(row.get("joints"), path, channel_id)
            open_preset = _float_list(row.get("open_preset"), path, f"{channel_id}.open_preset")
            close_preset = _float_list(row.get("close_preset"), path, f"{channel_id}.close_preset")
            if len(open_preset) != len(joints) or len(close_preset) != len(joints):
                raise ValueError(f"{path} {channel_id} preset length must match joints")
            return {
                "id": channel_id,
                "kind": kind,
                "size": 1,
                "part": str(row.get("part") or ""),
                "side": str(row.get("side") or ""),
                "open_value": float(row.get("open_value", 0)),
                "close_value": float(row.get("close_value", 1)),
                "state_source": str(row.get("state_source") or "scalar_topic"),
                "state_topic_key": row.get("state_topic_key"),
                "joints": joints,
                "open_preset": open_preset,
                "close_preset": close_preset,
            }
        raise ValueError(f"{path} unsupported channel kind: {kind}")

    @staticmethod
    def _trailing(row: Any, path: Path) -> dict[str, Any]:
        if not isinstance(row, dict):
            raise ValueError(f"{path} openvla_trailing entry must be an object")
        if str(row.get("kind") or "") != "phase":
            raise ValueError(f"{path} openvla trailing kind must be phase")
        size = int(row.get("size") or 0)
        if size != 1:
            raise ValueError(f"{path} phase trailing size must be 1")
        return {"id": str(row.get("id") or "PHASE"), "kind": "phase", "size": size}

    def component_order(self) -> list[str]:
        return [channel["id"] for channel in self.channels]

    def output_joint_names(self) -> list[str]:
        names: list[str] = []
        for channel in self.channels:
            names.extend(channel["joints"])
        return names

    def matches_openvla(self, length: int) -> bool:
        return length == self.openvla_dim or length == self.vector_dim

    def decode(self, numbers: list[float]) -> tuple[dict[str, float], float | None]:
        """Map a model vector to joint targets. Phase is set only for the OpenVLA length."""
        if len(numbers) == self.openvla_dim and self.phase_dim:
            phase = float(numbers[-1])
            body = numbers[: self.vector_dim]
        elif len(numbers) == self.vector_dim:
            phase = None
            body = list(numbers)
        else:
            raise ValueError(
                f"Action length {len(numbers)} does not match layout "
                f"{self.vector_dim} or {self.openvla_dim}"
            )
        targets: dict[str, float] = {}
        cursor = 0
        for channel in self.channels:
            size = channel["size"]
            chunk = body[cursor : cursor + size]
            cursor += size
            if channel["kind"] == "joints":
                for name, value in zip(channel["joints"], chunk):
                    targets[name] = float(value)
                continue
            scalar = float(chunk[0])
            closed = scalar >= self.gripper_threshold
            preset = channel["close_preset"] if closed else channel["open_preset"]
            for name, value in zip(channel["joints"], preset):
                targets[name] = float(value)
        return targets, phase

    def gripper_scalars(self, numbers: list[float]) -> dict[str, float]:
        """Left/right gripper keys from a 21- or 22-dim model vector."""
        if len(numbers) == self.openvla_dim and self.phase_dim:
            body = numbers[: self.vector_dim]
        elif len(numbers) == self.vector_dim:
            body = list(numbers)
        else:
            return {}
        scalars: dict[str, float] = {}
        cursor = 0
        for channel in self.channels:
            size = channel["size"]
            chunk = body[cursor : cursor + size]
            cursor += size
            if channel["kind"] == "gripper_scalar" and chunk:
                scalars[channel["id"]] = float(chunk[0])
        return scalars

    def lerobot_gripper_configs(self) -> dict[str, dict[str, Any]]:
        configs: dict[str, dict[str, Any]] = {}
        for channel in self.channels:
            if channel["kind"] != "gripper_scalar":
                continue
            configs[channel["id"]] = {
                "side": channel["side"],
                "command_mode": "hand_preset",
                "state_source": channel["state_source"],
                "state_topic_key": channel["state_topic_key"],
                "command_topic_key": None,
                "open_value": channel["open_value"],
                "close_value": channel["close_value"],
                "open_preset": list(channel["open_preset"]),
                "close_preset": list(channel["close_preset"]),
            }
        return configs


def _name_list(raw: Any, path: Path, label: str) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} {label} joints is empty")
    names = [str(item).strip() for item in raw]
    if any(not name for name in names):
        raise ValueError(f"{path} {label} has an empty joint name")
    return names


def _float_list(raw: Any, path: Path, label: str) -> list[float]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} {label} is empty")
    try:
        return [float(item) for item in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} {label} must be numbers") from exc


@lru_cache(maxsize=1)
def load_action_layout() -> ActionLayout:
    path = LAYOUT_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read action layout: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Action layout root must be an object: {path}")
    return ActionLayout(data, path)


def apply_to_lerobot_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Overwrite LeRobot vector order and gripper presets from the shared layout."""
    layout = load_action_layout()
    order = layout.component_order()
    observation = policy.setdefault("observation", {})
    if isinstance(observation, dict):
        observation["state_order"] = list(order)
    action = policy.setdefault("action", {})
    if isinstance(action, dict):
        action["order"] = list(order)
        action["gripper_threshold"] = layout.gripper_threshold
    policy["grippers"] = layout.lerobot_gripper_configs()
    contract = policy.setdefault("oap_model_contract", {})
    if isinstance(contract, dict):
        contract["state_dim"] = layout.vector_dim
        contract["action_dim"] = layout.vector_dim
        contract["state_order_summary"] = layout.summary
        features = contract.setdefault("input_features", {})
        if isinstance(features, dict):
            features["observation.state"] = [layout.vector_dim]
            features["action"] = [layout.vector_dim]
    return policy
