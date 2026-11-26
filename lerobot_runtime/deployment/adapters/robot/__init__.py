"""OAP onboard LeRobot: Agibot G2 adapter only."""

from .agibot_g2 import AgibotG2Adapter
from .base import (
    CAMERA_COMPONENTS,
    CANONICAL_COMMAND_TOPIC_KEYS,
    CANONICAL_GRIPPER_KEYS,
    CANONICAL_JOINT_KEYS,
    CANONICAL_STATE_TOPIC_KEYS,
    COMPONENT_TO_KEY,
    GRIPPER_TO_HAND,
    KEY_TO_COMPONENT,
    RobotComponent,
    RobotIOAdapter,
    component_from_name,
    component_key,
    component_list,
    component_names,
    has_component,
    joint_count,
    resolve_external_data_config,
    resolve_gripper_config,
    topic_enabled,
    validate_robot_schema,
    validate_topic_entry,
)

ADAPTER_REGISTRY = {
    AgibotG2Adapter.adapter_name: AgibotG2Adapter,
}

SUPPORTED_ADAPTERS = tuple(ADAPTER_REGISTRY.keys())


def create_robot_adapter(owner) -> RobotIOAdapter:
    adapter_name = owner.robot_config["adapter"]
    try:
        adapter_cls = ADAPTER_REGISTRY[adapter_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported robot adapter: {adapter_name!r}. "
            f"OAP lerobot_runtime supports only: {', '.join(SUPPORTED_ADAPTERS)}"
        ) from exc
    return adapter_cls(owner)
