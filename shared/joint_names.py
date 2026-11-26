"""Agibot G2 whole-body joints used by OpenVLA targets and the planner UI."""

from __future__ import annotations

BODY_JOINTS: tuple[str, ...] = (
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
)

HEAD_JOINTS: tuple[str, ...] = (
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
)

LEFT_ARM_JOINTS: tuple[str, ...] = (
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
)

RIGHT_ARM_JOINTS: tuple[str, ...] = (
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
)

LEFT_HAND_JOINTS: tuple[str, ...] = (
    "idx31_hand_l_thumb_roll_joint",
    "idx32_hand_l_thumb_abad_joint",
    "idx33_hand_l_thumb_mcp_joint",
    "idx36_hand_l_index_abad_joint",
    "idx37_hand_l_index_pip_joint",
    "idx39_hand_l_middle_pip_joint",
    "idx41_hand_l_ring_abad_joint",
    "idx42_hand_l_ring_pip_joint",
    "idx44_hand_l_pinky_abad_joint",
    "idx45_hand_l_pinky_pip_joint",
)

RIGHT_HAND_JOINTS: tuple[str, ...] = (
    "idx71_hand_r_thumb_roll_joint",
    "idx72_hand_r_thumb_abad_joint",
    "idx73_hand_r_thumb_mcp_joint",
    "idx76_hand_r_index_abad_joint",
    "idx77_hand_r_index_pip_joint",
    "idx79_hand_r_middle_pip_joint",
    "idx81_hand_r_ring_abad_joint",
    "idx82_hand_r_ring_pip_joint",
    "idx84_hand_r_pinky_abad_joint",
    "idx85_hand_r_pinky_pip_joint",
)

ALL_MODEL_JOINTS: tuple[str, ...] = (
    BODY_JOINTS + HEAD_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
)

# 39-DOF phased-data layout: body(5) + left_arm(7) + right_arm(7) + left_hand(10) + right_hand(10)
BODY_ARM_HAND_JOINTS: tuple[str, ...] = (
    BODY_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS
)


def joint_group(name: str) -> str | None:
    if name in BODY_JOINTS:
        return "body"
    if name in HEAD_JOINTS:
        return "head"
    if name in LEFT_ARM_JOINTS or name in RIGHT_ARM_JOINTS:
        return "arms"
    return None


# Finer groups for OpenVLA phase pose error (Σ|pose − predicted| per part).
PART_ERROR_KEYS: tuple[str, ...] = (
    "head",
    "body",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
)


def joint_part_group(name: str) -> str | None:
    if name in HEAD_JOINTS:
        return "head"
    if name in BODY_JOINTS:
        return "body"
    if name in LEFT_ARM_JOINTS:
        return "left_arm"
    if name in RIGHT_ARM_JOINTS:
        return "right_arm"
    if name in LEFT_HAND_JOINTS:
        return "left_hand"
    if name in RIGHT_HAND_JOINTS:
        return "right_hand"
    return None
