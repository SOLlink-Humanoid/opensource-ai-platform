"""Task planner health view using OpenVLA phase, joints, and YOLO detections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from web.yolo.yolo_service import Detection


@dataclass(frozen=True)
class PlannerCheck:
    id: str
    label: str
    ok: bool | None
    detail: str


@dataclass(frozen=True)
class PlannerSnapshot:
    phase_description: str
    phase_index: int | None
    joint_error_sum: float | None
    error_trend: str
    approaching_target: bool | None
    visible_labels: tuple[str, ...]
    checks: tuple[PlannerCheck, ...]


def build_planner_snapshot(
    *,
    openvla_phase_description: str,
    openvla_phase_index: int | None,
    joint_error_sum: float | None,
    error_trend: str,
    detections: tuple[Detection, ...] | None,
    joint_error_threshold: float = 0.35,
) -> PlannerSnapshot:
    labels = tuple(sorted({d.label for d in detections})) if detections else ()
    approaching: bool | None = None
    if joint_error_sum is not None:
        approaching = joint_error_sum <= joint_error_threshold or error_trend == "decreasing"

    checks: list[PlannerCheck] = [
        PlannerCheck(
            id="phase",
            label="OpenVLA phase active",
            ok=bool(openvla_phase_description or openvla_phase_index is not None),
            detail=openvla_phase_description or "—",
        ),
        PlannerCheck(
            id="joints",
            label="Moving toward target joints",
            ok=approaching,
            detail=(
                f"Σ|Δq|={joint_error_sum:.4f} ({error_trend})"
                if joint_error_sum is not None
                else "No joint feedback"
            ),
        ),
        PlannerCheck(
            id="vision",
            label="YOLO scene (planner view)",
            ok=True if labels else None,
            detail=", ".join(labels) if labels else "No detections in frame",
        ),
    ]

    return PlannerSnapshot(
        phase_description=openvla_phase_description,
        phase_index=openvla_phase_index,
        joint_error_sum=joint_error_sum,
        error_trend=error_trend,
        approaching_target=approaching,
        visible_labels=labels,
        checks=tuple(checks),
    )

def detections_to_json(detections: tuple[Detection, ...] | None) -> list[dict[str, Any]]:
    if not detections:
        return []
    return [
        {
            "class_id": d.class_id,
            "label": d.label,
            "confidence": d.confidence,
            "xyxy": list(d.xyxy),
        }
        for d in detections
    ]

