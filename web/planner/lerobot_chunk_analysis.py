"""Analyze a LeRobot action chunk vs OpenVLA reference (convergence / abort / truncate)."""

from __future__ import annotations

from typing import Any

from shared.action_layout import load_action_layout

from .openvla_joint_gap import error_sum, max_part_error, part_errors_vs_openvla_target

# Per-part rad; above this at any step → do not run policy (step 0 = immediate reject).
DEFAULT_PART_ABORT_RAD = 1.25
DEFAULT_WORSEN_STREAK = 4
DEFAULT_WORSEN_DELTA_RAD = 0.012
DEFAULT_IMPROVE_DELTA_RAD = 0.003


def analyze_lerobot_chunk_vs_openvla(
    action_chunk: list[list[float]],
    openvla_target_joints: dict[str, float],
    *,
    inferred_joints_by_part: dict[str, list[str]] | None = None,
    part_abort_rad: float = DEFAULT_PART_ABORT_RAD,
    worsen_streak: int = DEFAULT_WORSEN_STREAK,
    worsen_delta_rad: float = DEFAULT_WORSEN_DELTA_RAD,
) -> dict[str, Any]:
    layout = load_action_layout()
    target = {str(k): float(v) for k, v in (openvla_target_joints or {}).items()}
    if not target:
        return {
            "ok": False,
            "reject": True,
            "reason": "OpenVLA target_joints 없음",
            "max_publish_steps": 0,
            "steps_analyzed": 0,
        }

    series: list[dict[str, Any]] = []
    for index, row in enumerate(action_chunk or []):
        try:
            decoded, _phase = layout.decode([float(x) for x in row])
        except (TypeError, ValueError) as exc:
            return {
                "ok": False,
                "reject": True,
                "reason": f"청크 step {index} 디코드 실패: {exc}",
                "max_publish_steps": 0,
                "steps_analyzed": index,
            }
        parts = part_errors_vs_openvla_target(
            decoded,
            target,
            inferred_joints_by_part=inferred_joints_by_part,
        )
        total = error_sum(parts)
        peak = max_part_error(parts)
        series.append(
            {
                "step": index,
                "error_sum": total,
                "max_part_error": peak,
                "part_errors": parts,
            }
        )
        if peak is not None and peak > part_abort_rad:
            return {
                "ok": True,
                "reject": True,
                "reason": (
                    f"step {index} 부위 오차 {peak:.3f} rad > 한계 {part_abort_rad:.3f} rad"
                ),
                "max_publish_steps": 0,
                "steps_analyzed": len(series),
                "series": series,
                "abort_part_error_rad": part_abort_rad,
            }

    if not series:
        return {
            "ok": False,
            "reject": True,
            "reason": "action_chunk 비어 있음",
            "max_publish_steps": 0,
            "steps_analyzed": 0,
        }

    totals = [row["error_sum"] for row in series if row["error_sum"] is not None]
    if not totals:
        return {
            "ok": False,
            "reject": True,
            "reason": "OpenVLA와 비교 가능한 관절이 없음",
            "max_publish_steps": 0,
            "steps_analyzed": len(series),
            "series": series,
        }

    divergence_index: int | None = None
    streak = 0
    prev = totals[0]
    for i in range(1, len(totals)):
        cur = totals[i]
        if cur is None or prev is None:
            prev = cur
            streak = 0
            continue
        if cur > prev + worsen_delta_rad:
            streak += 1
            if streak >= worsen_streak and divergence_index is None:
                divergence_index = i - worsen_streak + 1
        else:
            streak = 0
        prev = cur

    if divergence_index is not None and divergence_index <= 0:
        return {
            "ok": True,
            "reject": True,
            "reason": "초기부터 OpenVLA 목표와 발산 — 실행하지 않음",
            "max_publish_steps": 0,
            "divergence_step": divergence_index,
            "steps_analyzed": len(series),
            "series": series,
            "initial_error_sum": totals[0],
            "final_error_sum": totals[-1],
        }

    max_steps = len(series)
    if divergence_index is not None and divergence_index > 0:
        max_steps = divergence_index

    converging = totals[-1] <= totals[0] + worsen_delta_rad
    return {
        "ok": True,
        "reject": False,
        "reason": "",
        "max_publish_steps": max_steps,
        "divergence_step": divergence_index,
        "steps_analyzed": len(series),
        "series": series,
        "initial_error_sum": totals[0],
        "final_error_sum": totals[-1],
        "converging": converging,
        "abort_part_error_rad": part_abort_rad,
    }