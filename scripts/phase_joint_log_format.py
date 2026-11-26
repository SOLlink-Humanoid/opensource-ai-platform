#!/usr/bin/env python3
"""Build fixed-layout phase joint .log (one file per phase transition)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.joint_names import (  # noqa: E402
    BODY_JOINTS,
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    LEFT_HAND_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_HAND_JOINTS,
)

PHASES: list[tuple[int, str]] = [
    (1, "start"),
    (2, "pick_empty_box"),
    (3, "lift_empty_box"),
    (4, "turn_to_bench"),
    (5, "place_empty_box"),
    (6, "ready_packing"),
    (7, "pack"),
    (8, "lift_full_box"),
    (9, "turn_to_ship"),
    (10, "place_to_ship"),
    (11, "complete"),
]

PHASE_NAME_WIDTH = max(len(name) for _, name in PHASES)

PART_LINES: list[tuple[str, tuple[str, ...]]] = [
    ("허리", BODY_JOINTS),
    ("머리", HEAD_JOINTS),
    ("왼팔", LEFT_ARM_JOINTS),
    ("오른팔", RIGHT_ARM_JOINTS),
    ("왼손", LEFT_HAND_JOINTS),
    ("오른손", RIGHT_HAND_JOINTS),
]

# float64 십진 표기: 소수 16자리, 부호·정수부 포함 고정 폭(최대 -999.… 수준).
VALUE_DECIMALS = 16
VALUE_WIDTH = 21


def fmt_value(v: float) -> str:
    return f"{v:{VALUE_WIDTH}.{VALUE_DECIMALS}f}"


def phase_log_filename(when: datetime, phase_name: str) -> str:
    """예: 20261116_201728_place_empty_box.log"""
    token = (phase_name or "").strip() or "unknown_phase"
    stamp = when.strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{token}.log"


def timestamp_line(when: datetime) -> str:
    return when.strftime("%Y/%m/%d %H:%M:%S")


def part_line(label: str, joint_count: int, seed: float) -> str:
    values = [seed + 1.0e-10 * i for i in range(joint_count)]
    text = " ".join(fmt_value(v) for v in values)
    return f"    {label}:\t{text}\n"


def render_phase_log(
    number: int,
    name: str,
    when: datetime,
    *,
    value_seed: float = 0.5234198574829103,
) -> str:
    padded = name.ljust(PHASE_NAME_WIDTH)
    out = [
        f"* Phase {number:02d}\n",
        f"* Phase 명: {padded}\n",
        f"* Time stamp: {timestamp_line(when)}\n",
        "* 출력 관절\n",
    ]
    seed = value_seed
    for label, joints in PART_LINES:
        out.append(part_line(label, len(joints), seed))
        seed += 0.1
    return "".join(out)


def main() -> None:
    out_dir = ROOT / "samples" / "phase_logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 샘플: 동일 시각으로 파일명만 phase명 차이
    when = datetime(2026, 11, 16, 20, 17, 28)

    sizes: list[tuple[str, int]] = []
    for num, name in PHASES:
        body = render_phase_log(num, name, when)
        raw = body.encode("utf-8")
        fname = phase_log_filename(when, name)
        path = out_dir / fname
        path.write_bytes(raw)
        sizes.append((fname, len(raw)))

    print(f"Output directory: {out_dir}")
    print(f"Phase name field width: {PHASE_NAME_WIDTH}")
    print(
        f"Value format: {VALUE_WIDTH} chars, {VALUE_DECIMALS} decimals "
        "(one line per part)"
    )
    print()
    for fname, nbytes in sizes:
        print(f"  {fname}: {nbytes} bytes")
    uniq = {n for _, n in sizes}
    if len(uniq) == 1:
        print(f"\nEach phase file: {uniq.pop()} bytes")
    print(f"\nExample: {out_dir / phase_log_filename(when, 'place_empty_box')}")


if __name__ == "__main__":
    main()