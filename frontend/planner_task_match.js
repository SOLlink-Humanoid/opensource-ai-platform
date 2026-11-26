/** Compare live YOLO + OpenVLA results to Task 설정 rules → LeRobot policy. */

import {
  openvlaHasResultPhase,
  openvlaPhaseRoundedNumber,
  openvlaPhaseRowForDisplayNumber,
} from "./openvla_phase_display.js";

export function yoloLabelCounts(yolo) {
  const targets = (yolo?.targets || []).filter((row) => row.enabled);
  const allowed = new Set(targets.map((row) => String(row.name || "").trim()).filter(Boolean));
  const counts = new Map();
  for (const det of yolo?.detections || []) {
    const label = String(det.label || "").trim();
    if (!allowed.has(label)) continue;
    counts.set(label, (counts.get(label) || 0) + 1);
  }
  return counts;
}

export function openvlaPredictedPhaseId(openvla, phaseRows) {
  if (!openvlaHasResultPhase(openvla)) return "";
  const num = openvlaPhaseRoundedNumber(openvla, phaseRows);
  if (num == null || num < 0) return "";
  const row = openvlaPhaseRowForDisplayNumber(num, phaseRows);
  const id = row?.id != null ? String(row.id).trim() : "";
  return id;
}

function ruleMatches(rule, snapshot) {
  const profileNeed = String(rule.openvla_profile || "").trim();
  if (profileNeed && profileNeed !== snapshot.phaseProfile) return false;

  const phaseNeed = String(rule.openvla_phase_id || "").trim();
  if (phaseNeed) {
    if (!snapshot.predictedPhaseId || phaseNeed !== snapshot.predictedPhaseId) {
      return false;
    }
  }

  const labelNeed = String(rule.yolo_label || "").trim();
  const minCount = Number(rule.yolo_min_count);
  const min = Number.isFinite(minCount) ? Math.max(0, Math.min(99, minCount)) : 1;
  if (labelNeed) {
    const have = snapshot.yoloCounts.get(labelNeed) || 0;
    if (have < min) return false;
  }

  const policyId = String(rule.policy_id || "").trim();
  if (!policyId) return false;

  return true;
}

/**
 * @returns {{ policyId: string, rule: object | null, index: number } | null}
 */
export function matchPlannerTaskPolicy(
  rules,
  { yolo, openvla, phaseRows, phaseProfile }
) {
  const list = Array.isArray(rules) ? rules : [];
  if (!list.length) return null;

  const snapshot = {
    phaseProfile: String(phaseProfile || "").trim(),
    predictedPhaseId: openvlaPredictedPhaseId(openvla, phaseRows),
    yoloCounts: yoloLabelCounts(yolo ?? {}),
  };

  for (let i = 0; i < list.length; i += 1) {
    const rule = list[i];
    if (!rule) continue;
    if (ruleMatches(rule, snapshot)) {
      return {
        policyId: String(rule.policy_id || "").trim(),
        rule,
        index: i,
      };
    }
  }
  return { policyId: "", rule: null, index: -1 };
}