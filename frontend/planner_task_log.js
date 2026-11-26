/** Task 실행 하단 작업 로그 (Task 시작 시 초기화·갱신). */

import { formatOpenvlaResultPhaseText, openvlaHasResultPhase } from "./openvla_phase_display.js";
import { matchPlannerTaskPolicy, yoloLabelCounts } from "./planner_task_match.js";
import {
  getEffectivePlannerRules,
  policyDisplayName,
} from "./planner_task_settings.js";

let lines = [];

function ts() {
  return new Date().toLocaleTimeString("ko-KR", { hour12: false });
}

function renderPlannerTaskLogView() {
  const el = document.getElementById("planner-task-log");
  if (!(el instanceof HTMLElement)) return;
  if (!lines.length) {
    el.textContent = "Task 시작을 누르면 작업 로그가 표시됩니다.";
    el.classList.add("is-placeholder");
    return;
  }
  el.classList.remove("is-placeholder");
  el.textContent = lines.join("\n");
  el.scrollTop = el.scrollHeight;
}

export function refreshPlannerTaskLog() {
  lines = [];
  renderPlannerTaskLogView();
}

export function appendPlannerTaskLog(message, { level = "info" } = {}) {
  const text = String(message ?? "").trim();
  if (!text) return;
  const prefix = level === "err" ? "[!] " : level === "ok" ? "[✓] " : "";
  lines.push(`[${ts()}] ${prefix}${text}`);
  renderPlannerTaskLogView();
}

/** 단계 전환·의사결정 (관절값은 넣지 않음). */
export function appendPlannerTaskStage(stage, detail, { level = "info" } = {}) {
  const tag = String(stage || "").trim();
  const body = String(detail || "").trim();
  if (!body) return;
  const line = tag ? `${tag} — ${body}` : body;
  appendPlannerTaskLog(line, { level });
}

export function formatYoloSummary(yolo) {
  if (yolo?.error) return `오류: ${yolo.error}`;
  if (!yolo?.ready) return "대기 중 (마지막 YOLO 스냅샷 없음)";
  const counts = yoloLabelCounts(yolo);
  const parts = [];
  for (const [label, count] of counts) {
    if (count > 0) parts.push(`${label} × ${count}`);
  }
  return parts.length ? parts.join(", ") : "탐지된 객체 없음";
}

export function beginPlannerTaskLog({ instruction, yolo, phaseProfile }) {
  refreshPlannerTaskLog();
  appendPlannerTaskLog("Task 시작");
  if (instruction) {
    appendPlannerTaskLog(`명령어: ${instruction}`);
  }
  const profile = String(phaseProfile || "").trim();
  if (profile) {
    appendPlannerTaskLog(`OpenVLA Phase 프로필: ${profile}`);
  }
  appendPlannerTaskLog(`YOLO: ${formatYoloSummary(yolo)}`);
  appendPlannerTaskStage(
    "준비",
    "사이클 루프 시작 (OpenVLA → 조건 매칭 → LeRobot 실행·검증)"
  );
}

export function describeTaskRuleBrief(rule, phaseRows = []) {
  if (!rule) return "—";
  const bits = [];
  const profile = String(rule.openvla_profile || "").trim();
  bits.push(profile ? `OpenVLA 프로필=${profile}` : "OpenVLA 프로필=무관");
  const phaseId = String(rule.openvla_phase_id || "").trim();
  if (phaseId) {
    const row = (phaseRows || []).find((r) => String(r.id) === phaseId);
    const label = row
      ? `P${row.number}${row.description ? ` ${row.description}` : ""}`
      : phaseId.slice(0, 8);
    bits.push(`Phase=${label}`);
  } else {
    bits.push("Phase=무관");
  }
  const yolo = String(rule.yolo_label || "").trim();
  const min = rule.yolo_min_count ?? 1;
  if (yolo) bits.push(`YOLO ${yolo}≥${min}`);
  else bits.push("YOLO=무관");
  const policy = String(rule.policy_id || "").trim();
  if (policy) bits.push(`→ Policy ${policyDisplayName(policy)}`);
  return bits.join(" · ");
}

export function logTaskPolicyMatch({ match, phaseRows, yolo, ruleCount }) {
  appendPlannerTaskStage("2. 조건 매칭", `YOLO 스냅샷: ${formatYoloSummary(yolo)}`);
  if (!ruleCount) {
    appendPlannerTaskLog("Task 설정에 규칙 없음", { level: "err" });
    return;
  }
  appendPlannerTaskLog(`등록된 조건 ${ruleCount}개 — 위에서부터 순서대로 검사`);
  if (match?.policyId) {
    const rule = match.rule;
    appendPlannerTaskStage(
      "2. 조건 매칭",
      `조건 #${match.index + 1} 일치 — ${describeTaskRuleBrief(rule, phaseRows)}`,
      { level: "ok" }
    );
  } else {
    appendPlannerTaskLog("모든 조건 불일치 — LeRobot Policy 없음", { level: "err" });
  }
}

export function logChunkAnalysisDecision(analysis, chunkLen) {
  appendPlannerTaskStage(
    "4. 궤적 검사",
    `청크 ${chunkLen} step · OpenVLA target_joints 기준 부위 오차 비교`
  );
  if (analysis.reject) {
    appendPlannerTaskStage("4. 궤적 검사", `실행 거부 — ${analysis.reason || "불일치"}`, {
      level: "err",
    });
    return;
  }
  const init = analysis.initial_error_sum;
  const fin = analysis.final_error_sum;
  if (init != null && fin != null) {
    const trend =
      fin < init - 0.01
        ? "목표 쪽으로 감소(수렴)"
        : fin > init + 0.01
          ? "목표에서 멀어짐"
          : "거의 변화 없음";
    appendPlannerTaskLog(
      `오차 합 ${init.toFixed(3)} → ${fin.toFixed(3)} rad — ${trend} (${analysis.steps_analyzed} step)`
    );
  }
  const truncated =
    analysis.divergence_step != null && Number(analysis.divergence_step) > 0;
  if (truncated) {
    appendPlannerTaskStage(
      "5. 실행 결정",
      `발산 step ${analysis.divergence_step} 이전까지만 로봇 구동 (tick 한도 ${analysis.max_publish_steps})`
    );
  } else if (analysis.converging) {
    appendPlannerTaskStage("5. 실행 결정", "수렴 — 정책 완료까지 전체 실행", { level: "ok" });
  } else {
    appendPlannerTaskStage(
      "5. 실행 결정",
      `비수렴이지만 한계 이내 — tick 한도 ${analysis.max_publish_steps} 로 실행`
    );
  }
}

export function finishPlannerTaskLog({ openvla, yolo, phaseRows, phaseProfile }) {
  if (openvla?.error) {
    appendPlannerTaskLog(`OpenVLA 오류: ${openvla.error}`, { level: "err" });
  } else if (!openvla?.ready) {
    appendPlannerTaskLog("OpenVLA 결과 없음", { level: "err" });
  } else if (openvlaHasResultPhase(openvla)) {
    const formatted = formatOpenvlaResultPhaseText(openvla, phaseRows);
    const phaseText =
      formatted && !formatted.error ? formatted.text : "Phase 결과를 해석할 수 없음";
    appendPlannerTaskLog(`OpenVLA 다음 Phase: ${phaseText}`, {
      level: formatted?.error ? "err" : "ok",
    });
  } else {
    appendPlannerTaskLog("OpenVLA 추론 완료 (Phase 결과 없음)");
  }

  const rules = getEffectivePlannerRules();
  if (!rules.length) {
    appendPlannerTaskLog("LeRobot: Task 설정에 조건 없음");
    return;
  }

  const yoloSnap = yolo ?? {};
  const match = matchPlannerTaskPolicy(rules, {
    yolo: yoloSnap,
    openvla,
    phaseRows,
    phaseProfile,
  });
  const policyId = match?.policyId ? String(match.policyId) : "";
  if (policyId) {
    const name = policyDisplayName(policyId);
    appendPlannerTaskLog(
      `LeRobot Policy 선택: ${name}${name !== policyId ? ` (${policyId})` : ""}`,
      { level: "ok" }
    );
    if (match.index >= 0) {
      appendPlannerTaskLog(`매칭 조건 #${match.index + 1}`);
    }
  } else {
    appendPlannerTaskLog("LeRobot: 일치하는 Policy 없음", { level: "err" });
  }
}

export function failPlannerTaskLog(err) {
  appendPlannerTaskLog(`Task 실패: ${String(err?.message || err)}`, { level: "err" });
}

export function initPlannerTaskLog() {
  renderPlannerTaskLogView();
}