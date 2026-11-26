/** Task 시작: OpenVLA ↔ YOLO 매칭 → LeRobot 실행·수렴 검사·루프. */

import { matchPlannerTaskPolicy } from "./planner_task_match.js";
import {
  appendPlannerTaskLog,
  appendPlannerTaskStage,
  beginPlannerTaskLog,
  formatYoloSummary,
  logChunkAnalysisDecision,
  logTaskPolicyMatch,
} from "./planner_task_log.js";
import {
  getEffectivePlannerRules,
  policyDisplayName,
} from "./planner_task_settings.js";
import { formatOpenvlaResultPhaseText, openvlaHasResultPhase } from "./openvla_phase_display.js";

const CHUNK_WAIT_MS = 20000;
const CHUNK_POLL_MS = 250;
const TICK_POLL_MS = 200;
const SETTLE_POLL_MS = 300;
const SETTLE_MAX_MS = 45000;
const SETTLE_EPS_RAD = 0.004;
const SETTLE_STABLE_POLLS = 6;
const MAX_TASK_CYCLES = 64;

let deps = null;
let taskRun = {
  active: false,
  abort: null,
  cycle: 0,
};

export function setPlannerTaskRunnerDeps(d) {
  deps = d && typeof d === "object" ? d : null;
}

export function isPlannerTaskRunning() {
  return taskRun.active;
}

export function stopPlannerTask(reason = "사용자 중지") {
  if (!taskRun.active) return;
  taskRun.abort?.abort();
  appendPlannerTaskLog(reason, { level: "err" });
}

function requireDeps() {
  if (!deps?.api || !deps?.apiPost) {
    throw new Error("Task runner가 초기화되지 않았습니다");
  }
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("aborted", "AbortError"));
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("aborted", "AbortError"));
      },
      { once: true }
    );
  });
}

function parseLastPublishTick(logText) {
  const text = String(logText || "");
  const re = /\btick=(\d+)\b/g;
  let last = 0;
  let m;
  while ((m = re.exec(text))) {
    const n = Number(m[1]);
    if (Number.isFinite(n)) last = Math.max(last, n);
  }
  return last;
}

function jointsDelta(a, b) {
  if (!a || !b) return Infinity;
  let sum = 0;
  let n = 0;
  for (const [key, va] of Object.entries(a)) {
    if (!(key in b)) continue;
    const vb = Number(b[key]);
    const vaN = Number(va);
    if (!Number.isFinite(vaN) || !Number.isFinite(vb)) continue;
    sum += Math.abs(vaN - vb);
    n += 1;
  }
  return n ? sum / n : Infinity;
}

async function fetchLive() {
  requireDeps();
  return deps.api("/api/live");
}

async function waitForActionChunk(signal) {
  appendPlannerTaskStage("3. LeRobot", "첫 action_chunk 대기 중…");
  const deadline = Date.now() + CHUNK_WAIT_MS;
  while (Date.now() < deadline) {
    if (signal.aborted) throw new DOMException("aborted", "AbortError");
    const body = await fetchLive();
    const chunk = body?.lerobot?.action_chunk;
    if (Array.isArray(chunk) && chunk.length > 0) {
      appendPlannerTaskStage(
        "3. LeRobot",
        `action_chunk 수신 (${chunk.length} step, dim ${chunk[0]?.length ?? "?"})`,
        { level: "ok" }
      );
      return chunk;
    }
    const st = await deps.api("/api/lerobot/policies/status");
    if (!st?.running) break;
    await sleep(CHUNK_POLL_MS, signal);
  }
  return null;
}

async function checkChunk({ chunk, openvla }) {
  requireDeps();
  return deps.apiPost("/api/planner/lerobot-chunk-check", {
    action_chunk: chunk,
    openvla_target_joints: openvla?.target_joints || {},
    inferred_joints_by_part: openvla?.inferred_joints_by_part || {},
  });
}

async function waitUntilPolicyStops(signal) {
  appendPlannerTaskStage("6. 구동", "정책 전체 실행 — 완료 대기");
  const deadline = Date.now() + 600000;
  while (Date.now() < deadline) {
    if (signal.aborted) throw new DOMException("aborted", "AbortError");
    const st = await deps.api("/api/lerobot/policies/status");
    if (!st?.running) {
      appendPlannerTaskStage("6. 구동", "정책 정상 종료", { level: "ok" });
      return;
    }
    await sleep(TICK_POLL_MS, signal);
  }
  appendPlannerTaskStage("6. 구동", "장시간 미종료 — 정책 중지", { level: "err" });
  await deps.lerobotPolicyStop({ silent: true });
}

async function pollPolicyTicksUntil(maxTicks, signal) {
  let lastTick = 0;
  const deadline = Date.now() + CHUNK_WAIT_MS + 120000;
  while (Date.now() < deadline) {
    if (signal.aborted) throw new DOMException("aborted", "AbortError");
    const st = await deps.api("/api/lerobot/policies/status");
    if (!st?.running) break;
    try {
      const log = await deps.api("/api/lerobot/policies/action-log?lines=120");
      const tail = String(log?.log || "");
      lastTick = parseLastPublishTick(tail);
      if (maxTicks > 0 && lastTick >= maxTicks) {
        appendPlannerTaskStage(
          "6. 구동",
          `tick 한도 도달 (${lastTick}/${maxTicks}) — 정책 중지`,
          { level: "ok" }
        );
        await deps.lerobotPolicyStop({ silent: true });
        return lastTick;
      }
    } catch {
      /* log optional */
    }
    await sleep(TICK_POLL_MS, signal);
  }
  if ((await deps.api("/api/lerobot/policies/status"))?.running) {
    appendPlannerTaskStage("6. 구동", "타임아웃 — 정책 강제 중지", { level: "err" });
    await deps.lerobotPolicyStop({ silent: true });
  }
  return lastTick;
}

async function waitForSettle(signal) {
  const deadline = Date.now() + SETTLE_MAX_MS;
  let prev = null;
  let stable = 0;
  while (Date.now() < deadline) {
    if (signal.aborted) throw new DOMException("aborted", "AbortError");
    const jointBody = await deps.api("/api/joints");
    const joints = jointBody?.positions || jointBody?.joints;
    if (joints && typeof joints === "object") {
      if (prev && jointsDelta(prev, joints) < SETTLE_EPS_RAD) {
        stable += 1;
        if (stable >= SETTLE_STABLE_POLLS) {
          appendPlannerTaskStage("7. 안정화", "관절 변화율 안정 — 다음 사이클 가능", {
            level: "ok",
          });
          return;
        }
      } else {
        stable = 0;
      }
      prev = joints;
    }
    await sleep(SETTLE_POLL_MS, signal);
  }
  appendPlannerTaskStage("7. 안정화", "대기 시간 초과 — 다음 OpenVLA 사이클로 진행");
}

async function runOpenvlaOnce() {
  requireDeps();
  const openvla = await deps.runOpenvlaInferCore();
  deps.renderPlannerAfterOpenvla?.(openvla);
  return openvla;
}

function logOpenvlaSummary(openvla, phaseRows) {
  if (openvla?.error) {
    appendPlannerTaskLog(`OpenVLA 오류: ${openvla.error}`, { level: "err" });
    return;
  }
  if (openvlaHasResultPhase(openvla)) {
    const formatted = formatOpenvlaResultPhaseText(openvla, phaseRows);
    if (formatted && !formatted.error) {
      appendPlannerTaskLog(`OpenVLA 다음 Phase: ${formatted.text}`, { level: "ok" });
    }
  }
}

function resolvePolicyMatch(liveBody, openvla, phaseRows, phaseProfile) {
  const rules = getEffectivePlannerRules();
  if (!rules.length) return null;
  return matchPlannerTaskPolicy(rules, {
    yolo: liveBody?.yolo,
    openvla,
    phaseRows,
    phaseProfile,
  });
}

async function runPlannerTaskLoop() {
  requireDeps();
  if (taskRun.active) return;
  const abort = new AbortController();
  taskRun = { active: true, abort, cycle: 0 };
  deps.setTaskUiRunning?.(true);

  const instruction = deps.openvlaInstructionValue?.() || "";
  const phaseProfile = deps.getOpenvlaInferenceProfileName?.() || "";
  let liveBody = await fetchLive();

  let completedCycles = 0;
  let stopReason = "";
  try {
    beginPlannerTaskLog({
      instruction,
      yolo: liveBody?.yolo,
      phaseProfile,
    });

    while (taskRun.cycle < MAX_TASK_CYCLES) {
      if (abort.signal.aborted) break;
      taskRun.cycle += 1;
      completedCycles = taskRun.cycle;
      appendPlannerTaskLog(`──────── 사이클 ${taskRun.cycle} ────────`);

      appendPlannerTaskStage("1. OpenVLA", "추론 시작");
      const openvla = await runOpenvlaOnce();
      if (openvla?.error) {
        stopReason = "OpenVLA 오류";
        appendPlannerTaskStage("1. OpenVLA", `실패 — ${openvla.error}`, { level: "err" });
        break;
      }
      appendPlannerTaskStage("1. OpenVLA", "추론 완료 — UI 갱신(관절은 비교용만)", { level: "ok" });

      const phaseRows = deps.getOpenvlaInferencePhaseRows?.() || [];
      logOpenvlaSummary(openvla, phaseRows);

      liveBody = await fetchLive();
      const rules = getEffectivePlannerRules();
      const match = resolvePolicyMatch(liveBody, openvla, phaseRows, phaseProfile);
      logTaskPolicyMatch({
        match,
        phaseRows,
        yolo: liveBody?.yolo,
        ruleCount: rules.length,
      });
      const policyId = match?.policyId ? String(match.policyId) : "";
      if (!policyId) {
        stopReason = "Policy 매칭 없음";
        deps.renderPlannerPanel?.();
        break;
      }

      const policyLabel = policyDisplayName(policyId);
      await deps.lerobotPolicyStop({ silent: true });
      appendPlannerTaskStage(
        "3. LeRobot",
        `정책 시작: ${policyLabel}${policyLabel !== policyId ? ` (${policyId})` : ""}`
      );
      const started = await deps.lerobotPolicyPlay(policyId, { silent: true });
      if (!started) {
        stopReason = "LeRobot 시작 실패";
        appendPlannerTaskStage("3. LeRobot", "정책 프로세스 기동 실패", { level: "err" });
        break;
      }
      appendPlannerTaskStage("3. LeRobot", "정책 실행 중", { level: "ok" });

      const chunk = await waitForActionChunk(abort.signal);
      if (!chunk?.length) {
        stopReason = "action_chunk 없음";
        appendPlannerTaskStage("3. LeRobot", "chunk 미수신 — Task 중단", { level: "err" });
        await deps.lerobotPolicyStop({ silent: true });
        break;
      }

      const analysis = await checkChunk({ chunk, openvla });
      logChunkAnalysisDecision(analysis, chunk.length);
      if (analysis.reject) {
        stopReason = analysis.reason || "궤적 검사 거부";
        await deps.lerobotPolicyStop({ silent: true });
        break;
      }

      const maxSteps = Number(analysis.max_publish_steps) || chunk.length;
      const truncated =
        analysis.divergence_step != null && analysis.divergence_step > 0;

      if (maxSteps <= 0) {
        stopReason = "안전 실행 구간 없음";
        await deps.lerobotPolicyStop({ silent: true });
        appendPlannerTaskStage("5. 실행 결정", "실행 step 0 — Task 중단", { level: "err" });
        break;
      }

      if (truncated && maxSteps < chunk.length) {
        await pollPolicyTicksUntil(maxSteps, abort.signal);
      } else {
        await waitUntilPolicyStops(abort.signal);
      }
      await waitForSettle(abort.signal);

      appendPlannerTaskStage(
        "다음",
        `사이클 ${taskRun.cycle} 완료 — YOLO(${formatYoloSummary(liveBody?.yolo)}) 기준으로 재매칭`
      );
      deps.renderPlannerPanel?.();
    }
    if (abort.signal.aborted) {
      stopReason = stopReason || "사용자 중지";
    } else if (taskRun.cycle >= MAX_TASK_CYCLES) {
      stopReason = `최대 사이클(${MAX_TASK_CYCLES}) 도달`;
    } else if (!stopReason && completedCycles > 0) {
      stopReason = "정상 종료";
    }
  } catch (e) {
    if (e?.name !== "AbortError") {
      appendPlannerTaskLog(`Task 오류: ${e.message || e}`, { level: "err" });
    }
    try {
      await deps.lerobotPolicyStop({ silent: true });
    } catch {
      /* ignore */
    }
  } finally {
    const cycles = completedCycles;
    taskRun = { active: false, abort: null, cycle: 0 };
    deps.setTaskUiRunning?.(false);
    deps.renderPlannerPanel?.();
    await deps.lerobotPolicyLoad?.();
    if (cycles > 0 || stopReason) {
      appendPlannerTaskStage(
        "종료",
        `${stopReason || "Task 끝"} · 실행 사이클 ${cycles}회`,
        stopReason && stopReason !== "정상 종료" ? { level: "err" } : { level: "ok" }
      );
    }
  }
}

export function startPlannerTask() {
  void runPlannerTaskLoop();
}