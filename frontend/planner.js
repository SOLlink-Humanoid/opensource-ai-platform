/** Task Planner tab: combined YOLO / OpenVLA / LeRobot inference status. */

import {
  applyOpenvlaResultPhaseDisplay,
  openvlaHasResultPhase,
} from "./openvla_phase_display.js";
import {
  clearOpenvlaPhasePartErrors,
  renderOpenvlaPhasePartErrorValues,
} from "./openvla_part_errors.js";
import { matchPlannerTaskPolicy } from "./planner_task_match.js";
import {
  getEffectivePlannerRules,
  policyDisplayName,
} from "./planner_task_settings.js";

const PLANNER_OPENVLA_ERR_PREFIX = "planner-openvla-phase-err-";

let plannerPaneChangeListener = null;

export function setPlannerPaneChangeListener(fn) {
  plannerPaneChangeListener = typeof fn === "function" ? fn : null;
}

function yoloDetectionCountsByTarget(yolo) {
  const targets = (yolo?.targets || []).filter((row) => row.enabled);
  const allowed = new Set(targets.map((row) => row.name));
  const counts = new Map();
  for (const det of yolo?.detections || []) {
    if (!allowed.has(det.label)) continue;
    counts.set(det.label, (counts.get(det.label) || 0) + 1);
  }
  return targets.map((row) => [row.name, counts.get(row.name) || 0]);
}

function setPlannerYoloMeta(meta, message, { err = false, show = true } = {}) {
  if (!meta) return;
  if (!show || !message) {
    meta.hidden = true;
    meta.textContent = "";
    meta.classList.remove("err");
    return;
  }
  meta.hidden = false;
  meta.textContent = message;
  meta.classList.toggle("err", err);
}

function renderPlannerYolo(yolo) {
  const meta = document.getElementById("planner-yolo-meta");
  const root = document.getElementById("planner-yolo-overview");
  if (!root) return;

  root.innerHTML = "";
  if (yolo?.error) {
    setPlannerYoloMeta(meta, `오류: ${yolo.error}`, { err: true });
    return;
  }
  if (!yolo?.ready) {
    setPlannerYoloMeta(meta, "대기 중…");
    const empty = document.createElement("p");
    empty.className = "planner-model-empty";
    empty.textContent = "—";
    root.appendChild(empty);
    return;
  }

  setPlannerYoloMeta(meta, null, { show: false });

  const counts = yoloDetectionCountsByTarget(yolo);
  const detected = counts.filter(([, count]) => count > 0);

  if (!counts.length) {
    const empty = document.createElement("p");
    empty.className = "planner-model-empty";
    empty.textContent = "탐지 타겟 없음";
    root.appendChild(empty);
    return;
  }

  if (!detected.length) {
    const empty = document.createElement("p");
    empty.className = "planner-model-empty";
    empty.textContent = "탐지된 객체 없음";
    root.appendChild(empty);
    return;
  }

  for (const [label, count] of detected) {
    const row = document.createElement("div");
    row.className = "openvla-metric-row planner-openvla-phase-row";
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = `${count}개`;
    row.appendChild(dt);
    row.appendChild(dd);
    root.appendChild(row);
  }
}

function renderPlannerLerobotPolicyMatch({ yolo, openvla, phaseRows, phaseProfile }) {
  const root = document.getElementById("planner-lerobot-match");
  if (!(root instanceof HTMLElement)) return;

  const rules = getEffectivePlannerRules();
  if (!rules.length) {
    root.textContent = "—";
    root.classList.remove("is-match", "is-empty");
    root.dataset.policyId = "";
    return;
  }

  const match = matchPlannerTaskPolicy(rules, {
    yolo,
    openvla,
    phaseRows,
    phaseProfile,
  });
  const policyId = match?.policyId ? String(match.policyId) : "";

  root.dataset.policyId = policyId;
  root.classList.toggle("is-match", Boolean(policyId));
  root.classList.toggle("is-empty", !policyId);

  if (policyId) {
    root.textContent = policyDisplayName(policyId);
    return;
  }
  root.textContent = "일치하는 Policy 없음";
}

function renderPlannerOpenvlaPhaseProfile(profileName) {
  const el = document.getElementById("planner-openvla-phase-profile");
  if (!el) return;
  const text = String(profileName || "").trim();
  el.textContent = text || "—";
}

function renderPlannerOpenvlaPhaseBlock(openvla, phaseRows) {
  const phaseEl = document.getElementById("planner-predicted-phase");
  if (!phaseEl) return;

  const rows = phaseRows ?? [];
  const hasResult = openvlaHasResultPhase(openvla);
  applyOpenvlaResultPhaseDisplay(phaseEl, openvla, rows, hasResult);

  if (!hasResult) {
    clearOpenvlaPhasePartErrors(PLANNER_OPENVLA_ERR_PREFIX);
    return;
  }
  renderOpenvlaPhasePartErrorValues(
    openvla,
    openvla.target_phase_pose,
    hasResult,
    PLANNER_OPENVLA_ERR_PREFIX
  );
}

function renderOpenvlaInference(openvla, phaseRows, phaseProfile) {
  renderPlannerOpenvlaPhaseProfile(phaseProfile);
  const grid = document.getElementById("planner-inference-grid");
  const meta = document.getElementById("planner-openvla-meta");

  if (meta) {
    if (openvla?.error) {
      meta.hidden = false;
      meta.textContent = `오류: ${openvla.error}`;
      meta.classList.add("err");
    } else if (openvla?.engine_status === "loading") {
      meta.hidden = false;
      meta.textContent = "모델 로딩 중…";
      meta.classList.remove("err");
    } else if (openvla?.ready) {
      meta.hidden = true;
      meta.textContent = "";
      meta.classList.remove("err");
    } else if (openvla?.engine_ready === false) {
      meta.hidden = false;
      meta.textContent = openvla.engine_error || "엔진 준비 안 됨";
      meta.classList.add("err");
    } else {
      meta.hidden = false;
      meta.textContent = "대기 중…";
      meta.classList.remove("err");
    }
  }

  if (!openvla?.ready) {
    if (grid) grid.hidden = true;
    renderPlannerOpenvlaPhaseBlock(openvla, phaseRows);
    return;
  }

  if (grid) grid.hidden = false;
  renderPlannerOpenvlaPhaseBlock(openvla, phaseRows);
}

export function renderPlannerPanel(body) {
  const openvla = body.openvla ?? {};
  const phaseRows = body.openvlaPhaseRows ?? [];
  const phaseProfile = body.openvlaPhaseProfile ?? "";
  const yolo = body.yolo ?? {};
  renderPlannerYolo(yolo);
  renderOpenvlaInference(openvla, phaseRows, phaseProfile);
  renderPlannerLerobotPolicyMatch({ yolo, openvla, phaseRows, phaseProfile });
}

function setPlannerSidePane(paneId) {
  const panes = {
    run: document.getElementById("planner-pane-run"),
    settings: document.getElementById("planner-pane-settings"),
  };
  const tabs = document.querySelectorAll(".openvla-bookmark[data-planner-pane]");
  for (const tab of tabs) {
    const active = tab.dataset.plannerPane === paneId;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  }
  for (const [key, el] of Object.entries(panes)) {
    if (!el) continue;
    el.hidden = key !== paneId;
  }
  plannerPaneChangeListener?.(paneId);
}

function bindPlannerSideBookmarks() {
  const nav = document.getElementById("planner-bookmarks");
  if (!nav || nav.dataset.bound === "1") return;
  nav.dataset.bound = "1";
  nav.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const tab = target.closest(".openvla-bookmark[data-planner-pane]");
    if (!tab) return;
    const paneId = tab.dataset.plannerPane;
    if (!paneId) return;
    setPlannerSidePane(paneId);
  });
}

export function initPlannerPanel() {
  bindPlannerSideBookmarks();
}