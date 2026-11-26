import {
  applyOpenvlaResultPhaseDisplay,
  openvlaHasResultPhase,
  phaseNumberFromModelTail,
} from "./openvla_phase_display.js";
import {
  OPENVLA_PHASE_ERR_PARTS,
  clearOpenvlaPhasePartErrors as clearOpenvlaPartErrorElements,
  openvlaPartHasModelOutput,
  renderOpenvlaPhasePartErrorValues,
} from "./openvla_part_errors.js";
import {
  initPlannerPanel,
  renderPlannerPanel,
  setPlannerPaneChangeListener,
} from "./planner.js";
import {
  initPlannerTaskRulesCache,
  initPlannerTaskSettings,
  onPlannerSettingsPaneOpen,
  syncPlannerPolicyCatalog,
} from "./planner_task_settings.js";
import { initPlannerTaskLog } from "./planner_task_log.js";
import {
  setPlannerTaskRunnerDeps,
  startPlannerTask,
  stopPlannerTask,
} from "./planner_task_runner.js";
import {
  applyYoloTagConditionsConfig,
  yoloTagConditionsSyncBlocked,
  prepareYoloTagRulesPaneFirstOpen,
  initYoloTagConditions,
  setYoloTagClassOptions,
  updateYoloActiveTagsFromDetections,
  yoloBoxOrientation,
} from "./yolo_tag_conditions.js";

const TAB_TITLES = {
  yolo: "YOLO11",
  openvla: "OpenVLA",
  lerobot: "LeRobot",
  planner: "Task Planner",
  dev: "Configuration",
};

const TAB_PATHS = {
  yolo: "/yolo",
  openvla: "/openvla",
  lerobot: "/lerobot",
  planner: "/planner",
  dev: "/dev",
};

const PATH_TO_TAB = Object.fromEntries(
  Object.entries(TAB_PATHS).map(([tab, path]) => [path, tab])
);

let activeTab = "yolo";

const YOLO_INFERENCE_TABS = new Set(["yolo", "planner"]);
const YOLO_FAST_INTERVAL_SEC = 0.2;
const YOLO_BACKGROUND_INTERVAL_SEC = 1.0;
let cameraId = null;
let pollTimer = null;
let openvlaPinned = null;
let openvlaBusy = false;
let openvlaMotionBusy = false;
let openvlaMotionEpoch = 0;
let openvlaEngineReady = false;
let lastLiveBody = null;
let studioRawPreview = false;

const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebar-toggle");
const panelTitle = document.getElementById("panel-title");
const frameBadge = document.getElementById("frame-badge");
const healthDot = document.getElementById("health-dot");
const healthText = document.getElementById("health-text");

sidebarToggle.addEventListener("click", () => {
  sidebar.classList.toggle("collapsed");
});

function tabFromPathname(pathname) {
  const normalized = pathname.replace(/\/+$/, "") || "/yolo";
  if (normalized === "/") {
    return "yolo";
  }
  return PATH_TO_TAB[normalized] || "yolo";
}

function pathForTab(tab) {
  return TAB_PATHS[tab] || TAB_PATHS.yolo;
}

const VIDEO_HEADER_TABS = new Set(["yolo", "openvla", "lerobot", "planner"]);

function placePanelHeader(tab) {
  const header = document.querySelector(".main-header");
  const main = document.querySelector(".main");
  if (!header || !main) return;
  if (VIDEO_HEADER_TABS.has(tab)) {
    const left = document.querySelector(`#panel-${tab} .panel-col-split-pane-left`);
    if (left) left.prepend(header);
    return;
  }
  const firstPanel = main.querySelector(":scope > .panel");
  if (firstPanel) main.insertBefore(header, firstPanel);
  else main.prepend(header);
}

document.querySelectorAll(".tab").forEach((link) => {
  link.addEventListener("click", (event) => {
    const tab = link.dataset.tab;
    if (!tab) return;
    if (
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey ||
      event.button !== 0
    ) {
      return;
    }
    event.preventDefault();
    setTab(tab, { pushState: true });
  });
});

function setTab(tab, { pushState = false } = {}) {
  if (!TAB_TITLES[tab]) {
    tab = "yolo";
  }
  activeTab = tab;
  document.body.dataset.page = tab;
  document.querySelectorAll(".tab").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === tab);
  });
  document.querySelectorAll(".panel").forEach((p) => {
    p.classList.toggle("active", p.id === `panel-${tab}`);
  });
  panelTitle.textContent = TAB_TITLES[tab] || tab;
  document.title = `${TAB_TITLES[tab] || tab} · Open Source AI Platform`;
  placePanelHeader(tab);
  const path = pathForTab(tab);
  if (pushState && window.location.pathname !== path) {
    history.pushState({ tab }, "", path);
  }
  refreshStreams();
  if (tab === "dev") {
    void refreshDevProfile();
  }
  syncYoloInferenceForTab(tab);
  refreshPanelColSplitLayout(tab);
}

async function syncYoloInferenceForTab(tab) {
  const fast = YOLO_INFERENCE_TABS.has(tab);
  try {
    await api("/api/yolo/worker", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        enabled: true,
        interval_sec: fast ? YOLO_FAST_INTERVAL_SEC : YOLO_BACKGROUND_INTERVAL_SEC,
      }),
    });
  } catch (err) {
    console.warn("yolo worker sync", err);
  }
}

function syncTabFromUrl() {
  const tab = tabFromPathname(window.location.pathname);
  setTab(tab, { pushState: false });
}

window.addEventListener("popstate", (event) => {
  const tab = event.state?.tab || tabFromPathname(window.location.pathname);
  setTab(tab, { pushState: false });
});

function streamUrl(kind) {
  if (!cameraId) return "";
  return `/api/stream/${kind}/${encodeURIComponent(cameraId)}`;
}

function updateYoloVideoFrameCorners() {
  const img = document.getElementById("video-yolo");
  const stage = document.getElementById("yolo-video-stage");
  if (!(img instanceof HTMLImageElement) || !stage) return;
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  if (!w || !h) return;
  const tl = stage.querySelector('[data-corner="tl"]');
  const tr = stage.querySelector('[data-corner="tr"]');
  const bl = stage.querySelector('[data-corner="bl"]');
  const br = stage.querySelector('[data-corner="br"]');
  if (tl) tl.textContent = `(0, 0)`;
  if (tr) tr.textContent = `(${w}, 0)`;
  if (bl) bl.textContent = `(0, ${h})`;
  if (br) br.textContent = `(${w}, ${h})`;
}

function bindYoloVideoFrameCorners() {
  const img = document.getElementById("video-yolo");
  if (!(img instanceof HTMLImageElement) || img.dataset.frameCornersBound === "1") {
    return;
  }
  img.dataset.frameCornersBound = "1";
  img.addEventListener("load", updateYoloVideoFrameCorners);
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(updateYoloVideoFrameCorners).observe(img);
  }
  updateYoloVideoFrameCorners();
}

function refreshStreams() {
  const yoloImg = document.getElementById("video-yolo");
  const rawImg = document.getElementById("video-openvla");
  const lerobotImg = document.getElementById("video-lerobot");
  const plannerImg = document.getElementById("video-planner");
  if (!cameraId) return;

  const rawStream = streamUrl("raw");
  const yoloStream = studioRawPreview ? rawStream : streamUrl("yolo");

  if (yoloImg instanceof HTMLImageElement) {
    yoloImg.src = activeTab === "yolo" ? yoloStream : "";
  }
  if (plannerImg instanceof HTMLImageElement) {
    plannerImg.src = activeTab === "planner" ? yoloStream : "";
  }
  if (rawImg instanceof HTMLImageElement) {
    rawImg.src =
      activeTab === "openvla" || activeTab === "lerobot" ? rawStream : "";
  }
  if (lerobotImg instanceof HTMLImageElement) {
    lerobotImg.src = activeTab === "lerobot" ? rawStream : "";
  }
}

async function api(path, options) {
  const res = await fetch(path, { cache: "no-store", ...options });
  if (!res.ok) {
    let detail = `${res.status} ${path}`;
    try {
      const errBody = await res.json();
      if (errBody.detail) detail = String(errBody.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

async function apiPost(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
}

let yoloTargetSaveTimer = null;
let yoloTargetEditing = false;
let yoloTargets = [];
let yoloDefaultConfidence = 0.7;

function normalizeYoloTargets(rawTargets, classes) {
  const byId = new Map((rawTargets || []).map((row) => [row.class_id, row]));
  const source = classes?.length ? classes : yoloTargets;
  return source.map((row) => {
    const saved = byId.get(row.class_id);
    return {
      class_id: row.class_id,
      name: saved?.name || row.name,
      enabled: saved?.enabled ?? true,
      threshold: Number(saved?.threshold ?? yoloDefaultConfidence),
    };
  });
}

function enabledTargetLabels() {
  return new Set(
    yoloTargets.filter((row) => row.enabled).map((row) => row.name)
  );
}

function renderYoloTargetBoard() {
  const board = document.getElementById("yolo-target-board");
  if (!board) return;
  board.innerHTML = "";
  if (!yoloTargets.length) {
    const empty = document.createElement("p");
    empty.className = "yolo-target-board-empty";
    empty.textContent = "모델 클래스 정보 로딩 중…";
    board.appendChild(empty);
    return;
  }
  for (const row of yoloTargets) {
    const slot = document.createElement("div");
    slot.className = "yolo-target-row" + (row.enabled ? "" : " disabled");
    slot.dataset.classId = String(row.class_id);

    const checkLabel = document.createElement("label");
    checkLabel.className = "yolo-target-check";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = row.enabled;
    checkbox.dataset.classId = String(row.class_id);
    const name = document.createElement("span");
    name.className = "yolo-target-name";
    name.textContent = row.name;
    checkLabel.appendChild(checkbox);
    checkLabel.appendChild(name);

    const thresholdWrap = document.createElement("div");
    thresholdWrap.className = "yolo-target-threshold-wrap";
    const thresholdLabel = document.createElement("span");
    thresholdLabel.className = "yolo-target-threshold-label";
    thresholdLabel.textContent = "confidence";
    const thresholdInput = document.createElement("input");
    thresholdInput.type = "number";
    thresholdInput.min = "0";
    thresholdInput.max = "1";
    thresholdInput.step = "0.01";
    thresholdInput.inputMode = "decimal";
    thresholdInput.className = "yolo-target-threshold";
    thresholdInput.value = Number(row.threshold).toFixed(2);
    thresholdInput.disabled = !row.enabled;
    thresholdInput.dataset.classId = String(row.class_id);
    thresholdWrap.appendChild(thresholdLabel);
    thresholdWrap.appendChild(thresholdInput);

    slot.appendChild(checkLabel);
    slot.appendChild(thresholdWrap);
    board.appendChild(slot);
  }
}

function queueSaveYoloTargets() {
  clearTimeout(yoloTargetSaveTimer);
  yoloTargetSaveTimer = setTimeout(async () => {
    try {
      const res = await api("/api/yolo/targets", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          targets: yoloTargets.map((row) => ({
            class_id: row.class_id,
            enabled: row.enabled,
            threshold: row.threshold,
          })),
        }),
      });
      if (!yoloTargetEditing) {
        yoloTargets = normalizeYoloTargets(res.targets, res.classes);
        renderYoloTargetBoard();
      }
    } catch (err) {
      console.warn("yolo targets", err);
    }
  }, 350);
}

function bindYoloTargetBoard() {
  const board = document.getElementById("yolo-target-board");
  if (!board || board.dataset.bound === "1") return;
  board.dataset.bound = "1";
  board.addEventListener("focusin", () => {
    yoloTargetEditing = true;
  });
  board.addEventListener("focusout", () => {
    window.setTimeout(() => {
      if (!board.contains(document.activeElement)) {
        yoloTargetEditing = false;
      }
    }, 0);
  });
  board.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    const classId = Number(target.dataset.classId);
    const row = yoloTargets.find((item) => item.class_id === classId);
    if (!row) return;
    if (target.type === "checkbox") {
      row.enabled = target.checked;
      const thresholdInput = board.querySelector(
        `.yolo-target-threshold[data-class-id="${classId}"]`
      );
      if (thresholdInput instanceof HTMLInputElement) {
        thresholdInput.disabled = !row.enabled;
      }
      const slot = target.closest(".yolo-target-row");
      slot?.classList.toggle("disabled", !row.enabled);
      queueSaveYoloTargets();
      return;
    }
    if (target.classList.contains("yolo-target-threshold")) {
      const value = Number(target.value);
      if (!Number.isFinite(value) || value < 0 || value > 1) return;
      row.threshold = value;
      queueSaveYoloTargets();
    }
  });
  board.addEventListener("input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.classList.contains("yolo-target-threshold")) return;
    const classId = Number(target.dataset.classId);
    const row = yoloTargets.find((item) => item.class_id === classId);
    if (!row) return;
    const value = Number(target.value);
    if (!Number.isFinite(value) || value < 0 || value > 1) return;
    row.threshold = value;
    queueSaveYoloTargets();
  });
}

const OPENVLA_MAX_PROFILE_TAGS = 5;
const OPENVLA_INFERENCE_PROFILE_KEY = "oap-openvla-inference-profile";
const OPENVLA_INFERENCE_AUTO_TAG_KEY = "oap-openvla-inference-auto-tag";

let openvlaPhases = [];
/** @type {Record<string, { phases: { id: string, description: string, pose_name: string, pose_file: string }[], tags: string[], hand_pose_file: string }>} */
let openvlaPhaseProfiles = {};
let openvlaActivePhaseProfile = "기본";
let openvlaInferenceProfile = "";
let openvlaInferenceAutoTag = true;
/** @type {string[]} */
let openvlaProfileTags = [];
let openvlaProfileHandPoseFile = "";
/** @type {{ id: string, name: string }[]} */
let openvlaHandPoseFiles = [];
/** @type {{ file: string, poses: { name: string, id: string | null, file: string }[] }[]} */
let openvlaPoseFiles = [];
let openvlaPhaseSaveTimer = null;
let openvlaPhaseEditing = false;
let openvlaPhaseProfileRenaming = false;
let openvlaHandPoseMetricsPending = false;

function newOpenvlaPhaseId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID().replace(/-/g, "").slice(0, 12);
  }
  return `phase-${Date.now().toString(36)}`;
}

function poseFilesFromResponse(res) {
  const groups = Array.isArray(res?.files) ? res.files : [];
  const mapGroup = (group) => ({
    file: String(group?.file || ""),
    poses: (group?.poses || [])
      .map((pose) => ({
        name: String(pose?.name || ""),
        id: pose?.id || null,
        file: String(pose?.file || group?.file || ""),
      }))
      .filter((pose) => pose.name)
      .sort((a, b) => a.name.localeCompare(b.name)),
  });
  if (groups.length) {
    return groups
      .map(mapGroup)
      .filter((group) => group.file)
      .sort((a, b) => a.file.localeCompare(b.file));
  }
  const byFile = new Map();
  const loose = [];
  for (const pose of res?.poses || []) {
    if (!pose?.name) continue;
    const file = String(pose?.file || "");
    if (!file) {
      loose.push(pose);
      continue;
    }
    if (!byFile.has(file)) byFile.set(file, []);
    byFile.get(file).push(pose);
  }
  const grouped = [...byFile.entries()]
    .map(([file, poses]) => mapGroup({ file, poses }))
    .sort((a, b) => a.file.localeCompare(b.file));
  if (grouped.length) return grouped;
  if (!loose.length) return [];
  return [mapGroup({ file: "", poses: loose })];
}

function openvlaNamedPoseFiles() {
  return openvlaPoseFiles.filter((group) => group.file);
}

function poseFileLabel(name) {
  const text = String(name || "");
  return text.toLowerCase().endsWith(".json") ? text.slice(0, -5) : text;
}

function openvlaPosesInFile(file) {
  if (!file) return [];
  const group = openvlaPoseFiles.find((item) => item.file === file);
  return group?.poses || [];
}

function resolveOpenvlaPoseFile(row) {
  const saved = String(row?.pose_file || "");
  if (saved && openvlaPoseFiles.some((item) => item.file === saved)) return saved;
  const poseName = String(row?.pose_name || "");
  if (!poseName) return "";
  const matches = openvlaPoseFiles.filter((item) =>
    item.poses.some((pose) => pose.name === poseName)
  );
  return matches.length === 1 ? matches[0].file : "";
}

function openvlaPoseIdForName(name, file) {
  if (!name) return null;
  const groups = file
    ? openvlaPoseFiles.filter((item) => item.file === file)
    : openvlaPoseFiles;
  for (const group of groups) {
    const row = group.poses.find((pose) => pose.name === name);
    if (row?.id) return row.id;
  }
  return null;
}

function fillPoseSelect(select, options, selected, blankLabel) {
  select.replaceChildren();
  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = blankLabel;
  select.appendChild(blank);
  let found = !selected;
  for (const option of options) {
    const opt = document.createElement("option");
    opt.value = option.value;
    opt.textContent = option.label;
    if (option.value === selected) {
      opt.selected = true;
      found = true;
    }
    select.appendChild(opt);
  }
  if (selected && !found) {
    const extra = document.createElement("option");
    extra.value = selected;
    extra.textContent = poseFileLabel(selected);
    extra.selected = true;
    select.appendChild(extra);
  }
}

function normalizeOpenvlaPhaseNumber(raw, fallback) {
  const n = Number(raw);
  if (Number.isFinite(n)) return Math.round(n * 10) / 10;
  const fb = Number(fallback);
  return Number.isFinite(fb) ? Math.round(fb * 10) / 10 : 1;
}

function cloneOpenvlaPhaseRows(rows) {
  return (rows || []).map((row, index) => ({
    id: row.id,
    number: normalizeOpenvlaPhaseNumber(row.number, index + 1),
    description: row.description || "",
    pose_name: row.pose_name || "",
    pose_file: row.pose_file || "",
  }));
}

function nextOpenvlaPhaseNumber(phases) {
  if (!phases.length) return 1;
  let maxInt = 0;
  for (const row of phases) {
    const n = Number(row.number);
    if (Number.isFinite(n)) {
      maxInt = Math.max(maxInt, Math.floor(n));
    }
  }
  return maxInt + 1;
}

function openvlaPhaseNumbersEqual(a, b) {
  return Math.abs(Number(a) - Number(b)) < 0.051;
}

function findDuplicateOpenvlaPhaseNumber(phaseId, number) {
  return openvlaPhases.some(
    (row) => row.id !== phaseId && openvlaPhaseNumbersEqual(row.number, number)
  );
}

function sortOpenvlaPhasesByNumber() {
  openvlaPhases.sort((a, b) => {
    const diff = Number(a.number) - Number(b.number);
    if (diff !== 0) return diff;
    return String(a.id).localeCompare(String(b.id));
  });
}

function applyOpenvlaPhaseNumberChange(phaseId, nextNum, input = null) {
  const row = openvlaPhases.find((item) => item.id === phaseId);
  if (!row) return false;
  const normalized = normalizeOpenvlaPhaseNumber(nextNum, row.number ?? 1);
  if (findDuplicateOpenvlaPhaseNumber(phaseId, normalized)) {
    setOpenvlaPhaseSaveStatus(
      `Phase 번호 ${normalized} 은(는) 이미 사용 중입니다. 다른 값을 입력하세요.`,
      true
    );
    if (input instanceof HTMLInputElement) {
      input.value = String(normalizeOpenvlaPhaseNumber(row.number, 1));
    }
    return false;
  }
  if (openvlaPhaseNumbersEqual(normalized, row.number)) {
    if (input instanceof HTMLInputElement) {
      input.value = String(normalized);
    }
    return true;
  }
  row.number = normalized;
  sortOpenvlaPhasesByNumber();
  setOpenvlaPhaseSaveStatus("", false);
  renderOpenvlaPhaseBoard();
  queueSaveOpenvlaPhases();
  return true;
}

function commitOpenvlaPhaseNumInput(input) {
  if (!(input instanceof HTMLInputElement)) return false;
  const phaseId = input.dataset.phaseId;
  const row = openvlaPhases.find((item) => item.id === phaseId);
  if (!row) return false;
  return applyOpenvlaPhaseNumberChange(
    phaseId,
    normalizeOpenvlaPhaseNumber(input.value, row.number ?? 1),
    input
  );
}

function emptyOpenvlaProfileBucket() {
  return { phases: [], tags: [], hand_pose_file: "" };
}

function normalizeOpenvlaTags(tags) {
  const cleaned = [];
  const seen = new Set();
  for (const item of tags || []) {
    const text = String(item || "").trim();
    if (!text || seen.has(text)) continue;
    seen.add(text);
    cleaned.push(text);
    if (cleaned.length >= OPENVLA_MAX_PROFILE_TAGS) break;
  }
  return cleaned;
}

function resolveOpenvlaProfileBucket(raw) {
  if (Array.isArray(raw)) {
    return { phases: cloneOpenvlaPhaseRows(raw), tags: [], hand_pose_file: "" };
  }
  if (!raw || typeof raw !== "object") {
    return emptyOpenvlaProfileBucket();
  }
  return {
    phases: cloneOpenvlaPhaseRows(raw.phases),
    tags: normalizeOpenvlaTags(raw.tags),
    hand_pose_file: String(raw.hand_pose_file || ""),
  };
}

function commitOpenvlaPhasesToActiveProfile() {
  if (!openvlaActivePhaseProfile) return;
  openvlaPhaseProfiles[openvlaActivePhaseProfile] = {
    phases: cloneOpenvlaPhaseRows(openvlaPhases),
    tags: normalizeOpenvlaTags(openvlaProfileTags),
    hand_pose_file: String(openvlaProfileHandPoseFile || ""),
  };
}

function loadOpenvlaActiveProfileView() {
  const bucket = resolveOpenvlaProfileBucket(
    openvlaPhaseProfiles[openvlaActivePhaseProfile]
  );
  openvlaPhases = bucket.phases;
  openvlaProfileTags = bucket.tags;
  openvlaProfileHandPoseFile = bucket.hand_pose_file || "";
}

function loadOpenvlaInferenceAutoTagPref() {
  const saved = localStorage.getItem(OPENVLA_INFERENCE_AUTO_TAG_KEY);
  if (saved === null) {
    openvlaInferenceAutoTag = true;
    return;
  }
  openvlaInferenceAutoTag = saved === "1" || saved === "true";
}

function yoloDetectionClassLabels() {
  const detections = lastLiveBody?.yolo?.detections || [];
  const labels = new Set();
  for (const det of detections) {
    const label = String(det?.label || "").trim().toLowerCase();
    if (label) {
      labels.add(label);
    }
  }
  return [...labels];
}

function scoreOpenvlaProfileTagsAgainstYoloClasses(profileName, classLabels) {
  const bucket = resolveOpenvlaProfileBucket(openvlaPhaseProfiles[profileName]);
  if (!classLabels.length) return 0;
  let score = 0;
  for (const tag of bucket.tags) {
    const needle = String(tag || "").trim().toLowerCase();
    if (!needle) continue;
    const matched = classLabels.some((className) => className.includes(needle));
    if (matched) {
      score += 1;
    }
  }
  return score;
}

function pickOpenvlaInferenceProfileFromYolo() {
  const classLabels = yoloDetectionClassLabels();
  if (!classLabels.length) return null;
  const names = Object.keys(openvlaPhaseProfiles);
  if (!names.length) return null;
  const scored = names.map((name) => ({
    name,
    score: scoreOpenvlaProfileTagsAgainstYoloClasses(name, classLabels),
  }));
  const bestScore = Math.max(0, ...scored.map((row) => row.score));
  if (bestScore === 0) return null;
  const winners = scored.filter((row) => row.score === bestScore);
  if (winners.length !== 1) return null;
  return winners[0].name;
}

function applyOpenvlaAutoProfileFromYoloIfNeeded() {
  if (!openvlaInferenceAutoTag) return false;
  const prev = openvlaInferenceProfile;
  ensureOpenvlaInferenceProfile();
  const changed = openvlaInferenceProfile !== prev;
  if (changed) {
    renderOpenvlaInferenceProfileSelect();
    touchPlannerPanel();
    void refreshOpenvlaInferenceMetrics();
  }
  return changed;
}

function syncOpenvlaInferenceProfileUi() {
  const select = document.getElementById("openvla-inference-profile");
  const checkbox = document.getElementById("openvla-inference-auto-tag");
  if (checkbox instanceof HTMLInputElement) {
    checkbox.checked = openvlaInferenceAutoTag;
  }
  if (select instanceof HTMLSelectElement) {
    const names = Object.keys(openvlaPhaseProfiles);
    select.disabled = openvlaInferenceAutoTag || !names.length;
  }
}

function onOpenvlaInferenceProfileSourceChanged() {
  ensureOpenvlaInferenceProfile();
  renderOpenvlaInferenceProfileSelect();
  touchPlannerPanel();
  void refreshOpenvlaInferenceMetrics();
}

function ensureOpenvlaInferenceProfile() {
  const names = Object.keys(openvlaPhaseProfiles);
  if (!names.length) {
    openvlaInferenceProfile = "";
    return;
  }
  if (openvlaInferenceAutoTag) {
    const matched = pickOpenvlaInferenceProfileFromYolo();
    if (matched) {
      openvlaInferenceProfile = matched;
      return;
    }
    if (openvlaInferenceProfile && names.includes(openvlaInferenceProfile)) {
      return;
    }
  }
  if (openvlaInferenceProfile && names.includes(openvlaInferenceProfile)) {
    return;
  }
  const saved = localStorage.getItem(OPENVLA_INFERENCE_PROFILE_KEY);
  if (saved && names.includes(saved)) {
    openvlaInferenceProfile = saved;
    return;
  }
  if (names.includes(openvlaActivePhaseProfile)) {
    openvlaInferenceProfile = openvlaActivePhaseProfile;
    return;
  }
  openvlaInferenceProfile = names[0];
}

function getOpenvlaInferencePhaseRows() {
  ensureOpenvlaInferenceProfile();
  const name = openvlaInferenceProfile || openvlaActivePhaseProfile;
  return resolveOpenvlaProfileBucket(openvlaPhaseProfiles[name]).phases;
}

function getOpenvlaInferenceProfileName() {
  ensureOpenvlaInferenceProfile();
  return openvlaInferenceProfile || "";
}

function renderOpenvlaInferenceProfileSelect() {
  const select = document.getElementById("openvla-inference-profile");
  if (!select) return;
  ensureOpenvlaInferenceProfile();
  const names = Object.keys(openvlaPhaseProfiles);
  const selected = openvlaInferenceProfile;
  select.innerHTML = "";
  if (!names.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "프로필 없음";
    select.appendChild(opt);
    syncOpenvlaInferenceProfileUi();
    return;
  }
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    opt.selected = name === selected;
    select.appendChild(opt);
  }
  syncOpenvlaInferenceProfileUi();
}

async function refreshOpenvlaInferenceMetrics() {
  const base = openvlaPinned ?? lastLiveBody?.openvla;
  if (!base?.ready) {
    renderOpenvla(base);
    return;
  }
  const target = base.target_joints;
  if (!target || !Object.keys(target).length) {
    renderOpenvla(base);
    return;
  }
  ensureOpenvlaInferenceProfile();
  if (!openvlaInferenceProfile) {
    renderOpenvla(base);
    return;
  }
  try {
    const actionTail = base.predicted_action?.length
      ? base.predicted_action[base.predicted_action.length - 1]
      : null;
    const phaseRows = getOpenvlaInferencePhaseRows();
    const scaledPhase =
      actionTail != null && actionTail !== ""
        ? phaseNumberFromModelTail(actionTail, phaseRows)
        : null;
    const metrics = await apiPost("/api/openvla/profile-metrics", {
      phase_profile: openvlaInferenceProfile,
      phase_index: scaledPhase != null ? null : base.phase_index ?? null,
      phase_text:
        scaledPhase != null ? String(scaledPhase) : base.phase_text ?? null,
      target_joints: target,
      predicted_action: base.predicted_action ?? [],
    });
    const merged = {
      ...base,
      predicted_phase: metrics.predicted_phase ?? base.predicted_phase,
      target_phase_pose: metrics.target_phase_pose ?? base.target_phase_pose ?? null,
      target_joints: metrics.target_joints ?? base.target_joints,
      phase_index: metrics.predicted_phase?.index ?? base.phase_index,
      phase_description:
        metrics.predicted_phase?.description ?? base.phase_description,
    };
    openvlaPinned = merged;
    renderOpenvla(merged);
    touchPlannerPanel();
  } catch (err) {
    console.warn("openvla profile metrics", err);
    renderOpenvla(base);
  }
}

function bindOpenvlaInferenceProfileSelect() {
  const select = document.getElementById("openvla-inference-profile");
  if (!select || select.dataset.bound === "1") return;
  select.dataset.bound = "1";
  select.addEventListener("change", () => {
    if (openvlaInferenceAutoTag) return;
    openvlaInferenceProfile = select.value;
    localStorage.setItem(OPENVLA_INFERENCE_PROFILE_KEY, openvlaInferenceProfile);
    touchPlannerPanel();
    void refreshOpenvlaInferenceMetrics();
  });
}

function bindOpenvlaInferenceAutoTag() {
  const checkbox = document.getElementById("openvla-inference-auto-tag");
  if (!checkbox || checkbox.dataset.bound === "1") return;
  checkbox.dataset.bound = "1";
  loadOpenvlaInferenceAutoTagPref();
  syncOpenvlaInferenceProfileUi();
  checkbox.addEventListener("change", () => {
    openvlaInferenceAutoTag = checkbox.checked;
    localStorage.setItem(
      OPENVLA_INFERENCE_AUTO_TAG_KEY,
      openvlaInferenceAutoTag ? "1" : "0"
    );
    onOpenvlaInferenceProfileSourceChanged();
  });
}

function uniqueOpenvlaProfileName(base) {
  const root = (base || "프로필").trim() || "프로필";
  if (!Object.prototype.hasOwnProperty.call(openvlaPhaseProfiles, root)) {
    return root;
  }
  let index = 2;
  while (Object.prototype.hasOwnProperty.call(openvlaPhaseProfiles, `${root} ${index}`)) {
    index += 1;
  }
  return `${root} ${index}`;
}

function applyOpenvlaPhaseDocument(res) {
  const profilesIn = res?.profiles && typeof res.profiles === "object" ? res.profiles : {};
  const nextProfiles = {};
  for (const [name, bucket] of Object.entries(profilesIn)) {
    const key = String(name).trim();
    if (!key) continue;
    nextProfiles[key] = resolveOpenvlaProfileBucket({
      phases: Array.isArray(bucket?.phases) ? bucket.phases : [],
      tags: bucket?.tags,
      hand_pose_file: bucket?.hand_pose_file,
    });
  }
  if (!Object.keys(nextProfiles).length && Array.isArray(res?.phases)) {
    const fallback = res.active_profile || openvlaActivePhaseProfile || "기본";
    nextProfiles[fallback] = resolveOpenvlaProfileBucket({ phases: res.phases, tags: [] });
  }
  openvlaPhaseProfiles = nextProfiles;
  const active = res?.active_profile || openvlaActivePhaseProfile;
  openvlaActivePhaseProfile =
    active && nextProfiles[active] ? active : Object.keys(nextProfiles)[0] || "기본";
  if (!openvlaPhaseProfiles[openvlaActivePhaseProfile]) {
    openvlaPhaseProfiles[openvlaActivePhaseProfile] = emptyOpenvlaProfileBucket();
  }
  loadOpenvlaActiveProfileView();
}

function buildOpenvlaPhaseSaveBody() {
  commitOpenvlaPhasesToActiveProfile();
  const profiles = {};
  for (const [name, bucket] of Object.entries(openvlaPhaseProfiles)) {
    const profile = resolveOpenvlaProfileBucket(bucket);
    profiles[name] = {
      phases: profile.phases.map((row) => {
        const phase = {
          id: row.id,
          number: normalizeOpenvlaPhaseNumber(row.number, 1),
          description: row.description,
          pose_name: row.pose_name,
        };
        if (row.pose_file) phase.pose_file = row.pose_file;
        return phase;
      }),
      tags: [...profile.tags],
      hand_pose_file: profile.hand_pose_file || "",
    };
  }
  return {
    active_profile: openvlaActivePhaseProfile,
    profiles,
  };
}

function renderOpenvlaPhaseProfileTabs() {
  const bar = document.getElementById("openvla-phase-profiles");
  if (!bar) return;
  if (openvlaPhaseProfileRenaming) return;
  bar.innerHTML = "";
  const names = Object.keys(openvlaPhaseProfiles);
  for (const name of names) {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = "openvla-phase-profile-tab";
    tab.role = "tab";
    tab.dataset.profile = name;
    tab.setAttribute("aria-selected", name === openvlaActivePhaseProfile ? "true" : "false");
    tab.setAttribute("aria-controls", "openvla-phase-profile-bundle");
    tab.title = name;
    tab.textContent = name;
    bar.appendChild(tab);
  }
  const add = document.createElement("button");
  add.type = "button";
  add.className = "openvla-phase-profile-tab openvla-phase-profile-tab--add";
  add.dataset.action = "add-profile";
  add.setAttribute("aria-label", "프로필 추가");
  add.textContent = "+";
  bar.appendChild(add);
}

function beginOpenvlaProfileRename(profileName) {
  const bar = document.getElementById("openvla-phase-profiles");
  if (!bar || openvlaPhaseProfileRenaming) return;
  openvlaPhaseProfileRenaming = true;
  bar.innerHTML = "";
  const input = document.createElement("input");
  input.type = "text";
  input.className = "openvla-phase-profile-rename";
  input.value = profileName;
  input.spellcheck = false;
  input.setAttribute("aria-label", "프로필 이름");
  bar.appendChild(input);
  input.focus();
  input.select();

  const cancel = () => {
    openvlaPhaseProfileRenaming = false;
    renderOpenvlaPhaseProfileTabs();
  };

  const commit = () => {
    const next = input.value.trim();
    openvlaPhaseProfileRenaming = false;
    if (!next || next === "active_profile" || next === profileName) {
      renderOpenvlaPhaseProfileTabs();
      return;
    }
    if (Object.prototype.hasOwnProperty.call(openvlaPhaseProfiles, next)) {
      setOpenvlaPhaseSaveStatus(`이미 있는 프로필 이름: ${next}`, true);
      renderOpenvlaPhaseProfileTabs();
      return;
    }
    commitOpenvlaPhasesToActiveProfile();
    const bucket = resolveOpenvlaProfileBucket(openvlaPhaseProfiles[profileName]);
    if (!bucket) {
      renderOpenvlaPhaseProfileTabs();
      return;
    }
    delete openvlaPhaseProfiles[profileName];
    openvlaPhaseProfiles[next] = bucket;
    if (openvlaActivePhaseProfile === profileName) {
      openvlaActivePhaseProfile = next;
      loadOpenvlaActiveProfileView();
    }
    renderOpenvlaPhaseProfileTabs();
    renderOpenvlaPhaseBoard();
    renderOpenvlaPhaseTags();
    queueSaveOpenvlaPhases();
  };

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      commit();
    } else if (event.key === "Escape") {
      event.preventDefault();
      cancel();
    }
  });
  input.addEventListener("blur", () => {
    window.setTimeout(commit, 0);
  });
}

function switchOpenvlaPhaseProfile(profileName) {
  if (!profileName || profileName === openvlaActivePhaseProfile) return;
  if (!Object.prototype.hasOwnProperty.call(openvlaPhaseProfiles, profileName)) return;
  commitOpenvlaPhasesToActiveProfile();
  openvlaActivePhaseProfile = profileName;
  loadOpenvlaActiveProfileView();
  openvlaPhaseEditing = false;
  renderOpenvlaPhaseProfileTabs();
  renderOpenvlaPhaseBoard();
  renderOpenvlaPhaseTags();
  queueSaveOpenvlaPhases();
}

function addOpenvlaPhaseProfile() {
  commitOpenvlaPhasesToActiveProfile();
  const name = uniqueOpenvlaProfileName("프로필");
  openvlaPhaseProfiles[name] = emptyOpenvlaProfileBucket();
  openvlaActivePhaseProfile = name;
  loadOpenvlaActiveProfileView();
  openvlaPhaseEditing = false;
  renderOpenvlaPhaseProfileTabs();
  renderOpenvlaPhaseBoard();
  renderOpenvlaPhaseTags();
  queueSaveOpenvlaPhases();
}

function applyOpenvlaHandPoseFiles(res) {
  const files = Array.isArray(res?.hand_pose_files) ? res.hand_pose_files : [];
  openvlaHandPoseFiles = files
    .map((item) => ({
      id: String(item?.id || ""),
      name: String(item?.name || item?.id || ""),
    }))
    .filter((item) => item.id);
}

function renderOpenvlaPhaseHandSelect() {
  const select = document.getElementById("openvla-phase-hand-file");
  if (!(select instanceof HTMLSelectElement)) return;
  const selected = String(openvlaProfileHandPoseFile || "");
  fillPoseSelect(
    select,
    openvlaHandPoseFiles.map((file) => ({
      value: file.id,
      label: file.name || poseFileLabel(file.id),
    })),
    selected,
    "핸드 자세 없음"
  );
}

function bindOpenvlaPhaseHandSelect() {
  const select = document.getElementById("openvla-phase-hand-file");
  if (!(select instanceof HTMLSelectElement) || select.dataset.bound === "1") return;
  select.dataset.bound = "1";
  select.addEventListener("change", () => {
    openvlaProfileHandPoseFile = select.value;
    commitOpenvlaPhasesToActiveProfile();
    openvlaHandPoseMetricsPending = openvlaInferenceProfile === openvlaActivePhaseProfile;
    queueSaveOpenvlaPhases();
  });
}

function renderOpenvlaPhaseTags() {
  const wrap = document.getElementById("openvla-phase-tags");
  const input = document.getElementById("openvla-phase-tag-input");
  if (!wrap) return;
  wrap.innerHTML = "";
  renderOpenvlaPhaseHandSelect();
  const tags = normalizeOpenvlaTags(openvlaProfileTags);
  if (!tags.length) {
    const empty = document.createElement("p");
    empty.className = "openvla-phase-tags-empty";
    empty.textContent = "태그 없음";
    wrap.appendChild(empty);
  } else {
    for (const label of tags) {
      const chip = document.createElement("span");
      chip.className = "openvla-phase-tag";
      const text = document.createElement("span");
      text.className = "openvla-phase-tag-label";
      text.textContent = label;
      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "openvla-phase-tag-remove";
      removeBtn.dataset.tag = label;
      removeBtn.setAttribute("aria-label", `태그 ${label} 삭제`);
      removeBtn.textContent = "×";
      chip.appendChild(text);
      chip.appendChild(removeBtn);
      wrap.appendChild(chip);
    }
  }
  if (input instanceof HTMLInputElement) {
    const full = tags.length >= OPENVLA_MAX_PROFILE_TAGS;
    input.disabled = full;
    input.placeholder = full
      ? "태그는 최대 5개까지 추가할 수 있습니다"
      : "태그 입력 후 Enter (최대 5개)";
  }
}

function addOpenvlaProfileTag(raw) {
  const text = String(raw || "").trim();
  if (!text) return;
  const tags = normalizeOpenvlaTags(openvlaProfileTags);
  if (tags.length >= OPENVLA_MAX_PROFILE_TAGS) {
    setOpenvlaPhaseSaveStatus("태그는 최대 5개까지 추가할 수 있습니다.", true);
    return;
  }
  if (tags.includes(text)) {
    setOpenvlaPhaseSaveStatus(`이미 있는 태그: ${text}`, true);
    return;
  }
  tags.push(text);
  openvlaProfileTags = tags;
  commitOpenvlaPhasesToActiveProfile();
  renderOpenvlaPhaseTags();
  queueSaveOpenvlaPhases();
  setOpenvlaPhaseSaveStatus("", false);
}

function removeOpenvlaProfileTag(tag) {
  const key = String(tag || "").trim();
  if (!key) return;
  openvlaProfileTags = normalizeOpenvlaTags(openvlaProfileTags).filter((item) => item !== key);
  commitOpenvlaPhasesToActiveProfile();
  renderOpenvlaPhaseTags();
  queueSaveOpenvlaPhases();
}

function bindOpenvlaPhaseTags() {
  const wrap = document.getElementById("openvla-phase-tags");
  const input = document.getElementById("openvla-phase-tag-input");
  if (!wrap || wrap.dataset.bound === "1") return;
  wrap.dataset.bound = "1";
  wrap.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const btn = target.closest(".openvla-phase-tag-remove");
    if (!btn) return;
    event.preventDefault();
    const tag = btn.dataset.tag;
    if (!tag) return;
    removeOpenvlaProfileTag(tag);
  });
  if (input instanceof HTMLInputElement) {
    input.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      const value = input.value;
      input.value = "";
      addOpenvlaProfileTag(value);
    });
  }
}

function renderOpenvlaPhaseBoard() {
  const board = document.getElementById("openvla-phase-board");
  if (!board) return;
  board.innerHTML = "";
  if (!openvlaPhases.length) {
    const empty = document.createElement("p");
    empty.className = "openvla-phase-board-empty";
    empty.textContent = "등록된 phase가 없습니다. 아래에서 추가하세요.";
    board.appendChild(empty);
    return;
  }
  openvlaPhases.forEach((row, index) => {
    const slot = document.createElement("div");
    slot.className = "openvla-phase-row";
    slot.dataset.phaseId = row.id;

    const numWrap = document.createElement("div");
    numWrap.className = "openvla-phase-num-wrap";
    const num = document.createElement("input");
    num.type = "number";
    num.step = "0.1";
    num.min = "0";
    num.inputMode = "decimal";
    num.className = "openvla-phase-num";
    num.value = String(normalizeOpenvlaPhaseNumber(row.number, index + 1));
    num.dataset.phaseId = row.id;
    num.setAttribute("aria-label", "Phase 번호");
    const spinCol = document.createElement("div");
    spinCol.className = "openvla-phase-num-spin";
    const stepUp = document.createElement("button");
    stepUp.type = "button";
    stepUp.className = "openvla-phase-num-step";
    stepUp.dataset.delta = "0.1";
    stepUp.dataset.phaseId = row.id;
    stepUp.setAttribute("aria-label", "Phase 번호 0.1 증가");
    stepUp.textContent = "▲";
    const stepDown = document.createElement("button");
    stepDown.type = "button";
    stepDown.className = "openvla-phase-num-step";
    stepDown.dataset.delta = "-0.1";
    stepDown.dataset.phaseId = row.id;
    stepDown.setAttribute("aria-label", "Phase 번호 0.1 감소");
    stepDown.textContent = "▼";
    spinCol.append(stepUp, stepDown);
    numWrap.append(num, spinCol);

    const fields = document.createElement("div");
    fields.className = "openvla-phase-fields";

    const desc = document.createElement("input");
    desc.type = "text";
    desc.className = "openvla-phase-desc";
    desc.placeholder = "description";
    desc.value = row.description;
    desc.dataset.phaseId = row.id;
    desc.spellcheck = false;

    const poseRow = document.createElement("div");
    poseRow.className = "openvla-phase-pose-row";
    const namedFiles = openvlaNamedPoseFiles();
    const poseFile = namedFiles.length ? resolveOpenvlaPoseFile(row) : "";
    if (poseFile && !row.pose_file) row.pose_file = poseFile;
    const poseOptions = namedFiles.length
      ? openvlaPosesInFile(poseFile)
      : openvlaPoseFiles.flatMap((group) => group.poses);

    const picks = document.createElement("div");
    picks.className = "openvla-phase-pose-picks";

    if (namedFiles.length) {
      const fileSelect = document.createElement("select");
      fileSelect.className = "openvla-phase-pose openvla-phase-pose-file";
      fileSelect.dataset.phaseId = row.id;
      fileSelect.setAttribute("aria-label", "pose 파일");
      fillPoseSelect(
        fileSelect,
        namedFiles.map((group) => ({
          value: group.file,
          label: poseFileLabel(group.file),
        })),
        poseFile,
        "파일 선택…"
      );
      picks.appendChild(fileSelect);
    }

    const poseSelect = document.createElement("select");
    poseSelect.className = "openvla-phase-pose";
    poseSelect.dataset.phaseId = row.id;
    poseSelect.dataset.poseFile = poseFile;
    poseSelect.setAttribute("aria-label", "pose");
    poseSelect.disabled = namedFiles.length > 0 && !poseFile;
    fillPoseSelect(
      poseSelect,
      poseOptions.map((pose) => ({
        value: pose.name,
        label: pose.name,
      })),
      row.pose_name,
      "pose 선택…"
    );

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "openvla-phase-remove";
    removeBtn.dataset.phaseId = row.id;
    removeBtn.title = "삭제";
    removeBtn.setAttribute("aria-label", "phase 삭제");
    removeBtn.textContent = "×";

    picks.appendChild(poseSelect);
    poseRow.appendChild(picks);
    poseRow.appendChild(removeBtn);
    fields.appendChild(desc);
    fields.appendChild(poseRow);
    slot.appendChild(numWrap);
    slot.appendChild(fields);
    board.appendChild(slot);
  });
}

function setOpenvlaPhaseSaveStatus(message, isError = false) {
  const board = document.getElementById("openvla-phase-board");
  if (!board) return;
  let status = document.getElementById("openvla-phase-save-status");
  if (!message) {
    if (status) {
      status.textContent = "";
      status.hidden = true;
    }
    return;
  }
  if (!status) {
    status = document.createElement("p");
    status.id = "openvla-phase-save-status";
    status.className = "openvla-phase-save-status";
    board.insertAdjacentElement("afterend", status);
  }
  status.hidden = false;
  status.textContent = message;
  status.classList.toggle("openvla-phase-save-status--err", isError);
}

function queueSaveOpenvlaPhases() {
  clearTimeout(openvlaPhaseSaveTimer);
  openvlaPhaseSaveTimer = setTimeout(async () => {
    try {
      const res = await api("/api/phases", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildOpenvlaPhaseSaveBody()),
      });
      setOpenvlaPhaseSaveStatus("", false);
      if (!openvlaPhaseEditing && !openvlaPhaseProfileRenaming) {
        const priorActive = openvlaActivePhaseProfile;
        applyOpenvlaPhaseDocument(res);
        if (priorActive && openvlaPhaseProfiles[priorActive]) {
          openvlaActivePhaseProfile = priorActive;
          loadOpenvlaActiveProfileView();
        }
        renderOpenvlaPhaseProfileTabs();
        renderOpenvlaPhaseBoard();
        renderOpenvlaPhaseTags();
        if (openvlaInferenceAutoTag) {
          onOpenvlaInferenceProfileSourceChanged();
        } else {
          renderOpenvlaInferenceProfileSelect();
        }
        if (openvlaHandPoseMetricsPending) {
          openvlaHandPoseMetricsPending = false;
          void refreshOpenvlaInferenceMetrics();
        }
      }
    } catch (err) {
      const msg = err?.message ? String(err.message) : String(err);
      setOpenvlaPhaseSaveStatus(`저장 실패: ${msg}`, true);
      console.warn("openvla phases", err);
    }
  }, 350);
}

function bindOpenvlaPhaseProfileTabs() {
  const bar = document.getElementById("openvla-phase-profiles");
  if (!bar || bar.dataset.bound === "1") return;
  bar.dataset.bound = "1";
  bar.addEventListener("click", (event) => {
    if (openvlaPhaseProfileRenaming) return;
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const addBtn = target.closest("[data-action='add-profile']");
    if (addBtn) {
      event.preventDefault();
      addOpenvlaPhaseProfile();
      return;
    }
    const tab = target.closest(".openvla-phase-profile-tab[data-profile]");
    if (!tab) return;
    const profile = tab.dataset.profile;
    if (!profile) return;
    switchOpenvlaPhaseProfile(profile);
  });
  bar.addEventListener("dblclick", (event) => {
    if (openvlaPhaseProfileRenaming) return;
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const tab = target.closest(".openvla-phase-profile-tab[data-profile]");
    if (!tab) return;
    const profile = tab.dataset.profile;
    if (!profile) return;
    event.preventDefault();
    beginOpenvlaProfileRename(profile);
  });
}

function bindOpenvlaPhaseBoard() {
  const board = document.getElementById("openvla-phase-board");
  if (!board || board.dataset.bound === "1") return;
  board.dataset.bound = "1";
  board.addEventListener("focusin", (event) => {
    const target = event.target;
    if (target instanceof HTMLInputElement && target.classList.contains("openvla-phase-num")) {
      const row = openvlaPhases.find((item) => item.id === target.dataset.phaseId);
      if (row) {
        target.dataset.phaseNumPrev = String(normalizeOpenvlaPhaseNumber(row.number, 1));
      }
    }
    openvlaPhaseEditing = true;
  });
  board.addEventListener("focusout", (event) => {
    const target = event.target;
    const related =
      event.relatedTarget instanceof HTMLElement ? event.relatedTarget : null;
    if (target instanceof HTMLInputElement && target.classList.contains("openvla-phase-num")) {
      if (
        related &&
        related.closest(".openvla-phase-num-wrap") === target.closest(".openvla-phase-num-wrap")
      ) {
        /* ▲/▼ 클릭 시 blur 만으로는 커밋하지 않음 (change·step 핸들러가 처리) */
      } else {
        window.setTimeout(() => {
          const row = openvlaPhases.find((item) => item.id === target.dataset.phaseId);
          if (
            row &&
            !openvlaPhaseNumbersEqual(
              normalizeOpenvlaPhaseNumber(target.value, row.number),
              row.number
            )
          ) {
            commitOpenvlaPhaseNumInput(target);
          }
        }, 0);
      }
    }
    window.setTimeout(() => {
      if (!board.contains(document.activeElement)) {
        openvlaPhaseEditing = false;
      }
    }, 0);
  });
  board.addEventListener("input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (target.classList.contains("openvla-phase-desc")) {
      const phaseId = target.dataset.phaseId;
      const row = openvlaPhases.find((item) => item.id === phaseId);
      if (!row) return;
      row.description = target.value;
      queueSaveOpenvlaPhases();
      return;
    }
  });
  board.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;
    if (target.classList.contains("openvla-phase-num-step")) {
      event.preventDefault();
      const wrap = target.closest(".openvla-phase-num-wrap");
      const input = wrap?.querySelector(".openvla-phase-num");
      const row = openvlaPhases.find((item) => item.id === target.dataset.phaseId);
      const delta = Number(target.dataset.delta);
      if (!(input instanceof HTMLInputElement) || !row || !Number.isFinite(delta)) return;
      const base = normalizeOpenvlaPhaseNumber(input.value, row.number ?? 1);
      applyOpenvlaPhaseNumberChange(
        row.id,
        normalizeOpenvlaPhaseNumber(base + delta, base),
        input
      );
      return;
    }
    if (!target.classList.contains("openvla-phase-remove")) return;
    event.preventDefault();
    const phaseId = target.dataset.phaseId;
    if (!phaseId) return;
    openvlaPhases = openvlaPhases.filter((item) => item.id !== phaseId);
    openvlaPhaseEditing = false;
    renderOpenvlaPhaseBoard();
    queueSaveOpenvlaPhases();
  });
  board.addEventListener("change", (event) => {
    const target = event.target;
    if (target instanceof HTMLInputElement && target.classList.contains("openvla-phase-num")) {
      commitOpenvlaPhaseNumInput(target);
      return;
    }
    if (!(target instanceof HTMLSelectElement)) return;
    const phaseId = target.dataset.phaseId;
    const row = openvlaPhases.find((item) => item.id === phaseId);
    if (!row) return;
    if (target.classList.contains("openvla-phase-pose-file")) {
      row.pose_file = target.value;
      if (
        row.pose_name &&
        !openvlaPosesInFile(row.pose_file).some((pose) => pose.name === row.pose_name)
      ) {
        row.pose_name = "";
      }
      renderOpenvlaPhaseBoard();
      queueSaveOpenvlaPhases();
      return;
    }
    if (target.classList.contains("openvla-phase-pose")) {
      row.pose_file = target.dataset.poseFile || row.pose_file || "";
      row.pose_name = target.value;
      const linkedId = openvlaPoseIdForName(target.value, row.pose_file);
      if (linkedId) {
        row.id = linkedId;
      }
      queueSaveOpenvlaPhases();
    }
  });
}

document.getElementById("openvla-phase-add")?.addEventListener("click", () => {
  const phaseId = newOpenvlaPhaseId();
  openvlaPhases.push({
    id: phaseId,
    number: nextOpenvlaPhaseNumber(openvlaPhases),
    description: "",
    pose_name: "",
    pose_file: "",
  });
  sortOpenvlaPhasesByNumber();
  renderOpenvlaPhaseBoard();
  queueSaveOpenvlaPhases();
  const board = document.getElementById("openvla-phase-board");
  const input = board?.querySelector(
    `input.openvla-phase-desc[data-phase-id="${CSS.escape(phaseId)}"]`
  );
  if (input instanceof HTMLInputElement) {
    input.focus();
  }
});

async function loadOpenvlaPhaseSettings() {
  try {
    const [phasesRes, posesRes] = await Promise.all([
      api("/api/phases"),
      api("/api/poses"),
    ]);
    openvlaPoseFiles = poseFilesFromResponse(posesRes);
    applyOpenvlaHandPoseFiles(phasesRes);
    if (!openvlaPhaseEditing && !openvlaPhaseProfileRenaming) {
      applyOpenvlaPhaseDocument(phasesRes);
      renderOpenvlaPhaseProfileTabs();
      renderOpenvlaPhaseBoard();
      renderOpenvlaPhaseTags();
      if (openvlaInferenceAutoTag) {
        onOpenvlaInferenceProfileSourceChanged();
      } else {
        renderOpenvlaInferenceProfileSelect();
        if (openvlaPinned?.ready) {
          void refreshOpenvlaInferenceMetrics();
        }
      }
    }
    if (openvlaPinned) {
      renderOpenvla(openvlaPinned);
    }
  } catch (err) {
    console.warn("openvla phase settings", err);
  }
}

async function loadYoloConfig() {
  try {
    const cfg = await api("/api/yolo/config");
    yoloDefaultConfidence = Number(cfg.confidence ?? 0.7);
    yoloTargets = normalizeYoloTargets(cfg.targets, cfg.classes);
    if (!yoloTargetEditing) {
      renderYoloTargetBoard();
    }
    setYoloTagClassOptions(cfg.classes);
    if (!yoloTagConditionsSyncBlocked()) {
      applyYoloTagConditionsConfig(cfg.tag_conditions);
    }
  } catch (err) {
    console.warn("yolo config", err);
  }
}

function groupDetectionsByLabel(detections) {
  const allowed = enabledTargetLabels();
  const byLabel = new Map();
  for (const det of detections || []) {
    if (!allowed.has(det.label)) continue;
    if (!byLabel.has(det.label)) {
      byLabel.set(det.label, []);
    }
    byLabel.get(det.label).push(det);
  }
  for (const items of byLabel.values()) {
    items.sort((a, b) => b.confidence - a.confidence);
  }
  return yoloTargets
    .filter((row) => row.enabled)
    .map((row) => [row.name, byLabel.get(row.name) || []]);
}

function renderYoloDetectionGroups(detections) {
  const root = document.getElementById("yolo-list");
  if (!root) return;
  root.innerHTML = "";
  const groups = groupDetectionsByLabel(detections);
  if (!groups.length) {
    const empty = document.createElement("p");
    empty.className = "yolo-det-empty";
    empty.textContent = "탐지 타겟을 하나 이상 선택하세요";
    root.appendChild(empty);
    return;
  }
  for (const [label, items] of groups) {
    const group = document.createElement("section");
    group.className = "yolo-det-group";
    const title = document.createElement("h3");
    title.className = "yolo-det-group-title panel-group-title";
    title.textContent = label;
    const list = document.createElement("ul");
    list.className = "det-list";
    if (!items.length) {
      const li = document.createElement("li");
      li.className = "yolo-det-group-empty";
      li.textContent = "—";
      list.appendChild(li);
    } else {
      items.forEach((d, index) => {
        const li = document.createElement("li");
        li.className = "yolo-det-row";
        const summary = document.createElement("span");
        summary.className = "yolo-det-summary";
        summary.textContent = `(${index + 1}) ${(d.confidence * 100).toFixed(1)}%`;
        const coords = document.createElement("span");
        coords.className = "yolo-det-coords";
        coords.textContent = `[${d.xyxy.join(", ")}]`;
        const orient = document.createElement("span");
        orient.className = "yolo-det-orient";
        orient.textContent = yoloBoxOrientation(d.xyxy);
        li.appendChild(summary);
        li.appendChild(coords);
        li.appendChild(orient);
        list.appendChild(li);
      });
    }
    group.appendChild(title);
    group.appendChild(list);
    root.appendChild(group);
  }
}

function renderYolo(yolo) {
  if (yolo.targets?.length) {
    if (!yoloTargetEditing) {
      yoloTargets = normalizeYoloTargets(yolo.targets, yolo.classes);
      renderYoloTargetBoard();
    }
  } else if (yolo.classes?.length && !yoloTargets.length) {
    yoloTargets = normalizeYoloTargets([], yolo.classes);
    renderYoloTargetBoard();
  }

  if (yolo.classes?.length) {
    setYoloTagClassOptions(yolo.classes);
  }
  if (Array.isArray(yolo.tag_conditions)) {
    applyYoloTagConditionsConfig(yolo.tag_conditions);
  }
  renderYoloDetectionGroups(yolo.detections);
  updateYoloActiveTagsFromDetections(yolo.detections, yolo.active_tags);
}

const OPENVLA_JOINT_DIM = 22;

function setOpenvlaResultPhase(el, v, hasResult) {
  applyOpenvlaResultPhaseDisplay(el, v, getOpenvlaInferencePhaseRows(), hasResult);
}

function setOpenvlaSidePane(paneId) {
  const panes = {
    inference: document.getElementById("openvla-pane-inference"),
    phases: document.getElementById("openvla-pane-phases"),
  };
  const tabs = document.querySelectorAll(".openvla-bookmark[data-openvla-pane]");
  for (const tab of tabs) {
    const active = tab.dataset.openvlaPane === paneId;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  }
  for (const [key, el] of Object.entries(panes)) {
    if (!el) continue;
    el.hidden = key !== paneId;
  }
}

function bindOpenvlaSideBookmarks() {
  const nav = document.getElementById("openvla-bookmarks");
  if (!nav || nav.dataset.bound === "1") return;
  nav.dataset.bound = "1";
  nav.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const tab = target.closest(".openvla-bookmark[data-openvla-pane]");
    if (!tab) return;
    const paneId = tab.dataset.openvlaPane;
    if (!paneId) return;
    setOpenvlaSidePane(paneId);
  });
}

function setYoloSidePane(paneId) {
  const panes = {
    detections: document.getElementById("yolo-pane-detections"),
    targets: document.getElementById("yolo-pane-targets"),
    "tag-rules": document.getElementById("yolo-pane-tag-rules"),
  };
  const tabs = document.querySelectorAll(".openvla-bookmark[data-yolo-pane]");
  for (const tab of tabs) {
    const active = tab.dataset.yoloPane === paneId;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  }
  for (const [key, el] of Object.entries(panes)) {
    if (!el) continue;
    el.hidden = key !== paneId;
  }
}

function setLerobotSidePane(paneId) {
  const panes = {
    run: document.getElementById("lerobot-pane-run"),
    hands: document.getElementById("lerobot-pane-hands"),
    settings: document.getElementById("lerobot-pane-settings"),
  };
  const tabs = document.querySelectorAll(".openvla-bookmark[data-lerobot-pane]");
  for (const tab of tabs) {
    const active = tab.dataset.lerobotPane === paneId;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  }
  for (const [key, el] of Object.entries(panes)) {
    if (!el) continue;
    el.hidden = key !== paneId;
  }
}

function bindLerobotSideBookmarks() {
  const nav = document.getElementById("lerobot-bookmarks");
  if (!nav || nav.dataset.bound === "1") return;
  nav.dataset.bound = "1";
  nav.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const tab = target.closest(".openvla-bookmark[data-lerobot-pane]");
    if (!tab) return;
    const paneId = tab.dataset.lerobotPane;
    if (!paneId) return;
    setLerobotSidePane(paneId);
  });
}

let yoloTagRulesPaneEverOpened = false;

function bindYoloSideBookmarks() {
  const nav = document.getElementById("yolo-bookmarks");
  if (!nav || nav.dataset.bound === "1") return;
  nav.dataset.bound = "1";
  nav.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    const tab = target.closest(".openvla-bookmark[data-yolo-pane]");
    if (!tab) return;
    const paneId = tab.dataset.yoloPane;
    if (!paneId) return;
    setYoloSidePane(paneId);
    if (paneId === "tag-rules") {
      if (!yoloTagRulesPaneEverOpened) {
        yoloTagRulesPaneEverOpened = true;
        prepareYoloTagRulesPaneFirstOpen();
      }
      void loadYoloConfig();
    }
  });
}

function syncOpenvlaInferenceGate(ov) {
  const pane = document.getElementById("openvla-pane-inference");
  const panel = document.getElementById("openvla-inference-panel");
  const banner = document.getElementById("openvla-load-banner");
  const btn = document.getElementById("openvla-run");
  const instructionInput = document.getElementById("openvla-instruction");
  const loading = ov?.engine_status === "loading";
  const ready = ov?.engine_ready === true;
  openvlaEngineReady = ready;
  const locked = !ready;
  pane?.classList.toggle("openvla-inference-locked", locked);
  panel?.classList.toggle("openvla-inference-locked", locked);
  if (banner instanceof HTMLElement) {
    banner.hidden = !loading;
  }
  if (btn instanceof HTMLButtonElement) {
    btn.disabled = !ready || openvlaBusy;
  }
  if (instructionInput instanceof HTMLInputElement) {
    instructionInput.disabled = !ready || openvlaBusy;
  }
  const plannerCommand = document.getElementById("planner-task-command");
  if (plannerCommand instanceof HTMLInputElement) {
    plannerCommand.disabled = !ready || openvlaBusy;
  }
  const plannerStart = document.getElementById("planner-task-start");
  if (plannerStart instanceof HTMLButtonElement) {
    plannerStart.disabled = !ready || openvlaBusy || plannerTaskUiRunning;
  }
  const plannerStop = document.getElementById("planner-task-stop");
  if (plannerStop instanceof HTMLButtonElement) {
    plannerStop.disabled = !plannerTaskUiRunning;
  }
}

let plannerTaskUiRunning = false;

function setPlannerTaskUiRunning(on) {
  plannerTaskUiRunning = Boolean(on);
  const start = document.getElementById("planner-task-start");
  const stop = document.getElementById("planner-task-stop");
  const plannerCommand = document.getElementById("planner-task-command");
  if (start instanceof HTMLButtonElement) {
    start.textContent = plannerTaskUiRunning ? "Task 실행 중…" : "Task 시작";
  }
  if (plannerCommand instanceof HTMLInputElement) {
    plannerCommand.disabled = plannerTaskUiRunning || !openvlaEngineReady || openvlaBusy;
  }
  syncOpenvlaInferenceGate(openvlaPinned ?? lastLiveBody?.openvla ?? {});
  if (stop instanceof HTMLButtonElement) {
    stop.disabled = !plannerTaskUiRunning;
  }
}

const OPENVLA_PHASE_ERR_ID_PREFIX = "openvla-phase-err-";

let openvlaPartMoveResultKey = null;

function openvlaPartMoveCheckbox(part) {
  return document.querySelector(
    `input.openvla-part-move-cb[data-openvla-part="${part}"]`
  );
}

function openvlaMoveResultKey(v) {
  if (!v?.ready) return "";
  return `${v.frame_id ?? ""}|${JSON.stringify(v.predicted_action ?? [])}`;
}

function openvlaPhasePoseMetricCount(pose) {
  if (!pose || typeof pose !== "object") return 0;
  return OPENVLA_PHASE_ERR_PARTS.reduce((n, [, key]) => {
    const raw = pose[key];
    return n + (raw != null && Number.isFinite(Number(raw)) ? 1 : 0);
  }, 0);
}

function pickRicherOpenvlaTargetPhasePose(incoming, pinned) {
  if (!incoming) return pinned ?? null;
  if (!pinned) return incoming;
  return openvlaPhasePoseMetricCount(incoming) >= openvlaPhasePoseMetricCount(pinned)
    ? incoming
    : pinned;
}

/** Keep profile-metrics pose errors when /api/live/state returns the same infer without them. */
function mergeOpenvlaLivePinned(incoming, pinned) {
  if (!incoming?.ready) return pinned ?? incoming;
  if (!pinned?.ready) return incoming;
  const sameInference =
    incoming.frame_id === pinned.frame_id &&
    openvlaMoveResultKey(incoming) === openvlaMoveResultKey(pinned);
  if (!sameInference) return incoming;
  return {
    ...incoming,
    target_phase_pose: pickRicherOpenvlaTargetPhasePose(
      incoming.target_phase_pose,
      pinned.target_phase_pose
    ),
    predicted_phase: pinned.predicted_phase ?? incoming.predicted_phase,
  };
}

function resetOpenvlaPartMoveCheckboxes() {
  for (const [part] of OPENVLA_PHASE_ERR_PARTS) {
    const cb = openvlaPartMoveCheckbox(part);
    if (cb instanceof HTMLInputElement) {
      cb.checked = false;
      cb.disabled = true;
    }
  }
  openvlaPartMoveResultKey = null;
}

function clearOpenvlaPhasePartErrors() {
  clearOpenvlaPartErrorElements(OPENVLA_PHASE_ERR_ID_PREFIX);
  resetOpenvlaPartMoveCheckboxes();
}

function renderOpenvlaPhasePartErrors(v, match, hasResult) {
  renderOpenvlaPhasePartErrorValues(v, match, hasResult, OPENVLA_PHASE_ERR_ID_PREFIX);
  const resultKey = hasResult ? openvlaMoveResultKey(v) : "";
  const resetDefaults = Boolean(resultKey) && resultKey !== openvlaPartMoveResultKey;
  if (resetDefaults) {
    openvlaPartMoveResultKey = resultKey;
  }
  for (const [part] of OPENVLA_PHASE_ERR_PARTS) {
    const available = hasResult && openvlaPartHasModelOutput(v, part);
    const cb = openvlaPartMoveCheckbox(part);
    if (cb instanceof HTMLInputElement) {
      cb.disabled = !available;
      if (!available) {
        cb.checked = false;
      } else if (resetDefaults) {
        cb.checked = true;
      }
    }
  }
}

function collectOpenvlaMoveTargets(openvla) {
  const byPart = openvla?.inferred_joints_by_part;
  const targets = openvla?.target_joints;
  if (!byPart || !targets || typeof targets !== "object") {
    return {};
  }
  const out = {};
  for (const [part, names] of Object.entries(byPart)) {
    const cb = openvlaPartMoveCheckbox(part);
    if (!(cb instanceof HTMLInputElement) || !cb.checked || cb.disabled) {
      continue;
    }
    if (!Array.isArray(names)) continue;
    for (const name of names) {
      if (targets[name] != null) {
        out[name] = targets[name];
      }
    }
  }
  return out;
}

function renderOpenvla(v) {
  const rawEl = document.getElementById("openvla-action");
  const phaseEl = document.getElementById("openvla-result-phase");

  if (v?.engine_status === "loading") {
    if (rawEl) rawEl.textContent = "";
    if (phaseEl) phaseEl.textContent = "—";
    clearOpenvlaPhasePartErrors();
    return;
  }

  if ((v.error || v.engine_error) && !v.ready) {
    const errText = v.engine_error || v.error;
    if (phaseEl) {
      phaseEl.classList.remove("openvla-phase-error");
      phaseEl.textContent = errText;
    }
    if (rawEl) rawEl.textContent = "";
    clearOpenvlaPhasePartErrors();
    return;
  }

  const hasResult = openvlaHasResultPhase(v);
  if (rawEl) {
    rawEl.textContent = hasResult
      ? JSON.stringify(v.predicted_action ?? [], null, 2)
      : "추론을 실행하면 raw action이 표시됩니다.";
  }
  setOpenvlaResultPhase(phaseEl, v, hasResult);
  renderOpenvlaPhasePartErrors(v, v.target_phase_pose, hasResult);
  syncOpenvlaMotionControls(v);
}

function setOpenvlaMotionStatus(text, kind) {
  const el = document.getElementById("openvla-motion-status");
  if (!(el instanceof HTMLElement)) return;
  if (!text) {
    el.hidden = true;
    el.textContent = "";
    el.classList.remove("is-ok", "is-err");
    return;
  }
  el.hidden = false;
  el.textContent = text;
  el.classList.toggle("is-ok", kind === "ok");
  el.classList.toggle("is-err", kind === "err");
}

function parseOpenvlaMotionSpeed() {
  const input = document.getElementById("openvla-motion-speed");
  const raw = input instanceof HTMLInputElement ? Number(input.value) : 0.05;
  if (!Number.isFinite(raw) || raw < 0.05) {
    return null;
  }
  return raw;
}

function syncOpenvlaMotionControls(v) {
  const moveBtn = document.getElementById("openvla-motion-move");
  const stopBtn = document.getElementById("openvla-motion-stop");
  const state = v ?? openvlaPinned ?? lastLiveBody?.openvla;
  const targets = collectOpenvlaMoveTargets(state);
  const hasTargets =
    state?.ready && targets && typeof targets === "object" && Object.keys(targets).length > 0;
  if (moveBtn instanceof HTMLButtonElement) {
    moveBtn.disabled = !hasTargets || openvlaMotionBusy || openvlaBusy;
  }
  if (stopBtn instanceof HTMLButtonElement) {
    stopBtn.disabled = !openvlaMotionBusy;
  }
}

async function openvlaMoveToResultPose() {
  const openvla = openvlaPinned ?? lastLiveBody?.openvla;
  const targets = collectOpenvlaMoveTargets(openvla);
  if (!targets || !Object.keys(targets).length) {
    setOpenvlaMotionStatus(
      "이동할 부위를 선택하세요. (모델이 출력한 부위만 체크 가능)",
      "err"
    );
    return;
  }
  const speed = parseOpenvlaMotionSpeed();
  if (speed == null) {
    setOpenvlaMotionStatus("속도는 0.05 rad/s 이상이어야 합니다.", "err");
    return;
  }
  const epoch = ++openvlaMotionEpoch;
  openvlaMotionBusy = true;
  setOpenvlaMotionStatus("", null);
  syncOpenvlaMotionControls(openvla);
  try {
    const res = await apiPost("/api/motion/move", {
      target_joints: targets,
      speed,
    });
    if (epoch !== openvlaMotionEpoch) return;
    if (!res.ok) {
      setOpenvlaMotionStatus(res.message || "이동에 실패했습니다.", "err");
    }
  } catch (err) {
    if (epoch !== openvlaMotionEpoch) return;
    setOpenvlaMotionStatus(String(err.message || err), "err");
  } finally {
    if (epoch === openvlaMotionEpoch) {
      openvlaMotionBusy = false;
      syncOpenvlaMotionControls(openvla);
    }
  }
}

async function openvlaStopMotion() {
  if (!openvlaMotionBusy) return;
  const epoch = openvlaMotionEpoch;
  const speed = parseOpenvlaMotionSpeed() ?? 0.05;
  let stopped = false;
  try {
    const res = await apiPost("/api/motion/stop", { speed });
    if (epoch !== openvlaMotionEpoch) return;
    stopped = Boolean(res.ok);
    if (stopped) {
      setOpenvlaMotionStatus("", null);
    } else {
      setOpenvlaMotionStatus(res.message || "중지에 실패했습니다.", "err");
    }
  } catch (err) {
    if (epoch !== openvlaMotionEpoch) return;
    setOpenvlaMotionStatus(String(err.message || err), "err");
  } finally {
    if (stopped && epoch === openvlaMotionEpoch) {
      openvlaMotionBusy = false;
      syncOpenvlaMotionControls(openvlaPinned ?? lastLiveBody?.openvla);
    }
  }
}

function bindOpenvlaMotionControls() {
  document
    .getElementById("openvla-motion-move")
    ?.addEventListener("click", openvlaMoveToResultPose);
  document
    .getElementById("openvla-motion-stop")
    ?.addEventListener("click", openvlaStopMotion);
  document
    .querySelector(".openvla-part-error-grid")
    ?.addEventListener("change", (ev) => {
      if (!(ev.target instanceof HTMLInputElement)) return;
      if (!ev.target.classList.contains("openvla-part-move-cb")) return;
      syncOpenvlaMotionControls(openvlaPinned ?? lastLiveBody?.openvla);
    });
}

function renderLerobot(_l) {
  /* Action chunk panel removed from LeRobot tab UI */
}

function renderDevProfile(profile) {
  const root = document.getElementById("dev-profile");
  if (!root) return;
  root.innerHTML = "";
  const sections = [
    {
      title: "실행 환경",
      rows: [
        ["Mode", profile.mode],
        ["Config", profile.config_file],
        ["Camera", profile.camera_source],
      ],
    },
    {
      title: "추론 장치",
      rows: [
        ["OpenVLA GPU", profile.openvla_device],
        ["YOLO GPU", profile.yolo_device],
        [
          "LeRobot GPU",
          profile.lerobot_device_note
            ? `${profile.lerobot_device ?? "—"} — ${profile.lerobot_device_note}`
            : profile.lerobot_device,
        ],
      ],
    },
    {
      title: "OpenVLA",
      rows: [
        ["사용 버전", profile.openvla_version],
        [
          "로드된 버전",
          profile.openvla_loaded_version
            ? profile.openvla_loaded_version
            : profile.mode === "mock"
              ? "(mock)"
              : "아직 로드 안 됨",
        ],
        ["체크포인트", profile.openvla_model_dir || "—"],
        ["모델 루트", profile.openvla_root || "—"],
      ],
    },
    {
      title: "모델 경로",
      rows: [
        ["YOLO", profile.yolo_model],
        ["LeRobot", profile.lerobot],
      ],
    },
  ];
  for (const section of sections) {
    const block = document.createElement("section");
    block.className = "config-profile-section";
    const heading = document.createElement("h2");
    heading.className = "panel-group-title";
    if (root.childElementCount > 0) {
      heading.classList.add("panel-group-title--spaced");
    }
    heading.textContent = section.title;
    const dl = document.createElement("dl");
    dl.className = "profile-dl";
    for (const [key, val] of section.rows) {
      const dt = document.createElement("dt");
      dt.textContent = key;
      const dd = document.createElement("dd");
      dd.textContent = val ?? "—";
      dl.appendChild(dt);
      dl.appendChild(dd);
    }
    block.appendChild(heading);
    block.appendChild(dl);
    root.appendChild(block);
  }
}

function renderState(body) {
  lastLiveBody = body;
  frameBadge.textContent = `frame ${body.frame_id ?? "—"}`;
  renderYolo(body.yolo);
  if (openvlaInferenceAutoTag) {
    applyOpenvlaAutoProfileFromYoloIfNeeded();
  }
  const openvlaLive = body.openvla ?? {};
  if (!openvlaBusy) {
    syncOpenvlaInferenceGate(openvlaLive);
  }
  if (body.openvla?.ready && !openvlaBusy) {
    openvlaPinned = mergeOpenvlaLivePinned(body.openvla, openvlaPinned);
  }
  if (!openvlaBusy) {
    renderOpenvla(openvlaPinned ?? body.openvla);
  }
  renderLerobot(body.lerobot);
  renderPlannerPanel(plannerPanelPayload(openvlaPinned ?? body.openvla, body));
}

let devProfileCache = null;

async function refreshDevProfile() {
  try {
    devProfileCache = await api("/api/dev/profile");
    renderDevProfile(devProfileCache);
  } catch (err) {
    console.warn("dev profile failed", err);
  }
}

async function pollLive() {
  try {
    const body = await api("/api/live/state");
    if (typeof body.studio_raw_preview === "boolean") {
      const prev = studioRawPreview;
      studioRawPreview = body.studio_raw_preview;
      if (prev !== studioRawPreview) {
        refreshStreams();
      }
    }
    renderState(body);
    if (activeTab === "dev" && !devProfileCache) {
      await refreshDevProfile();
    }
  } catch (err) {
    console.warn("live poll failed", err);
  }
}

let openvlaInstructionEntries = [];
let linkedInstructionSync = false;

function openvlaInstructionInputs() {
  const openvla = document.getElementById("openvla-instruction");
  const planner = document.getElementById("planner-task-command");
  return {
    openvla: openvla instanceof HTMLInputElement ? openvla : null,
    planner: planner instanceof HTMLInputElement ? planner : null,
  };
}

function syncLinkedInstructionFields(source) {
  if (linkedInstructionSync) return;
  const { openvla, planner } = openvlaInstructionInputs();
  if (!openvla && !planner) return;
  linkedInstructionSync = true;
  try {
    const value =
      source === "planner"
        ? planner?.value ?? ""
        : openvla?.value ?? planner?.value ?? "";
    if (openvla && source !== "openvla") openvla.value = value;
    if (planner && source !== "planner") planner.value = value;
  } finally {
    linkedInstructionSync = false;
  }
}

function bindLinkedInstructionSync() {
  const { openvla, planner } = openvlaInstructionInputs();
  if (!openvla || !planner) return;
  if (planner.dataset.instructionLinked === "1") return;
  planner.dataset.instructionLinked = "1";
  syncLinkedInstructionFields("openvla");
  openvla.addEventListener("input", () => syncLinkedInstructionFields("openvla"));
  planner.addEventListener("input", () => syncLinkedInstructionFields("planner"));
}

function openvlaInstructionValue() {
  const { openvla, planner } = openvlaInstructionInputs();
  const el = openvla ?? planner;
  if (!el) return "";
  return el.value.trim();
}

function openvlaInstructionFilter() {
  const query = openvlaInstructionValue().toLowerCase();
  if (!query) return openvlaInstructionEntries;
  return openvlaInstructionEntries.filter((row) =>
    row.text.toLowerCase().includes(query)
  );
}

function setOpenvlaInstructionDropdownOpen(open) {
  const input = document.getElementById("openvla-instruction");
  const list = document.getElementById("openvla-instruction-list");
  if (!(input instanceof HTMLInputElement) || !(list instanceof HTMLElement)) return;
  list.hidden = !open;
  input.setAttribute("aria-expanded", open ? "true" : "false");
}

function renderOpenvlaInstructionDropdown() {
  const list = document.getElementById("openvla-instruction-list");
  if (!(list instanceof HTMLElement)) return;
  const filtered = openvlaInstructionFilter();
  list.innerHTML = "";
  if (!filtered.length) {
    const empty = document.createElement("li");
    empty.className = "openvla-instruction-dropdown-empty";
    empty.textContent = openvlaInstructionEntries.length
      ? "일치하는 이력이 없습니다."
      : "저장된 작업 지시가 없습니다.";
    list.appendChild(empty);
    return;
  }
  for (const row of filtered) {
    const item = document.createElement("li");
    item.className = "openvla-instruction-option";
    item.setAttribute("role", "option");
    item.dataset.entryId = row.id;
    const text = document.createElement("span");
    text.className = "openvla-instruction-option-text";
    text.textContent = row.text;
    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "openvla-instruction-option-remove";
    removeBtn.setAttribute("aria-label", "이력 삭제");
    removeBtn.title = "삭제";
    removeBtn.textContent = "×";
    removeBtn.addEventListener("mousedown", (event) => event.preventDefault());
    removeBtn.addEventListener("click", async (event) => {
      event.stopPropagation();
      try {
        const updated = await api(
          `/api/openvla/instruction-history/${encodeURIComponent(row.id)}`,
          { method: "DELETE" }
        );
        openvlaInstructionEntries = updated.entries || [];
        renderOpenvlaInstructionDropdown();
      } catch (err) {
        console.warn("instruction delete", err);
      }
    });
    item.addEventListener("mousedown", (event) => event.preventDefault());
    item.addEventListener("click", () => {
      const input = document.getElementById("openvla-instruction");
      if (input instanceof HTMLInputElement) {
        input.value = row.text;
        input.dispatchEvent(new Event("input", { bubbles: true }));
      }
      setOpenvlaInstructionDropdownOpen(false);
    });
    item.appendChild(text);
    item.appendChild(removeBtn);
    list.appendChild(item);
  }
}

async function loadOpenvlaInstructionHistory() {
  const input = document.getElementById("openvla-instruction");
  if (!(input instanceof HTMLInputElement)) return;
  try {
    const data = await api("/api/openvla/instruction-history");
    openvlaInstructionEntries = data.entries || [];
    if (!input.value.trim() && data.last_instruction) {
      input.value = data.last_instruction;
    }
    syncLinkedInstructionFields("openvla");
    renderOpenvlaInstructionDropdown();
  } catch (err) {
    console.warn("instruction history", err);
  }
}

function bindOpenvlaInstructionCombo() {
  const input = document.getElementById("openvla-instruction");
  const list = document.getElementById("openvla-instruction-list");
  if (!(input instanceof HTMLInputElement) || !(list instanceof HTMLElement)) return;
  input.addEventListener("focus", () => {
    renderOpenvlaInstructionDropdown();
    setOpenvlaInstructionDropdownOpen(true);
  });
  input.addEventListener("input", () => {
    renderOpenvlaInstructionDropdown();
    setOpenvlaInstructionDropdownOpen(true);
  });
  input.addEventListener("blur", () => {
    window.setTimeout(() => setOpenvlaInstructionDropdownOpen(false), 140);
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.activeElement === input) {
      setOpenvlaInstructionDropdownOpen(false);
    }
  });
}

function plannerLerobotRuntimeSnapshot() {
  return {
    running: lerobotPolicyState.running,
    runningPolicy: lerobotPolicyState.runningPolicy,
    pid: lerobotPolicyState.pid,
  };
}

function plannerPanelPayload(openvla, live = lastLiveBody) {
  const base = live || { planner: {} };
  return {
    ...base,
    openvla,
    openvlaPhaseRows: getOpenvlaInferencePhaseRows(),
    openvlaPhaseProfile: getOpenvlaInferenceProfileName(),
    lerobotRuntime: plannerLerobotRuntimeSnapshot(),
  };
}

function touchPlannerPanel() {
  if (activeTab !== "planner") return;
  renderPlannerPanel(plannerPanelPayload(openvlaPinned ?? lastLiveBody?.openvla));
}

async function runOpenvlaInferCore() {
  const instruction = openvlaInstructionValue();
  if (!instruction) {
    throw new Error("명령어가 비어 있습니다");
  }
  if (!openvlaEngineReady) {
    throw new Error("OpenVLA 엔진이 준비되지 않았습니다");
  }
  if (openvlaBusy) {
    throw new Error("OpenVLA 추론이 이미 진행 중입니다");
  }
  openvlaBusy = true;
  try {
    ensureOpenvlaInferenceProfile();
    const result = await apiPost("/api/openvla/infer", {
      instruction,
      phase_profile: openvlaInferenceProfile || undefined,
    });
    await loadOpenvlaInstructionHistory();
    openvlaPinned = result.openvla;
    renderOpenvla(result.openvla);
    await refreshOpenvlaInferenceMetrics();
    return result.openvla;
  } catch (err) {
    openvlaPinned = {
      ready: false,
      status: "error",
      error: String(err.message || err),
      target_joints: {},
      predicted_action: [],
    };
    renderOpenvla(openvlaPinned);
    return openvlaPinned;
  } finally {
    openvlaBusy = false;
    if (!plannerTaskUiRunning) {
      syncOpenvlaInferenceGate(openvlaPinned ?? lastLiveBody?.openvla ?? {});
    }
    syncYoloInferenceForTab(activeTab);
  }
}

async function runOpenvlaInference() {
  const btn = document.getElementById("openvla-run");
  const instructionInput = document.getElementById("openvla-instruction");
  if (!openvlaEngineReady || openvlaBusy) return;
  if (!btn || btn.disabled) return;
  const instruction = openvlaInstructionValue();
  if (!instruction) {
    if (instructionInput instanceof HTMLInputElement) {
      instructionInput.focus();
    }
    return;
  }
  if (btn instanceof HTMLButtonElement) {
    btn.disabled = true;
    btn.textContent = "추론 중…";
  }
  if (instructionInput instanceof HTMLInputElement) {
    instructionInput.disabled = true;
  }
  const rawEl = document.getElementById("openvla-action");
  if (rawEl) rawEl.textContent = "…";
  const openvla = await runOpenvlaInferCore();
  renderPlannerPanel(plannerPanelPayload(openvla));
  if (btn instanceof HTMLButtonElement) {
    btn.textContent = "추론";
  }
  syncOpenvlaInferenceGate(openvla ?? {});
}

document.getElementById("openvla-run")?.addEventListener("click", runOpenvlaInference);

document.getElementById("planner-task-start")?.addEventListener("click", () => {
  if (!openvlaEngineReady || plannerTaskUiRunning) return;
  const instruction = openvlaInstructionValue();
  if (!instruction) {
    const plannerCommand = document.getElementById("planner-task-command");
    if (plannerCommand instanceof HTMLInputElement) {
      plannerCommand.focus();
    }
    return;
  }
  startPlannerTask();
});

document.getElementById("planner-task-stop")?.addEventListener("click", () => {
  stopPlannerTask("사용자 Stop — Task 중단");
});

document.getElementById("openvla-instruction")?.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    runOpenvlaInference();
  }
});

const PANEL_COL_SPLIT_CSS_VAR = "--panel-col-side-width";
const PANEL_COL_SPLIT_MIN_RIGHT = 220;
const PANEL_COL_SPLIT_MIN_LEFT = 200;
/** 영상(왼쪽) : 기능 패널(오른쪽) = 2 : 1 */
const PANEL_COL_SPLIT_RIGHT_RATIO = 1 / 3;
const panelColSplitRefreshById = new Map();

function initPanelColSplitResize({ panelId, handleId, storageKey }) {
  const panel = document.getElementById(panelId);
  const handle = document.getElementById(handleId);
  if (!panel || !handle) return;

  const clearFixedLayout = () => {
    panel.classList.remove("panel-col-split--fixed");
    panel.style.removeProperty(PANEL_COL_SPLIT_CSS_VAR);
  };

  const measureUsableWidth = () => {
    const rect = panel.getBoundingClientRect();
    const handleW = handle.offsetWidth || 10;
    const margin = 4;
    return Math.max(0, rect.width - handleW - margin);
  };

  const defaultRightWidth = () =>
    Math.round(measureUsableWidth() * PANEL_COL_SPLIT_RIGHT_RATIO);

  const applyRatioLayout = () => {
    const layout = () => {
      const usable = measureUsableWidth();
      if (usable < PANEL_COL_SPLIT_MIN_LEFT + PANEL_COL_SPLIT_MIN_RIGHT) {
        clearFixedLayout();
        return false;
      }
      clearFixedLayout();
      return true;
    };
    if (!layout()) {
      requestAnimationFrame(() => layout());
    }
  };

  const applySavedOrRatio = () => {
    if (!panel.classList.contains("active")) {
      return;
    }
    if (panel.getBoundingClientRect().width <= 0) {
      requestAnimationFrame(() => applySavedOrRatio());
      return;
    }
    const saved = Number(localStorage.getItem(storageKey));
    const target = defaultRightWidth();
    const minAcceptable = Math.max(
      PANEL_COL_SPLIT_MIN_RIGHT,
      Math.round(target * 0.82)
    );
    if (Number.isFinite(saved) && saved >= minAcceptable) {
      setRightWidth(saved, false);
    } else {
      if (Number.isFinite(saved) && saved < minAcceptable) {
        localStorage.removeItem(storageKey);
      }
      applyRatioLayout();
    }
  };

  const clampRightWidth = (widthPx) => {
    const rect = panel.getBoundingClientRect();
    const handleW = handle.offsetWidth || 10;
    const maxRight = Math.min(
      rect.width * 0.72,
      rect.width - PANEL_COL_SPLIT_MIN_LEFT - handleW - 8
    );
    return Math.max(PANEL_COL_SPLIT_MIN_RIGHT, Math.min(widthPx, maxRight));
  };

  const setRightWidth = (widthPx, persist) => {
    const clamped = clampRightWidth(widthPx);
    panel.classList.add("panel-col-split--fixed");
    panel.style.setProperty(PANEL_COL_SPLIT_CSS_VAR, `${clamped}px`);
    if (persist) {
      localStorage.setItem(storageKey, String(Math.round(clamped)));
    }
  };

  applySavedOrRatio();

  const currentRightWidth = () => {
    const right = panel.querySelector(".panel-col-split-pane-right");
    return right ? right.getBoundingClientRect().width : NaN;
  };

  let dragging = false;

  const onPointerMove = (event) => {
    if (!dragging) return;
    const rect = panel.getBoundingClientRect();
    setRightWidth(rect.right - event.clientX, false);
  };

  const stopDrag = () => {
    if (!dragging) return;
    dragging = false;
    document.body.classList.remove("panel-col-split-dragging");
    handle.releasePointerCapture?.(handle._activePointerId);
    handle._activePointerId = undefined;
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", stopDrag);
    window.removeEventListener("pointercancel", stopDrag);
    const current = parseFloat(
      getComputedStyle(panel).getPropertyValue(PANEL_COL_SPLIT_CSS_VAR)
    );
    if (Number.isFinite(current)) {
      localStorage.setItem(storageKey, String(Math.round(current)));
    }
  };

  handle.addEventListener("pointerdown", (event) => {
    if (window.matchMedia("(max-width: 900px)").matches) return;
    if (event.button !== 0) return;
    dragging = true;
    handle._activePointerId = event.pointerId;
    handle.setPointerCapture?.(event.pointerId);
    document.body.classList.add("panel-col-split-dragging");
    event.preventDefault();
    const fromVar = parseFloat(
      getComputedStyle(panel).getPropertyValue(PANEL_COL_SPLIT_CSS_VAR)
    );
    const measured = currentRightWidth();
    const seed =
      Number.isFinite(fromVar) && fromVar > 0
        ? fromVar
        : Number.isFinite(measured) && measured > 0
          ? measured
          : defaultRightWidth();
    setRightWidth(seed, false);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", stopDrag);
    window.addEventListener("pointercancel", stopDrag);
  });

  handle.addEventListener("keydown", (event) => {
    if (window.matchMedia("(max-width: 900px)").matches) return;
    const step = event.shiftKey ? 40 : 16;
    const fromVar = parseFloat(
      getComputedStyle(panel).getPropertyValue(PANEL_COL_SPLIT_CSS_VAR)
    );
    const base = Number.isFinite(fromVar) ? fromVar : currentRightWidth();
    if (event.key === "ArrowLeft") {
      setRightWidth(base + step, true);
      event.preventDefault();
    } else if (event.key === "ArrowRight") {
      setRightWidth(base - step, true);
      event.preventDefault();
    }
  });

  const onWindowResize = () => {
    if (!panel.classList.contains("active")) {
      return;
    }
    if (!localStorage.getItem(storageKey)) {
      applyRatioLayout();
      return;
    }
    const current = parseFloat(
      getComputedStyle(panel).getPropertyValue(PANEL_COL_SPLIT_CSS_VAR)
    );
    if (Number.isFinite(current)) {
      setRightWidth(current, true);
    }
  };
  window.addEventListener("resize", onWindowResize);

  panelColSplitRefreshById.set(panelId, applySavedOrRatio);
}

function refreshPanelColSplitLayout(tab) {
  const fn = panelColSplitRefreshById.get(`panel-${tab}`);
  if (fn) {
    requestAnimationFrame(() => fn());
  }
}

function initPanelColSplitResizes() {
  initPanelColSplitResize({
    panelId: "panel-yolo",
    handleId: "yolo-split-handle",
    storageKey: "oap-yolo-side-width-v3",
  });
  initPanelColSplitResize({
    panelId: "panel-openvla",
    handleId: "openvla-split-handle",
    storageKey: "oap-openvla-side-width-v3",
  });
  initPanelColSplitResize({
    panelId: "panel-lerobot",
    handleId: "lerobot-split-handle",
    storageKey: "oap-lerobot-side-width-v3",
  });
  initPanelColSplitResize({
    panelId: "panel-planner",
    handleId: "planner-split-handle",
    storageKey: "oap-planner-side-width-v3",
  });
}

async function bootstrap() {
  try {
    const health = await api("/api/health");
    studioRawPreview = !!health.studio_raw_preview;
    healthDot.classList.toggle("ok", health.ok);
    healthDot.classList.toggle("warn", !health.ok);
    healthText.textContent = health.mode || (health.mock ? "mock" : "live");

    const cams = await api("/api/cameras");
    cameraId = cams.default_camera_id || cams.cameras[0]?.id || null;
    refreshStreams();
    bindYoloVideoFrameCorners();
    bindYoloTargetBoard();
    initYoloTagConditions(api);
    bindYoloSideBookmarks();
    bindLerobotSideBookmarks();
    bindOpenvlaSideBookmarks();
    loadOpenvlaInferenceAutoTagPref();
    bindOpenvlaInferenceAutoTag();
    bindOpenvlaInferenceProfileSelect();
    bindOpenvlaPhaseProfileTabs();
    bindOpenvlaPhaseBoard();
    bindOpenvlaPhaseTags();
    bindOpenvlaPhaseHandSelect();
    initPanelColSplitResizes();
    bindOpenvlaInstructionCombo();
    bindLinkedInstructionSync();
    bindOpenvlaMotionControls();
    loadYoloConfig();
    loadOpenvlaPhaseSettings();
    loadOpenvlaInstructionHistory();
    pollLive();
    pollTimer = setInterval(pollLive, 250);
  } catch (err) {
    healthDot.classList.add("warn");
    healthText.textContent = "offline";
    console.error(err);
  }
}

syncTabFromUrl();
setPlannerPaneChangeListener((paneId) => {
  if (paneId === "settings") void onPlannerSettingsPaneOpen();
});
initPlannerPanel();
initPlannerTaskSettings();
initPlannerTaskLog();
void initPlannerTaskRulesCache();
setPlannerTaskRunnerDeps({
  api,
  apiPost,
  openvlaInstructionValue,
  getOpenvlaInferenceProfileName,
  getOpenvlaInferencePhaseRows,
  runOpenvlaInferCore,
  renderPlannerAfterOpenvla: (openvla) => {
    renderPlannerPanel(plannerPanelPayload(openvla));
  },
  renderPlannerPanel: () => {
    renderPlannerPanel(plannerPanelPayload(openvlaPinned ?? lastLiveBody?.openvla));
  },
  lerobotPolicyPlay,
  lerobotPolicyStop,
  lerobotPolicyLoad,
  setTaskUiRunning: setPlannerTaskUiRunning,
});
initLerobotPolicyPanel();
bootstrap();


/* ── LeRobot 정책 실행 (bwr-lerobot, Mission Studio policy_run 로컬 버전) ── */
const LEROBOT_RUN_SETTINGS_KEY = "oap-lerobot-run-settings-v2";
/** GDK arm_lead_rad per 100Hz tick (see agibot_g2._follow_one). */
const LEROBOT_ARM_LEAD_MIN = 0.002;
const LEROBOT_ARM_LEAD_MAX = 0.05;
const LEROBOT_ARM_LEAD_DEFAULT = 0.012;
const LEROBOT_ARM_LEAD_WARN = 0.04;
const LEROBOT_ARM_LEAD_STEP = 0.001;

let lerobotLastAcceptedArmLead = LEROBOT_ARM_LEAD_DEFAULT;

function lerobotSyncMotionSpeedFields(form, speed) {
  if (!form) return;
  const v = String(lerobotClampArmLead(speed));
  lerobotSyncingSpeedFields = true;
  const slider = form.elements.namedItem("motion_speed");
  const precise = form.elements.namedItem("motion_speed_precise");
  if (slider instanceof HTMLInputElement) slider.value = v;
  if (precise instanceof HTMLInputElement) precise.value = v;
  lerobotSyncingSpeedFields = false;
}

function lerobotClampArmLead(speed) {
  const s = Number(speed);
  if (!Number.isFinite(s)) return LEROBOT_ARM_LEAD_DEFAULT;
  const clamped = Math.min(
    LEROBOT_ARM_LEAD_MAX,
    Math.max(LEROBOT_ARM_LEAD_MIN, s),
  );
  return (
    Math.round(clamped / LEROBOT_ARM_LEAD_STEP) * LEROBOT_ARM_LEAD_STEP
  );
}

/** Pre-rad-era UI stored 0.25–1.0 multipliers; clamping 1.0 → 0.5 looked like “default”. */
function lerobotIsLegacyMotionSpeed(speed) {
  const s = Number(speed);
  if (!Number.isFinite(s)) return true;
  if (s > LEROBOT_ARM_LEAD_MAX + 1e-9) return true;
  if (s > 0.5 - 1e-9 && s <= 1.0 + 1e-9) return true;
  return false;
}

function lerobotResolveMotionSpeed(cliSpeed, runtimeSpeed) {
  const pick =
    cliSpeed != null && cliSpeed !== ""
      ? cliSpeed
      : runtimeSpeed != null && runtimeSpeed !== ""
        ? runtimeSpeed
        : null;
  if (pick == null || lerobotIsLegacyMotionSpeed(pick)) {
    return LEROBOT_ARM_LEAD_DEFAULT;
  }
  return lerobotClampArmLead(pick);
}

function lerobotConfirmHighArmLead(value) {
  if (value <= LEROBOT_ARM_LEAD_WARN + 1e-9) return true;
  return window.confirm(
    `동작 속도가 ${LEROBOT_ARM_LEAD_WARN} rad/tick을 초과합니다.\n` +
      "과도한 속도는 물체·관절에 손상을 줄 수 있습니다.\n\n" +
      `${value.toFixed(2)} rad/tick으로 계속하시겠습니까?`,
  );
}

function lerobotRoundSetting(value, decimals = 2) {
  if (!Number.isFinite(value)) return value;
  const scale = 10 ** decimals;
  return Math.round(value * scale) / scale;
}

function lerobotBasePublishHz(defaults) {
  const base = defaults || lerobotPolicyState.runDefaults;
  const hz = Number(base?.runtime?.publish_hz);
  return Number.isFinite(hz) && hz > 0 ? hz : 2.5;
}

function lerobotSpeedFromPublish(publishHz, defaults) {
  return LEROBOT_ARM_LEAD_DEFAULT;
}

/** Speed slider → GDK motion only (publish_hz / inference_hz stay at template). */
function lerobotComputeOptimizedRuntime(speed, defaults) {
  const base = defaults || lerobotPolicyState.runDefaults;
  const rt0 = base?.runtime || {};
  const arm0 = rt0.arm_action_postprocess || {};
  const clamped = lerobotClampArmLead(speed);
  const basePublish = lerobotBasePublishHz(base);
  const baseInference = Number(rt0.inference_hz);
  const inferRatio =
    Number.isFinite(baseInference) && basePublish > 0
      ? baseInference / basePublish
      : 0.4;

  const publish_hz = lerobotRoundSetting(basePublish);
  const inference_hz = lerobotRoundSetting(
    Number.isFinite(baseInference) && baseInference > 0
      ? baseInference
      : publish_hz * inferRatio,
  );
  const warming_up_time = lerobotRoundSetting(
    Number(rt0.warming_up_time ?? 0.3),
    2,
  );

  const cutoffScale = Math.sqrt(clamped);
  const baseCutoff = Number(arm0.min_cutoff ?? 0.4);
  const baseDeriv = Number(arm0.derivative_cutoff ?? baseCutoff);
  const min_cutoff = lerobotRoundSetting(baseCutoff * cutoffScale);
  const derivative_cutoff = lerobotRoundSetting(baseDeriv * cutoffScale);
  const beta = lerobotRoundSetting(Number(arm0.beta ?? 0.01), 3);

  const baseImageAge = Number(rt0.max_image_age_s ?? 1.0);
  const max_image_age_s = lerobotRoundSetting(
    Math.max(0.05, baseImageAge / Math.sqrt(clamped)),
    2,
  );

  return {
    motion_speed: clamped,
    publish_hz,
    inference_hz,
    warming_up_time,
    max_image_age_s,
    arm_action_postprocess: {
      enabled: arm0.enabled !== false,
      min_cutoff,
      derivative_cutoff,
      beta,
    },
  };
}

let lerobotSyncingSpeedFields = false;

const lerobotPolicyState = {
  policies: [],
  running: false,
  runningPolicy: "",
  pid: "",
  busy: false,
  pollTimer: null,
  runDefaults: null,
  runSettings: null,
  handPoseFiles: [],
  handMappings: {},
};

function lerobotMergeRunDefaults(defaults, saved) {
  const base = defaults || {};
  const baseRt = base.runtime || {};
  const savedRt = saved?.runtime || {};
  const rt = { ...baseRt, ...savedRt };
  rt.publish_consume = {
    ...(baseRt.publish_consume || {}),
    ...(savedRt.publish_consume || {}),
  };
  const arm = {
    ...(baseRt.arm_action_postprocess || {}),
    ...(savedRt.arm_action_postprocess || {}),
  };
  rt.arm_action_postprocess = arm;
  return {
    runtime: rt,
    action: { ...(base.action || {}), ...(saved?.action || {}) },
    cli: { ...(base.cli || {}), ...(saved?.cli || {}) },
  };
}

function lerobotFlatSettings(merged) {
  const rt = merged.runtime || {};
  const arm = rt.arm_action_postprocess || {};
  const cli = merged.cli || {};
  const motionSpeed = lerobotResolveMotionSpeed(
    cli.motion_speed,
    rt.motion_speed,
  );
  return {
    motion_speed: motionSpeed,
    publish_hz: rt.publish_hz,
    inference_hz: rt.inference_hz,
    warming_up_time: rt.warming_up_time,
    min_run_time_before_complete: rt.min_run_time_before_complete,
    max_image_age_s: rt.max_image_age_s,
    gripper_threshold: merged.action?.gripper_threshold,
    arm_pp_enabled: arm.enabled,
    arm_min_cutoff: arm.min_cutoff,
    arm_derivative_cutoff: arm.derivative_cutoff,
    arm_beta: arm.beta,
    timeout: cli.timeout,
    skip_warmup: Boolean(cli.skip_warmup),
    waypoint_approach_rad: rt.publish_consume?.waypoint_approach_rad,
  };
}

function lerobotUpdateSpeedSummary(flat) {
  const out = lerobotPolicyEl("lerobot-speed-summary");
  if (!out || !flat) return;
  const lead = lerobotClampArmLead(flat.motion_speed);
  out.textContent = `${lead.toFixed(3)} rad/tick (팔)`;
}

function lerobotApplyOptimizedFieldsToForm(optimized, { includeImageAge = true } = {}) {
  const form = lerobotPolicyEl("lerobot-run-settings");
  if (!form || !optimized) return;
  lerobotSyncingSpeedFields = true;
  try {
    const setNum = (name, value) => {
      const el = form.elements.namedItem(name);
      if (el instanceof HTMLInputElement && value != null && value !== "") {
        el.value = String(value);
      }
    };
    if (optimized.motion_speed != null) {
      lerobotSyncMotionSpeedFields(form, optimized.motion_speed);
    }
    setNum("publish_hz", optimized.publish_hz);
    setNum("inference_hz", optimized.inference_hz);
    setNum("warming_up_time", optimized.warming_up_time);
    if (includeImageAge && optimized.max_image_age_s != null) {
      setNum("max_image_age_s", optimized.max_image_age_s);
    }
    const arm = optimized.arm_action_postprocess;
    if (arm) {
      const pp = form.elements.namedItem("arm_pp_enabled");
      if (pp instanceof HTMLInputElement && pp.type === "checkbox") {
        pp.checked = Boolean(arm.enabled);
      }
      setNum("arm_min_cutoff", arm.min_cutoff);
      setNum("arm_derivative_cutoff", arm.derivative_cutoff);
      setNum("arm_beta", arm.beta);
    }
  } finally {
    lerobotSyncingSpeedFields = false;
  }
}

/** Update GDK motion_speed only — do not rewrite publish_hz / arm_pp from the slider. */
function lerobotCommitMotionSpeed(speed) {
  const clamped = lerobotClampArmLead(speed);
  lerobotLastAcceptedArmLead = clamped;
  const form = lerobotPolicyEl("lerobot-run-settings");
  if (form) {
    lerobotSyncMotionSpeedFields(form, clamped);
  }
  lerobotReadSettingsFromForm();
  lerobotUpdateSpeedSummary(lerobotFlatSettings(lerobotPolicyState.runSettings));
}

function lerobotOptimizeForSpeed(speed, { includeImageAge = true } = {}) {
  const clamped = lerobotClampArmLead(speed);
  const optimized = lerobotComputeOptimizedRuntime(clamped, lerobotPolicyState.runDefaults);
  lerobotApplyOptimizedFieldsToForm(optimized, { includeImageAge });
  const form = lerobotPolicyEl("lerobot-run-settings");
  if (form) {
    lerobotSyncMotionSpeedFields(form, clamped);
  }
  lerobotReadSettingsFromForm();
  lerobotUpdateSpeedSummary(lerobotFlatSettings(lerobotPolicyState.runSettings));
}

function lerobotApplySettingsToForm(flat) {
  const form = lerobotPolicyEl("lerobot-run-settings");
  if (!form || !flat) return;
  const setNum = (name, value) => {
    const el = form.elements.namedItem(name);
    if (el instanceof HTMLInputElement && value != null && value !== "") {
      el.value = String(value);
    }
  };
  const setCheck = (name, value) => {
    const el = form.elements.namedItem(name);
    if (el instanceof HTMLInputElement && el.type === "checkbox") {
      el.checked = Boolean(value);
    }
  };
  lerobotSyncMotionSpeedFields(form, flat.motion_speed);
  setNum("publish_hz", flat.publish_hz);
  setNum("inference_hz", flat.inference_hz);
  setNum("warming_up_time", flat.warming_up_time);
  setNum("min_run_time_before_complete", flat.min_run_time_before_complete);
  setNum("max_image_age_s", flat.max_image_age_s);
  setNum("gripper_threshold", flat.gripper_threshold);
  setNum("waypoint_approach_rad", flat.waypoint_approach_rad);
  setNum("arm_min_cutoff", flat.arm_min_cutoff);
  setNum("arm_derivative_cutoff", flat.arm_derivative_cutoff);
  setNum("arm_beta", flat.arm_beta);
  setNum("timeout", flat.timeout);
  setCheck("arm_pp_enabled", flat.arm_pp_enabled);
  setCheck("skip_warmup", flat.skip_warmup);
  lerobotUpdateSpeedSummary(flat);
}

function lerobotPickDefined(obj) {
  const out = {};
  for (const [key, value] of Object.entries(obj)) {
    if (value === undefined) continue;
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const nested = lerobotPickDefined(value);
      if (Object.keys(nested).length) out[key] = nested;
    } else {
      out[key] = value;
    }
  }
  return out;
}

function lerobotReadSettingsFromForm() {
  const form = lerobotPolicyEl("lerobot-run-settings");
  if (!form) return lerobotPolicyState.runSettings;
  const num = (name) => {
    const el = form.elements.namedItem(name);
    if (!(el instanceof HTMLInputElement) || el.value === "") return undefined;
    const v = Number(el.value);
    return Number.isFinite(v) ? v : undefined;
  };
  const check = (name) => {
    const el = form.elements.namedItem(name);
    return el instanceof HTMLInputElement && el.type === "checkbox" ? el.checked : false;
  };
  const speedEl = form.elements.namedItem("motion_speed_precise");
  const speedSlider = form.elements.namedItem("motion_speed");
  const rawSpeed =
    speedEl instanceof HTMLInputElement && speedEl.value !== ""
      ? speedEl.value
      : speedSlider instanceof HTMLInputElement && speedSlider.value !== ""
        ? speedSlider.value
        : "";
  const motionSpeed =
    rawSpeed !== "" ? lerobotClampArmLead(Number(rawSpeed)) : undefined;
  const approachRad = num("waypoint_approach_rad");
  const publishConsume = {
    ...(lerobotPolicyState.runDefaults?.runtime?.publish_consume || {}),
    ...(lerobotPolicyState.runSettings?.runtime?.publish_consume || {}),
  };
  if (approachRad != null) {
    publishConsume.waypoint_approach_rad = approachRad;
  }
  const patch = lerobotPickDefined({
    runtime: {
      motion_speed: motionSpeed,
      publish_hz: num("publish_hz"),
      inference_hz: num("inference_hz"),
      warming_up_time: num("warming_up_time"),
      min_run_time_before_complete: num("min_run_time_before_complete"),
      max_image_age_s: num("max_image_age_s"),
      publish_consume: lerobotPickDefined(publishConsume),
      arm_action_postprocess: {
        enabled: check("arm_pp_enabled"),
        min_cutoff: num("arm_min_cutoff"),
        derivative_cutoff: num("arm_derivative_cutoff"),
        beta: num("arm_beta"),
      },
    },
    action: { gripper_threshold: num("gripper_threshold") },
    cli: {
      motion_speed: motionSpeed,
      timeout: num("timeout"),
      skip_warmup: check("skip_warmup"),
    },
  });
  const merged = lerobotMergeRunDefaults(lerobotPolicyState.runDefaults, patch);
  lerobotPolicyState.runSettings = merged;
  try {
    localStorage.setItem(LEROBOT_RUN_SETTINGS_KEY, JSON.stringify(merged));
  } catch {
    /* ignore */
  }
  return merged;
}

function lerobotRunRequestBody(policyId) {
  const form = lerobotPolicyEl("lerobot-run-settings");
  if (form) {
    const speedEl = form.elements.namedItem("motion_speed_precise");
    const slider = form.elements.namedItem("motion_speed");
    const raw =
      speedEl instanceof HTMLInputElement && speedEl.value !== ""
        ? speedEl.value
        : slider instanceof HTMLInputElement
          ? slider.value
          : "";
    if (raw !== "") {
      lerobotCommitMotionSpeed(Number(raw));
    }
  }
  const settings = lerobotReadSettingsFromForm();
  const rt = { ...(settings.runtime || {}) };
  const arm = rt.arm_action_postprocess;
  if (arm) {
    rt.arm_action_postprocess = { ...arm };
  }
  const body = {
    policy: policyId,
    runtime: rt,
    action: settings.action || {},
    timeout: settings.cli?.timeout ?? -1,
    skip_warmup: Boolean(settings.cli?.skip_warmup),
  };
  return body;
}

function lerobotInitRunSettings(defaults) {
  if (!defaults) return;
  lerobotPolicyState.runDefaults = defaults;
  let saved = null;
  try {
    const raw = localStorage.getItem(LEROBOT_RUN_SETTINGS_KEY);
    if (raw) saved = JSON.parse(raw);
  } catch {
    saved = null;
  }
  const merged = lerobotMergeRunDefaults(defaults, saved);
  merged.runtime = merged.runtime || {};
  merged.cli = merged.cli || {};
  const resolved = lerobotResolveMotionSpeed(
    merged.cli.motion_speed,
    merged.runtime.motion_speed,
  );
  merged.runtime.motion_speed = resolved;
  merged.cli.motion_speed = resolved;
  if (saved == null) {
    const templateMotion = lerobotClampArmLead(
      Number(defaults.runtime?.motion_speed) || LEROBOT_ARM_LEAD_DEFAULT,
    );
    merged.runtime.motion_speed = templateMotion;
    merged.cli.motion_speed = templateMotion;
    merged.runtime.publish_consume = {
      ...(defaults.runtime?.publish_consume || {}),
      ...(merged.runtime.publish_consume || {}),
    };
  } else if (
    lerobotIsLegacyMotionSpeed(saved.cli?.motion_speed) ||
    lerobotIsLegacyMotionSpeed(saved.runtime?.motion_speed)
  ) {
    const templateMotion = lerobotClampArmLead(
      Number(defaults.runtime?.motion_speed) || LEROBOT_ARM_LEAD_DEFAULT,
    );
    merged.runtime.motion_speed = templateMotion;
    merged.cli.motion_speed = templateMotion;
  }
  lerobotPolicyState.runSettings = merged;
  try {
    localStorage.setItem(LEROBOT_RUN_SETTINGS_KEY, JSON.stringify(merged));
  } catch {
    /* ignore */
  }
  const flat = lerobotFlatSettings(merged);
  lerobotLastAcceptedArmLead = lerobotClampArmLead(flat.motion_speed);
  lerobotApplySettingsToForm(flat);
}

function bindLerobotRunSettingsForm() {
  const form = lerobotPolicyEl("lerobot-run-settings");
  if (!form || form.dataset.bound === "1") return;
  form.dataset.bound = "1";
  form.addEventListener("change", (ev) => {
    if (lerobotSyncingSpeedFields) return;
    const target = ev.target;
    if (
      target instanceof HTMLInputElement &&
      (target.name === "motion_speed" ||
        target.name === "motion_speed_precise") &&
      target.value !== ""
    ) {
      const requested = lerobotClampArmLead(Number(target.value));
      if (
        requested > LEROBOT_ARM_LEAD_WARN &&
        !lerobotConfirmHighArmLead(requested)
      ) {
        lerobotSyncMotionSpeedFields(form, lerobotLastAcceptedArmLead);
        lerobotUpdateSpeedSummary({ motion_speed: lerobotLastAcceptedArmLead });
        return;
      }
      lerobotCommitMotionSpeed(requested);
      return;
    }
    if (
      target instanceof HTMLInputElement &&
      target.name === "publish_hz" &&
      target.value !== ""
    ) {
      const speed = lerobotSpeedFromPublish(Number(target.value));
      const slider = form.elements.namedItem("motion_speed");
      if (slider instanceof HTMLInputElement) {
        lerobotSyncingSpeedFields = true;
        slider.value = String(speed);
        lerobotSyncingSpeedFields = false;
      }
    }
    lerobotReadSettingsFromForm();
    lerobotUpdateSpeedSummary(lerobotFlatSettings(lerobotPolicyState.runSettings));
  });
  form.addEventListener("input", (ev) => {
    if (lerobotSyncingSpeedFields) return;
    const target = ev.target;
    if (
      target instanceof HTMLInputElement &&
      (target.name === "motion_speed" ||
        target.name === "motion_speed_precise") &&
      target.value !== ""
    ) {
      const requested = lerobotClampArmLead(Number(target.value));
      lerobotSyncMotionSpeedFields(form, requested);
      lerobotUpdateSpeedSummary({ motion_speed: requested });
      if (requested <= LEROBOT_ARM_LEAD_WARN) {
        lerobotCommitMotionSpeed(requested);
      }
      return;
    }
    if (
      target instanceof HTMLInputElement &&
      target.name === "publish_hz" &&
      target.value !== ""
    ) {
      const speed = lerobotSpeedFromPublish(Number(target.value));
      const slider = form.elements.namedItem("motion_speed");
      if (slider instanceof HTMLInputElement) {
        lerobotSyncingSpeedFields = true;
        slider.value = String(speed);
        lerobotSyncingSpeedFields = false;
      }
    }
    lerobotReadSettingsFromForm();
    lerobotUpdateSpeedSummary(lerobotFlatSettings(lerobotPolicyState.runSettings));
  });
  const optimizeBtn = lerobotPolicyEl("lerobot-settings-optimize");
  if (optimizeBtn) {
    optimizeBtn.addEventListener("click", () => {
      const slider = form.elements.namedItem("motion_speed");
      const speed =
        slider instanceof HTMLInputElement && slider.value !== ""
          ? lerobotClampArmLead(Number(slider.value))
          : LEROBOT_ARM_LEAD_DEFAULT;
      if (!lerobotConfirmHighArmLead(speed)) return;
      lerobotLastAcceptedArmLead = speed;
      lerobotOptimizeForSpeed(speed, { includeImageAge: true });
    });
  }
  const resetBtn = lerobotPolicyEl("lerobot-settings-reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      if (!lerobotPolicyState.runDefaults) return;
      localStorage.removeItem(LEROBOT_RUN_SETTINGS_KEY);
      lerobotInitRunSettings(lerobotPolicyState.runDefaults);
    });
  }
}

function lerobotPolicyEl(id) {
  return document.getElementById(id);
}

function lerobotPolicySetError(msg) {
  const el = lerobotPolicyEl("lerobot-policy-error");
  if (!el) return;
  if (msg) {
    el.textContent = msg;
    el.hidden = false;
  } else {
    el.textContent = "";
    el.hidden = true;
  }
}

async function lerobotPolicyFetch(path, opts) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = body.detail;
    const msg =
      typeof detail === "string"
        ? detail
        : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return body;
}

function lerobotFormatGripperScalar(value) {
  if (value == null || !Number.isFinite(Number(value))) {
    return "—";
  }
  return Number(value).toFixed(4);
}

function lerobotFormatJointGrid(labelPrefix, values) {
  if (!Array.isArray(values) || !values.length) {
    return `<p class="lerobot-targets-empty">${labelPrefix}: —</p>`;
  }
  const cells = values
    .map(
      (v, i) =>
        `<span title="${labelPrefix}${i}">${i}:${Number(v).toFixed(3)}</span>`,
    )
    .join("");
  return `<div class="lerobot-targets-joints">${cells}</div>`;
}

function lerobotRenderTargetsPanel(state) {
  const panel = lerobotPolicyEl("lerobot-targets-panel");
  if (!panel) return;
  if (!state || !state.available || !state.state) {
    panel.innerHTML =
      '<p class="lerobot-targets-empty">실행 중이 아니거나 아직 목표 데이터가 없습니다.</p>';
    return;
  }
  const s = state.state;
  const pub = s.published || {};
  const obs = s.gripper_observation || {};
  const topic = s.gripper_topic || {};
  const err =
    s.waypoint_approach_err_rad != null
      ? Number(s.waypoint_approach_err_rad).toFixed(3)
      : "—";
  const age =
    s.timestamp != null
      ? `${Math.max(0, Date.now() / 1000 - Number(s.timestamp)).toFixed(1)}s 전`
      : "";
  panel.innerHTML = `
    <div class="lerobot-targets-meta">
      <span><strong>큐</strong> ${s.queue_remaining ?? "—"}</span>
      <span><strong>approach err</strong> ${err} rad</span>
      <span><strong>tick</strong> ${s.publish_tick ?? "—"}</span>
      <span><strong>infer</strong> #${s.inference_count ?? "—"}</span>
      <span>${age}</span>
    </div>
    <div class="lerobot-targets-grid">
      <section class="lerobot-targets-section">
        <h3>왼팔 · 명령</h3>
        ${lerobotFormatJointGrid("L", pub.LEFT_ARM)}
        <h3 style="margin-top:0.5rem">왼 그리퍼</h3>
        <p class="lerobot-targets-scalar">
          관측 <strong>${lerobotFormatGripperScalar(obs.LEFT_GRIPPER)}</strong>
          (GDK ${lerobotFormatGripperScalar(topic.LEFT_GRIPPER)})
        </p>
        <p class="lerobot-targets-scalar lerobot-targets-gripper-cmd">
          명령(chunk) ${lerobotFormatGripperScalar(pub.LEFT_GRIPPER)}
        </p>
      </section>
      <section class="lerobot-targets-section">
        <h3>오른팔 · 명령</h3>
        ${lerobotFormatJointGrid("R", pub.RIGHT_ARM)}
        <h3 style="margin-top:0.5rem">오른 그리퍼</h3>
        <p class="lerobot-targets-scalar">
          관측 <strong>${lerobotFormatGripperScalar(obs.RIGHT_GRIPPER)}</strong>
          (GDK ${lerobotFormatGripperScalar(topic.RIGHT_GRIPPER)})
        </p>
        <p class="lerobot-targets-scalar lerobot-targets-gripper-cmd">
          명령(chunk) ${lerobotFormatGripperScalar(pub.RIGHT_GRIPPER)}
        </p>
      </section>
    </div>`;
}

async function lerobotPolicyRefreshTargets() {
  const details = lerobotPolicyEl("lerobot-targets-details");
  if (details && !details.open) return;
  try {
    const data = await lerobotPolicyFetch("/api/lerobot/policies/live-targets");
    lerobotRenderTargetsPanel(data);
  } catch {
    lerobotRenderTargetsPanel(null);
  }
}

function lerobotPolicyRenderList() {
  const ul = lerobotPolicyEl("lerobot-policy-list");
  if (!ul) return;
  ul.innerHTML = "";
  if (!lerobotPolicyState.policies.length) {
    const li = document.createElement("li");
    li.className = "lerobot-policy-empty";
    li.textContent = "models/LeRobot 에 체크포인트가 없습니다.";
    ul.appendChild(li);
    return;
  }
  for (const p of lerobotPolicyState.policies) {
    const li = document.createElement("li");
    li.className = "lerobot-policy-item";
    const body = document.createElement("div");
    body.className = "lerobot-policy-item-body";
    const title = document.createElement("p");
    title.className = "lerobot-policy-name";
    title.textContent = p.name || p.id;
    body.appendChild(title);
    if (p.id && p.id !== (p.name || "")) {
      const sub = document.createElement("p");
      sub.className = "lerobot-policy-id";
      sub.textContent = p.id;
      body.appendChild(sub);
    }
    if (p.model_path && p.model_exists === false) {
      const bad = document.createElement("p");
      bad.className = "lerobot-policy-missing";
      bad.textContent = `모델 없음: ${p.model_path}`;
      body.appendChild(bad);
    }
    if (p.model_compatible === false && p.incompatible_reason) {
      const bad = document.createElement("p");
      bad.className = "lerobot-policy-missing";
      bad.textContent = `배포 설정 불일치: ${p.incompatible_reason}`;
      body.appendChild(bad);
    }
    li.appendChild(body);
    const btn = document.createElement("button");
    btn.type = "button";
    const isRun = lerobotPolicyState.running && lerobotPolicyState.runningPolicy === p.id;
    btn.className = isRun ? "btn-danger lerobot-policy-btn" : "btn-primary lerobot-policy-btn";
    const runnable =
      p.model_exists !== false && p.model_compatible !== false;
    btn.disabled =
      lerobotPolicyState.busy ||
      (lerobotPolicyState.running && !isRun) ||
      (!isRun && !runnable);
    btn.textContent = isRun ? "중지" : "실행";
    btn.addEventListener("click", () => {
      void (isRun ? lerobotPolicyStop() : lerobotPolicyPlay(p.id));
    });
    li.appendChild(btn);
    ul.appendChild(li);
  }
  touchPlannerPanel();
}

async function lerobotPolicyLoad() {
  lerobotPolicySetError("");
  try {
    const data = await lerobotPolicyFetch("/api/lerobot/policies");
    if (data.run_defaults) {
      lerobotInitRunSettings(data.run_defaults);
    }
    lerobotPolicyState.policies = data.policies || [];
    syncPlannerPolicyCatalog(lerobotPolicyState.policies);
    const st = data.status || {};
    lerobotPolicyState.running = Boolean(st.running);
    lerobotPolicyState.pid = st.pid || "";
    lerobotPolicyState.runningPolicy = st.policy || "";
    lerobotPolicyRenderList();
    void lerobotHandMapLoad();
    if (lerobotPolicyState.running) {
      lerobotPolicyStartPoll();
    } else lerobotPolicyStopPoll();
  } catch (e) {
    lerobotPolicySetError(e.message || String(e));
  }
}

async function lerobotPolicyPlay(policyId, { silent = false } = {}) {
  if (lerobotPolicyState.busy && !silent) return false;
  if (!silent && !lerobotHandMappingFileForPolicy(policyId)) {
    if (!lerobotConfirmRunWithoutHandMapping(policyId)) {
      return false;
    }
  }
  lerobotPolicyState.busy = true;
  if (!silent) {
    lerobotPolicyRenderList();
    lerobotPolicySetError("");
  }
  try {
    await lerobotPolicyFetch("/api/lerobot/policies/run", {
      method: "POST",
      body: JSON.stringify(lerobotRunRequestBody(policyId)),
    });
    lerobotPolicyState.running = true;
    lerobotPolicyState.runningPolicy = policyId;
    lerobotPolicyStartPoll();
    void lerobotPolicyRefreshTargets();
    return true;
  } catch (e) {
    const msg = e.message || String(e);
    if (!silent) {
      lerobotPolicySetError(
        msg.includes("바로 종료")
          ? "정책이 바로 종료되었습니다. 서버 run.log를 확인하세요."
          : msg
      );
    }
    lerobotPolicyState.running = false;
    lerobotRenderTargetsPanel(null);
    return false;
  } finally {
    lerobotPolicyState.busy = false;
    if (!plannerTaskUiRunning) {
      await lerobotPolicyLoad();
    }
  }
}

async function lerobotPolicyStop({ silent = false } = {}) {
  if (lerobotPolicyState.busy && !silent) return;
  lerobotPolicyState.busy = true;
  if (!silent) {
    lerobotPolicyRenderList();
  }
  try {
    await lerobotPolicyFetch("/api/lerobot/policies/stop", { method: "POST", body: "{}" });
    lerobotPolicyState.running = false;
    lerobotPolicyState.runningPolicy = "";
    lerobotPolicyStopPoll();
    lerobotRenderTargetsPanel(null);
  } catch (e) {
    if (!silent) {
      lerobotPolicySetError(e.message || String(e));
    }
  } finally {
    lerobotPolicyState.busy = false;
    if (!plannerTaskUiRunning) {
      await lerobotPolicyLoad();
    }
  }
}

function lerobotPolicyStartPoll() {
  lerobotPolicyStopPoll();
  void lerobotPolicyRefreshTargets();
  lerobotPolicyState.pollTimer = window.setInterval(() => {
    void lerobotPolicyPollTick();
  }, 500);
}

function lerobotPolicyStopPoll() {
  if (lerobotPolicyState.pollTimer != null) {
    window.clearInterval(lerobotPolicyState.pollTimer);
    lerobotPolicyState.pollTimer = null;
  }
}

async function lerobotPolicyPollTick() {
  try {
    const st = await lerobotPolicyFetch("/api/lerobot/policies/status");
    lerobotPolicyState.running = Boolean(st.running);
    lerobotPolicyState.pid = st.pid || "";
    lerobotPolicyState.runningPolicy = st.policy || lerobotPolicyState.runningPolicy;
    lerobotPolicyRenderList();
    if (lerobotPolicyState.running) {
      await lerobotPolicyRefreshTargets();
    } else {
      lerobotPolicyStopPoll();
      lerobotRenderTargetsPanel(null);
    }
  } catch {
    /* keep */
  }
}

function lerobotHandMapEl(id) {
  return document.getElementById(id);
}

function lerobotHandMapSetError(message) {
  const el = lerobotHandMapEl("lerobot-hand-map-error");
  if (!el) return;
  const text = String(message || "").trim();
  el.hidden = !text;
  el.textContent = text;
}

function lerobotHandMapRender() {
  const list = lerobotHandMapEl("lerobot-hand-map-list");
  if (!list || list.contains(document.activeElement)) return;
  list.innerHTML = "";
  const policies = lerobotPolicyState.policies || [];
  if (!policies.length) {
    const li = document.createElement("li");
    li.className = "lerobot-policy-empty";
    li.textContent = "models/LeRobot 에 체크포인트가 없습니다.";
    list.appendChild(li);
    return;
  }
  const files = lerobotPolicyState.handPoseFiles || [];
  for (const policy of policies) {
    const li = document.createElement("li");
    li.className = "lerobot-hand-map-item";
    const title = document.createElement("p");
    title.className = "lerobot-policy-name";
    title.textContent = policy.name || policy.id;
    li.appendChild(title);
    const select = document.createElement("select");
    select.className = "lerobot-hand-map-select";
    select.setAttribute("aria-label", `${policy.name || policy.id} 핸드 자세 파일`);
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = files.length ? "선택 안 함" : "핸드 자세 파일 없음";
    select.appendChild(empty);
    for (const file of files) {
      const option = document.createElement("option");
      option.value = file.id;
      option.textContent = file.name || file.id;
      select.appendChild(option);
    }
    const selected = lerobotPolicyState.handMappings?.[policy.id] || "";
    select.value = files.some((file) => file.id === selected) ? selected : "";
    select.addEventListener("change", () => {
      void lerobotHandMapSave(policy.id, select.value, select);
    });
    li.appendChild(select);
    list.appendChild(li);
  }
}

async function lerobotHandMapLoad() {
  try {
    const data = await lerobotPolicyFetch("/api/lerobot/hand-mappings");
    lerobotPolicyState.handPoseFiles = data.files || [];
    lerobotPolicyState.handMappings = data.selections || {};
    lerobotHandMapSetError("");
    lerobotHandMapRender();
  } catch (err) {
    lerobotHandMapSetError(err.message || String(err));
    lerobotHandMapRender();
  }
}

async function lerobotHandMapSave(policyId, fileId, select) {
  const previous = lerobotPolicyState.handMappings?.[policyId] || "";
  try {
    const data = await lerobotPolicyFetch("/api/lerobot/hand-mappings", {
      method: "PUT",
      body: JSON.stringify({ policy: policyId, file: fileId }),
    });
    lerobotPolicyState.handMappings = data.selections || {};
    if (Array.isArray(data.files)) lerobotPolicyState.handPoseFiles = data.files;
    lerobotHandMapSetError("");
  } catch (err) {
    if (select instanceof HTMLSelectElement) select.value = previous;
    lerobotHandMapSetError(err.message || String(err));
  }
}

function lerobotHandMappingFileForPolicy(policyId) {
  return String(lerobotPolicyState.handMappings?.[policyId] || "").trim();
}

function lerobotConfirmRunWithoutHandMapping(policyId) {
  const policies = lerobotPolicyState.policies || [];
  const policy = policies.find((item) => item.id === policyId);
  const label = policy?.name || policyId;
  return window.confirm(
    `이 정책(${label})의 핸드 자세 체계가 선택되지 않았습니다.\n` +
      "해당 체계가 없을 경우 핸드의 자세는 계속 고정된 채 진행됩니다.\n\n" +
      "그래도 정책을 실행할까요?"
  );
}

function initLerobotPolicyPanel() {
  bindLerobotRunSettingsForm();
  const refreshBtn = lerobotPolicyEl("lerobot-policy-refresh");
  if (refreshBtn) refreshBtn.addEventListener("click", () => void lerobotPolicyLoad());
  const targetsDetails = lerobotPolicyEl("lerobot-targets-details");
  if (targetsDetails) {
    targetsDetails.addEventListener("toggle", () => {
      if (targetsDetails.open) void lerobotPolicyRefreshTargets();
    });
  }
  void lerobotPolicyLoad();
}
