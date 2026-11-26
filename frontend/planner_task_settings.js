/** Task Planner → Task 설정: OpenVLA + YOLO 조건과 LeRobot Policy 매핑. */

const LOCAL_RULES_KEY = "oap-planner-task-policy-rules";

let rules = [];
let catalog = {
  profiles: {},
  activeProfile: "",
  yoloTargets: [],
  policies: [],
};
let saveTimer = null;
let loaded = false;

async function plannerApi(path, options) {
  const res = await fetch(path, { cache: "no-store", ...options });
  if (!res.ok) {
    let detail = `${res.status} ${path}`;
    try {
      const body = await res.json();
      if (body.detail) detail = String(body.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

function defaultRule() {
  const id =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : `rule-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return {
    id,
    openvla_profile: "",
    openvla_phase_id: "",
    yolo_label: "",
    yolo_min_count: 1,
    policy_id: "",
  };
}

function setSettingsStatus(message, { err = false } = {}) {
  const el = document.getElementById("planner-task-settings-status");
  if (!el) return;
  if (!message) {
    el.hidden = true;
    el.textContent = "";
    el.classList.remove("err");
    return;
  }
  el.hidden = false;
  el.textContent = message;
  el.classList.toggle("err", err);
}

function phasesForProfile(profileName) {
  const key = (profileName || "").trim();
  if (key && catalog.profiles[key]?.phases) {
    return catalog.profiles[key].phases;
  }
  const names = Object.keys(catalog.profiles);
  const fallback = (key || catalog.activeProfile || names[0] || "").trim();
  if (fallback && catalog.profiles[fallback]?.phases) {
    return catalog.profiles[fallback].phases;
  }
  return [];
}

function appendOption(select, value, label, { disabled = false, selected = false } = {}) {
  const opt = document.createElement("option");
  opt.value = value;
  opt.textContent = label;
  opt.disabled = disabled;
  opt.selected = selected;
  select.appendChild(opt);
}

function fillProfileSelect(select, value) {
  select.innerHTML = "";
  appendOption(select, "", "(프로필 무관)", { selected: value === "" });
  const names = Object.keys(catalog.profiles).sort((a, b) => a.localeCompare(b, "ko"));
  for (const name of names) {
    appendOption(select, name, name, { selected: value === name });
  }
}

function fillPhaseSelect(select, profileName, phaseId) {
  select.innerHTML = "";
  appendOption(select, "", "(Phase 무관)", { selected: phaseId === "" });
  const rows = phasesForProfile(profileName);
  for (const row of rows) {
    const id = String(row.id || "");
    const num = row.number != null ? `P${row.number}` : "";
    const desc = (row.description || "").trim();
    const label = [num, desc].filter(Boolean).join(" · ") || id || "—";
    if (!id) continue;
    appendOption(select, id, label, { selected: phaseId === id });
  }
}

function fillYoloSelect(select, labelValue) {
  select.innerHTML = "";
  appendOption(select, "", "(클래스 무관)", { selected: labelValue === "" });
  const names = catalog.yoloTargets
    .map((t) => String(t.name || "").trim())
    .filter(Boolean);
  const uniq = [...new Set(names)].sort((a, b) => a.localeCompare(b, "ko"));
  for (const name of uniq) {
    appendOption(select, name, name, { selected: labelValue === name });
  }
}

function fillPolicySelect(select, policyId) {
  select.innerHTML = "";
  appendOption(select, "", "Policy 선택…", { selected: policyId === "" });
  const items = [...(catalog.policies || [])].sort((a, b) =>
    String(a.name || a.id).localeCompare(String(b.name || b.id), "ko")
  );
  for (const p of items) {
    const id = String(p.id || "");
    if (!id) continue;
    const title = String(p.name || id);
    const runnable = p.model_exists !== false && p.model_compatible !== false;
    const suffix = runnable ? "" : " (실행 불가)";
    appendOption(select, id, `${title}${suffix}`, {
      selected: policyId === id,
      disabled: !runnable,
    });
  }
}

function ruleFromRowEl(rowEl) {
  const id = rowEl.dataset.ruleId || "";
  const profile = rowEl.querySelector('[data-field="openvla_profile"]')?.value ?? "";
  const phaseId = rowEl.querySelector('[data-field="openvla_phase_id"]')?.value ?? "";
  const yolo = rowEl.querySelector('[data-field="yolo_label"]')?.value ?? "";
  const minRaw = rowEl.querySelector('[data-field="yolo_min_count"]')?.value ?? "1";
  let minCount = Number.parseInt(String(minRaw), 10);
  if (!Number.isFinite(minCount) || minCount < 0) minCount = 0;
  if (minCount > 99) minCount = 99;
  const policy = rowEl.querySelector('[data-field="policy_id"]')?.value ?? "";
  return {
    id,
    openvla_profile: profile,
    openvla_phase_id: phaseId,
    yolo_label: yolo,
    yolo_min_count: minCount,
    policy_id: policy,
  };
}

function collectRulesFromDom() {
  const list = document.getElementById("planner-task-rules-list");
  if (!list) return [...rules];
  const out = [];
  for (const row of list.querySelectorAll(".planner-task-rule")) {
    out.push(ruleFromRowEl(row));
  }
  return out;
}

function persistLocal(data) {
  try {
    localStorage.setItem(LOCAL_RULES_KEY, JSON.stringify(data));
  } catch {
    /* ignore */
  }
}

function scheduleSave() {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    void flushSave();
  }, 450);
}

async function flushSave() {
  rules = collectRulesFromDom();
  persistLocal(rules);
  if (!catalog.policies.length) {
    try {
      const data = await plannerApi("/api/lerobot/policies");
      catalog.policies = data.policies || [];
    } catch {
      /* ignore */
    }
  }
  setSettingsStatus("저장 중…");
  try {
    const data = await plannerApi("/api/planner/task-policy-rules", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rules }),
    });
    rules = data.rules || rules;
    setSettingsStatus("저장됨");
    window.setTimeout(() => setSettingsStatus(""), 1200);
  } catch (e) {
    setSettingsStatus(`서버 저장 실패 (로컬만 유지): ${e.message}`, { err: true });
  }
}

function buildRuleRow(rule, index) {
  const li = document.createElement("li");
  li.className = "planner-task-rule";
  li.dataset.ruleId = rule.id;

  const head = document.createElement("div");
  head.className = "planner-task-rule-head";
  const title = document.createElement("span");
  title.className = "planner-task-rule-title";
  title.textContent = `조건 ${index + 1}`;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "btn-ghost planner-task-rule-remove";
  remove.textContent = "삭제";
  remove.addEventListener("click", () => {
    li.remove();
    renumberRuleTitles();
    rules = collectRulesFromDom();
    scheduleSave();
  });
  head.append(title, remove);

  const grid = document.createElement("div");
  grid.className = "planner-task-rule-grid";

  const mkField = (labelText, control) => {
    const wrap = document.createElement("label");
    wrap.className = "planner-task-rule-field";
    const cap = document.createElement("span");
    cap.className = "planner-sublabel";
    cap.textContent = labelText;
    wrap.append(cap, control);
    return wrap;
  };

  const profileSel = document.createElement("select");
  profileSel.dataset.field = "openvla_profile";
  fillProfileSelect(profileSel, rule.openvla_profile);
  profileSel.addEventListener("change", () => {
    const phaseSel = li.querySelector('[data-field="openvla_phase_id"]');
    if (phaseSel) {
      fillPhaseSelect(phaseSel, profileSel.value, phaseSel.value);
    }
    scheduleSave();
  });

  const phaseSel = document.createElement("select");
  phaseSel.dataset.field = "openvla_phase_id";
  fillPhaseSelect(phaseSel, rule.openvla_profile, rule.openvla_phase_id);
  phaseSel.addEventListener("change", scheduleSave);

  const yoloSel = document.createElement("select");
  yoloSel.dataset.field = "yolo_label";
  fillYoloSelect(yoloSel, rule.yolo_label);
  yoloSel.addEventListener("change", scheduleSave);

  const minInp = document.createElement("input");
  minInp.type = "number";
  minInp.min = "0";
  minInp.max = "99";
  minInp.step = "1";
  minInp.className = "planner-task-rule-min";
  minInp.dataset.field = "yolo_min_count";
  minInp.value = String(rule.yolo_min_count ?? 1);
  minInp.addEventListener("change", scheduleSave);

  const policySel = document.createElement("select");
  policySel.dataset.field = "policy_id";
  policySel.className = "planner-task-rule-policy";
  fillPolicySelect(policySel, rule.policy_id);
  policySel.addEventListener("change", scheduleSave);

  grid.append(
    mkField("OpenVLA Phase 프로필", profileSel),
    mkField("OpenVLA 다음 Phase", phaseSel),
    mkField("YOLO 탐지 클래스", yoloSel),
    mkField("YOLO 최소 개수", minInp),
    mkField("LeRobot Policy", policySel)
  );

  li.append(head, grid);
  return li;
}

function renumberRuleTitles() {
  const list = document.getElementById("planner-task-rules-list");
  if (!list) return;
  list.querySelectorAll(".planner-task-rule").forEach((row, i) => {
    const title = row.querySelector(".planner-task-rule-title");
    if (title) title.textContent = `조건 ${i + 1}`;
  });
}

function renderRulesList() {
  const list = document.getElementById("planner-task-rules-list");
  if (!list) return;
  list.innerHTML = "";
  const rows = rules.length ? rules : [];
  rows.forEach((rule, index) => {
    list.appendChild(buildRuleRow(rule, index));
  });
  if (!rows.length) {
    const empty = document.createElement("p");
    empty.className = "planner-meta planner-task-rules-empty";
    empty.textContent =
      "조건이 없습니다. 아래 「조건 추가」로 OpenVLA·YOLO 추론 결과에 맞는 LeRobot Policy를 연결하세요.";
    list.appendChild(empty);
  }
}

async function refreshCatalog() {
  const tasks = [
    plannerApi("/api/phases").then((data) => {
      catalog.profiles = data.profiles || {};
      catalog.activeProfile = data.active_profile || "";
    }),
    plannerApi("/api/yolo/config").then((data) => {
      catalog.yoloTargets = data.targets || [];
    }),
    plannerApi("/api/lerobot/policies").then((data) => {
      catalog.policies = data.policies || [];
    }),
  ];
  await Promise.all(tasks);
}

async function loadRules() {
  let serverRules = null;
  try {
    const data = await plannerApi("/api/planner/task-policy-rules");
    serverRules = data.rules;
  } catch {
    /* offline */
  }
  if (Array.isArray(serverRules)) {
    rules = serverRules.map((row) => ({ ...defaultRule(), ...row, id: row.id || defaultRule().id }));
    persistLocal(rules);
    return;
  }
  try {
    const raw = localStorage.getItem(LOCAL_RULES_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        rules = parsed.map((row) => ({
          ...defaultRule(),
          ...row,
          id: row.id || defaultRule().id,
        }));
      }
    }
  } catch {
    rules = [];
  }
}

export function getPlannerTaskPolicyRules() {
  return getEffectivePlannerRules();
}

export function getEffectivePlannerRules() {
  const list = document.getElementById("planner-task-rules-list");
  if (list?.querySelector(".planner-task-rule")) {
    return collectRulesFromDom();
  }
  return rules.map((row) => ({ ...row }));
}

export function policyDisplayName(policyId) {
  const token = String(policyId || "").trim();
  if (!token) return "";
  const hit = (catalog.policies || []).find((p) => String(p.id) === token);
  return hit ? String(hit.name || hit.id) : token;
}

export function syncPlannerPolicyCatalog(policies) {
  catalog.policies = Array.isArray(policies) ? policies : [];
}

export async function initPlannerTaskRulesCache() {
  if (!loaded) {
    await loadRules();
    loaded = true;
  }
  try {
    const data = await plannerApi("/api/lerobot/policies");
    catalog.policies = data.policies || [];
  } catch {
    /* policies optional for display */
  }
}

export async function onPlannerSettingsPaneOpen() {
  setSettingsStatus("목록 불러오는 중…");
  try {
    await refreshCatalog();
    if (!loaded) {
      await loadRules();
      loaded = true;
    }
    renderRulesList();
    setSettingsStatus("");
  } catch (e) {
    setSettingsStatus(e.message || String(e), { err: true });
  }
}

export function initPlannerTaskSettings() {
  const addBtn = document.getElementById("planner-task-rule-add");
  if (addBtn && addBtn.dataset.bound !== "1") {
    addBtn.dataset.bound = "1";
    addBtn.addEventListener("click", () => {
      const list = document.getElementById("planner-task-rules-list");
      if (!list) return;
      const empty = list.querySelector(".planner-task-rules-empty");
      if (empty) empty.remove();
      const rule = defaultRule();
      rules = collectRulesFromDom();
      rules.push(rule);
      const index = rules.length - 1;
      list.appendChild(buildRuleRow(rule, index));
      scheduleSave();
    });
  }
}