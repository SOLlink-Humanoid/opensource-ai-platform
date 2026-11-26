/** YOLO tag condition cards (bookmark: 태그 조건). */

let yoloTagConditions = [];
let yoloTagConditionSaveTimer = null;
let yoloTagConditionEditing = false;
let yoloTagConditionDirty = false;
let yoloTagConditionSaveInFlight = false;
const YOLO_TAG_CONDITION_SAVE_MS = 200;
/** 프로세스 시작 후 태그 조건 탭 첫 진입 시 규칙 접기 */
let yoloTagRulesInitialCollapsePending = false;
/** @type {string[]} */
let yoloActiveTags = [];
/** @type {{ class_id: number, name: string }[]} */
let yoloTagClassOptions = [];

const RULE_KIND_ORDER = ["class", "target", "region", "size", "orientation"];
const RULE_KIND_LABELS = {
  class: "포함되는 이름",
  target: "탐지된 타겟",
  region: "탐지된 위치",
  size: "탐지된 크기",
  orientation: "탐지 방향",
};

function newYoloTagConditionId() {
  return `ytag-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

function newRuleItemId() {
  return `rule-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}

function splitRuleItems(ruleItems) {
  const class_contains = [];
  const target_labels = [];
  const seenTargets = new Set();
  const regions = [];
  const size_rules = [];
  for (const item of ruleItems || []) {
    const kind = String(item?.kind || "").toLowerCase();
    if (kind === "class") {
      const text = String(item.text || "").trim();
      if (text) class_contains.push({ text });
    } else if (kind === "target") {
      for (const label of item.labels || []) {
        const name = String(label || "").trim();
        if (name && !seenTargets.has(name)) {
          seenTargets.add(name);
          target_labels.push(name);
        }
      }
    } else if (kind === "region") {
      regions.push({
        x1: Number(item.x1 ?? 0),
        y1: Number(item.y1 ?? 0),
        x2: Number(item.x2 ?? 0),
        y2: Number(item.y2 ?? 0),
        min_overlap_pct: Number(item.min_overlap_pct ?? 50),
      });
    } else if (kind === "orientation") {
      const orient = String(item.orientation || "").toLowerCase();
      if (orient === "horizontal" || orient === "vertical") {
        size_rules.push({ orientation: orient });
      }
    } else if (kind === "size") {
      const rule = {};
      for (const key of ["min_width", "max_width", "min_height", "max_height"]) {
        if (item[key] != null && item[key] !== "") rule[key] = Number(item[key]);
      }
      const orient = String(item.orientation || "").toLowerCase();
      if (orient === "horizontal" || orient === "vertical") rule.orientation = orient;
      if (Object.keys(rule).length) size_rules.push(rule);
    }
  }
  return { class_contains, target_labels, regions, size_rules };
}

function legacyToRuleItems(row) {
  const items = [];
  const targetLabels = (row?.target_labels || [])
    .map((x) => String(x || "").trim())
    .filter(Boolean);
  if (targetLabels.length) {
    items.push({ id: newRuleItemId(), kind: "target", labels: targetLabels });
  }
  for (const part of row.class_contains || []) {
    const text = String(part?.text || "").trim();
    if (text) items.push({ id: newRuleItemId(), kind: "class", text });
  }
  for (const part of row.regions || []) {
    items.push({
      id: newRuleItemId(),
      kind: "region",
      x1: Number(part.x1 ?? 0),
      y1: Number(part.y1 ?? 0),
      x2: Number(part.x2 ?? 0),
      y2: Number(part.y2 ?? 0),
      min_overlap_pct: Number(part.min_overlap_pct ?? 50),
    });
  }
  for (const part of row.size_rules || []) {
    const orient = String(part.orientation || "").toLowerCase();
  const dimKeys = ["min_width", "max_width", "min_height", "max_height"];
    const hasDim = dimKeys.some((key) => part[key] != null && part[key] !== "");
    if (orient === "horizontal" || orient === "vertical") {
      items.push({ id: newRuleItemId(), kind: "orientation", orientation: orient });
    }
    if (hasDim) {
      const item = { id: newRuleItemId(), kind: "size" };
      for (const key of dimKeys) {
        if (part[key] != null && part[key] !== "") item[key] = Number(part[key]);
      }
      items.push(item);
    }
  }
  return items;
}

function normalizeRuleItem(raw) {
  if (!raw || typeof raw !== "object") return null;
  const kind = String(raw.kind || "").toLowerCase();
  if (!["class", "target", "region", "size", "orientation"].includes(kind)) return null;
  const base = { id: String(raw.id || newRuleItemId()), kind };
  if (kind === "class") {
    return { ...base, text: String(raw.text || "").trim() };
  }
  if (kind === "target") {
    const labels = Array.isArray(raw.labels)
      ? raw.labels.map((x) => String(x || "").trim()).filter(Boolean)
      : [];
    return { ...base, labels };
  }
  if (kind === "region") {
    return {
      ...base,
      x1: Number(raw.x1 ?? 0),
      y1: Number(raw.y1 ?? 0),
      x2: Number(raw.x2 ?? 0),
      y2: Number(raw.y2 ?? 0),
      min_overlap_pct: Number(raw.min_overlap_pct ?? 50),
    };
  }
  if (kind === "orientation") {
    const orient = String(raw.orientation || "horizontal").toLowerCase();
    return {
      ...base,
      orientation: orient === "vertical" ? "vertical" : "horizontal",
    };
  }
  return {
    ...base,
    min_width: raw.min_width != null && raw.min_width !== "" ? Number(raw.min_width) : null,
    max_width: raw.max_width != null && raw.max_width !== "" ? Number(raw.max_width) : null,
    min_height: raw.min_height != null && raw.min_height !== "" ? Number(raw.min_height) : null,
    max_height: raw.max_height != null && raw.max_height !== "" ? Number(raw.max_height) : null,
  };
}

function expandSizeOrientationRuleItems(ruleItems) {
  const expanded = [];
  for (const raw of ruleItems || []) {
    const item = normalizeRuleItem(raw);
    if (!item) continue;
    if (item.kind !== "size") {
      expanded.push(item);
      continue;
    }
    const legacyOrient = ["horizontal", "vertical"].includes(
      String(raw?.orientation || "").toLowerCase()
    )
      ? String(raw.orientation).toLowerCase()
      : "";
    const dimKeys = ["min_width", "max_width", "min_height", "max_height"];
    const hasDim = dimKeys.some((key) => item[key] != null && item[key] !== "");
    if (hasDim) expanded.push(item);
    if (legacyOrient) {
      expanded.push({
        id: newRuleItemId(),
        kind: "orientation",
        orientation: legacyOrient,
        collapsed: item.collapsed,
      });
    }
    if (!hasDim && !legacyOrient) expanded.push(item);
  }
  return expanded;
}

export function normalizeYoloTagConditions(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.map((row) => {
    let rule_items = [];
    if (Array.isArray(row?.rule_items) && row.rule_items.length) {
      rule_items = expandSizeOrientationRuleItems(row.rule_items);
    } else {
      rule_items = legacyToRuleItems({
        class_contains: row?.class_contains,
        target_labels: row?.target_labels,
        regions: row?.regions,
        size_rules: row?.size_rules,
      });
    }
    const split = splitRuleItems(rule_items);
    return {
      id: String(row?.id || newYoloTagConditionId()),
      enabled: true,
      tag_name: String(row?.tag_name || "").trim(),
      rule_items,
      ...split,
    };
  });
}

function defaultRuleItem(kind) {
  const id = newRuleItemId();
  const collapsed = false;
  if (kind === "class") return { id, kind: "class", text: "", collapsed };
  if (kind === "target") return { id, kind: "target", labels: [], collapsed };
  if (kind === "region") {
    return { id, kind: "region", x1: 0, y1: 0, x2: 0, y2: 0, min_overlap_pct: 50, collapsed };
  }
  if (kind === "orientation") {
    return { id, kind: "orientation", orientation: "horizontal", collapsed };
  }
  return {
    id,
    kind: "size",
    min_width: null,
    max_width: null,
    min_height: null,
    max_height: null,
    collapsed,
  };
}

function buildYoloTagSizeNumField(field) {
  const label = document.createElement("label");
  label.className = "yolo-tag-subfield yolo-tag-subfield--size-num";
  const span = document.createElement("span");
  span.className = "yolo-tag-subfield-label";
  span.textContent = field.label;
  const numWrap = document.createElement("div");
  numWrap.className = "openvla-phase-num-wrap yolo-tag-size-num-wrap";
  const input = document.createElement("input");
  input.type = "number";
  input.min = "0";
  input.step = "1";
  input.inputMode = "numeric";
  input.className = "openvla-phase-num yolo-tag-size-num";
  input.dataset.field = field.key;
  if (field.value != null && field.value !== "") {
    input.value = String(field.value);
  }
  const spinCol = document.createElement("div");
  spinCol.className = "openvla-phase-num-spin";
  for (const [delta, glyph] of [
    [1, "▲"],
    [-1, "▼"],
  ]) {
    const step = document.createElement("button");
    step.type = "button";
    step.className = "openvla-phase-num-step yolo-tag-size-num-step";
    step.dataset.delta = String(delta);
    step.dataset.field = field.key;
    step.setAttribute("aria-label", delta > 0 ? "1 증가" : "1 감소");
    step.textContent = glyph;
    spinCol.appendChild(step);
  }
  numWrap.append(input, spinCol);
  label.append(span, numWrap);
  return label;
}

function bboxArea(xyxy) {
  const x1 = Number(xyxy[0]);
  const y1 = Number(xyxy[1]);
  const x2 = Number(xyxy[2]);
  const y2 = Number(xyxy[3]);
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
}

function intersectionArea(a, b) {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  return Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
}

function ruleItemMatches(detection, item) {
  const kind = String(item?.kind || "").toLowerCase();
  const label = String(detection?.label || "");
  const xyxy = detection?.xyxy;
  if (kind === "class") {
    const text = String(item.text || "").trim();
    if (!text) return true;
    return label.toLowerCase().includes(text.toLowerCase());
  }
  if (kind === "target") {
    const labels = (item.labels || []).map((x) => String(x || "").trim()).filter(Boolean);
    if (!labels.length) return true;
    return labels.includes(label.trim());
  }
  if (!Array.isArray(xyxy) || xyxy.length < 4) return false;
  if (kind === "region") {
    const area = bboxArea(xyxy);
    if (area <= 0) return false;
    const rx1 = Number(item.x1);
    const ry1 = Number(item.y1);
    const rx2 = Number(item.x2);
    const ry2 = Number(item.y2);
    const pct = Math.min(100, Math.max(0, Number(item.min_overlap_pct ?? 50)));
    const overlap = intersectionArea(xyxy, [rx1, ry1, rx2, ry2]);
    return (overlap / area) * 100 + 1e-9 >= pct;
  }
  if (kind === "orientation") {
    const width = Math.abs(Number(xyxy[2]) - Number(xyxy[0]));
    const height = Math.abs(Number(xyxy[3]) - Number(xyxy[1]));
    if (width <= 0 || height <= 0) return false;
    const orient = String(item.orientation || "horizontal").toLowerCase();
    if (orient === "vertical") return height > width;
    return width > height;
  }
  if (kind === "size") {
    const width = Math.abs(Number(xyxy[2]) - Number(xyxy[0]));
    const height = Math.abs(Number(xyxy[3]) - Number(xyxy[1]));
    if (width <= 0 || height <= 0) return false;
    let hasConstraint = false;
    let ok = true;
    for (const key of ["min_width", "max_width", "min_height", "max_height"]) {
      if (item[key] == null || item[key] === "") continue;
      hasConstraint = true;
      const bound = Number(item[key]);
      if (key === "min_width" && width < bound) ok = false;
      if (key === "max_width" && width > bound) ok = false;
      if (key === "min_height" && height < bound) ok = false;
      if (key === "max_height" && height > bound) ok = false;
    }
    if (!hasConstraint) return true;
    return ok;
  }
  return true;
}

function effectiveRuleItems(cond) {
  if (Array.isArray(cond?.rule_items) && cond.rule_items.length) {
    return cond.rule_items;
  }
  const items = [];
  for (const row of cond.class_contains || []) {
    const text = String(row?.text || "").trim();
    if (text) items.push({ kind: "class", text });
  }
  if (cond.target_labels?.length) {
    items.push({ kind: "target", labels: [...cond.target_labels] });
  }
  for (const region of cond.regions || []) {
    items.push({ kind: "region", ...region });
  }
  for (const rule of cond.size_rules || []) {
    const orient = String(rule.orientation || "").toLowerCase();
    const dimKeys = ["min_width", "max_width", "min_height", "max_height"];
    const hasDim = dimKeys.some((key) => rule[key] != null && rule[key] !== "");
    if (orient === "horizontal" || orient === "vertical") {
      items.push({ kind: "orientation", orientation: orient });
    }
    if (hasDim) items.push({ kind: "size", ...rule, orientation: undefined });
  }
  return items;
}

function detectionMatchesCondition(detection, cond) {
  const xyxy = detection?.xyxy;
  if (!Array.isArray(xyxy) || xyxy.length < 4) return false;
  const ruleItems = effectiveRuleItems(cond);
  for (const item of ruleItems) {
    if (!ruleItemMatches(detection, item)) return false;
  }
  return true;
}

export function evaluateYoloTagConditions(detections, conditions) {
  const active = [];
  const seen = new Set();
  for (const cond of conditions || []) {
    if (!cond) continue;
    const tag = String(cond.tag_name || "").trim();
    if (!tag || seen.has(tag)) continue;
    for (const det of detections || []) {
      if (detectionMatchesCondition(det, cond)) {
        active.push(tag);
        seen.add(tag);
        break;
      }
    }
  }
  return active;
}

function setRuleItemCollapsed(block, collapsed) {
  if (!(block instanceof HTMLElement)) return;
  block.classList.toggle("yolo-tag-rule-item--collapsed", collapsed);
  const foldBtn = block.querySelector(".yolo-tag-rule-fold");
  if (foldBtn instanceof HTMLButtonElement) {
    foldBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    foldBtn.title = collapsed ? "펼치기" : "접기";
  }
}

function toggleRuleItemCollapsed(block) {
  if (!(block instanceof HTMLElement)) return;
  const ruleId = block.dataset.ruleId;
  const condId = block.dataset.conditionId;
  const collapsed = !block.classList.contains("yolo-tag-rule-item--collapsed");
  setRuleItemCollapsed(block, collapsed);
  const cond = yoloTagConditions.find((row) => row.id === condId);
  const item = cond?.rule_items?.find((row) => row.id === ruleId);
  if (item) item.collapsed = collapsed;
}

function renderYoloTagActiveChips() {
  const wrap = document.getElementById("yolo-tag-active");
  if (!wrap) return;
  wrap.innerHTML = "";
  if (!yoloActiveTags.length) {
    const empty = document.createElement("span");
    empty.className = "yolo-tag-active-empty";
    empty.textContent = "활성 태그 없음";
    wrap.appendChild(empty);
    return;
  }
  for (const tag of yoloActiveTags) {
    const chip = document.createElement("span");
    chip.className = "yolo-tag-active-chip";
    chip.textContent = tag;
    wrap.appendChild(chip);
  }
}

function ruleItemCollapsed(item) {
  return item?.collapsed !== false;
}

function buildRuleItemElement(condId, item) {
  const collapsed = ruleItemCollapsed(item);
  const block = document.createElement("div");
  block.className = "yolo-tag-rule-item";
  if (collapsed) {
    block.classList.add("yolo-tag-rule-item--collapsed");
  }
  block.dataset.ruleId = item.id;
  block.dataset.kind = item.kind;
  block.dataset.conditionId = condId;

  const titleRow = document.createElement("div");
  titleRow.className = "yolo-tag-rule-item-head";

  const foldBtn = document.createElement("button");
  foldBtn.type = "button";
  foldBtn.className = "yolo-tag-rule-fold";
  foldBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
  foldBtn.title = collapsed ? "펼치기" : "접기";

  const title = document.createElement("button");
  title.type = "button";
  title.className = "yolo-tag-rule-item-title";
  title.textContent = RULE_KIND_LABELS[item.kind] || item.kind;

  const removeRuleBtn = document.createElement("button");
  removeRuleBtn.type = "button";
  removeRuleBtn.className = "yolo-tag-rule-remove";
  removeRuleBtn.dataset.ruleId = item.id;
  removeRuleBtn.dataset.conditionId = condId;
  removeRuleBtn.textContent = "삭제";

  titleRow.appendChild(foldBtn);
  titleRow.appendChild(title);
  titleRow.appendChild(removeRuleBtn);

  const body = document.createElement("div");
  body.className = "yolo-tag-subrow yolo-tag-rule-item-body";

  if (item.kind === "class") {
    const input = document.createElement("input");
    input.type = "text";
    input.className = "yolo-tag-class-text yolo-tag-class-text--solo";
    input.dataset.field = "text";
    input.placeholder = "이름에 포함되는 문자열";
    input.value = item.text || "";
    body.appendChild(input);
  } else if (item.kind === "target") {
    const picks = document.createElement("div");
    picks.className = "yolo-tag-target-picks";
    const selected = new Set(item.labels || []);
    if (!yoloTagClassOptions.length) {
      const empty = document.createElement("span");
      empty.className = "yolo-tag-target-empty";
      empty.textContent = "모델 클래스 로딩 중…";
      picks.appendChild(empty);
    } else {
      for (const row of yoloTagClassOptions) {
        const checkLabel = document.createElement("label");
        checkLabel.className = "yolo-tag-target-check";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.className = "yolo-tag-target-cb";
        cb.value = row.name;
        cb.checked = selected.has(row.name);
        const name = document.createElement("span");
        name.textContent = row.name;
        checkLabel.appendChild(cb);
        checkLabel.appendChild(name);
        picks.appendChild(checkLabel);
      }
    }
    body.appendChild(picks);
    body.classList.add("yolo-tag-rule-item-body--target");
  } else if (item.kind === "region") {
    for (const field of [
      { label: "x1", key: "x1", value: item.x1 },
      { label: "y1", key: "y1", value: item.y1 },
      { label: "x2", key: "x2", value: item.x2 },
      { label: "y2", key: "y2", value: item.y2 },
      { label: "겹침 %", key: "min_overlap_pct", value: item.min_overlap_pct, min: 0, max: 100 },
    ]) {
      const label = document.createElement("label");
      label.className = "yolo-tag-subfield";
      const span = document.createElement("span");
      span.className = "yolo-tag-subfield-label";
      span.textContent = field.label;
      const input = document.createElement("input");
      input.type = "number";
      input.className = `yolo-tag-region-${field.key}`;
      input.dataset.field = field.key;
      if (field.min != null) input.min = String(field.min);
      if (field.max != null) input.max = String(field.max);
      input.value = String(field.value ?? 0);
      label.appendChild(span);
      label.appendChild(input);
      body.appendChild(label);
    }
  } else if (item.kind === "size") {
    body.classList.add("yolo-tag-rule-item-body--size");
    for (const field of [
      { label: "최소 너비", key: "min_width", value: item.min_width },
      { label: "최대 너비", key: "max_width", value: item.max_width },
      { label: "최소 높이", key: "min_height", value: item.min_height },
      { label: "최대 높이", key: "max_height", value: item.max_height },
    ]) {
      body.appendChild(buildYoloTagSizeNumField(field));
    }
  } else if (item.kind === "orientation") {
    const orientWrap = document.createElement("div");
    orientWrap.className = "yolo-tag-orientation-field";
    const orientSelect = document.createElement("select");
    orientSelect.className = "yolo-tag-orientation-select";
    orientSelect.dataset.field = "orientation";
    for (const opt of [
      { value: "horizontal", label: "가로" },
      { value: "vertical", label: "세로" },
    ]) {
      const option = document.createElement("option");
      option.value = opt.value;
      option.textContent = opt.label;
      if ((item.orientation || "horizontal") === opt.value) option.selected = true;
      orientSelect.appendChild(option);
    }
    orientWrap.appendChild(orientSelect);
    body.appendChild(orientWrap);
  }

  block.appendChild(titleRow);
  block.appendChild(body);
  return block;
}

export function renderYoloTagConditionBoard() {
  const board = document.getElementById("yolo-tag-condition-board");
  if (!board) return;
  board.innerHTML = "";
  if (!yoloTagConditions.length) return;
  for (const cond of yoloTagConditions) {
    const card = document.createElement("article");
    card.className = "yolo-tag-condition-row";
    card.dataset.conditionId = cond.id;

    const head = document.createElement("div");
    head.className = "yolo-tag-condition-head";

    const tagInput = document.createElement("input");
    tagInput.type = "text";
    tagInput.className = "yolo-tag-condition-tag-input";
    tagInput.placeholder = "태그";
    tagInput.required = true;
    tagInput.value = cond.tag_name;
    tagInput.dataset.conditionId = cond.id;

    const actions = document.createElement("div");
    actions.className = "yolo-tag-condition-actions";

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "yolo-tag-condition-remove";
    removeBtn.dataset.conditionId = cond.id;
    removeBtn.textContent = "카드 삭제";

    actions.appendChild(removeBtn);

    head.appendChild(tagInput);
    head.appendChild(actions);

    const ruleList = document.createElement("div");
    ruleList.className = "yolo-tag-rule-list";
    ruleList.dataset.conditionId = cond.id;
    for (const item of cond.rule_items || []) {
      ruleList.appendChild(buildRuleItemElement(cond.id, item));
    }

    const addBar = document.createElement("div");
    addBar.className = "yolo-tag-rule-add-bar";

    const kindSelect = document.createElement("select");
    kindSelect.className = "yolo-tag-rule-kind-select";
    kindSelect.dataset.conditionId = cond.id;
    for (const value of RULE_KIND_ORDER) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = RULE_KIND_LABELS[value] || value;
      kindSelect.appendChild(option);
    }

    const ruleAddBtn = document.createElement("button");
    ruleAddBtn.type = "button";
    ruleAddBtn.className = "yolo-tag-rule-add";
    ruleAddBtn.dataset.conditionId = cond.id;
    ruleAddBtn.textContent = "+ 조건 추가";

    addBar.appendChild(kindSelect);
    addBar.appendChild(ruleAddBtn);

    card.appendChild(head);
    card.appendChild(ruleList);
    card.appendChild(addBar);
    board.appendChild(card);
  }
}

function readRuleItemFromDom(block) {
  if (!(block instanceof HTMLElement)) return null;
  const kind = String(block.dataset.kind || "").toLowerCase();
  const id = String(block.dataset.ruleId || newRuleItemId());
  if (kind === "class") {
    const input = block.querySelector(".yolo-tag-class-text");
    return {
      id,
      kind: "class",
      text: input instanceof HTMLInputElement ? input.value.trim() : "",
    };
  }
  if (kind === "target") {
    const labels = [];
    for (const cb of block.querySelectorAll(".yolo-tag-target-cb")) {
      if (cb instanceof HTMLInputElement && cb.checked && cb.value) {
        labels.push(cb.value);
      }
    }
    return { id, kind: "target", labels };
  }
  if (kind === "region") {
    const nums = {};
    for (const input of block.querySelectorAll("input[data-field]")) {
      if (!(input instanceof HTMLInputElement)) continue;
      nums[input.dataset.field] = Number(input.value);
    }
    return {
      id,
      kind: "region",
      x1: nums.x1 ?? 0,
      y1: nums.y1 ?? 0,
      x2: nums.x2 ?? 0,
      y2: nums.y2 ?? 0,
      min_overlap_pct: nums.min_overlap_pct ?? 50,
    };
  }
  if (kind === "size") {
    const rule = { id, kind: "size" };
    for (const input of block.querySelectorAll(".yolo-tag-size-num[data-field]")) {
      if (!(input instanceof HTMLInputElement)) continue;
      const field = input.dataset.field;
      if (!field || input.value.trim() === "") continue;
      rule[field] = Number(input.value);
    }
    rule.min_width = rule.min_width ?? null;
    rule.max_width = rule.max_width ?? null;
    rule.min_height = rule.min_height ?? null;
    rule.max_height = rule.max_height ?? null;
    return rule;
  }
  if (kind === "orientation") {
    const orient = block.querySelector(".yolo-tag-orientation-select");
    const value =
      orient instanceof HTMLSelectElement && orient.value
        ? orient.value
        : "horizontal";
    return {
      id,
      kind: "orientation",
      orientation: value === "vertical" ? "vertical" : "horizontal",
    };
  }
  return null;
}

function readConditionFromDom(condId) {
  const card = document.querySelector(`.yolo-tag-condition-row[data-condition-id="${CSS.escape(condId)}"]`);
  if (!card) return null;
  const tagInput = card.querySelector(".yolo-tag-condition-tag-input");
  const tag_name = tagInput instanceof HTMLInputElement ? tagInput.value.trim() : "";
  const enabled = true;

  const rule_items = [];
  for (const block of card.querySelectorAll(".yolo-tag-rule-item")) {
    const item = readRuleItemFromDom(block);
    if (item) rule_items.push(item);
  }
  const split = splitRuleItems(rule_items);
  return { id: condId, enabled, tag_name, rule_items, ...split };
}

function syncYoloTagConditionsFromDom() {
  const ids = yoloTagConditions.map((c) => c.id);
  const next = [];
  for (const id of ids) {
    const row = readConditionFromDom(id);
    if (row) next.push(row);
  }
  yoloTagConditions = next;
}

function tagConditionsPayload() {
  return yoloTagConditions.map(
    ({
      id,
      enabled,
      tag_name,
      rule_items,
      class_contains,
      target_labels,
      regions,
      size_rules,
    }) => ({
      id,
      enabled,
      tag_name,
      rule_items,
      class_contains,
      target_labels,
      regions,
      size_rules,
    })
  );
}

function tagConditionsForCompare(rows) {
  return (rows || []).map((cond) => ({
    id: cond.id,
    enabled: cond.enabled,
    tag_name: cond.tag_name,
    rule_items: (cond.rule_items || []).map((item) => {
      const copy = { ...item };
      delete copy.collapsed;
      return copy;
    }),
  }));
}

function mergeTagConditionsFromServer(raw) {
  const collapsedByRuleId = new Map();
  for (const cond of yoloTagConditions) {
    for (const item of cond.rule_items || []) {
      if (item?.id) collapsedByRuleId.set(item.id, item.collapsed);
    }
  }
  const next = normalizeYoloTagConditions(raw);
  for (const cond of next) {
    for (const item of cond.rule_items || []) {
      if (item?.id && collapsedByRuleId.has(item.id)) {
        item.collapsed = collapsedByRuleId.get(item.id);
      }
    }
  }
  return next;
}

export function yoloTagConditionsSyncBlocked() {
  return (
    yoloTagConditionEditing ||
    yoloTagConditionDirty ||
    yoloTagConditionSaveInFlight
  );
}

async function flushSaveYoloTagConditions(apiFn) {
  clearTimeout(yoloTagConditionSaveTimer);
  yoloTagConditionSaveTimer = null;
  syncYoloTagConditionsFromDom();
  yoloTagConditionSaveInFlight = true;
  try {
    const cfg = await apiFn("/api/yolo/tag-conditions", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tag_conditions: tagConditionsPayload() }),
    });
    if (cfg && Array.isArray(cfg.tag_conditions)) {
      const merged = mergeTagConditionsFromServer(cfg.tag_conditions);
      const same =
        JSON.stringify(tagConditionsForCompare(merged)) ===
        JSON.stringify(tagConditionsForCompare(yoloTagConditions));
      yoloTagConditions = merged;
      if (!same) {
        renderYoloTagConditionBoard();
      }
    }
    yoloTagConditionDirty = false;
  } catch (err) {
    console.warn("yolo tag conditions save", err);
  } finally {
    yoloTagConditionSaveInFlight = false;
  }
}

function queueSaveYoloTagConditions(apiFn, { immediate = false } = {}) {
  yoloTagConditionDirty = true;
  clearTimeout(yoloTagConditionSaveTimer);
  if (immediate) {
    void flushSaveYoloTagConditions(apiFn);
    return;
  }
  yoloTagConditionSaveTimer = setTimeout(
    () => void flushSaveYoloTagConditions(apiFn),
    YOLO_TAG_CONDITION_SAVE_MS
  );
}

function markYoloTagConditionDirty(apiFn, { immediate = false } = {}) {
  queueSaveYoloTagConditions(apiFn, { immediate });
}

export function bindYoloTagConditionBoard(apiFn) {
  const board = document.getElementById("yolo-tag-condition-board");
  const addBtn = document.getElementById("yolo-tag-condition-add");
  const pane = document.getElementById("yolo-pane-tag-rules");
  if (!board || board.dataset.bound === "1") return;
  board.dataset.bound = "1";

  const markEditing = () => {
    yoloTagConditionEditing = true;
  };
  const unmarkEditing = () => {
    window.setTimeout(() => {
      if (yoloTagConditionDirty || yoloTagConditionSaveInFlight) return;
      const active = document.activeElement;
      if (board.contains(active) || pane?.contains(active)) {
        return;
      }
      yoloTagConditionEditing = false;
    }, 120);
  };
  board.addEventListener("focusin", markEditing);
  board.addEventListener("focusout", unmarkEditing);
  board.addEventListener("mousedown", markEditing);
  pane?.addEventListener("mousedown", markEditing);

  addBtn?.addEventListener("click", () => {
    const id = newYoloTagConditionId();
    const defaultOrient = defaultRuleItem("orientation");
    yoloTagConditions.push({
      id,
      enabled: true,
      tag_name: "",
      rule_items: [defaultOrient],
      class_contains: [],
      regions: [],
      size_rules: [{ orientation: "horizontal" }],
    });
    renderYoloTagConditionBoard();
    const input = board.querySelector(
      `.yolo-tag-condition-tag-input[data-condition-id="${CSS.escape(id)}"]`
    );
    input?.focus();
    markYoloTagConditionDirty(apiFn, { immediate: true });
  });

  board.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.classList.contains("yolo-tag-size-num-step")) {
      event.preventDefault();
      const wrap = target.closest(".yolo-tag-size-num-wrap");
      const input = wrap?.querySelector("input[data-field]");
      const delta = Number(target.dataset.delta);
      if (!(input instanceof HTMLInputElement) || !Number.isFinite(delta)) return;
      const cur = input.value.trim() === "" ? 0 : Number(input.value);
      const base = Number.isFinite(cur) ? cur : 0;
      const next = Math.max(0, base + delta);
      input.value = String(next);
      markYoloTagConditionDirty(apiFn);
      return;
    }
    if (target.classList.contains("yolo-tag-condition-remove")) {
      const condId = target.dataset.conditionId;
      if (!condId) return;
      yoloTagConditions = yoloTagConditions.filter((row) => row.id !== condId);
      renderYoloTagConditionBoard();
      markYoloTagConditionDirty(apiFn, { immediate: true });
      return;
    }
    if (target.classList.contains("yolo-tag-rule-fold") || target.classList.contains("yolo-tag-rule-item-title")) {
      const itemEl = target.closest(".yolo-tag-rule-item");
      if (itemEl) toggleRuleItemCollapsed(itemEl);
      return;
    }
    if (target.classList.contains("yolo-tag-rule-remove")) {
      const condId = target.dataset.conditionId;
      const ruleId = target.dataset.ruleId;
      const card = target.closest(".yolo-tag-condition-row");
      if (!card || !condId || !ruleId) return;
      syncYoloTagConditionsFromDom();
      const cond = yoloTagConditions.find((row) => row.id === condId);
      if (cond) {
        cond.rule_items = (cond.rule_items || []).filter((item) => item.id !== ruleId);
        const split = splitRuleItems(cond.rule_items);
        Object.assign(cond, split);
      }
      const block = card.querySelector(`.yolo-tag-rule-item[data-rule-id="${CSS.escape(ruleId)}"]`);
      block?.remove();
      markYoloTagConditionDirty(apiFn, { immediate: true });
      return;
    }
    if (target.classList.contains("yolo-tag-rule-add")) {
      const condId = target.dataset.conditionId;
      const card = target.closest(".yolo-tag-condition-row");
      if (!card || !condId) return;
      const select = card.querySelector(".yolo-tag-rule-kind-select");
      const kind = select instanceof HTMLSelectElement ? select.value : "class";
      const item = defaultRuleItem(kind);
      const cond = yoloTagConditions.find((row) => row.id === condId);
      if (cond) {
        if (!Array.isArray(cond.rule_items)) cond.rule_items = [];
        cond.rule_items.push(item);
      }
      const list = card.querySelector(".yolo-tag-rule-list");
      const node = buildRuleItemElement(condId, item);
      list?.appendChild(node);
      const split = splitRuleItems(cond.rule_items);
      Object.assign(cond, split);
      const focusInput = node.querySelector("input, select");
      focusInput?.focus();
      markYoloTagConditionDirty(apiFn, { immediate: true });
      return;
    }
  });

  board.addEventListener("change", () => markYoloTagConditionDirty(apiFn));
  board.addEventListener("input", () => markYoloTagConditionDirty(apiFn));
}

export function setYoloTagClassOptions(classes) {
  yoloTagClassOptions = (classes || [])
    .map((row) => ({
      class_id: Number(row.class_id),
      name: String(row.name || "").trim(),
    }))
    .filter((row) => row.name);
  if (!yoloTagConditionsSyncBlocked()) {
    renderYoloTagConditionBoard();
  }
}

export function prepareYoloTagRulesPaneFirstOpen() {
  yoloTagRulesInitialCollapsePending = true;
}

function collapseAllYoloTagRuleItems() {
  for (const cond of yoloTagConditions) {
    for (const item of cond.rule_items || []) {
      item.collapsed = true;
    }
  }
}

export function applyYoloTagConditionsConfig(rawConditions) {
  if (yoloTagConditionsSyncBlocked()) return;
  const next = normalizeYoloTagConditions(rawConditions);
  const unchanged =
    JSON.stringify(tagConditionsForCompare(next)) ===
    JSON.stringify(tagConditionsForCompare(yoloTagConditions));
  if (unchanged && !yoloTagRulesInitialCollapsePending) return;
  yoloTagConditions = next;
  if (yoloTagRulesInitialCollapsePending) {
    collapseAllYoloTagRuleItems();
    yoloTagRulesInitialCollapsePending = false;
  }
  renderYoloTagConditionBoard();
}

export function yoloBoxOrientation(xyxy) {
  const width = Math.abs(Number(xyxy[2]) - Number(xyxy[0]));
  const height = Math.abs(Number(xyxy[3]) - Number(xyxy[1]));
  if (width <= 0 || height <= 0) return "—";
  if (width > height) return "가로";
  if (height > width) return "세로";
  return "정사각";
}

export function updateYoloActiveTagsFromDetections(detections, serverTags) {
  if (Array.isArray(serverTags)) {
    yoloActiveTags = serverTags;
  } else {
    yoloActiveTags = evaluateYoloTagConditions(detections, yoloTagConditions);
  }
  renderYoloTagActiveChips();
}

export function initYoloTagConditions(apiFn) {
  bindYoloTagConditionBoard(apiFn);
}