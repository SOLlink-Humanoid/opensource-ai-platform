/** Shared OpenVLA “result phase” formatting (1-based / next phase labels). */

const PHASE_TAIL_SCALE = 10;

export function parsePhaseNumber(raw) {
  if (raw == null) return null;
  const text = String(raw).trim().replace(/\u2212/g, "-");
  if (!text) return null;
  const n = Number(text);
  if (!Number.isFinite(n)) return null;
  return Math.round(n * 10) / 10;
}

/** Model tail ×10, then the nearest phase number (clamped to the profile list). */
export function phaseNumberFromModelTail(raw, rows) {
  const text = String(raw ?? "").trim().replace(/\u2212/g, "-");
  if (!text) return null;
  const n = Number(text);
  if (!Number.isFinite(n)) return null;
  const scaled = n * PHASE_TAIL_SCALE;
  const rounded = Math.round(scaled);
  if (rounded < 0 || Math.abs(rounded) >= 64) return rounded < 0 ? rounded : null;
  const list = rows ?? [];
  const count = list.length;
  if (count > 0) {
    const byNumber = list.find(
      (row) =>
        Math.abs(Number(row.number) - rounded) < 0.051 ||
        Math.abs(Number(row.number) - scaled) < 0.051
    );
    if (byNumber != null && Number.isFinite(Number(byNumber.number))) {
      return Math.round(Number(byNumber.number) * 10) / 10;
    }
    const idx = Math.min(count, Math.max(1, rounded)) - 1;
    const fromRow = Number(list[idx]?.number);
    if (Number.isFinite(fromRow)) return Math.round(fromRow * 10) / 10;
    return idx + 1;
  }
  return rounded;
}

export function openvlaPhaseRoundedNumber(v, rows) {
  const action = v.predicted_action ?? [];
  if (action.length) {
    const fromTail = phaseNumberFromModelTail(action[action.length - 1], rows);
    if (fromTail != null) return fromTail;
  }
  for (const raw of [v.phase_text, v.predicted_phase?.raw_text]) {
    const rounded = parsePhaseNumber(raw);
    if (rounded != null) return rounded;
  }
  const idx = v.predicted_phase?.index ?? v.phase_index;
  if (idx != null && idx !== "") {
    const n = Number(idx);
    if (Number.isFinite(n) && n >= 0) return n + 1;
  }
  return null;
}

export function openvlaPhaseRawSignals(v) {
  const signals = [];
  const push = (raw) => {
    if (raw == null) return;
    const text = String(raw).trim();
    if (text) signals.push(text);
  };
  if (v.phase_index != null && v.phase_index !== "") {
    push(v.phase_index);
  }
  push(v.phase_text);
  push(v.predicted_phase?.raw_text);
  const action = v.predicted_action ?? [];
  for (let i = action.length - 1; i >= 0; i -= 1) {
    if (typeof action[i] === "string") {
      push(action[i]);
      break;
    }
  }
  for (let i = action.length - 1; i >= 0; i -= 1) {
    if (typeof action[i] === "number" && Number.isFinite(action[i])) {
      push(action[i]);
      break;
    }
  }
  return signals;
}

export function openvlaPhaseRowForDisplayNumber(phaseNum, rows) {
  const list = rows ?? [];
  if (!list.length) return null;
  const n = Number(phaseNum);
  if (!Number.isFinite(n)) return null;
  const match = list.find((row) => {
    const rn = Number(row.number);
    return Number.isFinite(rn) && Math.abs(rn - n) < 0.051;
  });
  if (match) return match;
  const oneBased = Math.round(n);
  if (oneBased >= 1 && oneBased <= list.length) {
    return list[oneBased - 1];
  }
  if (n >= 0 && n < list.length) return list[n];
  return null;
}

export function openvlaPhaseNameForNumber(phaseNum, v, rows) {
  const row = openvlaPhaseRowForDisplayNumber(phaseNum, rows);
  const fromBoard = row && String(row.description || "").trim();
  if (fromBoard) return fromBoard;
  const apiIndex = v?.predicted_phase?.index ?? v?.phase_index;
  if (
    apiIndex != null &&
    (apiIndex === phaseNum || apiIndex + 1 === Number(phaseNum))
  ) {
    const fromApi = String(
      v?.predicted_phase?.description || v?.phase_description || ""
    ).trim();
    if (fromApi) return fromApi;
  }
  return null;
}

export function formatOpenvlaResultPhaseText(v, rows) {
  const rounded = openvlaPhaseRoundedNumber(v, rows);
  if (rounded == null) return null;
  if (rounded < 0) return { error: true, text: "Error" };
  const name = openvlaPhaseNameForNumber(rounded, v, rows);
  if (name) return { error: false, text: `${rounded} - ${name}` };
  return { error: false, text: `${rounded} - (unknown phase)` };
}

export function openvlaHasResultPhase(v) {
  return Boolean(
    v?.ready &&
      (Boolean(v.predicted_action?.length) ||
        v.phase_index != null ||
        v.phase_text != null ||
        v.predicted_phase?.raw_text != null)
  );
}

export function openvlaPoseNameForDisplay(v, rows) {
  const rounded = openvlaPhaseRoundedNumber(v, rows);
  if (rounded == null) return "";
  const row = openvlaPhaseRowForDisplayNumber(rounded, rows);
  const fromProfile = row && String(row.pose_name || "").trim();
  if (fromProfile) return fromProfile;
  return String(v?.predicted_phase?.pose_name || "").trim();
}

export function applyOpenvlaResultPhaseDisplay(el, v, rows, hasResult) {
  if (!(el instanceof HTMLElement)) return;
  el.classList.remove("openvla-phase-error");
  if (!hasResult) {
    el.textContent = "—";
    return;
  }
  const formatted = formatOpenvlaResultPhaseText(v, rows);
  if (formatted) {
    if (formatted.error) {
      el.textContent = "Error";
      el.classList.add("openvla-phase-error");
    } else {
      el.textContent = formatted.text;
    }
    return;
  }
  if (openvlaPhaseRawSignals(v).length === 0) {
    el.textContent = "—";
    return;
  }
  el.textContent = "Error";
  el.classList.add("openvla-phase-error");
}