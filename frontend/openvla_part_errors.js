/** OpenVLA phase joint error display (target pose vs inferred joints). */

export const OPENVLA_PHASE_ERR_PARTS = [
  ["left_arm", "error_left_arm"],
  ["right_arm", "error_right_arm"],
  ["left_hand", "error_left_hand"],
  ["right_hand", "error_right_hand"],
];

export function openvlaPhaseErrElementId(prefix, part) {
  return `${prefix}${part.replaceAll("_", "-")}`;
}

export function openvlaErrorTextColor(value) {
  const t = Math.min(1, Math.max(0, value) / 1.5);
  const hue = Math.round(142 * (1 - t));
  return `hsl(${hue} 70% 58%)`;
}

export function setOpenvlaPartErrorMetric(el, value, digits = 4) {
  if (!(el instanceof HTMLElement)) return;
  el.innerHTML = "";
  if (value == null || !Number.isFinite(value)) {
    el.textContent = "—";
    el.style.color = "";
    return;
  }
  el.style.color = openvlaErrorTextColor(value);
  const num = document.createElement("span");
  num.className = "openvla-metric-num";
  num.textContent = value.toFixed(digits);
  const unit = document.createElement("span");
  unit.className = "openvla-metric-unit";
  unit.textContent = " rad";
  el.appendChild(num);
  el.appendChild(unit);
}

export function openvlaPartHasModelOutput(v, part) {
  const names = v?.inferred_joints_by_part?.[part];
  return Array.isArray(names) && names.length > 0;
}

export function clearOpenvlaPhasePartErrors(elementIdPrefix) {
  for (const [part] of OPENVLA_PHASE_ERR_PARTS) {
    const el = document.getElementById(openvlaPhaseErrElementId(elementIdPrefix, part));
    if (el) {
      el.textContent = "—";
      el.style.color = "";
    }
  }
}

export function renderOpenvlaPhasePartErrorValues(
  v,
  match,
  hasResult,
  elementIdPrefix
) {
  const pose = hasResult ? match : null;
  for (const [part, key] of OPENVLA_PHASE_ERR_PARTS) {
    const el = document.getElementById(openvlaPhaseErrElementId(elementIdPrefix, part));
    const available = hasResult && openvlaPartHasModelOutput(v, part);
    const raw = available && pose?.[key] != null ? Number(pose[key]) : null;
    setOpenvlaPartErrorMetric(el, raw);
  }
}