"""Onboard LeRobot ACT: shared policy template + checkpoints under models/LeRobot."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from . import hand_pose_map

WORK = Path("/tmp/oap_lerobot_policy")
PID_FILE = WORK / "run.pid"
PROCESS_PID_FILE = WORK / "process.pid"
LOG_FILE = WORK / "run.log"
SCRIPT_FILE = WORK / "run.sh"
POLICY_FILE = WORK / "current_policy"
ACTIVE_POLICY_FILE = WORK / "active_policy.json"
TMUX_SESSION = "oap_lerobot_policy"
SYSTEMD_UNIT = "oap-lerobot-policy.service"

REMOTE_PID_FILE_WAIT_ITER = 200
POLICY_STARTUP_ALIVE_SECONDS = 30.0
POLICY_STARTUP_POLL_SECONDS = 0.5
POLICY_STARTUP_FAILURE_LOG_LINES = 500
LOG_FLUSH_BEFORE_TAIL_SECONDS = 1.0

_PLATFORM_ROOT = Path(__file__).resolve().parents[2]
LEROBOT_ACTION_LOG_FILE = _PLATFORM_ROOT / "lerobot_action.log"
_DEFAULT_LEROBOT_RUNTIME = _PLATFORM_ROOT / "lerobot_runtime"
_DEFAULT_MODELS_ROOT = _PLATFORM_ROOT / "models" / "LeRobot"
_DEFAULT_POLICY_TEMPLATE = "agibot_g2_act.json"
_DEFAULT_ROBOT_CONFIG = "deployment/configs/robot/agibot_g2_opensource_ai.json"
_DEFAULT_GENERATOR = "deployment/main/general_policy_action_generator_agibot_g2.py"

_LEGACY_BWR_ROOT = "/home/agi/bwr-lerobot"
_LEGACY_POLICY_DIR = "bwr/deployment/workflow/action/policy_action/opensource-ai"


def platform_root() -> Path:
    raw = (
        os.environ.get("OAP_PLATFORM_ROOT")
        or os.environ.get("OAP_VLA_ROOT")
        or ""
    ).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _PLATFORM_ROOT


def lerobot_runtime_root() -> Path:
    raw = (
        os.environ.get("OAP_LEROBOT_ROOT")
        or os.environ.get("OAP_BWR_ROOT")
        or ""
    ).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    env_rel = os.environ.get("OAP_LEROBOT_RUNTIME", "").strip()
    if env_rel:
        path = Path(env_rel).expanduser()
        if not path.is_absolute():
            path = platform_root() / path
        return path.resolve()
    return _DEFAULT_LEROBOT_RUNTIME


def models_root() -> Path:
    raw = (
        os.environ.get("OAP_LEROBOT_MODELS_DIR")
        or os.environ.get("OAP_LEROBOT_MODELS_ROOT")
        or ""
    ).strip()
    if raw:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = platform_root() / path
        return path.resolve()
    return _DEFAULT_MODELS_ROOT


def policy_template_relative() -> str:
    return (
        os.environ.get("OAP_LEROBOT_POLICY_TEMPLATE")
        or _DEFAULT_POLICY_TEMPLATE
    ).strip().lstrip("/")


def policy_template_path() -> Path:
    return lerobot_runtime_root() / policy_template_relative()


def _default_activate(runtime_root: Path) -> str:
    lr = shlex.quote(str(runtime_root))
    py_path = (
        f"{lr}/deployment/main:"
        f"{lr}/deployment"
    )
    return "\n".join(
        [
            "source /home/agi/app/env.sh",
            "source /home/agi/venv-g2/bin/activate",
            "source /opt/ros/kilted/setup.bash",
            f'export PYTHONPATH="{py_path}:${{PYTHONPATH:-}}"',
            "pkill -f 'agibot_g2.*teleop' || true",
            "pkill -f g2_bridge || true",
        ]
    )


def deployment_config() -> dict[str, str]:
    runtime = lerobot_runtime_root()
    cfg: dict[str, str] = {
        "root": str(runtime),
        "platform_root": str(platform_root()),
        "models_dir": str(models_root()),
        "policy_template": policy_template_relative(),
        "robot_config": (
            os.environ.get("OAP_LEROBOT_ROBOT_CONFIG")
            or os.environ.get("OAP_BWR_ROBOT_CONFIG")
            or _DEFAULT_ROBOT_CONFIG
        ).strip(),
        "generator": _DEFAULT_GENERATOR,
        "activate": os.environ.get("OAP_LEROBOT_ACTIVATE", "").strip()
        or _default_activate(runtime),
    }
    return cfg


_MODEL_LABEL_RE = re.compile(r"^run_simple_(.+)_\d{8}$", re.IGNORECASE)


def model_display_name(folder_name: str) -> str:
    match = _MODEL_LABEL_RE.match(folder_name)
    if match:
        return match.group(1).replace("_", " ")
    return folder_name.replace("_", " ")


def task_description_for_model(folder_name: str) -> str:
    label = model_display_name(folder_name)
    return f"pick and place into the {label}"


def is_lerobot_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    name = path.name
    if name.startswith(".") or name == "README.txt":
        return False
    return (path / "config.json").is_file() and (path / "model.safetensors").is_file()


def load_policy_template() -> dict[str, Any]:
    path = policy_template_path()
    if not path.is_file():
        raise ValueError(f"공통 정책 JSON이 없습니다: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"공통 정책 JSON을 읽지 못했습니다: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"공통 정책 JSON 루트가 객체가 아닙니다: {path}")
    from shared.action_layout import apply_to_lerobot_policy

    return apply_to_lerobot_policy(data)


def _normalize_shape(raw: Any) -> list[int] | None:
    if not isinstance(raw, (list, tuple)):
        return None
    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            return None
    return out


def _feature_shape(features: dict[str, Any], key: str) -> list[int] | None:
    entry = features.get(key)
    if not isinstance(entry, dict):
        return None
    return _normalize_shape(entry.get("shape"))


def model_contract_from_template() -> dict[str, Any]:
    contract = load_policy_template().get("oap_model_contract")
    if isinstance(contract, dict):
        return contract
    return {}


def validate_checkpoint_against_contract(
    model_dir: Path,
) -> tuple[bool, str | None]:
    """Compare checkpoint config.json with agibot_g2_act.json oap_model_contract."""
    cfg_path = model_dir / "config.json"
    try:
        model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"config.json 읽기 실패: {exc}"

    contract = model_contract_from_template()
    expected = contract.get("input_features")
    if not isinstance(expected, dict) or not expected:
        return True, None

    input_feats = model_cfg.get("input_features") or {}
    output_feats = model_cfg.get("output_features") or {}
    mismatches: list[str] = []
    for feat_key, exp_raw in expected.items():
        exp_shape = _normalize_shape(exp_raw)
        if exp_shape is None:
            continue
        if feat_key == "action":
            got = _feature_shape(output_feats, "action")
        else:
            got = _feature_shape(input_feats, feat_key)
        if got != exp_shape:
            got_txt = got if got is not None else "없음"
            mismatches.append(f"{feat_key}: 기대 {exp_shape}, 체크포인트 {got_txt}")

    if mismatches:
        summary = contract.get("state_order_summary") or "oap_model_contract"
        return False, f"{summary} — " + "; ".join(mismatches)
    return True, None


def run_defaults() -> dict[str, Any]:
    """Default runtime/action fields from agibot_g2_act.json (UI + run overrides)."""
    template = load_policy_template()
    runtime = dict(template.get("runtime") or {})
    action = template.get("action") or {}
    gripper = action.get("gripper_threshold", 0.5)
    arm_pp = dict((runtime.get("arm_action_postprocess") or {}))
    motion_speed = float(runtime.get("motion_speed", 0.012))
    return {
        "runtime": {
            "publish_hz": float(runtime.get("publish_hz", 2.5)),
            "inference_hz": float(runtime.get("inference_hz", 0.5)),
            "max_publish_hz": float(runtime.get("max_publish_hz", 2.5)),
            "warming_up_time": float(runtime.get("warming_up_time", 0.3)),
            "min_run_time_before_complete": float(
                runtime.get("min_run_time_before_complete", 160.0)
            ),
            "max_image_age_s": float(runtime.get("max_image_age_s", 1.0)),
            "reinfer_threshold": int(runtime.get("reinfer_threshold", -1)),
            "action_chunk_execute_steps": runtime.get("action_chunk_execute_steps"),
            "clear_action_queue_on_reinfer": bool(
                runtime.get("clear_action_queue_on_reinfer", False)
            ),
            "motion_speed": motion_speed,
            "publish_consume": dict(runtime.get("publish_consume") or {}),
            "arm_action_postprocess": {
                "enabled": bool(arm_pp.get("enabled", True)),
                "min_cutoff": float(arm_pp.get("min_cutoff", 0.4)),
                "beta": float(arm_pp.get("beta", 0.01)),
                "derivative_cutoff": float(arm_pp.get("derivative_cutoff", 0.4)),
            },
        },
        "action": {"gripper_threshold": float(gripper)},
        "cli": {
            "motion_speed": motion_speed,
            "timeout": -1.0,
            "skip_warmup": False,
        },
    }


def _merge_policy_overrides(
    policy: dict[str, Any],
    *,
    runtime_overrides: dict[str, Any] | None,
    action_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    if runtime_overrides:
        rt = policy.setdefault("runtime", {})
        for key, value in runtime_overrides.items():
            if key == "arm_action_postprocess" and isinstance(value, dict):
                arm = rt.setdefault("arm_action_postprocess", {})
                if isinstance(arm, dict):
                    arm.update(value)
                else:
                    rt[key] = value
            elif key == "publish_consume" and isinstance(value, dict):
                pc = rt.setdefault("publish_consume", {})
                if isinstance(pc, dict):
                    pc.update(value)
                else:
                    rt[key] = value
            else:
                rt[key] = value
    if action_overrides:
        policy.setdefault("action", {}).update(action_overrides)
    _clamp_runtime_hz_ceiling(policy)
    return policy


def _clamp_runtime_hz_ceiling(policy: dict[str, Any]) -> None:
    """Never run faster than agibot_g2_act.json caps (slower is OK)."""
    template_rt = load_policy_template().get("runtime") or {}
    rt = policy.setdefault("runtime", {})
    try:
        cap_pub = float(rt.get("max_publish_hz", template_rt.get("max_publish_hz", 0)))
    except (TypeError, ValueError):
        cap_pub = 0.0
    if cap_pub <= 0:
        try:
            cap_pub = float(template_rt.get("publish_hz", 2.5))
        except (TypeError, ValueError):
            cap_pub = 2.5
    try:
        base_inf = float(template_rt.get("inference_hz", 0.5))
    except (TypeError, ValueError):
        base_inf = 0.5
    try:
        base_pub = float(template_rt.get("publish_hz", 2.5))
    except (TypeError, ValueError):
        base_pub = 2.5
    cap_inf = base_inf if base_pub <= 0 else base_inf * (cap_pub / base_pub)

    for key, cap in (("publish_hz", cap_pub), ("inference_hz", cap_inf)):
        if key not in rt:
            continue
        try:
            value = float(rt[key])
        except (TypeError, ValueError):
            continue
        if value > cap + 1e-9:
            rt[key] = cap


def merge_policy_for_model(model_dir: Path) -> dict[str, Any]:
    if not is_lerobot_checkpoint(model_dir):
        raise ValueError(f"유효한 LeRobot 체크포인트가 아닙니다: {model_dir}")
    ok, reason = validate_checkpoint_against_contract(model_dir)
    if not ok:
        raise ValueError(
            f"체크포인트가 공통 정책(agibot_g2_act.json)과 맞지 않습니다. {reason}"
        )
    policy = load_policy_template()
    policy["name"] = model_dir.name
    policy["model_path"] = str(model_dir.resolve())
    policy["task_description"] = task_description_for_model(model_dir.name)
    return policy


def write_active_policy(
    model_id: str,
    *,
    runtime_overrides: dict[str, Any] | None = None,
    action_overrides: dict[str, Any] | None = None,
) -> Path:
    model_dir = models_root() / model_id
    merged = merge_policy_for_model(model_dir)
    _merge_policy_overrides(
        merged,
        runtime_overrides=runtime_overrides,
        action_overrides=action_overrides,
    )
    hand_table = hand_pose_map.table_for_policy(model_id)
    if hand_table is not None:
        merged["hand_key_poses"] = hand_table
        merged["hand_pose_transition_sec"] = 1.0
    WORK.mkdir(parents=True, exist_ok=True)
    ACTIVE_POLICY_FILE.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return ACTIVE_POLICY_FILE


def resolve_model_id(selection: str) -> str:
    """API selection: checkpoint folder name under models/LeRobot."""
    token = (selection or "").strip().strip("/")
    if not token:
        raise ValueError("모델을 선택하세요")
    if ".." in token.split("/"):
        raise ValueError("잘못된 모델 경로입니다")
    # Legacy: relative policy JSON path → use parent folder name if under models/LeRobot
    if token.endswith(".json"):
        raise ValueError(
            "정책 JSON 경로 대신 models/LeRobot 아래 체크포인트 폴더 이름을 선택하세요 "
            f"(예: run_simple_small_empty_box_20261111). 받은 값: {token!r}"
        )
    return token.split("/")[-1]


def build_run_script(
    bwr: dict[str, str],
    policy_path: str,
    *,
    skip_warmup: bool = False,
    timeout: float | None = None,
) -> str:
    root = (bwr.get("root") or "").strip()
    if not root.startswith("/"):
        raise ValueError("lerobot runtime root 는 절대경로여야 합니다")
    generator = (bwr.get("generator") or "").strip()
    robot_config = (bwr.get("robot_config") or "").strip()
    policy = (policy_path or "").strip()
    if not generator or not robot_config or not policy:
        raise ValueError("generator / robot_config / policy 경로가 없습니다")
    activate = (bwr.get("activate") or "").strip()
    pid_path = shlex.quote(str(PROCESS_PID_FILE))
    lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        f"cd {shlex.quote(root)}",
    ]
    if activate:
        lines.append("set +e")
        lines.extend(activate.splitlines())
        lines.append("set -eo pipefail")
    lines.append("export PYTHONUNBUFFERED=1")
    lines.append("export OAP_LEROBOT_STATE_FILE=/tmp/oap_lerobot_live.json")
    lines.append(f"echo $$ > {pid_path}")
    lines.append("echo '[oap] starting python'")
    cli_tail: list[str] = []
    if timeout is not None:
        cli_tail.append(f"--timeout {float(timeout)}")
    if skip_warmup:
        cli_tail.append("--skip-warmup")
    extra = f" {' '.join(cli_tail)}" if cli_tail else ""
    lines.append(
        "exec python3 -u "
        f"{shlex.quote(generator)} --robot {shlex.quote(robot_config)} "
        f"--policy {shlex.quote(policy)}{extra}"
    )
    return "\n".join(lines) + "\n"


def display_command(
    bwr: dict[str, str],
    policy_path: str,
    *,
    skip_warmup: bool = False,
    timeout: float | None = None,
) -> str:
    return build_run_script(
        bwr,
        policy_path,
        skip_warmup=skip_warmup,
        timeout=timeout,
    ).strip()


def _run_shell(command: str, timeout: int = 60, check: bool = True) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "명령 실패").strip()
        raise ValueError(err)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _list_legacy_bwr_policies(deployment: dict[str, str]) -> list[dict[str, Any]]:
    """Optional fallback when OAP_LEROBOT_LEGACY_BWR=1."""
    root = (deployment.get("root") or "").strip()
    if root != _LEGACY_BWR_ROOT and not os.environ.get("OAP_LEROBOT_LEGACY_BWR"):
        return []
    legacy_root = Path(
        os.environ.get("OAP_BWR_ROOT", _LEGACY_BWR_ROOT)
    ).expanduser()
    sub = _LEGACY_POLICY_DIR.strip("/")
    folder = legacy_root / sub
    if not folder.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        path = folder / name
        label = path.stem
        model = ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if data.get("name"):
                    label = str(data["name"])
                model = str(data.get("model_path") or "").strip()
        except (OSError, json.JSONDecodeError):
            pass
        if model and not os.path.isabs(model):
            model = os.path.normpath(str(path.parent / model))
        items.append(
            {
                "id": f"{sub}/{name}",
                "name": label,
                "path": f"{sub}/{name}",
                "model_path": model,
                "model_exists": bool(model and os.path.exists(model)),
                "kind": "legacy_policy_json",
            }
        )
    return items


def list_policies(deployment: dict[str, str] | None = None) -> list[dict[str, Any]]:
    _ = deployment
    root = models_root()
    if not root.is_dir():
        raise ValueError(f"LeRobot 모델 폴더가 없습니다: {root}")
    template_ok = policy_template_path().is_file()
    if not template_ok:
        raise ValueError(f"공통 정책 템플릿이 없습니다: {policy_template_path()}")

    items: list[dict[str, Any]] = []
    for name in sorted(os.listdir(root)):
        model_dir = root / name
        if not is_lerobot_checkpoint(model_dir):
            continue
        compatible, incompatible_reason = validate_checkpoint_against_contract(model_dir)
        contract = model_contract_from_template()
        items.append(
            {
                "id": name,
                "name": model_display_name(name),
                "path": name,
                "model_path": str(model_dir.resolve()),
                "model_exists": True,
                "model_compatible": compatible,
                "incompatible_reason": incompatible_reason,
                "expected_state_dim": contract.get("state_dim"),
                "expected_action_dim": contract.get("action_dim"),
                "kind": "checkpoint",
                "policy_template": policy_template_relative(),
            }
        )
    if not items:
        legacy = _list_legacy_bwr_policies(deployment or deployment_config())
        if legacy:
            return legacy
        raise ValueError(
            f"models/LeRobot 에 config.json + model.safetensors 가 있는 체크포인트가 없습니다: {root}"
        )
    return items


def ensure_model(bwr: dict[str, str], model_id: str) -> str:
    _ = bwr
    token = resolve_model_id(model_id)
    model_dir = models_root() / token
    if not is_lerobot_checkpoint(model_dir):
        raise ValueError(f"모델이 없거나 불완전합니다: {model_dir}")
    ok, reason = validate_checkpoint_against_contract(model_dir)
    if not ok:
        raise ValueError(
            f"모델이 공통 정책(agibot_g2_act.json)과 호환되지 않습니다. {reason}"
        )
    return str(model_dir.resolve())


def _read_policy_line_from_file() -> str:
    try:
        return POLICY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def policy_status() -> dict[str, Any]:
    if PROCESS_PID_FILE.is_file():
        pid_s = PROCESS_PID_FILE.read_text(encoding="utf-8").strip()
        if pid_s.isdigit() and _pid_alive(int(pid_s)):
            return {
                "running": True,
                "pid": pid_s,
                "policy": _read_policy_line_from_file(),
            }
    try:
        _, out, _ = _run_shell(
            (
                f"if command -v tmux >/dev/null 2>&1 && tmux has-session -t {shlex.quote(TMUX_SESSION)} 2>/dev/null; then "
                f"pid=$(tmux list-panes -t {shlex.quote(TMUX_SESSION)} -F '#{{pane_pid}}' 2>/dev/null | head -n 1); "
                'echo "RUNNING:${pid:-tmux}"; '
                f"cat {shlex.quote(str(POLICY_FILE))} 2>/dev/null || true; "
                "else echo STOPPED; fi"
            ),
            timeout=10,
            check=False,
        )
        lines = out.strip().splitlines()
        if lines and lines[0].startswith("RUNNING:"):
            return {
                "running": True,
                "pid": lines[0].split(":", 1)[1].strip(),
                "policy": lines[1].strip() if len(lines) > 1 else "",
            }
    except (ValueError, subprocess.TimeoutExpired):
        pass
    if PID_FILE.is_file():
        pid_s = PID_FILE.read_text(encoding="utf-8").strip()
        if pid_s.isdigit() and _pid_alive(int(pid_s)):
            return {
                "running": True,
                "pid": pid_s,
                "policy": _read_policy_line_from_file(),
            }
    return {"running": False, "pid": None, "policy": ""}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_policy(
    model_id: str,
    deployment: dict[str, str] | None = None,
    *,
    runtime_overrides: dict[str, Any] | None = None,
    action_overrides: dict[str, Any] | None = None,
    skip_warmup: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    bwr = deployment or deployment_config()
    status = policy_status()
    if status.get("running"):
        raise ValueError(f"이미 정책이 실행 중입니다 (pid {status.get('pid')})")
    token = resolve_model_id(model_id)
    ensure_model(bwr, token)
    active_policy = write_active_policy(
        token,
        runtime_overrides=runtime_overrides,
        action_overrides=action_overrides,
    )
    script = build_run_script(
        bwr,
        str(active_policy),
        skip_warmup=skip_warmup,
        timeout=timeout if timeout is not None else -1.0,
    )
    WORK.mkdir(parents=True, exist_ok=True)
    SCRIPT_FILE.write_text(script, encoding="utf-8")
    SCRIPT_FILE.chmod(0o755)
    POLICY_FILE.write_text(token, encoding="utf-8")
    PROCESS_PID_FILE.unlink(missing_ok=True)
    header = (
        "===== 실행 =====\n"
        + script
        + "\n\n===== 로그 =====\n"
    )
    LOG_FILE.write_text(header, encoding="utf-8")
    log_q = shlex.quote(str(LOG_FILE))
    run_in_tmux = f"bash {shlex.quote(str(SCRIPT_FILE))} >> {log_q} 2>&1"
    command = (
        f"if command -v tmux >/dev/null 2>&1; then "
        f"tmux kill-session -t {shlex.quote(TMUX_SESSION)} 2>/dev/null || true; "
        "pgrep -f 'general_policy_action_generator_agibot_g2\\.py' | xargs -r kill -INT 2>/dev/null || true; "
        "pgrep -f 'general_policy_action_generator\\.py' | xargs -r kill -INT 2>/dev/null || true; "
        "sleep 0.3; "
        f"tmux new-session -d -s {shlex.quote(TMUX_SESSION)} {shlex.quote(run_in_tmux)}; "
        f"i=0; while [ ! -s {shlex.quote(str(PROCESS_PID_FILE))} ] && [ \"$i\" -lt {REMOTE_PID_FILE_WAIT_ITER} ]; do sleep 0.1; i=$((i+1)); done; "
        f"if [ ! -s {shlex.quote(str(PROCESS_PID_FILE))} ]; then "
        "echo '정책 프로세스가 PID를 쓰지 못했습니다' >&2; exit 1; fi; "
        f"pid=$(cat {shlex.quote(str(PROCESS_PID_FILE))}); "
        f"echo \"$pid\" > {shlex.quote(str(PID_FILE))}; "
        'echo "$pid"; '
        "else "
        f"nohup setsid bash {shlex.quote(str(SCRIPT_FILE))} >> {log_q} 2>&1 < /dev/null & "
        f"i=0; while [ ! -s {shlex.quote(str(PROCESS_PID_FILE))} ] && [ \"$i\" -lt {REMOTE_PID_FILE_WAIT_ITER} ]; do sleep 0.1; i=$((i+1)); done; "
        f"if [ ! -s {shlex.quote(str(PROCESS_PID_FILE))} ]; then "
        "echo '정책 프로세스가 PID를 쓰지 못했습니다' >&2; exit 1; fi; "
        f"pid=$(cat {shlex.quote(str(PROCESS_PID_FILE))}); "
        f"echo \"$pid\" > {shlex.quote(str(PID_FILE))}; echo \"$pid\"; "
        "fi"
    )
    _, stdout, _ = _run_shell(command, timeout=60)
    pid = stdout.strip().splitlines()[-1].strip() if stdout.strip() else ""
    deadline = time.time() + POLICY_STARTUP_ALIVE_SECONDS
    alive = False
    while time.time() < deadline:
        if policy_status().get("running"):
            alive = True
            break
        time.sleep(POLICY_STARTUP_POLL_SECONDS)
    if not alive:
        time.sleep(LOG_FLUSH_BEFORE_TAIL_SECONDS)
        log = policy_log(POLICY_STARTUP_FAILURE_LOG_LINES).strip()
        marker = "[oap] starting python"
        legacy = "[ms] starting python"
        if (marker in log or legacy in log) and "General policy" not in log and "Traceback" not in log:
            log += (
                "\n\n힌트: Python이 로그를 남기기 전에 종료되었습니다. "
                "10초 후 재시도하거나 tail -f /tmp/oap_lerobot_policy/run.log 로 확인하세요."
            )
        raise ValueError("정책이 바로 종료되었습니다.\n" + log)
    return {
        "ok": True,
        "pid": pid,
        "policy": token,
        "model_path": str((models_root() / token).resolve()),
        "active_policy_file": str(active_policy),
        "command": display_command(bwr, str(active_policy)),
        "log_path": str(LOG_FILE),
    }


def stop_policy() -> dict[str, Any]:
    pid_path = shlex.quote(str(PROCESS_PID_FILE))
    rid_path = shlex.quote(str(PID_FILE))
    log_q = shlex.quote(str(LOG_FILE))
    command = (
        f"pid=$(cat {pid_path} 2>/dev/null || cat {rid_path} 2>/dev/null || true); "
        "pgid=''; "
        "if [ -n \"$pid\" ]; then pgid=$(ps -o pgid= -p \"$pid\" 2>/dev/null | tr -d ' '); fi; "
        f"if command -v tmux >/dev/null 2>&1 && tmux has-session -t {shlex.quote(TMUX_SESSION)} 2>/dev/null; then "
        f"tmux send-keys -t {shlex.quote(TMUX_SESSION)} C-c; "
        "fi; "
        "if [ -n \"$pgid\" ]; then kill -INT -- \"-$pgid\" 2>/dev/null || true; "
        "elif [ -n \"$pid\" ]; then kill -INT \"$pid\" 2>/dev/null || true; "
        "fi; "
        "sleep 0.5; "
        f"tmux kill-session -t {shlex.quote(TMUX_SESSION)} 2>/dev/null || true; "
        f"rm -f {rid_path} {pid_path} {shlex.quote(str(POLICY_FILE))}; "
        f"printf '\\n===== 중지 완료 =====\\n' >> {log_q}; "
        "echo STOPPED"
    )
    _, stdout, _ = _run_shell(command, timeout=20)
    if stdout.strip().startswith("STILL_RUNNING"):
        raise ValueError("정책 프로세스를 중지하지 못했습니다")
    return {
        "ok": True,
        "running": False,
        "message": "정책 프로세스가 중지되었습니다.",
    }


def read_live_policy_targets() -> dict[str, Any]:
    """Latest LeRobot publish targets written to OAP_LEROBOT_STATE_FILE."""
    import os

    path = Path(
        os.environ.get("OAP_LEROBOT_STATE_FILE", "/tmp/oap_lerobot_live.json")
    )
    if not path.is_file():
        return {"ok": True, "available": False}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "available": False, "error": str(exc)}
    if not isinstance(raw, dict):
        return {"ok": True, "available": False}
    return {"ok": True, "available": True, "state": raw}


def lerobot_action_log_tail(lines: int = 300) -> dict[str, Any]:
    n = max(10, min(int(lines), 5000))
    path = LEROBOT_ACTION_LOG_FILE
    if not path.is_file():
        return {"ok": True, "path": str(path), "log": ""}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"ok": True, "path": str(path), "log": ""}
    log_lines = text.splitlines()
    if len(log_lines) > n:
        log_lines = log_lines[-n:]
    return {"ok": True, "path": str(path), "log": "\n".join(log_lines) + ("\n" if log_lines else "")}


def policy_log(lines: int = 2000) -> str:
    n = max(10, min(int(lines), 20000))
    if not LOG_FILE.is_file():
        return ""
    try:
        text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    log_lines = text.splitlines()
    if len(log_lines) <= n:
        return text if text.endswith("\n") else text + ("\n" if text else "")
    return "\n".join(log_lines[-n:]) + "\n"
