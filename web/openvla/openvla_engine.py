"""OpenVLA loader. Prefers the bundled action regression head when present."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from shared.action_layout import load_action_layout

from .action_parser import phase_text_from_action_tail
from .openvla_offline import ensure_custom_code, ensure_model_root

_log = logging.getLogger(__name__)

_PROMPT = "In: What action should the robot take to {instruction}?\nOut:"
_G2_UNNORM_KEY = "agibot_g2_full_body"
_PROMPT_MAX_LENGTH = 512


def build_inference_prompt(
    instruction: str,
    *,
    unnorm_key: str,
    regression_head: bool,
) -> str:
    """G2 regression / full-body SFT uses the raw instruction string."""
    text = instruction.strip()
    if regression_head or unnorm_key == _G2_UNNORM_KEY:
        return text
    return _PROMPT.format(instruction=text)


def pool_last_token(hidden_states, attention_mask):
    """Pool the last non-pad token, matching openvla-model-factory/train/regression.py."""
    import torch

    if attention_mask is None:
        return hidden_states[:, -1, :]
    last_idx = attention_mask.long().sum(dim=1) - 1
    last_idx = last_idx.clamp(min=0)
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_idx, last_idx, :]


@dataclass(frozen=True)
class OpenVLAInference:
    joint_values: list[float]
    phase_text: str | None


def _regression_action_dim(model_dir: Path, head_path: Path) -> int:
    cfg_path = model_dir / "action_regression_config.json"
    if cfg_path.is_file():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            dim = int(raw.get("action_dim", 0))
            if dim > 0:
                return dim
        except (OSError, ValueError, TypeError) as exc:
            _log.warning("Could not parse %s: %s", cfg_path.name, exc)
    from safetensors import safe_open

    with safe_open(str(head_path), framework="pt") as tensors:
        for key in tensors.keys():
            if key.endswith("weight") and tensors.get_tensor(key).ndim == 2:
                return int(tensors.get_tensor(key).shape[0])
    return load_action_layout().openvla_dim


class _ActionHead:
    def __init__(
        self,
        path: Path,
        *,
        action_dim: int,
        hidden_size: int = 4096,
        head_layers: int = 2,
    ):
        import torch
        from torch import nn

        self._torch = torch
        self.action_dim = action_dim
        if head_layers < 1:
            raise ValueError("head_layers must be >= 1")
        if head_layers == 1:
            self.net: nn.Module = nn.Linear(hidden_size, action_dim)
        else:
            layers: list[nn.Module] = []
            for _ in range(head_layers - 1):
                layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
            layers.append(nn.Linear(hidden_size, action_dim))
            self.net = nn.Sequential(*layers)
        from safetensors.torch import load_file

        state = load_file(str(path))
        self.net.load_state_dict(state)
        self.net.eval()

    def to(self, device: str):
        self.net.to(device=device, dtype=self._torch.float32)
        return self

    def __call__(self, hidden):
        return self.net(hidden.float())


class OpenVLAEngine:
    def __init__(
        self,
        root: Path | None,
        *,
        default_unnorm_key: str,
        mock: bool,
        device_preference: str,
    ) -> None:
        self._root = Path(root) if root is not None else None
        self._unnorm_key = default_unnorm_key
        self._mock = mock
        self._preference = device_preference or "cpu"
        self._model = None
        self._processor = None
        self._head: _ActionHead | None = None
        self._loaded_version: str | None = None
        self.device = device_preference or "cpu"
        self.last_error: str | None = None
        if not mock and self._root is not None:
            try:
                ensure_model_root(self._root)
            except FileNotFoundError as exc:
                _log.warning("OpenVLA custom code bootstrap: %s", exc)

    @property
    def ready(self) -> bool:
        return self._mock or self._model is not None

    @property
    def loaded_version(self) -> str | None:
        return self._loaded_version

    def _resolve_device(self) -> str:
        import torch

        pref = self._preference
        if pref == "cuda" and torch.cuda.is_available():
            return "cuda"
        if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if pref not in {"cpu", "cuda", "mps"}:
            return "cpu"
        if pref == "cuda":
            _log.warning("CUDA requested but torch.cuda.is_available() is false; using CPU")
        return "cpu" if pref == "cuda" else pref

    def _version_dir(self, version: str) -> Path:
        if self._root is None:
            raise FileNotFoundError("OPENVLA_MODEL_ROOT is not set")
        model_dir = self._root / version / "model"
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"OpenVLA checkpoint not found: {model_dir}")
        return model_dir

    def load(self, version: str) -> None:
        if self._mock:
            self._loaded_version = version
            return
        if self._loaded_version == version and self._model is not None:
            return
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        model_dir = self._version_dir(version)
        ensure_custom_code(model_dir)
        self.device = self._resolve_device()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        _log.info("Loading OpenVLA %s on %s (%s)", version, self.device, dtype)
        local_only = True
        processor = AutoProcessor.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=local_only,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_only,
        )
        model.to(self.device)
        model.eval()
        head_path = model_dir / "action_regression_head.safetensors"
        head = None
        if head_path.is_file():
            hidden_size = 4096
            head_layers = 2
            cfg_path = model_dir / "action_regression_config.json"
            if cfg_path.is_file():
                try:
                    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                    hidden_size = int(raw.get("hidden_size", hidden_size))
                    head_layers = int(raw.get("head_layers", head_layers))
                except (OSError, ValueError, TypeError):
                    pass
            action_dim = _regression_action_dim(model_dir, head_path)
            head = _ActionHead(
                head_path,
                action_dim=action_dim,
                hidden_size=hidden_size,
                head_layers=head_layers,
            ).to(self.device)
            _log.info(
                "Loaded action regression head (%s, action_dim=%s, head_layers=%s)",
                head_path.name,
                action_dim,
                head_layers,
            )
        self._processor = processor
        self._model = model
        self._head = head
        self._loaded_version = version
        self.last_error = None

    def warmup(self, version: str, *, unnorm_key: str | None = None) -> None:
        self.load(version)
        if self._mock:
            return
        image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        self.predict(image, "stay still", version=version, unnorm_key=unnorm_key or self._unnorm_key)

    def predict(
        self,
        image: Image.Image,
        instruction: str,
        *,
        version: str,
        unnorm_key: str | None = None,
    ) -> OpenVLAInference:
        if self._mock:
            return OpenVLAInference(
                joint_values=[0.0] * load_action_layout().openvla_dim,
                phase_text=None,
            )
        self.load(version)
        assert self._model is not None and self._processor is not None
        import torch

        unnorm_key = unnorm_key or self._unnorm_key
        use_regression = self._head is not None
        prompt = build_inference_prompt(
            instruction,
            unnorm_key=unnorm_key,
            regression_head=use_regression,
        )
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=_PROMPT_MAX_LENGTH,
        )
        if not use_regression:
            input_ids = inputs["input_ids"]
            if not torch.all(input_ids[:, -1] == 29871):
                empty = torch.tensor([[29871]], dtype=input_ids.dtype)
                inputs["input_ids"] = torch.cat((input_ids, empty), dim=1)
                if "attention_mask" in inputs:
                    mask = inputs["attention_mask"]
                    extra = torch.ones((mask.shape[0], 1), dtype=mask.dtype)
                    inputs["attention_mask"] = torch.cat((mask, extra), dim=1)
        model_dtype = next(self._model.parameters()).dtype
        moved = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved[key] = value
                continue
            if value.dtype.is_floating_point:
                moved[key] = value.to(self.device, dtype=model_dtype)
            else:
                moved[key] = value.to(self.device)

        with torch.inference_mode():
            if self._head is not None:
                outputs = self._model(
                    input_ids=moved.get("input_ids"),
                    attention_mask=moved.get("attention_mask"),
                    pixel_values=moved.get("pixel_values"),
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = outputs.hidden_states[-1]
                pooled = pool_last_token(hidden, moved.get("attention_mask"))
                action = self._head(pooled)[0].float().cpu().tolist()
                values = [float(v) for v in action]
                layout = load_action_layout()
                phase = (
                    phase_text_from_action_tail(values)
                    if len(values) == layout.openvla_dim
                    else None
                )
                return OpenVLAInference(joint_values=values, phase_text=phase)

            key = unnorm_key or self._unnorm_key
            try:
                actions = self._model.predict_action(**moved, unnorm_key=key, do_sample=False)
            except AssertionError:
                actions = self._model.predict_action(**moved, unnorm_key=None, do_sample=False)
        values = [float(v) for v in actions.tolist()]
        return OpenVLAInference(joint_values=values, phase_text=None)
