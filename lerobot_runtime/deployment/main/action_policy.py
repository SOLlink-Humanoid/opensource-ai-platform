#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Policy inference adapter for LeRobot ACT checkpoint deployment.

The deployment code keeps the old PolicyAgent public API, while this module
loads policy weights and processor pipelines using the LeRobot processor layout:

    config.json
    model.safetensors
    policy_preprocessor.json
    policy_postprocessor.json

The imports are intentionally lazy/fallback-based so the same deployment file
can run on Python 3.10 LeRobot 0.4.x and Python 3.12 LeRobot 0.5.x layouts.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from pathlib import Path
import sys
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from depth_colormap import depth_to_turbo_rgb_from_config, unpack_camera_config
from vendored_lerobot import setup_vendored_lerobot

setup_vendored_lerobot()

try:
    from lerobot.utils.constants import (
        OBS_STATE,
        POLICY_POSTPROCESSOR_DEFAULT_NAME,
        POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
except ImportError:
    OBS_STATE = "observation.state"
    POLICY_PREPROCESSOR_DEFAULT_NAME = "policy_preprocessor"
    POLICY_POSTPROCESSOR_DEFAULT_NAME = "policy_postprocessor"


def _get_pretrained_config_class():
    try:
        from lerobot.configs import PreTrainedConfig
    except ImportError:
        from lerobot.configs.policies import PreTrainedConfig
    return PreTrainedConfig


def _install_act_only_policies_namespace() -> None:
    """Avoid importing optional policy packages when only ACT is needed.

    LeRobot 0.4.x executes ``lerobot.policies.__init__`` before importing
    submodules. That ``__init__`` imports optional policies such as GROOT, which
    can require heavy dependencies unrelated to ACT deployment. Installing a
    lightweight package module with the same search path lets Python import
    ``lerobot.policies.act.*`` and ``lerobot.policies.utils`` directly.
    """
    if "lerobot.policies" in sys.modules:
        return

    import lerobot

    lerobot_file = getattr(lerobot, "__file__", None)
    if lerobot_file is None:
        return

    policies_dir = Path(lerobot_file).resolve().parent / "policies"
    if not policies_dir.exists():
        return

    policies_pkg = types.ModuleType("lerobot.policies")
    policies_pkg.__file__ = str(policies_dir / "__init__.py")
    policies_pkg.__path__ = [str(policies_dir)]
    policies_pkg.__package__ = "lerobot.policies"
    sys.modules["lerobot.policies"] = policies_pkg
    setattr(lerobot, "policies", policies_pkg)


def _install_lightweight_processor_namespace() -> None:
    """Avoid importing tokenizer/robot processors when loading ACT processors."""
    if "lerobot.processor" in sys.modules:
        return

    import lerobot

    lerobot_file = getattr(lerobot, "__file__", None)
    if lerobot_file is None:
        return

    processor_dir = Path(lerobot_file).resolve().parent / "processor"
    if not processor_dir.exists():
        return

    processor_pkg = types.ModuleType("lerobot.processor")
    processor_pkg.__file__ = str(processor_dir / "__init__.py")
    processor_pkg.__path__ = [str(processor_dir)]
    processor_pkg.__package__ = "lerobot.processor"
    sys.modules["lerobot.processor"] = processor_pkg
    setattr(lerobot, "processor", processor_pkg)

    # Re-export the lightweight type aliases expected by lerobot.policies.utils
    # without executing lerobot.processor.__init__, which imports optional
    # tokenizer processors.
    from lerobot.processor.core import (  # type: ignore[import-not-found]
        EnvAction,
        EnvTransition,
        PolicyAction,
        RobotAction,
        RobotObservation,
        TransitionKey,
    )

    processor_pkg.EnvAction = EnvAction
    processor_pkg.EnvTransition = EnvTransition
    processor_pkg.PolicyAction = PolicyAction
    processor_pkg.RobotAction = RobotAction
    processor_pkg.RobotObservation = RobotObservation
    processor_pkg.TransitionKey = TransitionKey


def _install_training_config_stub() -> None:
    """PreTrainedPolicy imports TrainPipelineConfig, but deployment never uses it."""
    if "lerobot.configs.train" in sys.modules:
        return

    train_mod = types.ModuleType("lerobot.configs.train")

    class TrainPipelineConfig:
        def save_pretrained(self, *args, **kwargs):
            raise RuntimeError("TrainPipelineConfig is unavailable in ACT deployment mode.")

    train_mod.TrainPipelineConfig = TrainPipelineConfig
    sys.modules["lerobot.configs.train"] = train_mod


def _install_lerobot_act_deploy_shims() -> None:
    _install_act_only_policies_namespace()
    _install_lightweight_processor_namespace()
    _install_training_config_stub()


def _get_act_policy_class():
    _install_lerobot_act_deploy_shims()
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import LeRobot ACTPolicy. This deployment adapter only supports ACT, "
            "so optional policies should not be required. Check the missing ACT dependency in "
            "the original traceback."
        ) from exc
    return ACTPolicy


def _ensure_act_config_registered() -> None:
    """Import ACTConfig before decoding a polymorphic policy config."""
    _install_lerobot_act_deploy_shims()
    try:
        from lerobot.policies.act.configuration_act import ACTConfig  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import LeRobot ACTConfig before loading config.json."
        ) from exc


def _load_pretrained_processors(config, model_path: Path, device: torch.device):
    _install_lerobot_act_deploy_shims()
    try:
        # Import only processor pieces needed by default ACT checkpoints. Importing
        # lerobot.processor.__init__ pulls tokenizer processors and optional
        # transformers symbols that are not needed for ACT deployment.
        import lerobot.processor.batch_processor  # noqa: F401
        import lerobot.processor.device_processor  # noqa: F401
        import lerobot.processor.normalize_processor  # noqa: F401
        import lerobot.processor.rename_processor  # noqa: F401
        from lerobot.processor.pipeline import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import LeRobot processor pipeline. Check that minimal processor "
            "dependencies are installed for the selected LeRobot version."
        ) from exc

    device_override = {"device": str(device)}
    return (
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            overrides={"device_processor": device_override},
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        ),
        PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            overrides={"device_processor": {"device": "cpu"}},
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def _prepare_observation_for_inference(raw_obs, device):
    _install_lerobot_act_deploy_shims()
    try:
        from lerobot.policies.utils import prepare_observation_for_inference
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import LeRobot prepare_observation_for_inference. Check that the "
            "selected LeRobot version and its minimal inference dependencies are installed."
        ) from exc
    return prepare_observation_for_inference(
        raw_obs,
        device,
        task="",
        robot_type="humanoid_upper_body",
    )


class PolicyAgent:
    """Load a LeRobot ACT policy and run deployment inference."""

    CAMERA_CONFIGS: Tuple[Tuple[str, str, Tuple[int, int], bool], ...] = (
        ("camera_top", "observation.images.camera_top", (640, 360), False),
        ("camera_head", "observation.images.camera_head", (640, 360), False),
        ("camera_wrist", "observation.images.camera_wrist", (320, 240), False),
    )

    def __init__(
        self,
        model_path: str,
        camera_configs: Optional[
            List[Tuple[str, str, Optional[Tuple[int, int]], bool]]
        ] = None,
        profile_inference: bool = False,
        temporal_ensemble_device: str = "model",
        cudnn_benchmark: bool = False,
        allow_tf32: bool = False,
        fp16_autocast: bool = False,
    ):
        startup_started = time.perf_counter()
        phase_started = startup_started
        self.model_path = Path(model_path).expanduser().resolve()
        self.camera_configs = (
            tuple(camera_configs) if camera_configs is not None else self.CAMERA_CONFIGS
        )
        self.device = self._get_device()
        device_ms = (time.perf_counter() - phase_started) * 1000.0
        self.profile_inference = profile_inference
        self.cudnn_benchmark = cudnn_benchmark
        self.allow_tf32 = allow_tf32
        self.fp16_autocast = fp16_autocast and self.device.type == "cuda"
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = self.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
            print(f"cuDNN benchmark: {self.cudnn_benchmark}")
            print(f"TF32: {self.allow_tf32}")
            print(f"FP16 autocast: {self.fp16_autocast}")
        if temporal_ensemble_device not in {"model", "cpu"}:
            raise ValueError(
                "temporal_ensemble_device must be 'model' or 'cpu'"
            )
        self.temporal_ensemble_device = temporal_ensemble_device
        self.last_inference_timings_ms: Dict[str, float] = {}
        phase_started = time.perf_counter()
        self.config = self._load_config()
        config_ms = (time.perf_counter() - phase_started) * 1000.0
        phase_started = time.perf_counter()
        self.policy = self._load_policy()
        weights_ms = (time.perf_counter() - phase_started) * 1000.0
        phase_started = time.perf_counter()
        self.preprocessor, self.postprocessor = self._load_processors()
        processors_ms = (time.perf_counter() - phase_started) * 1000.0
        self.cnt = 0
        print(
            "Policy startup timing | "
            f"device={device_ms:.1f}ms config={config_ms:.1f}ms "
            f"model_and_weights={weights_ms:.1f}ms processors={processors_ms:.1f}ms "
            f"total={(time.perf_counter() - startup_started) * 1000.0:.1f}ms"
        )

    def _profile_sync(self) -> None:
        if self.profile_inference and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _autocast_context(self):
        if self.fp16_autocast:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available. Device set to:", device)
        else:
            device = torch.device("cpu")
            print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
        return device

    def _require_latest_checkpoint_files(self) -> None:
        required_files = [
            "config.json",
            "model.safetensors",
            f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        ]
        missing = [name for name in required_files if not (self.model_path / name).exists()]
        if missing:
            raise FileNotFoundError(
                "Latest LeRobot policy checkpoint is incomplete. "
                f"Missing files in {self.model_path}: {missing}. "
                "Expected a latest-format checkpoint with config.json, model.safetensors, "
                "policy_preprocessor.json, and policy_postprocessor.json. "
                "For older LeRobot checkpoints, run the latest LeRobot migration script first."
            )

    def _load_config(self):
        self._require_latest_checkpoint_files()
        print(f"Loading policy config from: {self.model_path}")
        _ensure_act_config_registered()
        PreTrainedConfig = _get_pretrained_config_class()
        config = PreTrainedConfig.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        if config.type != "act":
            raise ValueError(
                f"This deployment adapter currently supports latest LeRobot ACT checkpoints only; "
                f"got policy type '{config.type}'."
            )
        config.device = str(self.device)
        return config

    def _load_policy(self):
        print(f"Loading policy weights from: {self.model_path}")
        policy_cls = _get_act_policy_class()
        load_config = copy.copy(self.config)
        if hasattr(load_config, "pretrained_backbone_weights"):
            load_config.pretrained_backbone_weights = None
        policy = policy_cls.from_pretrained(
            self.model_path,
            config=load_config,
            local_files_only=True,
        )
        policy.eval()
        policy.to(self.device)
        return policy

    @staticmethod
    def _feature_shape(feature) -> Optional[Tuple[int, ...]]:
        if feature is None:
            return None
        shape = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", None)
        if shape is None:
            return None
        return tuple(int(value) for value in shape)

    def validate_deployment_layout(
        self,
        expected_state_dim: int,
        expected_action_dim: int,
        expected_images: Dict[str, Optional[Tuple[int, int, int]]],
        unused_trailing_action_dim: int = 0,
    ) -> None:
        input_features = getattr(self.config, "input_features", {}) or {}
        output_features = getattr(self.config, "output_features", {}) or {}
        state_feature = getattr(self.config, "robot_state_feature", None)
        if state_feature is None:
            state_feature = input_features.get(OBS_STATE)
        action_feature = getattr(self.config, "action_feature", None)
        if action_feature is None:
            action_feature = output_features.get("action")

        state_shape = self._feature_shape(state_feature)
        action_shape = self._feature_shape(action_feature)
        if state_shape is None or not state_shape:
            raise ValueError("ACT config has no observation.state feature shape")
        if action_shape is None or not action_shape:
            raise ValueError("ACT config has no action feature shape")
        if state_shape[0] != expected_state_dim:
            raise ValueError(
                f"ACT state dimension mismatch: model={state_shape[0]}, "
                f"deployment={expected_state_dim}"
            )
        commanded_action_dim = expected_action_dim + int(unused_trailing_action_dim)
        if action_shape[0] != commanded_action_dim:
            raise ValueError(
                f"ACT action dimension mismatch: model={action_shape[0]}, "
                f"deployment={expected_action_dim}"
                + (
                    f"+unused_trailing={unused_trailing_action_dim}"
                    if unused_trailing_action_dim
                    else ""
                )
            )

        model_images = getattr(self.config, "image_features", None)
        if model_images is None:
            model_images = {
                key: feature
                for key, feature in input_features.items()
                if str(key).startswith("observation.images.")
            }
        model_keys = set(model_images)
        expected_keys = set(expected_images)
        if model_keys != expected_keys:
            raise ValueError(
                "ACT image key mismatch: "
                f"model={sorted(model_keys)}, deployment={sorted(expected_keys)}"
            )
        for key, expected_shape in expected_images.items():
            if expected_shape is None:
                continue
            model_shape = self._feature_shape(model_images[key])
            if model_shape is not None and model_shape != expected_shape:
                raise ValueError(
                    f"ACT image shape mismatch for {key}: "
                    f"model={model_shape}, deployment={expected_shape}"
                )

    def _load_processors(self):
        return _load_pretrained_processors(self.config, self.model_path, self.device)

    def reset(self):
        """Reset policy and processor episode state."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()
        if hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()

    def generate_obs(self) -> Dict[str, Any]:
        """Generate simulated observation data for smoke tests."""
        return {
            "images": {
                "camera_head": (np.random.rand(360, 640, 3) * 255).astype(np.uint8),
                "camera_wrist": (np.random.rand(240, 320, 3) * 255).astype(np.uint8),
            },
            "qpos": np.random.randn(16),
            "arm_gripper_joints": np.random.randn(16),
        }

    def inference(self, obs: Optional[Dict[str, Any]]):
        """Return one postprocessed action tensor on CPU."""
        if obs is None:
            print("Using simulated observation data")
            obs = self.generate_obs()

        input_data = self.prepare_inference_obs(obs)
        with torch.inference_mode():
            processed_obs = self.preprocessor(input_data)
            with self._autocast_context():
                action = self.policy.select_action(processed_obs)
            if action.is_floating_point() and action.dtype != torch.float32:
                action = action.float()
            action = self.postprocessor(action)
        return action.detach().cpu()

    def inference_batch(self, obs: Optional[Dict[str, Any]], n_consumed=None) -> List[np.ndarray]:
        """Return a raw ACT prediction chunk without temporal ensembling."""
        if obs is None:
            print("Using simulated observation data")
            obs = self.generate_obs()

        total_start = time.perf_counter()
        stage_start = total_start
        input_data = self.prepare_inference_obs(obs)
        self._profile_sync()
        prepare_ms = (time.perf_counter() - stage_start) * 1000.0
        with torch.inference_mode():
            stage_start = time.perf_counter()
            processed_obs = self.preprocessor(input_data)
            self._profile_sync()
            preprocess_ms = (time.perf_counter() - stage_start) * 1000.0
            stage_start = time.perf_counter()
            with self._autocast_context():
                actions = self.policy.predict_action_chunk(
                    processed_obs,
                    profile=self.profile_inference,
                )
            if actions.is_floating_point() and actions.dtype != torch.float32:
                actions = actions.float()
            self._profile_sync()
            model_ms = (time.perf_counter() - stage_start) * 1000.0
            model_details = getattr(
                self.policy.model, "last_profile_timings_ms", {}
            )
            stage_start = time.perf_counter()
            actions = self.postprocessor(actions)
            self._profile_sync()
            postprocess_ms = (time.perf_counter() - stage_start) * 1000.0
            stage_start = time.perf_counter()
            actions = actions.detach().cpu()
            transfer_ms = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        result = self._action_chunk_to_numpy(actions)
        numpy_ms = (time.perf_counter() - stage_start) * 1000.0
        self.last_inference_timings_ms = {
            "prepare": prepare_ms,
            "preprocess": preprocess_ms,
            "model": model_ms,
            "postprocess": postprocess_ms,
            "transfer": transfer_ms,
            "numpy": numpy_ms,
            "total": (time.perf_counter() - total_start) * 1000.0,
            **{
                key: float(value)
                for key, value in model_details.items()
            },
        }
        return result

    def inference_temporal_ensemble_batch(
        self,
        obs: Optional[Dict[str, Any]],
        n_consumed: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Return a temporally ensembled action batch using consumed-step alignment."""
        if self.config.temporal_ensemble_coeff is None:
            raise ValueError(
                "temporal_ensemble_batch requires config.temporal_ensemble_coeff"
            )
        if not hasattr(self.policy, "select_actions_batch"):
            raise RuntimeError(
                "Loaded ACTPolicy does not provide select_actions_batch. "
                "Use this repository's patched lerobot package."
            )
        if obs is None:
            print("Using simulated observation data")
            obs = self.generate_obs()

        total_start = time.perf_counter()
        stage_start = total_start
        input_data = self.prepare_inference_obs(obs)
        self._profile_sync()
        prepare_ms = (time.perf_counter() - stage_start) * 1000.0
        with torch.inference_mode():
            stage_start = time.perf_counter()
            processed_obs = self.preprocessor(input_data)
            self._profile_sync()
            preprocess_ms = (time.perf_counter() - stage_start) * 1000.0
            with self._autocast_context():
                actions = self.policy.select_actions_batch(
                    processed_obs,
                    n_consumed=n_consumed,
                    ensemble_device=self.temporal_ensemble_device,
                    profile=self.profile_inference,
                )
            policy_timings = getattr(self.policy, "last_profile_timings_ms", {})
            stage_start = time.perf_counter()
            actions = self.postprocessor(actions)
            self._profile_sync()
            postprocess_ms = (time.perf_counter() - stage_start) * 1000.0
            stage_start = time.perf_counter()
            actions = actions.detach().cpu()
            transfer_ms = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        result = self._action_chunk_to_numpy(actions)
        numpy_ms = (time.perf_counter() - stage_start) * 1000.0
        self.last_inference_timings_ms = {
            "prepare": prepare_ms,
            "preprocess": preprocess_ms,
            "model": float(policy_timings.get("model", 0.0)),
            "ensemble_transfer": float(
                policy_timings.get("ensemble_transfer", 0.0)
            ),
            "ensemble": float(policy_timings.get("ensemble", 0.0)),
            "postprocess": postprocess_ms,
            "transfer": transfer_ms,
            "numpy": numpy_ms,
            "total": (time.perf_counter() - total_start) * 1000.0,
            **{
                key: float(value)
                for key, value in policy_timings.items()
                if key not in {"model", "ensemble"}
            },
        }
        return result

    def _action_chunk_to_numpy(self, actions: torch.Tensor) -> List[np.ndarray]:
        if actions.ndim == 3:
            actions = actions.squeeze(0)
        elif actions.ndim == 1:
            actions = actions.unsqueeze(0)
        return [self._to_numpy_action(action) for action in actions]

    def prepare_inference_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert deployment observations to latest LeRobot inference input."""
        raw_obs: Dict[str, Any] = {}
        images = obs.get("images", {})
        cameras_processed = 0

        for config in self.camera_configs:
            camera_key, obs_key, target_size, is_depth, depth_preprocess = (
                unpack_camera_config(config)
            )
            if camera_key not in images:
                continue
            image = self._prepare_image(
                images[camera_key], target_size, is_depth, depth_preprocess
            )
            raw_obs[obs_key] = image
            cameras_processed += 1

        if cameras_processed == 0:
            raise KeyError(f"No valid camera found in obs['images']. Available keys: {list(images.keys())}")

        qpos = np.asarray(obs["arm_gripper_joints"], dtype=np.float32)
        raw_obs[OBS_STATE] = qpos
        self.cnt += 1

        return _prepare_observation_for_inference(raw_obs, self.device)

    def _prepare_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]],
        is_depth: bool = False,
        depth_preprocess: Any = None,
    ) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            image = np.asarray(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if image.ndim != 3:
            raise ValueError(f"Camera image must be HWC, got shape {image.shape}")
        metric_depth = is_depth and image.shape[2] == 1
        if target_size is not None:
            width, height = target_size
            if image.shape[1] != width or image.shape[0] != height:
                interpolation = cv2.INTER_NEAREST if metric_depth else cv2.INTER_LINEAR
                image = cv2.resize(image, dsize=target_size, interpolation=interpolation)
                if image.ndim == 2:
                    image = image[:, :, np.newaxis]
        if depth_preprocess is not None and metric_depth:
            return depth_to_turbo_rgb_from_config(image, depth_preprocess)
        if is_depth and image.dtype == np.uint16:
            image = image.astype(np.float32) * (255.0 / 65535.0)
        if is_depth and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image

    def _to_numpy_action(self, action: torch.Tensor) -> np.ndarray:
        action = action.detach().cpu()
        if action.ndim > 1:
            action = action.squeeze(0)
        return action.numpy()
