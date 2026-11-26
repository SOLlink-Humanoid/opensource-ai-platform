#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgiBot G2 전용 정책 generator.

공용 general_policy_action_generator 를 상속만 하고 파일은 건드리지 않는다.
G2 는 ROS 카메라 토픽이 없어서 세 가지만 갈아끼운다:

  1. _setup_camera_interfaces  ROS 구독 대신 GDK 카메라 폴링 스레드를 띄운다.
  2. warm_up                   공용 구현은 walker_s2 에서만 관측을 기다린다.
                               G2 는 GDK 스트림이 붙는 데 시간이 걸리므로 기다린다.
  3. cv_bridge                 공용 constructor 가 ROS Image 변환용 CvBridge 를
                               무조건 import 한다. G2 는 이미지를 GDK NumPy 로
                               받으므로, 패키지가 없으면 stub 만 넣는다.

관절 상태 읽기와 서보 명령은 adapters/robot/agibot_g2.py 가 전부 처리하므로
이 파일에는 GDK 호출이 없다.

카메라 폴링은 robot config 의 topics.cameras.<COMPONENT>.gdk_camera_id 로 대상을
정하고, 받은 BGR 프레임을 공용 _store_camera_image 에 그대로 넘긴다. 리사이즈,
크롭, shape 검증, 수신 시각 기록은 공용 구현이 이미 하고 있어 다시 만들지 않는다.

로봇 온보드에서만 동작한다. 실행 전 source /home/agi/app/env.sh 가 필요하다.

워크플로에서는 정책 JSON 의 generator_class 로 이 클래스를 고른다:

    "generator_class": "general_policy_action_generator_agibot_g2:AgibotG2PolicyActionGenerator"

사용법 (워크플로 없이 단독 실행):
  python3 general_policy_action_generator_agibot_g2.py \
      --robot ../configs/robot/agibot_g2.json \
      --policy ../workflow/action/policy_action/default/agibot_g2_dual_arm_example.json
"""
import sys
import threading
import time
import types
from pathlib import Path
from typing import List

_DEPLOYMENT_DIR = Path(__file__).resolve().parent.parent
_MAIN_DIR = Path(__file__).resolve().parent
for _path in (_MAIN_DIR, _DEPLOYMENT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from adapters.robot import RobotComponent
from general_policy_action_generator import GeneralPolicyActionGenerator


def _ensure_cv_bridge_importable() -> None:
    """Shared constructor imports CvBridge even though G2 never converts ROS images."""
    try:
        import cv_bridge  # noqa: F401
        return
    except ImportError:
        pass

    stub = types.ModuleType("cv_bridge")

    class CvBridge:
        def imgmsg_to_cv2(self, *args, **kwargs):
            raise RuntimeError("agibotG2 does not convert ROS Image messages")

    stub.CvBridge = CvBridge
    sys.modules["cv_bridge"] = stub


class AgibotG2PolicyActionGenerator(GeneralPolicyActionGenerator):
    """GDK 카메라를 폴링해서 공용 추론 파이프라인에 넣는 G2 정책 generator."""

    # 카메라 폴링이 연속 실패할 때 로그를 남기는 최소 간격(초)
    CAMERA_ERROR_LOG_PERIOD_S = 5.0

    def __init__(self, *args, **kwargs):
        _ensure_cv_bridge_importable()
        super().__init__(*args, **kwargs)

    def _setup_camera_interfaces(self):
        if not self.image_specs:
            self.get_logger().warning("No camera specs in the policy config")
            return

        cameras = self.robot_config["topics"]["cameras"]
        poll_hz = float(self.robot_config.get("gdk", {}).get("camera_poll_hz", 30.0))
        self._g2_camera_stop = threading.Event()
        self._g2_camera_threads: List[threading.Thread] = []
        self._g2_camera_last_error = {}

        self.robot_io.prepare_cameras()
        for spec in self.image_specs:
            camera_id = cameras[spec.component.name].get("gdk_camera_id")
            if not camera_id:
                raise ValueError(
                    f"robot config topics.cameras.{spec.component.name} "
                    "missing gdk_camera_id"
                )
            thread = threading.Thread(
                target=self._g2_camera_loop,
                args=(spec, camera_id, poll_hz),
                daemon=True,
                name=f"agibot_g2_cam_{spec.component.name.lower()}",
            )
            self._g2_camera_threads.append(thread)
            thread.start()
            self.get_logger().info(
                f"Polling GDK camera {spec.component.name}: {camera_id} at {poll_hz:.0f}Hz"
            )

    def _g2_camera_loop(self, spec, camera_id: str, poll_hz: float):
        is_depth = spec.component == RobotComponent.CAMERA_HEAD_DEPTH
        period = 1.0 / max(1.0, poll_hz)
        while not self._g2_camera_stop.is_set():
            started = time.monotonic()
            try:
                image = (
                    self.robot_io.read_camera_depth_u16(camera_id)
                    if is_depth
                    else self.robot_io.read_camera_bgr(camera_id)
                )
                if image is not None:
                    self._store_camera_image(spec, image)
            except Exception as exc:
                self._g2_log_camera_error(camera_id, exc)
            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                self._g2_camera_stop.wait(remaining)

    def _g2_log_camera_error(self, camera_id: str, exc: BaseException):
        now = time.monotonic()
        last = self._g2_camera_last_error.get(camera_id, 0.0)
        if now - last < self.CAMERA_ERROR_LOG_PERIOD_S:
            return
        self._g2_camera_last_error[camera_id] = now
        self.get_logger().error(f"GDK camera {camera_id} poll failed: {exc}")

    def warm_up(self):
        """GDK 상태와 첫 카메라 프레임이 모두 들어올 때까지 기다린다."""
        self.get_logger().info("Warming up, waiting for GDK data...")
        deadline = time.time() + self._state_wait_timeout
        last_log_time = 0.0
        missing = []
        while time.time() < deadline and not self._stop_requested:
            missing = self._missing_observations()
            stale = self.robot_io.stale_state_components(self.state_order)
            missing.extend(
                f"stale_state:{component.name}"
                for component in stale
                if self._component_state(component) is not None
            )
            if not missing:
                self.get_logger().info("Warm-up completed")
                return
            now = time.time()
            if now - last_log_time >= 1.0:
                self.get_logger().warning(f"Waiting for observations: {', '.join(missing)}")
                last_log_time = now
            time.sleep(0.05)
        if self._stop_requested:
            return
        raise TimeoutError(
            f"Observations were not ready within {self._state_wait_timeout}s: "
            f"{', '.join(missing)}"
        )

    def destroy_node(self):
        stop = getattr(self, "_g2_camera_stop", None)
        if stop is not None:
            stop.set()
        for thread in getattr(self, "_g2_camera_threads", []):
            if thread is not threading.current_thread():
                thread.join(timeout=1.0)
        self._g2_camera_threads = []
        return super().destroy_node()


def main():
    import argparse

    from general_policy_action_generator import describe_mapping, load_and_validate_configs

    parser = argparse.ArgumentParser(
        description="AgiBot G2 policy generator - GDK 상태/카메라로 추론한다",
    )
    parser.add_argument("--robot", required=True, help="Robot JSON config")
    parser.add_argument("--policy", required=True, help="Policy JSON config")
    parser.add_argument(
        "--timeout", type=float, default=-1.0,
        help="정책 전체 실행 timeout 초. -1 = 무제한",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="JSON 검증과 벡터 매핑만 출력한다. GDK 도 ROS 도 필요 없다.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print(describe_mapping(load_and_validate_configs(args.robot, args.policy)))
        return 0

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init()
    generator = None
    executor = None
    try:
        generator = AgibotG2PolicyActionGenerator(
            args.robot,
            args.policy,
            async_model_load=True,
        )
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(generator)
        threading.Thread(target=executor.spin, daemon=True).start()

        success = generator.start(
            timeout=args.timeout,
            skip_warmup=args.skip_warmup,
        )
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        if generator is not None:
            generator.stop()
        return 1
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=1.0)
        if generator is not None:
            generator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
