# Open Source AI Platform (G2)

Agibot **G2** / NVIDIA Thor에서 동작하는 **4탭 통합 VLA 플랫폼**입니다.  
공유 헤드 카메라 허브 위에서 **YOLO11**, **OpenVLA**, **LeRobot ACT**, **Task Planner**를 한 UI에서 운영합니다.

## 기능 요약

| 탭 | 역할 |
|----|------|
| **YOLO11** | 산업 박스 등 클래스 검출, 태그 조건·타깃 설정, 실시간 오버레이 |
| **OpenVLA** | 언어 지시 기반 조작, Phase 프로필, 라이브 추론·관절 갭 분석 |
| **LeRobot** | ACT 정책 배포, 핸드 포즈 매핑, GDK 브리지를 통한 액션 실행 |
| **Task Planner** | 태스크 보드·스냅샷·적용, 정책 자동화 규칙 |

**Configuration** 탭에서 `config.yaml` 프로필, 디바이스, 카메라 크롭 등을 확인·조정합니다.

## 아키텍처

```text
frontend/          정적 UI (4탭 + planner 서브모듈 JS)
backend/           FastAPI 앱, 탭별 HTTP 라우트·스토어
web/               카메라·ROS/GDK 런타임·추론 엔진 (OpenVLA, YOLO)
shared/            관절 이름·액션 레이아웃 (모델·UI 공통)
config/            UI에서 편집하는 JSON (yolo, openvla, lerobot, planner)
lerobot_runtime/   G2용 LeRobot 배포 번들 (vendor + deployment)
data/              참조 트레이스·샘플 JSON
config.yaml        루트 프로필 (robot 모드, 경로, GPU, 카메라)
```

- **서버**: `backend/app.py` — 단일 FastAPI 프로세스가 정적 `frontend/`와 API를 함께 제공합니다.
- **로봇 모드**: GDK 헤드 카메라, joint state, CUDA 추론 (`mode: robot` in `config.yaml`).
- **모델 가중치**: 이 저장소 밖 — `models/YOLO11/`, `models/OpenVLA/`, `models/LeRobot/` 등 (경로는 `config.yaml`).

## 빠른 시작

### 1. 설정

루트 `config.yaml`에서 호스트·포트(기본 `8095`), 모델 경로, `openvla` / `yolo` / `lerobot` 옵션을 맞춥니다.  
탭별 세부 JSON은 [`config/README.md`](config/README.md)를 참고하세요.

### 2. 서버 실행

```bash
# 프로젝트 루트에서
export OAP_CONFIG=./config.yaml   # 선택
python -m backend --host 0.0.0.0 --port 8095
```

브라우저에서 `http://<host>:8095` — 기본 리다이렉트 및 `/yolo`, `/openvla`, `/lerobot`, `/planner`.

### 3. 셸 alias (선택)

`install.sh`는 `platform-start` 등 alias를 `~/.bashrc` / `~/.zshrc`에 등록합니다.  
실제 기동 스크립트(`start.sh` 등)는 배포 환경의 venv·Thor GDK 설정과 함께 사용합니다.

```bash
./install.sh && source ~/.zshrc   # 또는 ~/.bashrc
```

## LeRobot 런타임

온보드 ACT 추론용 최소 번들은 [`lerobot_runtime/README.md`](lerobot_runtime/README.md)에 정리되어 있습니다.  
체크포인트는 `models/LeRobot/<folder>/`에 두고, UI·브리지가 런타임에 `model_path`를 주입합니다.

## 개발·유틸

| 스크립트 | 설명 |
|----------|------|
| `scripts/patch_openvla_offline.py` | 오프라인 OpenVLA 동작 패치 |
| `scripts/phase_joint_log_format.py` | Phase·관절 로그 포맷 정리 |

백엔드 테스트 예: `backend/tests/test_yolo_tag_eval.py` (YOLO 태그 규칙 평가).

## 환경 변수 (자주 쓰는 것)

| 변수 | 의미 |
|------|------|
| `OAP_CONFIG` | YAML 설정 파일 경로 |
| `OAP_MODE` | `robot` 시 GDK·GPU 런타임 부트스트랩 |
| `OAP_HOST` / `OAP_PORT` | 바인드 주소·포트 오버라이드 |
| `OAP_LOG` | `install.sh` alias용 로그 경로 |

## 저장소 구성 참고

- Python `__pycache__`, 런타임 로그, 로컬 `.venv`는 `.gitignore` 대상입니다.
- Hand pose JSON은 `config.yaml`의 `hand_poses_dir`(보통 `../poses/hand_poses`)에서 브리지와 공유합니다.

---

**Bigwave Robotics** — G2 Open Source AI Platform