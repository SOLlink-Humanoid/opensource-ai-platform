# LeRobot runtime (OAP onboard)

Minimal bundle to run **LeRobot ACT** policies on Agibot G2 from the Open Source AI Platform UI.

| Path | Role |
|------|------|
| `agibot_g2_act.json` | Shared deploy spec (21-DOF, cameras, runtime); per-checkpoint `model_path` is injected at run time |
| `deployment/` | GDK policy generator, robot adapters, depth/inference stream helpers |
| `vendor/` | Vendored **inference-only** LeRobot (ACT, processors, no dataset training stack) |

Checkpoints live under `../models/LeRobot/<folder>/` (not in this tree).
