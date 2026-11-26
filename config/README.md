# Platform model & tab settings

UI-editable JSON lives here (not under model weight folders).

| Path | Tab / feature |
|------|----------------|
| `yolo.json` | YOLO — class targets, tag conditions |
| `openvla/phases.json` | OpenVLA — Phase profiles |
| `openvla/instruction_history.json` | OpenVLA — instruction history |
| `lerobot/hand_mappings.json` | LeRobot — policy → hand pose file |
| `planner/task_policy_rules.json` | Task planner — automation rules |

Hand pose JSON files live at **`../poses/hand_poses`** (`paths.hand_poses_dir` in `config.yaml`) for bridge sharing — not under this folder.

On first startup, missing files are copied from legacy paths (`data/`, `../poses/`, etc.) without overwriting existing files here.