# Live task switcher

A small local web app for showing what a multitask, language-conditioned policy (SmolVLA) can
do on the SO-101: the policy keeps running and you change its instruction on the fly.

- **Setup:** pick a checkpoint, sync or RTC inference and its settings, and a starting task.
  *Launch* loads the policy and connects the robot. The arm stays still until *Start*.
- **Task:** type a new instruction, or click one of the training tasks (read from the
  checkpoint's dataset, with their exact wording). It applies from the next policy inference,
  and nothing is reloaded.
- **Live view:** the front/wrist camera stream and a 3D SO-101 (URDF) driven by the measured
  joint angles.
- *Reset* sends the arm home and restores the starting task. *Shut down* returns the arm to its
  initial position and disconnects.

## Run

```bash
cd ~/robot_playground/lerobot
uv run python ../lerobot-learning-lab/04-language-multitask/live_task_switcher/app.py
# open http://127.0.0.1:8765
```

Requirements:
- The lerobot checkout next to this repo, with its `.venv`.
- The SO-101 URDF in `~/.cache/huggingface/lerobot/robot-urdfs/so101/`. If it's missing, run
  `../../02-ee-workspace-coverage/so101_fk.py` once to fetch it.
- Internet access in the browser, to load three.js and urdf-loader from jsDelivr.

## How it works

| File | Role |
| --- | --- |
| `app.py` | Standard-library HTTP server. Starts the rollout as a child process and talks to it. |
| `rollout_telemetry.py` | The child: the stock `lerobot-rollout` CLI plus a hook that streams the latest observation (JPEG frames and joint angles) back to the app over a local socket. |
| `index.html` | The page: controls, live camera, three.js URDF view. |

The task switching itself is lerobot's own interactive rollout mode (`--interactive=true`):
the app writes `/start`, `/subtask <text>`, `/reset` and `/stop` to the child's stdin.
Anything that produces text can change the task with `POST /api/task {"task": "..."}`, which
is where speech-to-text would plug in later.

Endpoints: `/stream/<camera>.mjpg` (MJPEG), `/events` (server-sent joint angles in degrees),
`/api/state`, `/api/launch`, `/api/start`, `/api/reset`, `/api/stop`, `/api/task`.
