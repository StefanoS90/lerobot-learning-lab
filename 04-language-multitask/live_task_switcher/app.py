"""Live task switching for a language-conditioned policy on the SO-101.

A small local web app over ``lerobot-rollout --interactive=true``:

1. pick a checkpoint, the inference settings and a starting task, then *Launch*
   (loads the policy, connects the robot — the arm stays still);
2. *Start* runs the policy;
3. type a new instruction (or click one of the training tasks) and it is applied from
   the next policy inference — the robot keeps moving, nothing is reloaded.

The rollout runs as a child process (``rollout_telemetry.py``: the stock CLI plus a telemetry
hook); this app writes interactive commands (``/start``, ``/subtask <text>``, ``/reset``,
``/stop``) to its stdin and streams its output back to the page. The child also sends its
latest observation over a local socket, which the page shows as a live camera stream
(``/stream/<camera>.mjpg``) and an animated URDF of the arm (joint angles on ``/events``).
Anything that produces text — a keyboard today, speech-to-text later — can drive the task
through ``POST /api/task``.

Run from the lerobot checkout (so its venv and ``outputs/train`` are found):

    cd ~/robot_playground/lerobot
    uv run python ../lerobot-learning-lab/04-language-multitask/live_task_switcher/app.py
    # then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import ast
import functools
import json
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

DEFAULT_LEROBOT_ROOT = Path(__file__).resolve().parents[3] / "lerobot"
HF_LEROBOT_CACHE = Path(
    os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot")
)
URDF_DIR = HF_LEROBOT_CACHE / "robot-urdfs/so101"
URDF_NAME = "so101_new_calib.urdf"
HERE = Path(__file__).resolve().parent
TELEMETRY_FPS = 15

# Defaults mirror the command used in this chapter; all of them are editable in the page.
DEFAULTS = {
    "model": "outputs/train/smolvla_multitask_v4_unfrozen/checkpoints/last/pretrained_model",
    "task": "Grab the blue object and drop it inside the cube",
    "inference": "rtc",
    "n_action_steps": 50,
    "execution_horizon": 20,
    "queue_threshold": 30,
    "fps": 30,
    "duration": 0,
    "display_data": False,
    "robot_port": "/dev/ttyACM0",
    "robot_id": "my_awesome_follower_arm",
    "cameras": (
        "{front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, warmup_s: 3}, "
        "wrist: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30, warmup_s: 3}}"
    ),
    "rename_map": (
        '{"observation.images.front": "observation.images.camera1", '
        '"observation.images.wrist": "observation.images.camera2"}'
    ),
}

# Lines printed by the interactive session (src/lerobot/rollout/interactive.py) that move the state.
_STATE_MARKERS = [
    ("Interactive rollout session", "ready"),
    ("Rollout running", "running"),
    ("Rollout run ended", "ready"),
    ("Robot reset", "ready"),
    ("Robot paused", "ready"),
    ("Resetting", "resetting"),
]
_TASK_CHANGED = re.compile(r"^Task: .* → ('.*'|\".*\") \(applies")
_TASK_RESTORED = re.compile(r"^Task restored to ('.*'|\".*\")")
_TASK_BANNER = re.compile(r"^Task: ('.*'|\".*\")$")


def find_checkpoints(lerobot_root: Path) -> list[str]:
    """Every ``pretrained_model`` dir under ``outputs/train``, newest first, relative to the root."""
    found = sorted(
        (lerobot_root / "outputs/train").glob("*/checkpoints/*/pretrained_model"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [
        str(p.relative_to(lerobot_root)) for p in found if (p / "config.json").exists()
    ]


_READ_TASKS = "import sys, pandas as pd; print('\\n'.join(map(str, pd.read_parquet(sys.argv[1]).index)))"


@functools.cache
def training_tasks(lerobot_root: Path, model: str) -> tuple[str, ...]:
    """The task strings of the checkpoint's training dataset, if it is in the local cache.

    A language-conditioned policy follows the exact wording it was trained on best, so these
    become one-click presets in the page. Read in a throwaway process: pyarrow segfaulted when
    called from this server's request threads.
    """
    try:
        train_cfg = json.loads((lerobot_root / model / "train_config.json").read_text())
        ds = train_cfg["dataset"]
        root = Path(ds["root"]) if ds.get("root") else HF_LEROBOT_CACHE / ds["repo_id"]
        python = lerobot_root / ".venv/bin/python"
        out = subprocess.run(
            [
                str(python) if python.exists() else sys.executable,
                "-c",
                _READ_TASKS,
                str(root / "meta/tasks.parquet"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return tuple(line for line in out.stdout.splitlines() if line)
    except Exception:  # noqa: BLE001 - missing config, dataset not cached, ...: just no presets
        return ()


class Telemetry:
    """Receives the child's observation feed and keeps the latest frame per camera + joints."""

    def __init__(self) -> None:
        self._server = socket.create_server(("127.0.0.1", 0))
        self.address = f"127.0.0.1:{self._server.getsockname()[1]}"
        self.cond = threading.Condition()
        self.seq = 0
        self.joints: dict[str, float] = {}
        self.frames: dict[str, bytes] = {}
        threading.Thread(
            target=self._accept, daemon=True, name="TelemetryAccept"
        ).start()

    def _accept(self) -> None:
        while True:
            conn, _ = self._server.accept()
            threading.Thread(
                target=self._read, args=(conn,), daemon=True, name="TelemetryRead"
            ).start()

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("telemetry stream closed")
            buf += chunk
        return bytes(buf)

    def _read(self, conn: socket.socket) -> None:
        with conn:
            try:
                while True:
                    (n,) = struct.unpack(">I", self._recv_exact(conn, 4))
                    header = json.loads(self._recv_exact(conn, n))
                    frames = {
                        name: self._recv_exact(conn, size)
                        for name, size in header["cams"].items()
                    }
                    with self.cond:
                        self.joints = header["joints"]
                        self.frames.update(frames)
                        self.seq += 1
                        self.cond.notify_all()
            except (ConnectionError, OSError, ValueError):
                pass

    def wait(self, last_seq: int, timeout: float = 2.0) -> int:
        """Block until a message newer than ``last_seq`` arrives (or timeout); return the latest seq."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout=timeout)
            return self.seq


class Rollout:
    """One ``lerobot-rollout --interactive=true`` child process and its output."""

    def __init__(self, lerobot_root: Path, telemetry: Telemetry) -> None:
        self.lerobot_root = lerobot_root
        self.telemetry = telemetry
        self.proc: subprocess.Popen[str] | None = None
        self.lines: deque[tuple[int, str]] = deque(maxlen=2000)
        self.seq = 0
        self.state = "idle"
        self.task = ""
        self.config: dict = {}
        self._lock = threading.Lock()

    # -- process lifecycle -------------------------------------------------

    def build_command(self, cfg: dict) -> list[str]:
        python = self.lerobot_root / ".venv/bin/python"
        cmd = [
            str(python) if python.exists() else sys.executable,
            str(HERE / "rollout_telemetry.py"),
            "--strategy.type=base",
            f"--policy.path={cfg['model']}",
            f"--policy.n_action_steps={int(cfg['n_action_steps'])}",
            "--robot.type=so101_follower",
            f"--robot.port={cfg['robot_port']}",
            f"--robot.id={cfg['robot_id']}",
            f"--robot.cameras={cfg['cameras']}",
            f"--task={cfg['task']}",
            f"--fps={cfg['fps']}",
            f"--duration={cfg['duration']}",
            f"--inference.type={cfg['inference']}",
            f"--display_data={'true' if cfg['display_data'] else 'false'}",
            "--interactive=true",
        ]
        if cfg.get("rename_map", "").strip():
            cmd.append(f"--rename_map={cfg['rename_map']}")
        if cfg["inference"] == "rtc":
            cmd += [
                f"--inference.rtc.execution_horizon={int(cfg['execution_horizon'])}",
                f"--inference.queue_threshold={int(cfg['queue_threshold'])}",
            ]
        return cmd

    def launch(self, cfg: dict) -> None:
        with self._lock:
            if self.alive:
                raise RuntimeError("A rollout is already running — stop it first.")
            cmd = self.build_command(cfg)
            self.config = cfg
            self.task = cfg["task"]
            self.state = "loading"
            self._append("$ " + " ".join(cmd))
            self.proc = subprocess.Popen(
                cmd,
                cwd=self.lerobot_root,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "LIVE_TASK_TELEMETRY": self.telemetry.address,
                    "LIVE_TASK_FPS": str(TELEMETRY_FPS),
                },
                start_new_session=True,  # our Ctrl-C must not hit the robot process directly
            )
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def send(self, line: str) -> None:
        if not self.alive or self.proc.stdin is None:
            raise RuntimeError("No rollout is running — launch one first.")
        self._append(f"> {line}")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def set_task(self, task: str) -> None:
        # One line per command: a newline inside the task would be read as a second command.
        task = " ".join(task.split())
        if not task:
            raise ValueError("Empty task.")
        self.send(f"/subtask {task}")

    def shutdown(self, timeout_s: float = 30.0) -> None:
        """``/stop`` (robot returns home and disconnects), then SIGINT, then SIGKILL as a last resort."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            self.send("/stop")
            proc.wait(timeout=timeout_s)
        except (subprocess.TimeoutExpired, RuntimeError, BrokenPipeError):
            self._append("[app] /stop timed out — sending SIGINT")
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._append("[app] SIGINT timed out — killing")
                os.killpg(proc.pid, signal.SIGKILL)

    # -- output ------------------------------------------------------------

    def _append(self, line: str) -> None:
        self.seq += 1
        self.lines.append((self.seq, line))

    def _pump(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            self._append(line)
            self._track(line)
        code = proc.wait()
        self._append(f"[app] rollout exited with code {code}")
        self.state = "exited" if code == 0 else "failed"

    def _track(self, line: str) -> None:
        for marker, state in _STATE_MARKERS:
            if line.startswith(marker):
                self.state = state
        for pattern in (_TASK_CHANGED, _TASK_RESTORED, _TASK_BANNER):
            m = pattern.match(line)
            if m:
                try:
                    self.task = ast.literal_eval(m.group(1))
                except (ValueError, SyntaxError):
                    pass

    def snapshot(self, since: int) -> dict:
        return {
            "state": self.state if self.proc is not None else "idle",
            "alive": self.alive,
            "task": self.task,
            "seq": self.seq,
            "lines": [line for s, line in list(self.lines) if s > since],
            "cameras": sorted(self.telemetry.frames),
        }


def make_handler(rollout: Rollout, page: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal for the rollout
            pass

        def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            path, query = url.path, {k: v[0] for k, v in parse_qs(url.query).items()}
            if path == "/":
                body = page.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/options":
                self._json(
                    {
                        "defaults": DEFAULTS,
                        "models": find_checkpoints(rollout.lerobot_root),
                        "urdf": f"/urdf/{URDF_NAME}"
                        if (URDF_DIR / URDF_NAME).exists()
                        else None,
                    }
                )
            elif path == "/api/tasks":
                self._json(
                    {
                        "tasks": list(
                            training_tasks(rollout.lerobot_root, query.get("model", ""))
                        )
                    }
                )
            elif path.startswith("/stream/") and path.endswith(".mjpg"):
                self._mjpeg(path.removeprefix("/stream/").removesuffix(".mjpg"))
            elif path == "/events":
                self._joint_events()
            elif path.startswith("/urdf/"):
                self._urdf_file(path.removeprefix("/urdf/"))
            elif path == "/api/state":
                self._json(rollout.snapshot(int(query.get("since", 0))))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _mjpeg(self, camera: str) -> None:
            """multipart/x-mixed-replace: an <img> tag shows it as live video."""
            telemetry = rollout.telemetry
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while True:
                    seq = telemetry.wait(seq)
                    frame = telemetry.frames.get(camera)
                    if frame is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode()
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _joint_events(self) -> None:
            """Server-sent events: one ``{joint: degrees}`` message per telemetry update."""
            telemetry = rollout.telemetry
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while True:
                    new_seq = telemetry.wait(seq)
                    # A comment line on timeout keeps dead connections from piling up.
                    msg = (
                        f"data: {json.dumps(telemetry.joints)}\n\n"
                        if new_seq != seq
                        else ": ping\n\n"
                    )
                    seq = new_seq
                    self.wfile.write(msg.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _urdf_file(self, rel: str) -> None:
            path = (URDF_DIR / rel).resolve()
            if not path.is_relative_to(URDF_DIR.resolve()) or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, f"{rel} not found in {URDF_DIR}")
                return
            body = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type",
                "application/xml" if rel.endswith(".urdf") else "model/stl",
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            try:
                data = self._body()
                if self.path == "/api/launch":
                    rollout.launch({**DEFAULTS, **data})
                elif self.path == "/api/task":
                    rollout.set_task(str(data.get("task", "")))
                elif self.path in ("/api/start", "/api/reset"):
                    rollout.send("/" + self.path.rsplit("/", 1)[1])
                elif self.path == "/api/stop":
                    threading.Thread(target=rollout.shutdown, daemon=True).start()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._json({"ok": True})
            except Exception as exc:  # noqa: BLE001 - report any failure to the page
                self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lerobot-root", type=Path, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    rollout = Rollout(args.lerobot_root.resolve(), Telemetry())
    if not (URDF_DIR / URDF_NAME).exists():
        print(
            f"No SO-101 URDF in {URDF_DIR} — the 3D view stays empty. Fetch it once with\n"
            "  uv run python ../lerobot-learning-lab/02-ee-workspace-coverage/so101_fk.py"
        )
    page = (HERE / "index.html").read_text()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(rollout, page))
    print(
        f"Live task app on http://{args.host}:{args.port}  (lerobot root: {rollout.lerobot_root})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(
            "\nShutting down — stopping the rollout (robot returns to its initial position)..."
        )
        rollout.shutdown()
        # Give the pump thread a moment to print the exit line.
        time.sleep(0.3)
        for _, line in list(rollout.lines)[-5:]:
            print(line)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
