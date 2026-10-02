"""``lerobot-rollout`` plus a live telemetry feed for ``app.py``.

Runs the stock rollout CLI unchanged (same arguments), after hooking the robot wrapper so the
latest observation — joint positions and camera frames — is streamed to the app over a local
TCP socket (address in ``LIVE_TASK_TELEMETRY=host:port``).

The control loop is never slowed down: the hook only keeps a reference to the observation the
policy already read; JPEG encoding and sending happen on a separate thread at ``LIVE_TASK_FPS``.
While the policy is idle (before Start, after Reset) nothing reads the robot, so that thread
reads it itself — through the wrapper's lock — to keep the view live.

Wire format, per message: 4-byte big-endian header length, JSON header
``{"t": ..., "joints": {name: deg}, "cams": {name: n_bytes}}``, then the JPEG bytes of each
camera in header order.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.scripts.lerobot_rollout import main as rollout_main

_STALE_S = (
    0.1  # policy (30 fps) hasn't read the robot for this long -> poll it ourselves
)
_JPEG_QUALITY = 75

_latest: dict = {"obs": None, "t": 0.0, "wrapper": None}

_original_init = ThreadSafeRobot.__init__
_original_get_observation = ThreadSafeRobot.get_observation


def _init(self, robot) -> None:
    _original_init(self, robot)
    _latest["wrapper"] = self


def _get_observation(self):
    obs = _original_get_observation(self)
    _latest["obs"], _latest["t"] = obs, time.monotonic()
    return obs


def _encode(obs: dict) -> bytes:
    joints = {
        k.removesuffix(".pos"): float(v) for k, v in obs.items() if k.endswith(".pos")
    }
    images = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim == 3:
            ok, buf = cv2.imencode(
                ".jpg",
                cv2.cvtColor(value, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
            )
            if ok:
                images[key] = buf.tobytes()
    header = json.dumps(
        {
            "t": time.time(),
            "joints": joints,
            "cams": {k: len(v) for k, v in images.items()},
        }
    ).encode()
    return struct.pack(">I", len(header)) + header + b"".join(images.values())


def _telemetry_loop(address: str, fps: float) -> None:
    host, port = address.rsplit(":", 1)
    try:
        sock = socket.create_connection((host, int(port)), timeout=5)
    except OSError:
        return  # no app listening: run as a plain rollout
    period = 1.0 / fps
    while True:
        start = time.monotonic()
        obs = _latest["obs"]
        wrapper = _latest["wrapper"]
        if start - _latest["t"] > _STALE_S and wrapper is not None:
            try:
                if wrapper.is_connected:
                    obs = (
                        wrapper.get_observation()
                    )  # also refreshes _latest via the hook
            except Exception:  # noqa: BLE001, S110 - connecting, disconnecting or a bus hiccup
                pass  # keep the last frame
        if obs is not None:
            try:
                sock.sendall(_encode(obs))
            except OSError:
                return  # app went away
        time.sleep(max(0.0, period - (time.monotonic() - start)))


if __name__ == "__main__":
    ThreadSafeRobot.__init__ = _init
    ThreadSafeRobot.get_observation = _get_observation
    address = os.environ.get("LIVE_TASK_TELEMETRY")
    if address:
        fps = float(os.environ.get("LIVE_TASK_FPS", "15"))
        threading.Thread(
            target=_telemetry_loop, args=(address, fps), daemon=True, name="Telemetry"
        ).start()
    rollout_main()
