#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pull each episode's opening camera frame, for the cell you clicked in the coverage viewer.

A coverage cell tells you *which* episodes went through it. This answers *why*: the external
camera at t=0 shows the scene each of those episodes started from -- for a pick-and-place set,
where the object was sitting. Cells that look identical on the heatmap often turn out to have
completely different causes.

One performance note that is not obvious. `decode_video_frames` accepts a list of timestamps, but
handing it several at once makes it walk the video between them: measured on an AV1 episode file,
one call per frame costs ~3 ms while a single call for 12 frames costs ~350 ms *each*. So frames
are fetched one call at a time, and cached, which is the opposite of the usual batching advice.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Substrings that mark a camera as being on the robot rather than looking at it.
_ONBOARD_HINTS = ("wrist", "hand", "gripper", "eye_in_hand")


class NoVideoError(RuntimeError):
    """The dataset has no video feature to show, or the requested one does not exist."""


def _pick_camera(info: dict, requested: str | None) -> str:
    video_keys = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    if not video_keys:
        raise NoVideoError("This dataset has no video features, so there are no frames to show.")
    if requested is not None:
        # Accept either the full feature key or just the camera name.
        for key in video_keys:
            if requested in (key, key.rsplit(".", 1)[-1]):
                return key
        raise NoVideoError(f"No video feature matches {requested!r}. Available: {', '.join(video_keys)}")
    # Default to a camera looking *at* the robot rather than one mounted on it.
    external = [k for k in video_keys if not any(h in k.lower() for h in _ONBOARD_HINTS)]
    return (external or video_keys)[0]


@dataclass
class EpisodeFrameLoader:
    """Decodes and caches the first frame of each episode of one dataset."""

    repo_id: str
    camera: str | None = None
    _cache: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        self.root = Path(LeRobotDatasetMetadata(self.repo_id).root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        self.camera = _pick_camera(self.info, self.camera)
        self.fps = float(self.info["fps"])

        files = sorted((self.root / "meta" / "episodes").rglob("*.parquet"))
        if not files:
            raise NoVideoError(f"No episode metadata under {self.root / 'meta' / 'episodes'}")
        cols = [
            "episode_index",
            f"videos/{self.camera}/chunk_index",
            f"videos/{self.camera}/file_index",
            f"videos/{self.camera}/from_timestamp",
        ]
        meta = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
        self._episodes = meta.set_index("episode_index")

    def _video_path(self, episode_index: int) -> tuple[Path, float]:
        row = self._episodes.loc[episode_index]
        # Episodes are packed into several video files per camera, so resolve per episode.
        path = self.root / self.info["video_path"].format(
            video_key=self.camera,
            chunk_index=int(row[f"videos/{self.camera}/chunk_index"]),
            file_index=int(row[f"videos/{self.camera}/file_index"]),
        )
        return path, float(row[f"videos/{self.camera}/from_timestamp"])

    def first_frame(self, episode_index: int) -> np.ndarray:
        """(H, W, 3) uint8 RGB of this episode's opening frame. Cached."""
        episode_index = int(episode_index)
        if episode_index in self._cache:
            return self._cache[episode_index]

        from lerobot.datasets.video_utils import decode_video_frames

        path, timestamp = self._video_path(episode_index)
        # One timestamp per call -- see the module docstring.
        frames = decode_video_frames(path, [timestamp], 1.0 / self.fps, return_uint8=True)
        frame = frames[0].permute(1, 2, 0).numpy()
        self._cache[episode_index] = frame
        return frame


def _grid_shape(n: int) -> tuple[int, int]:
    """Rows and columns for a near-square layout of n tiles, biased wider than tall."""
    cols = int(np.ceil(np.sqrt(n * 1.4)))
    return int(np.ceil(n / cols)), cols


def show_episode_frames(
    loader: EpisodeFrameLoader,
    episodes,
    title: str,
    max_thumbs: int = 16,
    figure=None,
):
    """Draw the opening frame of each episode in a reusable pop-up figure. Returns the figure.

    Pass the previous return value back as `figure` so repeated clicks reuse one window instead of
    burying the screen in them.
    """
    import matplotlib.pyplot as plt

    episodes = [int(e) for e in episodes]
    shown, extra = episodes[:max_thumbs], max(len(episodes) - max_thumbs, 0)
    if not shown:
        return figure

    if figure is None or not plt.fignum_exists(figure.number):
        figure = plt.figure(figsize=(11, 7))
        with contextlib.suppress(AttributeError):
            figure.canvas.manager.set_window_title("episode first frames")
    figure.clear()

    rows, cols = _grid_shape(len(shown))
    for i, episode in enumerate(shown):
        ax = figure.add_subplot(rows, cols, i + 1)
        try:
            ax.imshow(loader.first_frame(episode))
        except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the click
            ax.text(0.5, 0.5, f"ep {episode}\n(decode failed)", ha="center", va="center",
                    fontsize=8, color="#c62828", transform=ax.transAxes)
            print(f"Could not decode episode {episode}: {exc}")
        ax.set_title(f"ep {episode}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    suffix = f"   (+{extra} more)" if extra else ""
    figure.suptitle(f"{title} — {loader.camera}{suffix}", fontsize=10)
    figure.tight_layout()
    figure.canvas.draw_idle()
    # Headless backends have nothing to show; the figure is still returned and drawable.
    with contextlib.suppress(Exception):
        figure.show()
    return figure
