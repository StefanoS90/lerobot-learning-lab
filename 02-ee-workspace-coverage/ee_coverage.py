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
End-effector workspace coverage for SO-101 LeRobotDatasets, aggregated over every episode.

Joint-space statistics do not tell you whether a dataset actually covers the region of space
you care about, and the dataset viewer only replays one episode at a time. This runs forward
kinematics over every frame of one or more datasets and bins the resulting end-effector poses
into horizontal (x-y) slices of z, so you can see -- for the whole dataset at once -- where the
gripper has been, how long it lingered, how many distinct episodes went there, and at which
orientations it arrived.

Orientation is folded into the same picture through the colour channel (--panel-color):
`tilt` is the mean approach angle off straight-down and `tilt_spread` / `roll_spread` measure
how *varied* the approach was in each cell, which is what tells you whether a region was only
ever entered from one pose.

Coverage percentages are quoted against the robot's reachable workspace, estimated by sampling
the URDF joint limits, so "34% covered" means 34% of what the arm can actually get to at that
height -- not 34% of an arbitrary bounding box.

    --interactive   the main event: a window with a z slider, a colour-channel selector,
                    a dataset/diff switch, linked side views, and hover/click readout
    --out-dir       static PNG + summary for a write-up
    --rerun         3D aggregate point cloud with the arm and the slice planes for reference

Usage:
    # Interactive explorer over one dataset.
    python ee_coverage.py ssabats/object-dropping-cube-merged --interactive

    # Two datasets: the dataset switch gains a "B \\ A" diff view showing the new coverage.
    python ee_coverage.py \
        ssabats/object-dropping-cube-merged \
        ssabats/object-dropping-cube-recovery_20260909_175555 \
        --interactive

    # Static figure for a README.
    python ee_coverage.py ssabats/object-dropping-cube-merged \
        --z-slices 6 --bins 64 --panel-color count --log --out-dir assets/

    # Where did the gripper approach from only one angle?
    python ee_coverage.py ssabats/object-dropping-cube-merged \
        --panel-color tilt_spread --out-dir assets/

    # 3D aggregate view in rerun.
    python ee_coverage.py ssabats/object-dropping-cube-merged --rerun --color-by tilt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from so101_fk import DEFAULT_TIP_LINK, load_so101_chain

# Channels the colour map can show. `count` is the default; the rest are all derived from the
# same per-frame arrays, so switching between them in the viewer is a dict lookup.
CHANNELS = ("count", "dwell", "episodes", "tilt", "tilt_spread", "roll_spread", "gripper")

CHANNEL_LABELS = {
    "count": "frames",
    "dwell": "dwell (s)",
    "episodes": "distinct episodes",
    "tilt": "approach tilt off vertical (deg)",
    "tilt_spread": "approach direction spread (deg)",
    "roll_spread": "wrist-roll spread (deg)",
    "gripper": "mean gripper opening",
}

# Above this, a body joint cannot be RANGE_M100_100 (which clamps to +/-100), so the dataset
# must have been recorded with use_degrees=True. See lerobot motors_bus.py::_normalize.
NORMALIZED_CLAMP = 100.5
STS3215_RESOLUTION = 4096


@dataclass
class EEPoses:
    """Per-frame end-effector quantities for one dataset."""

    name: str
    pos: np.ndarray  # (N, 3) metres, robot base frame
    approach: np.ndarray  # (N, 3) unit tool axis
    tilt_deg: np.ndarray  # (N,) angle between approach and straight down
    roll_rad: np.ndarray  # (N,) rotation about the approach axis
    gripper: np.ndarray  # (N,)
    episode_index: np.ndarray  # (N,)
    q_deg: np.ndarray  # (N, n_joints) the FK input, kept so the rest pose can be drawn
    fps: float

    def __len__(self) -> int:
        return len(self.pos)


@dataclass
class SliceView:
    """Everything needed to draw one horizontal slab, computed on demand for arbitrary bounds.

    Grids are indexed [y_bin, x_bin] so they can be handed straight to imshow with
    origin="lower" and `extent`.
    """

    name: str
    z0: float
    z1: float
    channels: dict[str, np.ndarray]
    counts: np.ndarray
    tilt_std: np.ndarray
    approach_mean: np.ndarray  # (ny, nx, 3)
    episodes_in_cell: dict[tuple[int, int], np.ndarray]
    reach: np.ndarray | None
    # For a diff slice: how many cells the *opposite* direction would light up. An empty diff is
    # a real answer, but it is only interpretable next to the reverse count.
    reverse_cells: int | None = None

    @property
    def thickness(self) -> float:
        return self.z1 - self.z0

    @property
    def label(self) -> str:
        return f"z ∈ [{self.z0:.3f}, {self.z1:.3f}) m"


@dataclass
class _Binned:
    """One dataset's frames reduced to x-y cell indices, so a z query is a mask plus bincounts."""

    poses: EEPoses
    cell: np.ndarray  # (N,) flat iy * nx + ix, -1 outside the x-y extent
    ok_xy: np.ndarray  # (N,) bool
    z: np.ndarray


@dataclass
class CoverageModel:
    """Binned datasets plus a reachability volume, queryable at any pair of z bounds.

    Nothing is precomputed per slice: `query` re-bins the frames for whatever bounds it is given,
    which measures at ~2 ms for 30k frames -- far below the cost of drawing the result. That keeps
    the interactive viewer free to use continuous bounds while the static export asks for the same
    discrete slices as before, through the same code path.
    """

    names: list[str]
    x_edges: np.ndarray
    y_edges: np.ndarray
    z_range: tuple[float, float]
    z_edges: np.ndarray  # the discrete slicing used by the static export
    fps: float
    source: str
    side_xz: dict[str, np.ndarray]
    side_yz: dict[str, np.ndarray]
    z_edges_fine: np.ndarray
    poses: dict[str, EEPoses] = field(default_factory=dict)
    diff: tuple[str, str] | None = None
    reach_volume: np.ndarray | None = None
    reach_z_edges: np.ndarray | None = None
    reach_points: np.ndarray | None = field(default=None, repr=False)
    reach_min_thickness_m: float = 0.008
    chain: object | None = None
    rest_pose_deg: np.ndarray | None = None
    _binned: dict[str, _Binned] = field(default_factory=dict, repr=False)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.x_edges[0], self.x_edges[-1], self.y_edges[0], self.y_edges[-1])

    @property
    def n_slices(self) -> int:
        return len(self.z_edges) - 1

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.y_edges) - 1, len(self.x_edges) - 1)

    def slice_bounds(self, s: int) -> tuple[float, float]:
        return (float(self.z_edges[s]), float(self.z_edges[s + 1]))

    @property
    def bins(self) -> int:
        return len(self.x_edges) - 1

    @property
    def cell_size_m(self) -> float:
        return float(self.x_edges[1] - self.x_edges[0])

    def set_bins(self, bins: int) -> None:
        """Re-grid everything at a new resolution, over the same physical extent.

        The extent is deliberately held fixed so the view never jumps: only the cell size
        changes. Re-binning the frames is trivial; the reachable workspace is the expensive part,
        which is why the Monte-Carlo samples are kept on the model rather than thrown away -- they
        are re-binned here instead of re-running forward kinematics over millions of configs.
        """
        if bins == self.bins:
            return
        x0, x1, y0, y1 = self.extent
        self.x_edges = np.linspace(x0, x1, bins + 1)
        self.y_edges = np.linspace(y0, y1, bins + 1)
        self.z_edges_fine = np.linspace(self.z_edges_fine[0], self.z_edges_fine[-1], bins + 1)

        for name, b in self._binned.items():
            ix = _bin_index(b.poses.pos[:, 0], self.x_edges)
            iy = _bin_index(b.poses.pos[:, 1], self.y_edges)
            b.ok_xy = (ix >= 0) & (iy >= 0)
            b.cell = np.where(b.ok_xy, iy * bins + ix, 0).astype(np.int64)
            self.side_xz[name], self.side_yz[name] = _side_histograms(
                b.poses, self.x_edges, self.y_edges, self.z_edges_fine
            )
        if self.diff is not None:
            label = _diff_label(*self.diff)
            self.side_xz[label] = self.side_xz[self.diff[0]]
            self.side_yz[label] = self.side_yz[self.diff[0]]

        if self.reach_points is not None and self.reach_z_edges is not None:
            self.reach_volume = _bin_reach_points(
                self.reach_points, self.reach_z_edges, self.y_edges, self.x_edges
            )
            _check_reachability(self)

    def query(self, name: str, z0: float, z1: float) -> SliceView:
        """Channel grids for `name` over the slab [z0, z1)."""
        if self.diff is not None and name == _diff_label(*self.diff):
            b, a = self.diff
            return self.diff_view(self._query_one(b, z0, z1), self._query_one(a, z0, z1))
        return self._query_one(name, z0, z1)

    def diff_view(self, view_b: SliceView, view_a: SliceView) -> SliceView:
        """B's values restricted to the cells B reached and A never did.

        Takes the two slices rather than re-querying them, so a side-by-side comparison showing
        A, B and the diff costs two queries instead of four. Returns a new view: the inputs are
        still on display in their own panels and must not be mutated.
        """
        new = (view_b.counts > 0) & (view_a.counts == 0)
        reverse = int(((view_a.counts > 0) & (view_b.counts == 0)).sum())
        return SliceView(
            name=_diff_label(*self.diff),
            z0=view_b.z0,
            z1=view_b.z1,
            channels={k: np.where(new, v, np.nan) for k, v in view_b.channels.items()},
            counts=np.where(new, view_b.counts, 0.0),
            tilt_std=view_b.tilt_std,
            approach_mean=view_b.approach_mean,
            episodes_in_cell={c: e for c, e in view_b.episodes_in_cell.items() if new[c]},
            reach=view_b.reach,
            reverse_cells=reverse,
        )

    def _query_one(self, name: str, z0: float, z1: float) -> SliceView:
        b = self._binned[name]
        # The top of the data range is inclusive, matching how the x/y binner folds the right
        # edge into the last bin -- otherwise the highest frame in the dataset would be
        # unreachable by any query.
        upper = (b.z <= z1) if z1 >= self.z_range[1] - 1e-12 else (b.z < z1)
        mask = b.ok_xy & (b.z >= z0) & upper

        channels, counts, tilt_std, approach_mean, episodes = _channel_grids(
            b.poses, b.cell, mask, self.shape, self.fps
        )
        return SliceView(
            name=name,
            z0=float(z0),
            z1=float(z1),
            channels=channels,
            counts=counts,
            tilt_std=tilt_std,
            approach_mean=approach_mean,
            episodes_in_cell=episodes,
            reach=self.reach_at(z0, z1),
        )

    def reach_at(self, z0: float, z1: float) -> np.ndarray | None:
        """Reachable cells for the slab, as a 2D mask.

        Evaluated over at least `reach_min_thickness_m` of z, centred on the band: a 1 mm slab
        holds too few Monte-Carlo samples to estimate reachability from, and the reachable set
        changes slowly with height, so widening the window is harmless and errs towards calling
        a cell reachable -- the honest direction for a coverage denominator.
        """
        if self.reach_volume is None or self.reach_z_edges is None:
            return None
        from scipy import ndimage

        if z1 - z0 < self.reach_min_thickness_m:
            mid = 0.5 * (z0 + z1)
            half = self.reach_min_thickness_m / 2
            z0, z1 = mid - half, mid + half

        edges = self.reach_z_edges
        a = int(np.clip(np.searchsorted(edges, z0, "right") - 1, 0, len(edges) - 2))
        b = int(np.clip(np.searchsorted(edges, z1, "left"), a + 1, len(edges) - 1))
        mask = np.any(self.reach_volume[a:b], axis=0)
        return ndimage.binary_fill_holes(ndimage.binary_closing(mask, np.ones((3, 3), dtype=bool)))

    def clim(self, channel: str, thickness: float, n_probe: int = 8) -> tuple[float, float]:
        """Colour limits for a given slab thickness, sampled across the whole z range.

        Counts scale with how thick the slab is, so the limits are a function of thickness --
        recomputed when you thicken or thin the band, and frozen while you translate it, which is
        what keeps two heights comparable.
        """
        lo_z, hi_z = self.z_range
        thickness = max(float(thickness), 1e-6)
        starts = np.linspace(lo_z, max(hi_z - thickness, lo_z), n_probe)
        values = []
        for z0 in starts:
            for name in self.names:
                grid = self.query(name, float(z0), float(z0 + thickness)).channels[channel]
                finite = grid[np.isfinite(grid)]
                if len(finite):
                    values.append(finite)
        if not values:
            return (0.0, 1.0)
        stacked = np.concatenate(values)
        if channel in ("count", "dwell", "episodes"):
            lo, hi = float(stacked.min()), float(stacked.max())
        else:
            # Percentile clipping keeps a handful of single-frame cells from eating the colour bar.
            lo, hi = float(np.percentile(stacked, 1)), float(np.percentile(stacked, 99))
        return (lo, hi + 1e-9) if hi <= lo else (lo, hi)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _dataset_root(repo_id: str) -> Path:
    """Local root of a LeRobotDataset, downloading metadata+data if it is not cached yet."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    return Path(meta.root)


def _load_frames(repo_id: str, source: str) -> tuple[pd.DataFrame, dict]:
    """Read the frame table straight from parquet -- no video decoding, no LeRobotDataset item path."""
    root = _dataset_root(repo_id)
    info = json.loads((root / "meta" / "info.json").read_text())

    key = "observation.state" if source == "state" else "action"
    if key not in info["features"]:
        raise KeyError(f"{repo_id} has no feature {key!r}")

    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet data files under {root / 'data'}")
    frames = pd.concat([pd.read_parquet(f, columns=[key, "episode_index"]) for f in files], ignore_index=True)
    return frames, info


def _joint_columns(info: dict, key: str, joint_names: list[str]) -> list[int]:
    """Map URDF joint names onto dataset feature columns by name, so column order cannot drift."""
    names = info["features"][key].get("names") or []
    stripped = [n[: -len(".pos")] if n.endswith(".pos") else n for n in names]
    missing = [j for j in joint_names if j not in stripped]
    if missing:
        raise KeyError(f"Dataset feature {key!r} is missing joints {missing}; it has {stripped}")
    return [stripped.index(j) for j in joint_names]


def _gripper_column(info: dict, key: str) -> int | None:
    names = info["features"][key].get("names") or []
    stripped = [n[: -len(".pos")] if n.endswith(".pos") else n for n in names]
    return stripped.index("gripper") if "gripper" in stripped else None


def _load_calibration(path: Path | None, robot_type: str) -> dict:
    if path is None:
        from lerobot.utils.constants import HF_LEROBOT_HOME

        candidates = sorted((Path(HF_LEROBOT_HOME) / "calibration" / "robots" / robot_type).glob("*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"--joint-units normalized needs a calibration file; none found for robot type "
                f"{robot_type!r}. Pass --calibration explicitly."
            )
        path = candidates[0]
        print(f"Using calibration {path}")
    return json.loads(Path(path).read_text())


def _normalized_to_degrees(values: np.ndarray, joint_names: list[str], calibration: dict) -> np.ndarray:
    """Invert MotorsBus._normalize for RANGE_M100_100, then express the ticks in degrees.

    lerobot normalises as norm = (ticks - min) / (max - min) * 200 - 100 and reports DEGREES as
    (ticks - (min + max) / 2) * 360 / (resolution - 1). Composing the inverse of the first with
    the second is what turns a recorded -100..100 column into an FK-ready angle.
    """
    out = np.empty_like(values, dtype=np.float64)
    for i, name in enumerate(joint_names):
        cal = calibration[name]
        lo, hi = float(cal["range_min"]), float(cal["range_max"])
        ticks = (values[:, i] + 100.0) / 200.0 * (hi - lo) + lo
        out[:, i] = (ticks - (lo + hi) / 2.0) * 360.0 / (STS3215_RESOLUTION - 1)
    return out


def compute_ee_poses(
    repo_id: str,
    chain,
    source: str = "state",
    joint_units: str = "auto",
    calibration_path: Path | None = None,
    approach_axis: str = "z",
    episodes: np.ndarray | None = None,
) -> EEPoses:
    """Run FK over every frame of a dataset and derive the per-frame end-effector quantities."""
    frames, info = _load_frames(repo_id, source)
    key = "observation.state" if source == "state" else "action"

    if episodes is not None:
        frames = frames[frames["episode_index"].isin(episodes)].reset_index(drop=True)
        if frames.empty:
            raise ValueError(f"No frames left in {repo_id} after the --episodes filter")

    raw = np.stack(frames[key].to_numpy()).astype(np.float64)
    joint_names = chain.joint_names
    cols = _joint_columns(info, key, joint_names)
    body = raw[:, cols]

    units = joint_units
    if units == "auto":
        units = "degrees" if np.abs(body).max() > NORMALIZED_CLAMP else "normalized"
        print(f"{repo_id}: joint units detected as {units} (max |body joint| = {np.abs(body).max():.2f})")
    if units == "normalized":
        calibration = _load_calibration(calibration_path, info.get("robot_type", "so_follower"))
        body = _normalized_to_degrees(body, joint_names, calibration)

    t = chain.fk(np.deg2rad(body))
    pos = t[:, :3, 3]

    axis_col = {"x": 0, "y": 1, "z": 2}[approach_axis]
    approach = t[:, :3, axis_col]
    approach /= np.linalg.norm(approach, axis=1, keepdims=True)

    # 0 deg = pointing straight down, which is the useful reference for a tabletop task.
    tilt_deg = np.degrees(np.arccos(np.clip(-approach[:, 2], -1.0, 1.0)))

    # Roll about the approach axis, read off the two remaining tool axes.
    other = [c for c in (0, 1, 2) if c != axis_col]
    roll_rad = np.arctan2(t[:, 2, other[0]], t[:, 2, other[1]])

    gcol = _gripper_column(info, key)
    gripper = raw[:, gcol] if gcol is not None else np.zeros(len(raw))

    return EEPoses(
        name=repo_id,
        pos=pos,
        approach=approach,
        tilt_deg=tilt_deg,
        roll_rad=roll_rad,
        gripper=gripper,
        episode_index=frames["episode_index"].to_numpy(),
        q_deg=body,
        fps=float(info["fps"]),
    )


# --------------------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------------------


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Uniform-edge bin index, -1 for anything outside the range."""
    lo, hi, n = edges[0], edges[-1], len(edges) - 1
    idx = np.floor((values - lo) / (hi - lo) * n).astype(np.int64)
    idx[(values < lo) | (values > hi)] = -1
    idx[idx == n] = n - 1  # the right edge belongs to the last bin
    return idx


def _flat_index(
    pos: np.ndarray, z_edges: np.ndarray, y_edges: np.ndarray, x_edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (slice, y_bin, x_bin) to one index, and return the mask of in-range points."""
    iz = _bin_index(pos[:, 2], z_edges)
    iy = _bin_index(pos[:, 1], y_edges)
    ix = _bin_index(pos[:, 0], x_edges)
    ok = (iz >= 0) & (iy >= 0) & (ix >= 0)
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    flat = np.full(len(pos), -1, dtype=np.int64)
    flat[ok] = (iz[ok] * ny + iy[ok]) * nx + ix[ok]
    return flat, ok


def _resultant_to_degrees(resultant: np.ndarray) -> np.ndarray:
    """Circular standard deviation, in degrees, from a mean-resultant length in [0, 1]."""
    r = np.clip(resultant, 1e-12, 1.0)
    return np.degrees(np.sqrt(-2.0 * np.log(r)))


def _mean_by_bin(flat: np.ndarray, weights: np.ndarray, counts: np.ndarray, size: int) -> np.ndarray:
    total = np.bincount(flat, weights=weights, minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)


def _channel_grids(
    poses: EEPoses,
    cell: np.ndarray,
    mask: np.ndarray,
    shape: tuple[int, int],
    fps: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict]:
    """Every channel for the frames selected by `mask`, as (ny, nx) grids.

    This is the hot path -- it runs on every drag of the z bounds -- so it is one `np.bincount`
    per accumulator over the selected frames and nothing else.
    """
    ny, nx = shape
    size = ny * nx
    f = cell[mask]

    counts = np.bincount(f, minlength=size).astype(np.float64)
    empty = counts == 0

    grids: dict[str, np.ndarray] = {}
    grids["count"] = np.where(empty, np.nan, counts)
    grids["dwell"] = grids["count"] / fps

    # Distinct episodes per cell: unique (cell, episode) pairs, counted per cell.
    pairs = np.unique(np.stack([f, poses.episode_index[mask]], axis=1), axis=0)
    ep_counts = np.bincount(pairs[:, 0].astype(np.int64), minlength=size).astype(np.float64)
    grids["episodes"] = np.where(empty, np.nan, ep_counts)

    grids["tilt"] = _mean_by_bin(f, poses.tilt_deg[mask], counts, size)
    grids["gripper"] = _mean_by_bin(f, poses.gripper[mask], counts, size)

    # Angular dispersion of the approach direction, reported as an angle. The resultant length
    # R = ||mean unit vector|| is 1 when every visit approached identically; the standard
    # circular-spread measure sqrt(-2 ln R) turns that into radians, which is far easier to read
    # on a colour bar than a unitless 0..1 and is directly comparable to the `tilt` channel.
    mean_vec = np.stack(
        [_mean_by_bin(f, poses.approach[mask, k], counts, size) for k in range(3)],
        axis=1,
    )
    grids["tilt_spread"] = _resultant_to_degrees(np.linalg.norm(np.nan_to_num(mean_vec), axis=1))
    grids["tilt_spread"][empty] = np.nan

    roll_c = _mean_by_bin(f, np.cos(poses.roll_rad[mask]), counts, size)
    roll_s = _mean_by_bin(f, np.sin(poses.roll_rad[mask]), counts, size)
    grids["roll_spread"] = _resultant_to_degrees(
        np.hypot(np.nan_to_num(roll_c), np.nan_to_num(roll_s))
    )
    grids["roll_spread"][empty] = np.nan

    # Second moment of the tilt, for the hover readout only.
    tilt_sq = _mean_by_bin(f, poses.tilt_deg[mask] ** 2, counts, size)
    tilt_std = np.sqrt(np.maximum(tilt_sq - grids["tilt"] ** 2, 0.0))

    episodes_in_cell: dict[tuple[int, int], np.ndarray] = {}
    if len(pairs):
        order = np.argsort(pairs[:, 0], kind="stable")
        pairs = pairs[order]
        cell_ids, starts = np.unique(pairs[:, 0], return_index=True)
        stops = list(starts[1:]) + [len(pairs)]
        for cid, start, stop in zip(cell_ids, starts, stops, strict=True):
            iy, ix = divmod(int(cid), nx)
            episodes_in_cell[(iy, ix)] = pairs[start:stop, 1]

    grids = {k: v.reshape(shape) for k, v in grids.items()}
    return grids, counts.reshape(shape), tilt_std.reshape(shape), mean_vec.reshape((*shape, 3)), episodes_in_cell


def _side_histograms(
    poses: EEPoses, x_edges: np.ndarray, y_edges: np.ndarray, z_edges_fine: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """x-z and y-z projections over every frame, indexed [z_bin, horizontal_bin] for imshow."""
    xz, _, _ = np.histogram2d(poses.pos[:, 2], poses.pos[:, 0], bins=[z_edges_fine, x_edges])
    yz, _, _ = np.histogram2d(poses.pos[:, 2], poses.pos[:, 1], bins=[z_edges_fine, y_edges])
    return xz, yz


def _diff_label(b: str, a: str) -> str:
    return f"{_short(b)} \\ {_short(a)}"


def build_coverage(
    all_poses: list[EEPoses],
    bins: int = 64,
    z_slices: int = 6,
    z_min: float | None = None,
    z_max: float | None = None,
    z_edges: np.ndarray | None = None,
    margin: float = 0.02,
    reach_samples: int = 2_000_000,
    chain=None,
    diff: tuple[str, str] | None = None,
    source: str = "state",
    seed: int = 0,
    include_base: bool = True,
    rest_pose: str = "home",
    reach_z_bins: int = 48,
    reach_store: int = 2_000_000,
) -> CoverageModel:
    """Bin every dataset onto one shared x-y grid and sample the reachable volume once."""
    pos_all = np.concatenate([p.pos for p in all_poses])

    if z_edges is None:
        lo = np.percentile(pos_all[:, 2], 1) if z_min is None else z_min
        hi = np.percentile(pos_all[:, 2], 99) if z_max is None else z_max
        z_edges = np.linspace(lo, hi, z_slices + 1)
    z_edges = np.asarray(z_edges, dtype=float)

    # One square extent for x and y keeps the aspect ratio honest and the frame fixed. The base
    # sits outside the reached workspace, so include it by default: a top view that crops the
    # robot out gives you no way to tell which side of the arm the data is on.
    x_lo, x_hi = pos_all[:, 0].min() - margin, pos_all[:, 0].max() + margin
    y_lo, y_hi = pos_all[:, 1].min() - margin, pos_all[:, 1].max() + margin
    if include_base:
        x_lo, x_hi = min(x_lo, -margin), max(x_hi, margin)
        y_lo, y_hi = min(y_lo, -margin), max(y_hi, margin)
    span = max(x_hi - x_lo, y_hi - y_lo)
    x_mid, y_mid = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    x_edges = np.linspace(x_mid - span / 2, x_mid + span / 2, bins + 1)
    y_edges = np.linspace(y_mid - span / 2, y_mid + span / 2, bins + 1)

    # The slider spans everything the data touches, not just the discrete slicing.
    z_range = (float(pos_all[:, 2].min()), float(pos_all[:, 2].max()))
    z_edges_fine = np.linspace(z_range[0] - margin, z_range[1] + margin, bins + 1)

    names: list[str] = []
    binned: dict[str, _Binned] = {}
    side_xz: dict[str, np.ndarray] = {}
    side_yz: dict[str, np.ndarray] = {}
    poses_by_name: dict[str, EEPoses] = {}

    nx = bins
    for poses in all_poses:
        ix = _bin_index(poses.pos[:, 0], x_edges)
        iy = _bin_index(poses.pos[:, 1], y_edges)
        ok_xy = (ix >= 0) & (iy >= 0)
        cell = np.where(ok_xy, iy * nx + ix, 0).astype(np.int64)
        names.append(poses.name)
        binned[poses.name] = _Binned(poses=poses, cell=cell, ok_xy=ok_xy, z=poses.pos[:, 2])
        side_xz[poses.name], side_yz[poses.name] = _side_histograms(poses, x_edges, y_edges, z_edges_fine)
        poses_by_name[poses.name] = poses

    if diff is None and len(all_poses) >= 2:
        diff = (all_poses[-1].name, all_poses[0].name)
    if diff is not None:
        label = _diff_label(*diff)
        side_xz[label], side_yz[label] = side_xz[diff[0]], side_yz[diff[0]]
        names.append(label)

    model = CoverageModel(
        names=names,
        x_edges=x_edges,
        y_edges=y_edges,
        z_range=z_range,
        z_edges=z_edges,
        fps=all_poses[0].fps,
        source=source,
        side_xz=side_xz,
        side_yz=side_yz,
        z_edges_fine=z_edges_fine,
        poses=poses_by_name,
        diff=diff,
        chain=chain,
        rest_pose_deg=_rest_pose(all_poses, rest_pose),
        _binned=binned,
    )

    if reach_samples > 0 and chain is not None:
        reach_edges = np.linspace(z_range[0] - margin, z_range[1] + margin, reach_z_bins + 1)
        model.reach_volume, model.reach_points = _sample_reach_points(
            chain, reach_samples, reach_edges, y_edges, x_edges, seed, store_max=reach_store
        )
        model.reach_z_edges = reach_edges
        _check_reachability(model)

    return model


def _check_reachability(model: CoverageModel) -> None:
    """Guard against a unit, axis or tip-frame error, which would put data outside the workspace.

    Checked in projection (union over all heights) rather than per z bin: at the volume's fine z
    resolution a per-bin comparison would trip on Monte-Carlo sparsity instead of on the bugs this
    is here to catch.
    """
    reach_any = np.any(model.reach_volume, axis=0)
    visited = np.zeros(model.shape, dtype=bool)
    for name in model._binned:
        b = model._binned[name]
        flat = np.unique(b.cell[b.ok_xy])
        v = np.zeros(model.shape[0] * model.shape[1], dtype=bool)
        v[flat] = True
        visited |= v.reshape(model.shape)

    outside = int((visited & ~reach_any).sum())
    if not outside:
        return
    frac = outside / max(int(visited.sum()), 1)
    if frac > 0.05:
        raise RuntimeError(
            f"{outside} visited cells ({frac:.1%}) fall outside the sampled reachable workspace. "
            "That is too many to be sampling noise -- the joint units, the joint->column mapping "
            "or the tip frame is wrong."
        )
    print(
        f"Note: {outside} visited cells ({frac:.1%}) were missed by the reachability sampler; "
        "folding them into the mask. Raise --reach-samples to reduce this."
    )
    # Fold them in at every height they were actually visited.
    for name in model._binned:
        b = model._binned[name]
        iz = _bin_index(b.z, model.reach_z_edges)
        ok = b.ok_xy & (iz >= 0)
        model.reach_volume.reshape(len(model.reach_z_edges) - 1, -1)[iz[ok], b.cell[ok]] = True


def _bin_reach_points(
    points: np.ndarray, z_edges: np.ndarray, y_edges: np.ndarray, x_edges: np.ndarray
) -> np.ndarray:
    """Bin stored reachable positions into a (nz, ny, nx) occupancy volume."""
    nz, ny, nx = len(z_edges) - 1, len(y_edges) - 1, len(x_edges) - 1
    mask = np.zeros(nz * ny * nx, dtype=bool)
    flat, ok = _flat_index(points, z_edges, y_edges, x_edges)
    mask[flat[ok]] = True
    return mask.reshape((nz, ny, nx))


def _sample_reach_points(
    chain,
    max_samples: int,
    z_edges: np.ndarray,
    y_edges: np.ndarray,
    x_edges: np.ndarray,
    seed: int,
    store_max: int = 2_000_000,
    chunk: int = 250_000,
    tol: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the reachable workspace, returning both the occupancy volume and the raw points.

    Uniform joint-space sampling is very non-uniform in Cartesian space, and the binned volume is
    only the slab the data occupies -- roughly 12% of samples land in it for a tabletop dataset.
    So rather than fixing a sample count, keep drawing chunks until a chunk stops discovering new
    cells. Pinholes are closed at query time, once the bins spanning the requested band have been
    combined; closing here, per thin z bin, would be fighting sparsity instead of fixing it.

    The points themselves are kept (float32, ~12 MB per million) so that changing the grid
    resolution can re-bin them in ~90 ms instead of re-running this, which takes seconds.
    """
    nz, ny, nx = len(z_edges) - 1, len(y_edges) - 1, len(x_edges) - 1
    rng = np.random.default_rng(seed)
    mask = np.zeros(nz * ny * nx, dtype=bool)
    kept: list[np.ndarray] = []
    kept_n = 0

    drawn = 0
    while drawn < max_samples:
        take = min(chunk, max_samples - drawn)
        drawn += take
        before = int(mask.sum())
        pts = chain.sample_reachable_positions(take, rng)
        flat, ok = _flat_index(pts, z_edges, y_edges, x_edges)
        mask[flat[ok]] = True
        # Only points inside the binned volume are ever useful for re-binning.
        inside = pts[ok].astype(np.float32)
        if kept_n < store_max:
            room = store_max - kept_n
            kept.append(inside[:room])
            kept_n += len(kept[-1])
        found = int(mask.sum())
        if before and (found - before) <= tol * before:
            break

    points = np.concatenate(kept) if kept else np.zeros((0, 3), dtype=np.float32)
    print(f"Reachability volume: {drawn} samples, {int(mask.sum())} reachable cells over {nz} z "
          f"bins ({len(points)} points kept, {points.nbytes / 1e6:.0f} MB, for re-gridding).")
    return mask.reshape((nz, ny, nx)), points


def _rest_pose(all_poses: list[EEPoses], mode: str) -> np.ndarray | None:
    """The joint vector to draw as the arm's reference pose on every panel.

    "home" uses the median configuration at the first frame of every episode, which is the pose
    the operator actually parks the arm in -- far more informative than the URDF zero config,
    which has the arm stuck straight out.
    """
    if mode == "none":
        return None
    if mode == "zero":
        return np.zeros(all_poses[0].q_deg.shape[1])
    starts = np.concatenate([
        np.stack([p.q_deg[p.episode_index == e][0] for e in np.unique(p.episode_index)])
        for p in all_poses
    ])
    return np.median(starts, axis=0)


def _short(repo_id: str) -> str:
    return repo_id.split("/")[-1]


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------


def summarize(model: CoverageModel) -> str:
    lines: list[str] = []
    for name in model.names:
        poses = model.poses.get(name)
        lines.append(f"\n=== {name} ===")
        if poses is not None:
            lines.append(
                f"frames {len(poses)}  episodes {len(np.unique(poses.episode_index))}  fps {poses.fps:g}"
            )
            for axis, label in enumerate("xyz"):
                v = poses.pos[:, axis]
                lines.append(
                    f"  {label} (m): min {v.min():7.3f}  p1 {np.percentile(v, 1):7.3f}  "
                    f"median {np.median(v):7.3f}  p99 {np.percentile(v, 99):7.3f}  max {v.max():7.3f}"
                )
            radius = np.linalg.norm(poses.pos, axis=1)
            lines.append(f"  radius (m): median {np.median(radius):.3f}  max {radius.max():.3f}")
            q = np.percentile(poses.tilt_deg, [25, 50, 75])
            lines.append(f"  approach tilt off vertical (deg): q25 {q[0]:.1f}  median {q[1]:.1f}  q75 {q[2]:.1f}")

        lines.append(f"  {'slice':<26} {'frames':>8} {'cells':>7} {'reachable':>10} {'coverage':>9} {'tilt':>7}")
        for s_idx in range(model.n_slices):
            view = model.query(name, *model.slice_bounds(s_idx))
            visited = view.counts > 0
            n_cells = int(visited.sum())
            if view.reach is not None:
                n_reach = int(view.reach.sum())
                pct = f"{n_cells / n_reach:.1%}" if n_reach else "n/a"
            else:
                n_reach, pct = 0, "n/a"
            tilt = view.channels["tilt"]
            tilt_median = np.nanmedian(tilt) if np.isfinite(tilt).any() else np.nan
            lines.append(
                f"  {view.label:<26} {int(view.counts.sum()):>8} {n_cells:>7} "
                f"{n_reach:>10} {pct:>9} {tilt_median:>6.1f}°"
            )
    return "\n".join(lines)


def _parse_episodes(spec: str | None) -> np.ndarray | None:
    if not spec:
        return None
    if ":" in spec:
        lo, hi = spec.split(":")
        return np.arange(int(lo), int(hi))
    return np.array([int(v) for v in spec.split(",")])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("repo_ids", nargs="+", help="One or more LeRobotDataset repo ids.")
    parser.add_argument("--source", choices=("state", "action"), default="state",
                        help="Use measured joint positions or commanded actions (default: state).")
    parser.add_argument("--episodes", default=None, help="Episode filter, e.g. '0:40' or '0,4,12'.")
    parser.add_argument("--joint-units", choices=("auto", "degrees", "normalized"), default="auto",
                        help="How to read the joint columns (default: auto-detect).")
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Calibration JSON, required only for --joint-units normalized.")
    parser.add_argument("--urdf", type=Path, default=None, help="URDF path (default: fetch the SO-101 one).")
    parser.add_argument("--tip-link", default=DEFAULT_TIP_LINK, help="End-effector frame in the URDF.")
    parser.add_argument("--approach-axis", choices=("x", "y", "z"), default="z",
                        help="Which tool axis counts as the approach direction (default: z).")

    parser.add_argument("--bins", type=int, default=32,
                        help="Bins per axis in the top view (default: 32, ~13 mm cells).")
    parser.add_argument("--z-slices", type=int, default=6, help="Number of z slices (default: 6).")
    parser.add_argument("--z-min", type=float, default=None, help="Lowest slice edge (default: 1st percentile).")
    parser.add_argument("--z-max", type=float, default=None, help="Highest slice edge (default: 99th percentile).")
    parser.add_argument("--z-edges", default=None, help="Explicit slice edges, e.g. '0.0,0.05,0.1,0.2'.")
    parser.add_argument("--reach-store", type=int, default=2_000_000,
                        help="Cap on reachability samples kept in memory for re-gridding when the "
                             "grid resolution changes (~12 MB per million).")
    parser.add_argument("--reach-samples", type=int, default=2_000_000,
                        help="Sample budget for the reachability mask (it stops early once "
                             "chunks stop finding new cells); 0 disables the mask.")
    parser.add_argument("--diff", default=None, help="'B:A' -- show coverage in B that is absent from A.")
    parser.add_argument("--no-include-base", dest="include_base", action="store_false",
                        help="Crop to the reached workspace instead of always showing the robot base.")
    parser.add_argument("--z-bounds", default="-0.03,0.03",
                        help="Initial slab for the viewer, e.g. '0.05,0.09' "
                             "(default: -0.03,0.03 -- a 60 mm band about table height).")
    parser.add_argument("--rest-pose", choices=("home", "zero", "none"), default="home",
                        help="Arm pose drawn for reference: the median episode-start pose, the URDF "
                             "zero config, or nothing (default: home).")

    parser.add_argument("--panel-color", choices=CHANNELS, default="episodes",
                        help="What the colour map means (default: episodes).")
    parser.add_argument("--log", action=argparse.BooleanOptionalAction, default=True,
                        help="Log colour scale for count/dwell/episodes (default: on).")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name.")
    parser.add_argument("--quiver", action="store_true",
                        help="Overlay mean approach direction as arrows on the top view.")

    parser.add_argument("--interactive", action="store_true", help="Open the interactive viewer window.")
    parser.add_argument("--no-blit", dest="blit", action="store_false",
                        help="Redraw the whole figure on every update instead of blitting (slower; a fallback).")
    parser.add_argument("--camera", default=None,
                        help="Video feature whose frames to pop up on click (default: the first "
                             "camera that is not wrist-mounted).")
    parser.add_argument("--no-frames", dest="show_frames", action="store_false",
                        help="Do not pop up episode camera frames when a cell is clicked.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Write static PNG + summary here.")
    parser.add_argument("--rerun", action="store_true", help="Log the 3D aggregate view to rerun.")
    parser.add_argument("--color-by", choices=("tilt", "episode", "z"), default="tilt",
                        help="Point colouring for the rerun view (default: tilt).")
    parser.add_argument("--arrow-stride", type=int, default=50,
                        help="Log one approach arrow every N frames in the rerun view.")
    parser.add_argument("--rerun-save", type=Path, default=None,
                        help="Write the rerun recording to this .rrd instead of opening the viewer.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the reachability sampler.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    chain = load_so101_chain(args.urdf, tip_link=args.tip_link)
    episodes = _parse_episodes(args.episodes)

    all_poses = [
        compute_ee_poses(
            repo_id,
            chain,
            source=args.source,
            joint_units=args.joint_units,
            calibration_path=args.calibration,
            approach_axis=args.approach_axis,
            episodes=episodes,
        )
        for repo_id in args.repo_ids
    ]
    for poses in all_poses:
        print(f"{poses.name}: {len(poses)} frames, {len(np.unique(poses.episode_index))} episodes")

    z_edges = np.array([float(v) for v in args.z_edges.split(",")]) if args.z_edges else None
    diff = tuple(args.diff.split(":")) if args.diff else None

    model = build_coverage(
        all_poses,
        bins=args.bins,
        z_slices=args.z_slices,
        z_min=args.z_min,
        z_max=args.z_max,
        z_edges=z_edges,
        reach_samples=args.reach_samples,
        chain=chain,
        diff=diff,
        source=args.source,
        seed=args.seed,
        include_base=args.include_base,
        rest_pose=args.rest_pose,
        reach_store=args.reach_store,
    )

    summary = summarize(model)
    print(summary)

    did_something = False
    if args.out_dir:
        from coverage_render import export_static

        paths = export_static(
            model,
            args.out_dir,
            channel=args.panel_color,
            log=args.log,
            cmap=args.cmap,
            quiver=args.quiver,
        )
        (args.out_dir / "summary.txt").write_text(summary.lstrip("\n") + "\n")
        for path in paths:
            print(f"Wrote {path}")
        print(f"Wrote {args.out_dir / 'summary.txt'}")
        did_something = True

    if args.rerun or args.rerun_save:
        from coverage_render import log_to_rerun

        log_to_rerun(model, chain, color_by=args.color_by, arrow_stride=args.arrow_stride,
                     urdf_path=args.urdf, save_path=args.rerun_save)
        did_something = True

    if args.interactive or not did_something:
        from coverage_viewer import launch_viewer

        bounds = tuple(float(v) for v in args.z_bounds.split(",")) if args.z_bounds else None
        launch_viewer(model, channel=args.panel_color, log=args.log, cmap=args.cmap,
                      bounds=bounds, blit=args.blit, show_frames=args.show_frames,
                      camera=args.camera)


if __name__ == "__main__":
    main()
