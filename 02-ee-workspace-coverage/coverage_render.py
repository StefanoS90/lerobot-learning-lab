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
Rendering for `ee_coverage.py`: the static PNG grid and the rerun 3D aggregate view.

The drawing primitives here (`draw_panel_background`, `draw_robot`, `make_norm`) are
shared with the interactive viewer so a cell looks and reads identically whichever way you
opened it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap, LogNorm, Normalize

from ee_coverage import CHANNEL_LABELS, CoverageModel, SliceView, _short

# Three states per cell: reachable-but-never-visited (white), unreachable (grey), and visited
# (the colormap, drawn on top). Making "never went there" visually distinct from "cannot go
# there" is the whole point of the reachability mask.
UNVISITED_COLOR = "#ffffff"
UNREACHABLE_COLOR = "#e2e2e6"
_BACKGROUND_CMAP = ListedColormap([UNVISITED_COLOR, UNREACHABLE_COLOR])

LOG_CHANNELS = ("count", "dwell", "episodes")


def make_norm(clim: tuple[float, float], channel: str, log: bool) -> Normalize:
    lo, hi = clim
    if log and channel in LOG_CHANNELS:
        return LogNorm(vmin=max(lo, 1e-6), vmax=hi)
    return Normalize(vmin=lo, vmax=hi)


def make_cmap(name: str) -> matplotlib.colors.Colormap:
    cmap = matplotlib.colormaps[name].copy()
    cmap.set_bad(alpha=0.0)  # empty cells let the background through
    return cmap


def background_grid(model: CoverageModel, view: SliceView, show_mask: bool = True) -> np.ndarray:
    """0 where reachable, 1 where not -- the two-level backdrop behind the data."""
    if view.reach is not None and show_mask:
        return (~view.reach).astype(float)
    return np.zeros(model.shape)


def draw_panel_background(ax, model: CoverageModel, view: SliceView, show_mask: bool = True):
    """Paint the reachable / unreachable backdrop for one slab. Returns the image handle."""
    bg = background_grid(model, view, show_mask)
    return ax.imshow(
        bg,
        origin="lower",
        extent=model.extent,
        cmap=_BACKGROUND_CMAP,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        zorder=0,
    )


ROBOT_COLOR = "#c62828"


def draw_robot(ax, model: CoverageModel, arrow_frac: float = 0.46) -> tuple[list, list]:
    """Show where the robot is and which way it faces, on every top-view panel.

    A coverage heatmap with no robot in it is impossible to orient: you cannot tell whether the
    data sits to the arm's left or right. So this draws a long pale heading arrow along +x
    underneath everything, the arm itself projected top-down at its reference pose, and the base
    at the origin -- which together make "the data is all on one side" legible at a glance.
    """
    x0, x1, y0, y1 = model.extent
    span = x1 - x0
    artists: list = []
    # The text labels are split out because they are the expensive part to redraw, and in a
    # side-by-side comparison every panel would otherwise repeat the same four words.
    labels: list = []

    # Heading arrow: long and pale, and drawn first so the arm reads on top of it.
    length = span * arrow_frac
    artists.append(ax.annotate(
        "", xy=(length, 0.0), xytext=(0.0, 0.0), zorder=4,
        arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.6", color=ROBOT_COLOR,
                        lw=2.0, alpha=0.30, shrinkA=0, shrinkB=0),
    ))
    # Clear of the arm, which lies along the same line.
    labels.append(ax.text(length * 0.62, span * 0.045, "arm forward", color=ROBOT_COLOR,
                          fontsize=6.5, ha="center", va="bottom", alpha=0.6, zorder=9))

    # The arm at its reference pose, projected onto the table.
    chain, rest = model.chain, model.rest_pose_deg
    if chain is not None and rest is not None:
        links = chain.fk_links(np.deg2rad(np.asarray(rest, dtype=float))[None, :])[0]
        artists += ax.plot(links[:, 0], links[:, 1], "-", color="white", lw=4.0,
                           solid_capstyle="round", zorder=6)
        artists += ax.plot(links[:, 0], links[:, 1], "-", color=ROBOT_COLOR, lw=2.0, alpha=0.95,
                           solid_capstyle="round", zorder=7)
        artists += ax.plot(links[1:-1, 0], links[1:-1, 1], "o", color=ROBOT_COLOR, ms=2.8, zorder=8)
        artists += ax.plot(links[-1, 0], links[-1, 1], "o", mfc="white", mec=ROBOT_COLOR, ms=5.5,
                           mew=1.4, zorder=8)

    # The base.
    artists += ax.plot(0, 0, marker="o", ms=9, mfc="white", mec=ROBOT_COLOR, mew=2.0, zorder=9)
    artists += ax.plot(0, 0, marker="o", ms=3, color=ROBOT_COLOR, zorder=10)
    # Offset below-right: at a parked pose the arm folds back over the base, so a label centred
    # under the origin lands on top of the elbow.
    labels.append(ax.text(span * 0.018, -span * 0.052, "base", color=ROBOT_COLOR, fontsize=6.5,
                          ha="left", va="top", zorder=10))

    # Which side is which -- the whole point of the exercise.
    labels.append(ax.text(x0 + span * 0.012, y1 - span * 0.012, "left (+y)", color=ROBOT_COLOR,
                          fontsize=6.5, ha="left", va="top", alpha=0.75, zorder=9))
    labels.append(ax.text(x0 + span * 0.012, y0 + span * 0.012, "right (−y)", color=ROBOT_COLOR,
                          fontsize=6.5, ha="left", va="bottom", alpha=0.75, zorder=9))
    return artists, labels


def draw_quiver(ax, model: CoverageModel, view: SliceView, stride: int = 4):
    """Mean approach direction projected on x-y, subsampled so the arrows stay readable."""
    counts = view.counts[::stride, ::stride]
    mean_vec = view.approach_mean[::stride, ::stride]
    xc = 0.5 * (model.x_edges[:-1] + model.x_edges[1:])[::stride]
    yc = 0.5 * (model.y_edges[:-1] + model.y_edges[1:])[::stride]
    gx, gy = np.meshgrid(xc, yc)

    valid = (counts > 0) & np.isfinite(mean_vec).all(axis=-1)
    u, v = mean_vec[..., 0], mean_vec[..., 1]
    if not valid.any():
        return None
    return ax.quiver(
        gx[valid], gy[valid], u[valid], v[valid],
        color="#37474f", scale=12, width=0.004, zorder=4,
    )


def _style_panel(ax, model: CoverageModel):
    ax.set_aspect("equal")
    ax.set_xlim(model.x_edges[0], model.x_edges[-1])
    ax.set_ylim(model.y_edges[0], model.y_edges[-1])
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#b0b0b8")


def export_static(
    model: CoverageModel,
    out_dir: Path,
    channel: str = "count",
    log: bool = False,
    cmap: str = "viridis",
    quiver: bool = False,
) -> list[Path]:
    """Write one PNG: rows are datasets (plus any diff row), columns are z slices.

    The slices come from `model.z_edges`, queried through exactly the same path the interactive
    viewer uses for arbitrary bounds -- so the figure and the window can never disagree.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, cols = len(model.names), model.n_slices
    views = [[model.query(name, *model.slice_bounds(s)) for s in range(cols)] for name in model.names]

    thickness = float(np.mean(np.diff(model.z_edges)))
    norm = make_norm(model.clim(channel, thickness), channel, log)
    colormap = make_cmap(cmap)

    fig, axes = plt.subplots(
        rows, cols, figsize=(2.5 * cols + 1.6, 2.7 * rows + 0.9), squeeze=False, constrained_layout=True
    )
    for r, name in enumerate(model.names):
        for s in range(cols):
            ax = axes[r][s]
            view = views[r][s]
            draw_panel_background(ax, model, view)
            ax.imshow(
                view.channels[channel],
                origin="lower",
                extent=model.extent,
                cmap=colormap,
                norm=norm,
                interpolation="nearest",
                zorder=2,
            )
            if quiver:
                draw_quiver(ax, model, view)
            draw_robot(ax, model)  # static figure: labels on every panel
            _style_panel(ax, model)

            visited = int((view.counts > 0).sum())
            if view.reach is not None:
                reach = int(view.reach.sum())
                pct = f"{visited / reach:.0%} of reachable" if reach else "n/a"
            else:
                pct = f"{visited} cells"
            ax.set_title(f"{view.label}\n{pct}" if r == 0 else pct, fontsize=8)
            if s == 0:
                ax.set_ylabel(f"{_short(name)}\ny (m)", fontsize=8)
            if r == rows - 1:
                ax.set_xlabel("x (m)", fontsize=8)

    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.62, aspect=28, pad=0.012)
    cbar.set_label(CHANNEL_LABELS[channel], fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"SO-101 end-effector coverage — {CHANNEL_LABELS[channel]} — source: {model.source}",
        fontsize=11,
    )

    path = out_dir / f"coverage_zslices_{model.source}_{channel}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def _colors_for(model: CoverageModel, name: str, color_by: str) -> np.ndarray:
    poses = model.poses[name]
    if color_by == "tilt":
        values = poses.tilt_deg / max(poses.tilt_deg.max(), 1e-6)
    elif color_by == "z":
        z = poses.pos[:, 2]
        values = (z - z.min()) / max(np.ptp(z), 1e-6)
    else:
        ep = poses.episode_index.astype(float)
        values = (ep % 20) / 20.0
    cmap = matplotlib.colormaps["turbo" if color_by == "episode" else "viridis"]
    return (np.asarray(cmap(values))[:, :3] * 255).astype(np.uint8)


def log_to_rerun(
    model: CoverageModel,
    chain,
    color_by: str = "tilt",
    arrow_stride: int = 50,
    urdf_path: Path | None = None,
    save_path: Path | None = None,
) -> None:
    """The aggregate 3D view: the arm for reference, every EE position, and the slice planes."""
    import rerun as rr

    from so101_fk import ensure_so101_urdf

    rr.init("so101_ee_coverage", spawn=save_path is None)
    if save_path is not None:
        rr.save(str(save_path))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    path = Path(urdf_path) if urdf_path is not None else ensure_so101_urdf()
    try:
        tree = rr.urdf.UrdfTree.from_file_path(str(path))
        tree.log_urdf_to_recording()
    except Exception as exc:  # noqa: BLE001 - the point cloud is still useful without the arm
        print(f"Could not log the URDF for reference ({exc}); continuing without it.")

    for name in model.names:
        if name not in model.poses:
            continue  # the diff pseudo-dataset has no per-frame poses of its own
        poses = model.poses[name]
        entity = f"coverage/{_short(name)}"
        rr.log(
            f"{entity}/points",
            rr.Points3D(poses.pos, colors=_colors_for(model, name, color_by), radii=0.002),
            static=True,
        )
        step = max(arrow_stride, 1)
        origins = poses.pos[::step]
        rr.log(
            f"{entity}/approach",
            rr.Arrows3D(origins=origins, vectors=poses.approach[::step] * 0.03,
                        colors=np.tile(np.array([[120, 144, 156]], dtype=np.uint8), (len(origins), 1))),
            static=True,
        )

    # The slice planes, so the 3D view and the top-down panels share a vocabulary.
    x0, x1 = model.x_edges[0], model.x_edges[-1]
    y0, y1 = model.y_edges[0], model.y_edges[-1]
    strips = [
        np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [x0, y0, z]])
        for z in model.z_edges
    ]
    rr.log(
        "coverage/slice_planes",
        rr.LineStrips3D(strips, radii=0.0008,
                        colors=np.tile(np.array([[200, 200, 210]], dtype=np.uint8), (len(strips), 1))),
        static=True,
    )
    total = sum(len(model.poses[n]) for n in model.poses)
    where = f" -> {save_path}" if save_path is not None else ""
    print(f"Logged {total} EE poses to rerun{where}.")
