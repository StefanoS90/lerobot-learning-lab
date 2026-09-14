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
Interactive explorer for the coverage model built by `ee_coverage.py`.

One window: a large top view, two linked side views showing where the active slab sits in the
workspace, and widgets for the z bounds, the colour channel, the dataset (including the "new
coverage" diff), and the scale.

The two z bounds are set directly and continuously, at 1 mm resolution, rather than stepped
between fixed slices -- drag either handle to move one bound, drag the middle to translate the
slab, or use the arrow keys to step it by its own thickness. Each update re-bins the frames for
the new bounds (~2 ms at 30k frames), which is far cheaper than drawing the result, so there is
no reason to restrict the bounds to a precomputed set.

Updates are blitted: only the artists that actually change are redrawn, over a cached background.
That is what keeps a continuous drag smooth -- a full figure redraw costs ~100 ms, a blitted one
~25 ms. Pass `blit=False` (CLI `--no-blit`) to fall back to full redraws.

The frame is deliberately fixed -- axes limits never move, and with "fixed scale" checked the
colour limits are re-derived only when the slab *thickness* changes, not when it translates, so
two heights stay directly comparable.
"""

from __future__ import annotations

import numpy as np

from coverage_render import (
    LOG_CHANNELS,
    UNREACHABLE_COLOR,
    UNVISITED_COLOR,
    background_grid,
    draw_panel_background,
    draw_robot,
    make_cmap,
    make_norm,
)
from ee_coverage import CHANNEL_LABELS, CHANNELS, CoverageModel, _diff_label, _short

_HELP = ("hover a cell for its numbers · click to list its episodes · "
         "↑/↓ step the slab · +/− thicken")

def _elide(text: str, max_len: int = 22) -> str:
    """Keep the tail, which is where recording runs differ (dates, suffixes)."""
    return text if len(text) <= max_len else "…" + text[-(max_len - 1):]


def _radio_labels(names: list[str]) -> list[str]:
    """Short, unique labels for the dataset selector -- long repo ids would run off the axes."""
    labels = []
    for name in names:
        if " \\ " in name:
            # Mark the diff entry as such: without it, "B \\ A" reads like a third dataset.
            b, a = name.split(" \\ ")
            label = f"diff: {_elide(b, 14)} \\ {_elide(a, 14)}"
        else:
            label = _elide(name, 34)
        while label in labels:
            label += " "
        labels.append(label)
    return labels


class _Panel:
    """One top-view map: its axes and the artists that change when the bounds move."""

    def __init__(self, name: str, ax, bg_im, im, marker, robot_artists, robot_labels, note):
        self.name = name
        self.ax = ax
        self.bg_im = bg_im
        self.im = im
        self.marker = marker
        self.robot_artists = robot_artists
        self.robot_labels = robot_labels
        self.note = note
        self.view = None

    @property
    def artists(self) -> list:
        # Draw order: backdrop, data, robot on top of the data, then the selection marker.
        # Hidden labels are skipped -- text is by far the most expensive thing to re-draw.
        return [self.bg_im, self.im, *self.robot_artists,
                *[t for t in self.robot_labels if t.get_visible()],
                self.marker, self.ax.title,
                *([self.note] if self.note.get_visible() else [])]


class CoverageViewer:
    """The window. All state lives here; every control funnels into `_refresh`."""

    MIN_THICKNESS = 0.001  # 1 mm -- the slider's resolution and the thinnest slab allowed
    # Grid resolutions the slider steps through, coarse to fine. Labelled by the resulting cell
    # size in mm, which is the quantity you actually reason about on a workspace.
    BIN_STEPS = (24, 32, 48, 64, 96, 128)
    COMPARE = "compare all"

    def __init__(self, model: CoverageModel, channel: str = "count", log: bool = False,
                 cmap: str = "viridis", bounds: tuple[float, float] | None = None,
                 blit: bool = True, show_frames: bool = True, camera: str | None = None):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        from matplotlib.widgets import CheckButtons, RadioButtons, RangeSlider, Slider

        self.model = model
        self.channel = channel
        self.log = log
        self.show_mask = model.reach_volume is not None
        self.fixed_scale = True
        self.cmap = make_cmap(cmap)
        self.blit = blit
        self.show_frames = show_frames
        self.camera = camera
        self._loaders: dict[str, object] = {}
        self._frames_fig = None
        self._diff_hint_shown = None

        # With more than one dataset loaded, show them all at once -- that side-by-side reading,
        # with the diff panel next to its two inputs, is the whole point of loading two.
        self.layout = self.COMPARE if len(model.names) > 1 else model.names[0]
        self.focus = model.names[0]

        self.z0, self.z1 = bounds if bounds is not None else model.slice_bounds(0)
        self._clim_thickness = None
        self._clim = model.clim(self.channel, self.z1 - self.z0)
        self._updating = False
        self._bg = None

        n_panels = len(model.names)
        width = min(14.5 + 2.0 * (n_panels - 1), 18.0)
        self.fig = plt.figure(figsize=(width, 8.2))
        self.fig.canvas.manager.set_window_title("SO-101 end-effector coverage")
        ctrl_w = 0.9 * 14.5 / width  # keep the control column the same physical width
        gs = GridSpec(
            3, 3, figure=self.fig,
            width_ratios=[ctrl_w, 2.6 + 1.5 * (n_panels - 1), 1.15],
            height_ratios=[1.0, 1.0, 0.42],
            left=0.035, right=0.985, top=0.90, bottom=0.06, wspace=0.22, hspace=0.32,
        )

        # The panel strip is laid out by hand inside this cell, so the number of panels can
        # change without rebuilding the figure.
        region_ax = self.fig.add_subplot(gs[0:2, 1])
        self._panel_region = region_ax.get_position()
        region_ax.remove()
        self.ax_xz = self.fig.add_subplot(gs[0, 2])
        self.ax_yz = self.fig.add_subplot(gs[1, 2])

        # --- one panel per dataset (plus the diff) --------------------------------------
        self.norm = make_norm(self._clim, self.channel, self.log)
        self.panels: list[_Panel] = []
        for name in model.names:
            ax = self.fig.add_axes(self._panel_region.bounds)
            view = model.query(name, self.z0, self.z1)
            bg_im = draw_panel_background(ax, model, view, self.show_mask)
            im = ax.imshow(
                view.channels[self.channel], origin="lower", extent=model.extent,
                cmap=self.cmap, norm=self.norm, interpolation="nearest", zorder=2,
            )
            robot, robot_labels = draw_robot(ax, model)
            marker, = ax.plot([], [], marker="s", ms=9, mfc="none", mec="#ff1744",
                              mew=1.8, zorder=11)
            # Shown when a panel has nothing to draw, so an empty map reads as an answer rather
            # than as a broken dataset.
            note = ax.text(0.5, 0.70, "", transform=ax.transAxes, ha="center", va="center",
                           fontsize=8, color="#78909c", zorder=12, visible=False,
                           linespacing=1.6)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m) — robot forward", fontsize=9)
            panel = _Panel(name, ax, bg_im, im, marker, robot, robot_labels, note)
            panel.view = view
            self.panels.append(panel)

        self.cax = self.fig.add_axes([0.0, 0.0, 0.01, 0.1])  # positioned by _layout_panels
        self.cbar = self.fig.colorbar(self.panels[0].im, cax=self.cax)

        # --- side views ----------------------------------------------------------------
        self._init_side_views()

        # --- widgets -------------------------------------------------------------------
        sliders = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 1], hspace=0.9)
        ax_slider = self.fig.add_subplot(sliders[0])
        lo, hi = model.z_range
        self.slider = RangeSlider(
            ax_slider, "z bounds", lo, hi, valinit=(self.z0, self.z1),
            valstep=self.MIN_THICKNESS, color="#5c6bc0",
        )
        self.slider.on_changed(self._on_bounds)
        # The widget would otherwise force a full canvas redraw on every value change -- which
        # is the single most expensive thing that can happen while dragging. Turn that off and
        # blit the slider's own artists along with everything else instead.
        self.slider.drawon = False

        ax_grid = self.fig.add_subplot(sliders[1])
        steps = list(self.BIN_STEPS)
        if model.bins not in steps:
            steps = sorted({*steps, model.bins})
        self._bin_steps = steps
        self.grid_slider = Slider(
            ax_grid, "grid", min(steps), max(steps), valinit=model.bins,
            valstep=steps, color="#8d6e63",
        )
        self.grid_slider.on_changed(self._on_bins)
        self.grid_slider.drawon = False

        ax_channel = self.fig.add_subplot(gs[0, 0])
        ax_channel.set_title("colour map shows", fontsize=9, loc="left")
        self.radio_channel = RadioButtons(ax_channel, CHANNELS, active=CHANNELS.index(channel))
        self.radio_channel.on_clicked(self._on_channel)

        ax_dataset = self.fig.add_subplot(gs[1, 0])
        ax_dataset.set_title("view", fontsize=9, loc="left")
        options = ([self.COMPARE] + [_short(n) for n in model.names]
                   if len(model.names) > 1 else [_short(model.names[0])])
        labels = _radio_labels(options)
        self.radio_dataset = RadioButtons(ax_dataset, labels, active=0)
        self._label_to_name = {labels[0]: self.layout}
        self._label_to_name.update(dict(zip(labels[1:], model.names, strict=True))
                                   if len(model.names) > 1 else {})
        self.radio_dataset.on_clicked(self._on_dataset)
        for text in self.radio_dataset.labels:
            text.set_fontsize(7.5)
        for text in self.radio_channel.labels:
            text.set_fontsize(9)

        ax_checks = self.fig.add_subplot(gs[2, 0])
        self.checks = CheckButtons(
            ax_checks, ["log scale", "reach mask", "fixed scale"],
            [self.log, self.show_mask, self.fixed_scale],
        )
        self.checks.on_clicked(self._on_check)

        for ax in (ax_channel, ax_dataset, ax_checks):
            ax.set_facecolor("#fafafa")
            for spine in ax.spines.values():
                spine.set_visible(False)

        # --- text ------------------------------------------------------------------------
        self.suptitle = self.fig.text(0.5, 0.955, "", fontsize=11, ha="center", va="bottom")
        # The status line owns the whole bottom row; the legend sits up in the header, so a long
        # hover readout can never collide with it.
        self.status = self.fig.text(0.035, 0.012, _HELP, fontsize=8.5, family="monospace",
                                    color="#455a64")
        self.legend_text = self.fig.text(
            0.99, 0.972,
            f"■ unreachable ({UNREACHABLE_COLOR})    □ reachable, never visited ({UNVISITED_COLOR})",
            fontsize=8, color="#78909c", ha="right", va="bottom",
        )

        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("resize_event", lambda _evt: self._invalidate_background())

        self._layout_panels()
        self._refresh(full=True)

    # --- panels ------------------------------------------------------------------------

    @property
    def visible_panels(self) -> list:
        if self.layout == self.COMPARE:
            return self.panels
        return [p for p in self.panels if p.name == self.layout]

    @property
    def dataset(self) -> str:
        """The dataset the side views, hover and click act on."""
        return self.focus

    @property
    def focused_panel(self):
        """The panel hover, click and the side views act on."""
        focused = [p for p in self.visible_panels if p.name == self.focus]
        return (focused or self.visible_panels)[0]

    @property
    def view(self):
        """The focused panel's slice, kept for the hover/click readouts."""
        return self.focused_panel.view

    # Convenience handles onto the focused panel, so callers that think in terms of a single
    # map -- the tests, mostly -- do not have to reach through `panels`.
    @property
    def ax_main(self):
        return self.focused_panel.ax

    @property
    def im(self):
        return self.focused_panel.im

    @property
    def marker(self):
        return self.focused_panel.marker

    def _layout_panels(self):
        """Place the visible panels side by side across the reserved strip."""
        box = self._panel_region
        vis = self.visible_panels
        gap = 0.014
        width = (box.width - gap * (len(vis) - 1)) / len(vis)
        for panel in self.panels:
            show = panel in vis
            panel.ax.set_visible(show)
            if show:
                i = vis.index(panel)
                panel.ax.set_position([box.x0 + i * (width + gap), box.y0, width, box.height])
            # Only the leftmost panel carries the y label and the robot's text labels; the
            # panels share axes, so repeating them costs redraw time and says nothing new.
            for text in panel.robot_labels:
                text.set_visible(show and vis.index(panel) == 0)
            # Only the leftmost panel carries the y label; the rest share its axis.
            panel.ax.set_ylabel("y (m) — robot left" if show and vis.index(panel) == 0 else "",
                                fontsize=9)
            if show and vis.index(panel) > 0:
                panel.ax.tick_params(labelleft=False)
            else:
                panel.ax.tick_params(labelleft=True)

        last = vis[-1].ax.get_position()
        self.cax.set_position([last.x1 + 0.006, box.y0 + box.height * 0.15,
                               0.009, box.height * 0.7])
        self._rebuild_animated()

    def _rebuild_animated(self):
        """Artists blitted on every update, in draw order.

        The slider parts are in here because `drawon = False` means nothing else repaints them.
        """
        artists = []
        for panel in self.visible_panels:
            artists += panel.artists
        self._animated = artists + [
            self.band_xz, self.band_yz, self.suptitle, self.status,
            self.slider.poly, *getattr(self.slider, "_handles", []), self.slider.valtext,
            self.grid_slider.poly, self.grid_slider.valtext,
        ]

    def _init_side_views(self):
        from matplotlib.colors import LogNorm

        model = self.model
        side_max = max(float(np.max(model.side_xz[n])) for n in model.side_xz)
        self._side_cmap = make_cmap("Greys")
        self._side_norm = LogNorm(vmin=1.0, vmax=max(side_max, 2.0))

        for ax, hist, edges, label in (
            (self.ax_xz, model.side_xz[self.focus], model.x_edges, "x (m)"),
            (self.ax_yz, model.side_yz[self.focus], model.y_edges, "y (m)"),
        ):
            data = np.where(hist > 0, hist, np.nan)
            im = ax.imshow(
                data, origin="lower",
                extent=(edges[0], edges[-1], model.z_edges_fine[0], model.z_edges_fine[-1]),
                cmap=self._side_cmap, norm=self._side_norm,
                interpolation="nearest", aspect="auto", zorder=2,
            )
            ax.set_xlabel(label, fontsize=8)
            ax.set_ylabel("z (m)", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.axhline(0.0, color="#c62828", lw=0.8, ls=":", zorder=3)
            if ax is self.ax_xz:
                self.im_xz = im
            else:
                self.im_yz = im

        # The band marking the active slab -- this is what ties the side views to the bounds.
        self.band_xz = self.ax_xz.axhspan(0, 0, color="#ff1744", alpha=0.22, zorder=4)
        self.band_yz = self.ax_yz.axhspan(0, 0, color="#ff1744", alpha=0.22, zorder=4)

    # --- bounds ------------------------------------------------------------------------

    @property
    def thickness(self) -> float:
        return self.z1 - self.z0

    def set_bounds(self, z0: float, z1: float):
        """Clamp to the data's z range, enforce a minimum thickness, and push to the slider."""
        lo, hi = self.model.z_range
        z0, z1 = float(min(z0, z1)), float(max(z0, z1))
        if z1 - z0 < self.MIN_THICKNESS:
            z1 = z0 + self.MIN_THICKNESS
        width = min(z1 - z0, hi - lo)
        z0 = float(np.clip(z0, lo, hi - width))
        z1 = z0 + width
        self._updating = True
        try:
            self.slider.set_val((z0, z1))
        finally:
            self._updating = False
        self._apply_bounds(z0, z1)

    def _apply_bounds(self, z0: float, z1: float):
        thickness_changed = (
            self._clim_thickness is None
            or abs((z1 - z0) - self._clim_thickness) > self.MIN_THICKNESS / 2
        )
        self.z0, self.z1 = z0, z1
        if thickness_changed and self.fixed_scale:
            # Counts scale with slab thickness, so the scale is a function of thickness: derived
            # once here, then frozen while the band merely translates.
            self._clim = self.model.clim(self.channel, self.thickness)
            self._clim_thickness = self.thickness
            self._refresh(full=True)
        else:
            self._clim_thickness = self.thickness
            self._refresh()

    # --- drawing -----------------------------------------------------------------------

    def _panel_title(self, panel) -> str:
        view = panel.view
        visited = int((view.counts > 0).sum())
        total = int(view.counts.sum())
        # Titles sit side by side, so names have to be elided or adjacent panels collide.
        width = 50 if len(self.visible_panels) == 1 else 40
        if self.model.diff is not None and panel.name == _diff_label(*self.model.diff):
            b_name, a_name = (_elide(_short(n), width) for n in self.model.diff)
            return f"only in {b_name}\nnot in {a_name}\n{total} frames in {visited} new cells"
        if view.reach is not None and int(view.reach.sum()):
            pct = f"{visited / int(view.reach.sum()):.1%} of reachable"
        else:
            pct = f"{visited} cells"
        return f"{_elide(_short(panel.name), width)}\n{total} frames — {pct}\n"

    def _empty_note(self, panel) -> str:
        """What to write across a panel that has nothing in it. Empty string when it has data.

        Lines are kept short on purpose: the note has to fit inside one panel of the comparison.
        """
        view = panel.view
        if int(view.counts.sum()) > 0:
            return ""
        if view.reverse_cells is None:
            return "no frames\nin this slab"
        b_name, a_name = (_short(n) for n in self.model.diff)
        note = (f"no new cells here\n\n{_elide(b_name, 26)}\nadds nothing over\n"
                f"{_elide(a_name, 26)}")
        if view.reverse_cells:
            note += f"\n\nthe other direction\nhas {view.reverse_cells} cells — flip it\nwith --diff (see console)"
            if self._diff_hint_shown != self.model.diff:
                self._diff_hint_shown = self.model.diff
                print(f"\nThe diff panel is empty: {_short(b_name)} adds no new cells over "
                      f"{_short(a_name)} at these bounds.\nThe reverse direction has "
                      f"{view.reverse_cells}. To see it, rerun with:\n"
                      f"  --diff {self.model.diff[1]}:{self.model.diff[0]}")
        return note

    def _refresh(self, full: bool = False):
        model = self.model

        if full:
            self.norm = make_norm(self._clim, self.channel, self.log)

        # Query each real dataset once and derive the diff from those, rather than letting the
        # diff panel re-query both of its inputs: in the three-panel comparison that is two
        # queries instead of four.
        diff_name = _diff_label(*model.diff) if model.diff is not None else None
        cache: dict[str, object] = {}
        for panel in self.visible_panels:
            if panel.name != diff_name:
                cache[panel.name] = model.query(panel.name, self.z0, self.z1)
        for panel in self.visible_panels:
            if panel.name == diff_name:
                b, a = model.diff
                view = model.diff_view(
                    cache.get(b) or model.query(b, self.z0, self.z1),
                    cache.get(a) or model.query(a, self.z0, self.z1),
                )
            else:
                view = cache[panel.name]
            panel.view = view
            grid = view.channels[self.channel]

            if full:
                panel.im.set_norm(self.norm)
            panel.im.set_data(grid)
            if self.fixed_scale or not np.isfinite(grid).any():
                panel.im.set_clim(*self._clim)
            else:
                lo, hi = float(np.nanmin(grid)), float(np.nanmax(grid))
                if self.log and self.channel in LOG_CHANNELS:
                    lo = max(lo, 1e-6)
                panel.im.set_clim(lo, hi if hi > lo else lo + 1e-9)
            panel.bg_im.set_data(background_grid(model, view, self.show_mask))
            panel.ax.set_title(self._panel_title(panel), fontsize=9)
            panel.note.set_text(self._empty_note(panel))
            panel.note.set_visible(bool(panel.note.get_text()))

        if full:
            self.cbar.update_normal(self.visible_panels[0].im)
            self.cbar.set_label(CHANNEL_LABELS[self.channel], fontsize=8)
            self.cbar.ax.tick_params(labelsize=7)

        side = self.focus if self.focus in model.side_xz else self.visible_panels[0].name
        self.im_xz.set_data(np.where(model.side_xz[side] > 0, model.side_xz[side], np.nan))
        self.im_yz.set_data(np.where(model.side_yz[side] > 0, model.side_yz[side], np.nan))
        self.ax_xz.set_title(f"x–z projection — {_elide(_short(side), 26)}", fontsize=8)
        self.ax_yz.set_title(f"y–z projection — {_elide(_short(side), 26)}", fontsize=8)

        for band in (self.band_xz, self.band_yz):
            band.set_y(self.z0)
            band.set_height(self.thickness)

        # Said once, above the strip, rather than repeated in every panel title.
        self.suptitle.set_text(
            f"z ∈ [{self.z0:.3f}, {self.z1:.3f}) m  ·  {self.thickness * 1000:.0f} mm thick"
            f"  ·  {model.cell_size_m * 1000:.1f} mm cells"
        )
        self.grid_slider.valtext.set_text(
            f"{model.cell_size_m * 1000:.1f} mm cells ({model.bins}×{model.bins})"
        )
        self.slider.valtext.set_text(
            f"{self.z0:+.3f} … {self.z1:+.3f} m  ({self.thickness * 1000:.0f} mm)"
        )

        self._draw(full)

    def _draw(self, full: bool):
        if not self.blit:
            self.fig.canvas.draw_idle()
            return
        if full or self._bg is None:
            self._capture_background()
        self._blit()

    def _invalidate_background(self):
        self._bg = None

    def _capture_background(self):
        """Render everything except the animated artists, and keep the pixels."""
        try:
            for artist in self._animated:
                artist.set_animated(True)
            self.fig.canvas.draw()
            self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        except Exception:  # noqa: BLE001 - any backend without blitting support falls back
            self.blit = False
            for artist in self._animated:
                artist.set_animated(False)
            self.fig.canvas.draw_idle()

    def _blit(self):
        try:
            self.fig.canvas.restore_region(self._bg)
            for artist in self._animated:
                target = artist.axes if getattr(artist, "axes", None) is not None else self.fig
                target.draw_artist(artist)
            self.fig.canvas.blit(self.fig.bbox)
        except Exception:  # noqa: BLE001 - fall back rather than lose the window
            self.blit = False
            for artist in self._animated:
                artist.set_animated(False)
            self.fig.canvas.draw_idle()

    # --- widget callbacks --------------------------------------------------------------

    def _on_bounds(self, val):
        if self._updating:
            return
        z0, z1 = float(val[0]), float(val[1])
        if z1 - z0 < self.MIN_THICKNESS:
            self.set_bounds(z0, z0 + self.MIN_THICKNESS)
            return
        self._clear_markers()
        self._apply_bounds(z0, z1)

    def _on_bins(self, value):
        """Re-grid at a new cell size.

        Everything downstream of the grid changes: the image shape, the colour limits (counts
        scale with cell area) and any selected cell. So this takes the full-refresh path, which
        also re-captures the blit background.
        """
        bins = int(value)
        if bins == self.model.bins:
            return
        self.model.set_bins(bins)
        self._clear_markers()
        self._clim = self.model.clim(self.channel, self.thickness)
        self._clim_thickness = self.thickness
        self._refresh(full=True)

    def _on_key(self, event):
        """Step the slab with the keyboard: it is the fastest way to walk up the workspace."""
        if event.key in ("up", "down"):
            step = self.thickness if event.key == "up" else -self.thickness
            self.set_bounds(self.z0 + step, self.z1 + step)
        elif event.key in ("+", "=", "-", "_"):
            factor = 1.25 if event.key in ("+", "=") else 1 / 1.25
            mid = 0.5 * (self.z0 + self.z1)
            half = max(self.thickness * factor, self.MIN_THICKNESS) / 2
            self.set_bounds(mid - half, mid + half)

    def _on_channel(self, label):
        self.channel = label
        if self.fixed_scale:
            self._clim = self.model.clim(self.channel, self.thickness)
            self._clim_thickness = self.thickness
        self._refresh(full=True)

    def _on_dataset(self, label):
        """Switch between the side-by-side comparison and a single enlarged panel."""
        choice = self._label_to_name[label]
        self.layout = choice
        if choice != self.COMPARE:
            self.focus = choice
        elif self.model.diff is not None and self.focus == _diff_label(*self.model.diff):
            # The diff has no frames of its own, so leaving the side views pointed at it after
            # returning to the comparison is more confusing than useful.
            self.focus = self.panels[0].name
        self._clear_markers()
        self._layout_panels()
        self._refresh(full=True)

    def _clear_markers(self):
        for panel in self.panels:
            panel.marker.set_data([], [])

    def _on_check(self, label):
        if label == "log scale":
            self.log = not self.log
        elif label == "reach mask":
            self.show_mask = not self.show_mask
        else:
            self.fixed_scale = not self.fixed_scale
            if self.fixed_scale:
                self._clim = self.model.clim(self.channel, self.thickness)
                self._clim_thickness = self.thickness
        self._refresh(full=True)

    # --- hover / click -----------------------------------------------------------------

    def _cell_at(self, event):
        """Which panel the pointer is over, and which cell -- or None."""
        model = self.model
        for panel in self.visible_panels:
            if event.inaxes is panel.ax and event.xdata is not None:
                ny, nx = model.shape
                ix = int(np.floor((event.xdata - model.x_edges[0])
                                  / (model.x_edges[-1] - model.x_edges[0]) * nx))
                iy = int(np.floor((event.ydata - model.y_edges[0])
                                  / (model.y_edges[-1] - model.y_edges[0]) * ny))
                if 0 <= ix < nx and 0 <= iy < ny:
                    return panel, iy, ix
        return None

    def _on_hover(self, event):
        hit = self._cell_at(event)
        if hit is None:
            if self.status.get_text() != _HELP:
                self.status.set_text(_HELP)
                self._draw(full=False)
            return

        panel, iy, ix = hit
        # The side views follow the panel you are pointing at, so they always describe the map
        # you are reading.
        refocus = panel.name != self.focus and panel.name in self.model.side_xz
        if refocus:
            self.focus = panel.name

        view = panel.view
        n = view.counts[iy, ix]
        head = f"{_short(panel.name)}  x={event.xdata:+.3f} y={event.ydata:+.3f}"
        if n == 0:
            reachable = view.reach is None or view.reach[iy, ix]
            self.status.set_text(
                f"{head}  ·  never visited ({'reachable' if reachable else 'unreachable'})"
            )
        else:
            eps = view.channels["episodes"][iy, ix]
            tilt = view.channels["tilt"][iy, ix]
            std = view.tilt_std[iy, ix]
            spread = view.channels["tilt_spread"][iy, ix]
            self.status.set_text(
                f"{head}  ·  {int(n)} frames · {n / self.model.fps:.1f} s · {int(eps)} episodes"
                f" · tilt {tilt:.1f}° ± {std:.1f}° · approach spread {spread:.1f}°"
            )
        if refocus:
            self._refresh()
        else:
            self._draw(full=False)

    def _on_click(self, event):
        hit = self._cell_at(event)
        if hit is None:
            return
        panel, iy, ix = hit
        model, view = self.model, panel.view
        xc = 0.5 * (model.x_edges[ix] + model.x_edges[ix + 1])
        yc = 0.5 * (model.y_edges[iy] + model.y_edges[iy + 1])
        # Mark the same cell on every panel: the point of the comparison is reading one location
        # across all of them at once.
        for other in self.visible_panels:
            other.marker.set_data([xc], [yc])

        episodes = view.episodes_in_cell.get((iy, ix))
        if episodes is None or view.counts[iy, ix] == 0:
            print(f"\n{_short(panel.name)} · cell x={xc:.3f} y={yc:.3f} {view.label}: no frames")
        else:
            listed = ", ".join(str(int(e)) for e in episodes[:40])
            more = "" if len(episodes) <= 40 else f", … (+{len(episodes) - 40} more)"
            repo = panel.name if panel.name in model.poses else (model.diff or (panel.name,))[0]
            print(
                f"\n{_short(panel.name)} · cell x={xc:.3f} y={yc:.3f} {view.label}"
                f" · {int(view.counts[iy, ix])} frames"
                f"\n  episodes ({len(episodes)}): {listed}{more}"
                f"\n  inspect one with: lerobot-dataset-viz --repo-id {repo} "
                f"--episode-index {int(episodes[0])}"
            )
            self._show_frames(panel.name, episodes, f"x={xc:.3f} y={yc:.3f}  {view.label}")
        self._draw(full=False)

    def _frame_loader(self, name: str):
        """One loader per dataset, built on first use -- no video is touched until you click."""
        from episode_frames import EpisodeFrameLoader

        # The diff entry is a pseudo-dataset with no frames of its own; its cells come from the
        # first half of the diff pair, so show that dataset's episodes.
        repo_id = name if name in self.model.poses else (self.model.diff or (name,))[0]
        if repo_id in self._loaders:
            return self._loaders[repo_id]
        try:
            loader = EpisodeFrameLoader(repo_id, camera=self.camera)
        except Exception as exc:  # noqa: BLE001 - a missing camera must never break the click
            print(f"Camera frames unavailable for {repo_id}: {exc}")
            loader = None
        self._loaders[repo_id] = loader
        return loader

    def _show_frames(self, name: str, episodes, title: str):
        if not self.show_frames:
            return
        loader = self._frame_loader(name)
        if loader is None:
            return
        from episode_frames import show_episode_frames

        self._frames_fig = show_episode_frames(loader, episodes, title, figure=self._frames_fig)


def launch_viewer(model: CoverageModel, channel: str = "count", log: bool = False,
                  cmap: str = "viridis", bounds: tuple[float, float] | None = None,
                  blit: bool = True, show_frames: bool = True,
                  camera: str | None = None) -> CoverageViewer:
    import matplotlib.pyplot as plt

    viewer = CoverageViewer(model, channel=channel, log=log, cmap=cmap, bounds=bounds, blit=blit,
                            show_frames=show_frames, camera=camera)
    plt.show()
    return viewer
