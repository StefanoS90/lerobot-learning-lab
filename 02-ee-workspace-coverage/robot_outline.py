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
The robot's top-down silhouette, rasterised from the URDF's own CAD meshes.

Every visual mesh is placed with forward kinematics, projected onto the table plane and filled
into a boolean mask. Two details make that come out as a clean solid shape rather than a speckle:

- binning mesh *vertices* is not enough, because flat faces carry almost none; the projected
  triangles are sampled in proportion to their area instead, so every face is filled;
- the union is then closed and hole-filled -- the same scipy steps the reachability mask uses.

Drawing one mask rather than one polygon per mesh keeps the transparency uniform: overlapping
parts do not stack into darker patches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from so101_fk import joint_limits_deg, link_transforms, load_visual_meshes

_MASK_CACHE: dict[tuple, np.ndarray] = {}


def gripper_to_jaw_deg(urdf_path: str | Path, opening: float, joint: str = "gripper") -> float:
    """Map a dataset gripper value (0 closed .. 100 open) onto the jaw joint's URDF range."""
    lo, hi = joint_limits_deg(urdf_path).get(joint, (0.0, 0.0))
    return lo + float(np.clip(opening, 0.0, 100.0)) / 100.0 * (hi - lo)


def top_view_mask(
    urdf_path: str | Path,
    q_deg: dict[str, float],
    extent: tuple[float, float, float, float],
    resolution: int = 420,
    samples_per_cell: float = 4.0,
    seed: int = 0,
) -> np.ndarray:
    """(resolution, resolution) bool mask of where the robot covers the table, indexed [y, x].

    Deterministic for a given pose and extent, and cached, so every panel of a figure shares one.
    """
    from scipy import ndimage

    key = (str(Path(urdf_path).resolve()), tuple(sorted((k, round(v, 4)) for k, v in q_deg.items())),
           tuple(round(e, 6) for e in extent), resolution)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    frames = link_transforms(urdf_path, q_deg)
    projected = []
    for link, triangles in load_visual_meshes(urdf_path):
        pose = frames.get(link)
        if pose is None:
            continue
        projected.append((triangles @ pose[:3, :3].T + pose[:3, 3])[:, :, :2])
    tri = np.concatenate(projected)

    x0, x1, y0, y1 = extent
    cell = min(x1 - x0, y1 - y0) / resolution
    area = 0.5 * np.abs(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
    # At least one sample per triangle, more for big faces, capped so a single huge face cannot
    # blow the budget.
    counts = np.minimum(np.ceil(area / (cell * cell) * samples_per_cell).astype(int) + 1, 400)
    index = np.repeat(np.arange(len(tri)), counts)
    r1, r2 = np.random.default_rng(seed).random((2, len(index)))
    s = np.sqrt(r1)  # uniform barycentric sampling inside each triangle
    points = (tri[index, 0] * (1.0 - s)[:, None] + tri[index, 1] * (s * (1.0 - r2))[:, None]
              + tri[index, 2] * (s * r2)[:, None])

    ix = np.floor((points[:, 0] - x0) / (x1 - x0) * resolution).astype(int)
    iy = np.floor((points[:, 1] - y0) / (y1 - y0) * resolution).astype(int)
    ok = (ix >= 0) & (ix < resolution) & (iy >= 0) & (iy < resolution)
    mask = np.zeros((resolution, resolution), dtype=bool)
    mask[iy[ok], ix[ok]] = True
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, np.ones((3, 3), dtype=bool)))

    _MASK_CACHE[key] = mask
    return mask
