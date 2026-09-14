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
Vectorised forward kinematics for a URDF serial chain (SO-101 by default).

LeRobot ships `lerobot.model.RobotKinematics`, which wraps `placo`. That is the right tool
when you need IK, but it is a poor fit for dataset-wide analysis: it is an optional extra
with an awkward ABI pin on Ubuntu 24.04 (see lerobot's pyproject.toml), and its FK sets one
joint at a time and re-runs `update_kinematics` per call, so scoring 30k frames means 30k
python round-trips.

The SO-101 is a plain 5-joint serial chain
(base_link -> shoulder_pan -> shoulder_lift -> elbow_flex -> wrist_flex -> wrist_roll ->
gripper_frame_link), so the whole of FK is a fixed 4x4 per joint followed by a rotation about
that joint's axis. Done with batched Rodrigues that is one matmul per joint for the entire
dataset at once -- ~30k frames in milliseconds, which is also what makes the 200k-sample
reachability mask in `ee_coverage.py` affordable.

Conventions match lerobot's (`robot_kinematic_processor.compute_forward_kinematics_joints_to_ee`):
positions in metres in the robot base frame, and the tool axis is the +z column of the
`gripper_frame_link` rotation.

Usage:
    # Fetch the URDF, print the chain, and self-check the geometry.
    python so101_fk.py --self-test

    # Cross-check against placo, if you have installed it (uv sync --extra placo-dep).
    python so101_fk.py --self-test --placo
"""

from __future__ import annotations

import argparse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The bucket lerobot's own isaac example syncs (examples/isaac_teleop_to_so101/common.py).
# It carries the meshes too, which the rerun 3D view wants.
SO101_URDF_BUCKET = "hf://buckets/lerobot/robot-urdfs/so101"
SO101_URDF_NAME = "so101_new_calib.urdf"
# Meshless fallback: enough for FK, not enough for a textured 3D view.
SO101_URDF_RAW_URL = (
    "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101/so101_new_calib.urdf"
)

DEFAULT_TIP_LINK = "gripper_frame_link"
DEFAULT_BASE_LINK = "base_link"


def _default_urdf_dir() -> Path:
    """Cache URDFs next to lerobot's own, so a sync done by either side is reused."""
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME) / "robot-urdfs" / "so101"


def ensure_so101_urdf(cache_dir: Path | None = None) -> Path:
    """Return a local path to the SO-101 URDF, downloading it once if needed."""
    cache_dir = Path(cache_dir) if cache_dir is not None else _default_urdf_dir()
    urdf_path = cache_dir / SO101_URDF_NAME
    if urdf_path.exists():
        return urdf_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import sync_bucket

        print(f"Syncing {SO101_URDF_BUCKET} -> {cache_dir} ...")
        sync_bucket(SO101_URDF_BUCKET, str(cache_dir), quiet=True)
    except Exception as exc:  # noqa: BLE001 - any hub/network failure should fall back
        print(f"Bucket sync failed ({exc}); falling back to the raw URDF (no meshes).")
        urllib.request.urlretrieve(SO101_URDF_RAW_URL, urdf_path)

    if not urdf_path.exists():
        raise FileNotFoundError(f"Could not obtain {SO101_URDF_NAME} in {cache_dir}")
    return urdf_path


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw (i.e. R = Rz(yaw) @ Ry(pitch) @ Rx(roll))."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _floats(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    return np.asarray([float(v) for v in text.split()], dtype=float)


@dataclass
class _Joint:
    name: str
    is_fixed: bool
    origin: np.ndarray  # 4x4, the constant parent->joint transform
    axis: np.ndarray  # unit rotation axis in the joint frame
    lower: float  # radians
    upper: float


class UrdfChain:
    """The serial chain of a URDF between two links, with batched forward kinematics."""

    def __init__(self, joints: list[_Joint], tip_link: str, base_link: str):
        self._joints = joints
        self.tip_link = tip_link
        self.base_link = base_link

    @classmethod
    def from_file(
        cls,
        urdf_path: str | Path,
        tip_link: str = DEFAULT_TIP_LINK,
        base_link: str = DEFAULT_BASE_LINK,
    ) -> UrdfChain:
        root = ET.parse(str(urdf_path)).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}

        # Walk tip -> base, then reverse: a URDF lists joints in arbitrary order, and the
        # child->joint map is what makes the walk unambiguous.
        walked: list[ET.Element] = []
        link = tip_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(f"Link {link!r} has no parent joint; {base_link!r} unreachable.")
            joint = by_child[link]
            walked.append(joint)
            link = joint.find("parent").get("link")
        walked.reverse()

        joints: list[_Joint] = []
        for element in walked:
            origin_el = element.find("origin")
            xyz = _floats(origin_el.get("xyz") if origin_el is not None else None, (0.0, 0.0, 0.0))
            rpy = _floats(origin_el.get("rpy") if origin_el is not None else None, (0.0, 0.0, 0.0))
            origin = np.eye(4)
            origin[:3, :3] = _rpy_to_matrix(*rpy)
            origin[:3, 3] = xyz

            joint_type = element.get("type")
            is_fixed = joint_type == "fixed"
            if joint_type not in ("fixed", "revolute", "continuous"):
                raise NotImplementedError(f"Joint {element.get('name')!r} has unsupported type {joint_type!r}")

            axis_el = element.find("axis")
            axis = _floats(axis_el.get("xyz") if axis_el is not None else None, (1.0, 0.0, 0.0))
            norm = np.linalg.norm(axis)
            axis = axis / norm if norm > 0 else np.array([1.0, 0.0, 0.0])

            limit_el = element.find("limit")
            lower = float(limit_el.get("lower")) if limit_el is not None and limit_el.get("lower") else -np.pi
            upper = float(limit_el.get("upper")) if limit_el is not None and limit_el.get("upper") else np.pi

            joints.append(
                _Joint(
                    name=element.get("name"),
                    is_fixed=is_fixed,
                    origin=origin,
                    axis=axis,
                    lower=lower,
                    upper=upper,
                )
            )
        return cls(joints, tip_link=tip_link, base_link=base_link)

    @property
    def joint_names(self) -> list[str]:
        """Actuated joint names, base -> tip. This is the column order `fk` expects."""
        return [j.name for j in self._joints if not j.is_fixed]

    @property
    def limits_rad(self) -> np.ndarray:
        """(n_joints, 2) lower/upper limits in radians, in `joint_names` order."""
        return np.array([(j.lower, j.upper) for j in self._joints if not j.is_fixed])

    def fk(self, q_rad: np.ndarray) -> np.ndarray:
        """Batched forward kinematics.

        Args:
            q_rad: (N, n_joints) actuated joint angles in radians, in `joint_names` order.

        Returns:
            (N, 4, 4) tip-link poses in the base frame.
        """
        q_rad = np.asarray(q_rad, dtype=np.float64)
        if q_rad.ndim == 1:
            q_rad = q_rad[None, :]
        n_actuated = len(self.joint_names)
        if q_rad.shape[1] != n_actuated:
            raise ValueError(f"Expected (N, {n_actuated}) joint angles, got {q_rad.shape}")

        n = q_rad.shape[0]
        t = np.tile(np.eye(4), (n, 1, 1))
        col = 0
        for joint in self._joints:
            t = t @ joint.origin
            if joint.is_fixed:
                continue
            theta = q_rad[:, col]
            col += 1
            # Rodrigues about a constant axis, vectorised over the batch.
            ax = joint.axis
            k = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
            rot = np.eye(3) + np.sin(theta)[:, None, None] * k + (1.0 - np.cos(theta))[:, None, None] * (k @ k)
            step = np.tile(np.eye(4), (n, 1, 1))
            step[:, :3, :3] = rot
            t = t @ step
        return t

    def fk_links(self, q_rad: np.ndarray) -> np.ndarray:
        """Positions of every link frame along the chain, base first, tip last.

        Args:
            q_rad: (N, n_joints) actuated joint angles in radians.

        Returns:
            (N, n_links + 1, 3) link-origin positions, for drawing the arm as a polyline.
        """
        q_rad = np.asarray(q_rad, dtype=np.float64)
        if q_rad.ndim == 1:
            q_rad = q_rad[None, :]
        n = q_rad.shape[0]
        t = np.tile(np.eye(4), (n, 1, 1))
        points = [t[:, :3, 3].copy()]
        col = 0
        for joint in self._joints:
            t = t @ joint.origin
            if not joint.is_fixed:
                theta = q_rad[:, col]
                col += 1
                ax = joint.axis
                k = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
                rot = (np.eye(3) + np.sin(theta)[:, None, None] * k
                       + (1.0 - np.cos(theta))[:, None, None] * (k @ k))
                step = np.tile(np.eye(4), (n, 1, 1))
                step[:, :3, :3] = rot
                t = t @ step
            points.append(t[:, :3, 3].copy())
        return np.stack(points, axis=1)

    def sample_reachable_positions(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """(n_samples, 3) tip positions from configurations drawn uniformly inside the joint limits."""
        limits = self.limits_rad
        q = rng.uniform(limits[:, 0], limits[:, 1], size=(n_samples, limits.shape[0]))
        return self.fk(q)[:, :3, 3]


def load_so101_chain(urdf_path: str | Path | None = None, tip_link: str = DEFAULT_TIP_LINK) -> UrdfChain:
    """Convenience: fetch the SO-101 URDF if no path is given, and parse its chain."""
    path = Path(urdf_path) if urdf_path is not None else ensure_so101_urdf()
    return UrdfChain.from_file(path, tip_link=tip_link)


def _self_test(check_placo: bool) -> int:
    urdf_path = ensure_so101_urdf()
    print(f"URDF: {urdf_path}")
    chain = load_so101_chain(urdf_path)
    print(f"chain: {chain.base_link} -> {chain.tip_link}")
    print(f"actuated joints: {chain.joint_names}")
    print("limits (deg):")
    for name, (lo, hi) in zip(chain.joint_names, np.rad2deg(chain.limits_rad), strict=True):
        print(f"  {name:<16} [{lo:7.1f}, {hi:7.1f}]")

    failures = 0

    # Geometry: at the zero configuration the SO-101 points forward along +x.
    zero = chain.fk(np.zeros((1, len(chain.joint_names))))[0]
    print(f"\nzero-config tip position (m): {zero[:3, 3].round(4)}")
    reach = float(np.linalg.norm(zero[:3, 3]))
    if not 0.30 < reach < 0.55:
        print(f"FAIL: zero-config reach {reach:.3f} m is implausible for an SO-101")
        failures += 1

    # Rigid-transform sanity over random configurations.
    rng = np.random.default_rng(0)
    limits = chain.limits_rad
    q = rng.uniform(limits[:, 0], limits[:, 1], size=(1000, limits.shape[0]))
    t = chain.fk(q)
    rot = t[:, :3, :3]
    orthonormal = np.abs(rot @ rot.transpose(0, 2, 1) - np.eye(3)).max()
    dets = np.linalg.det(rot)
    print(f"random configs: max |R R^T - I| = {orthonormal:.2e}, det(R) in [{dets.min():.6f}, {dets.max():.6f}]")
    if orthonormal > 1e-9 or np.abs(dets - 1.0).max() > 1e-9:
        print("FAIL: rotation blocks are not proper rotations")
        failures += 1

    radii = np.linalg.norm(t[:, :3, 3], axis=1)
    print(f"sampled reach (m): min {radii.min():.3f}, median {np.median(radii):.3f}, max {radii.max():.3f}")
    if radii.max() > 0.6:
        print(f"FAIL: max sampled reach {radii.max():.3f} m exceeds the SO-101 envelope")
        failures += 1

    # Batched FK must agree with one-at-a-time FK.
    single = np.stack([chain.fk(q[i : i + 1])[0] for i in range(20)])
    batch_err = np.abs(single - t[:20]).max()
    print(f"batched vs per-sample FK: max abs diff {batch_err:.2e}")
    if batch_err > 1e-12:
        print("FAIL: batching changes the result")
        failures += 1

    if check_placo:
        try:
            from lerobot.model import RobotKinematics
        except ImportError as exc:
            print(f"\nplaco cross-check skipped: {exc}")
        else:
            try:
                kin = RobotKinematics(str(urdf_path), chain.tip_link, joint_names=chain.joint_names)
            except ImportError as exc:
                print(f"\nplaco cross-check skipped (placo not installed): {exc}")
            else:
                ours = chain.fk(q[:100])
                theirs = np.stack([kin.forward_kinematics(np.rad2deg(row)) for row in q[:100]])
                err = np.abs(ours - theirs).max()
                print(f"\nplaco cross-check over 100 configs: max abs diff {err:.2e}")
                if err > 1e-9:
                    print("FAIL: disagrees with lerobot's RobotKinematics")
                    failures += 1

    print("\nself-test: " + ("PASSED" if failures == 0 else f"{failures} FAILURE(S)"))
    return 1 if failures else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--self-test", action="store_true", help="Fetch the URDF and check the chain geometry.")
    parser.add_argument("--placo", action="store_true", help="Also cross-check FK against lerobot's placo wrapper.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.self_test:
        build_arg_parser().print_help()
        return
    raise SystemExit(_self_test(check_placo=args.placo))


if __name__ == "__main__":
    main()
