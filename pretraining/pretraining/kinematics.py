"""Forward kinematics of the H1-2 from its URDF: world pose and velocity of chosen bodies from the pelvis state and the
27 joints.

Replaces mjlab's ``ForwardKinematics`` (mjlab_research, commit 6ea3e57, ``src/mjlab/utils/forward_kinematics.py``),
which is built from a MuJoCo model; ``__call__`` has its signature and outputs (world-frame positions, wxyz
quaternions, linear and angular velocities, ``(..., B, 3|4)``). Here the tree is parsed from the URDF with
``xml.etree``; joints not among the given joint names (fixed joints, the gripper hinges) are held at zero, so they
merge into their parent's transform. The pass goes body by body in tree order, from the pelvis (the floating base).
Importable without Isaac.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from pretraining.math import quat_apply, quat_from_euler_xyz, quat_mul

# Restates main's locomanipulation_game.assets.h1_2.CL_ASSETS_DIR (that module needs Isaac).
_REPO_ROOT = Path(__file__).resolve().parents[2]
CL_ASSETS_DIR = Path(os.environ.get("CL_ASSETS_DIR", _REPO_ROOT / "third_party" / "CL_Assets"))
URDF = CL_ASSETS_DIR / "ros_assets" / "h1_2_magpie.urdf"


@dataclass(frozen=True)
class UrdfTree:
    """Bodies in depth-first order from the root, each with its parent's index (-1 for the root), its joint's origin
    in the parent frame (position, wxyz quaternion), its joint axis in its own frame, and the joint's column in
    ``joint_names`` (-1 for a fixed joint or one not in ``joint_names``)."""

    body_names: list[str]
    parent: list[int]
    origin_pos: list[tuple[float, float, float]]
    origin_quat: list[tuple[float, float, float, float]]
    axis: list[tuple[float, float, float]]
    joint_index: list[int]
    joint_names: list[str]

    @classmethod
    def from_file(cls, path: str | Path, joint_names: Sequence[str]) -> UrdfTree:
        robot = ET.parse(path).getroot()
        col = {name: i for i, name in enumerate(joint_names)}
        links = {link.get("name") for link in robot.findall("link")}
        children: dict[str, list[ET.Element]] = {}
        child_links = set()
        for joint in robot.findall("joint"):
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in links or joint.get("type") == "floating":
                continue  # the world link and the floating base
            children.setdefault(parent, []).append(joint)
            child_links.add(child)
        roots = sorted(links - child_links - {"world"})
        if len(roots) != 1:
            raise ValueError(f"{path}: expected one root link below the floating base, found {roots}")

        tree = cls([roots[0]], [-1], [(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [-1],
                   list(joint_names))
        stack = [(j, 0) for j in reversed(children.get(roots[0], []))]
        while stack:
            joint, parent = stack.pop()
            origin = joint.find("origin")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
            rpy = torch.tensor([float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0")
                                .split()], dtype=torch.float64)
            axis = joint.find("axis")
            moving = joint.get("type") in ("revolute", "continuous") and joint.get("name") in col
            child = joint.find("child").get("link")
            tree.body_names.append(child)
            tree.parent.append(parent)
            tree.origin_pos.append(tuple(xyz))
            tree.origin_quat.append(tuple(quat_from_euler_xyz(*rpy).tolist()))  # URDF rpy: R = Rz(y) Ry(p) Rx(r)
            tree.axis.append(tuple(float(v) for v in axis.get("xyz").split()) if moving else (0.0, 0.0, 0.0))
            tree.joint_index.append(col[joint.get("name")] if moving else -1)
            index = len(tree.body_names) - 1
            stack.extend((j, index) for j in reversed(children.get(child, [])))
        missing = set(joint_names) - {joint_names[i] for i in tree.joint_index if i >= 0}
        if missing:
            raise ValueError(f"{path}: no revolute joints named {sorted(missing)}")
        return tree


class ForwardKinematics:
    """World pose and twist of ``body_names`` of ``tree``, computed over the bodies on their paths from the root."""

    def __init__(self, tree: UrdfTree, body_names: Sequence[str], device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        unknown = [b for b in body_names if b not in tree.body_names]
        if unknown:
            raise ValueError(f"bodies not in the URDF tree: {unknown}")
        needed = set()
        for name in body_names:
            i = tree.body_names.index(name)
            while i >= 0 and i not in needed:
                needed.add(i)
                i = tree.parent[i]
        self._order = sorted(needed)  # depth-first order: parents before children
        slot = {body: k for k, body in enumerate(self._order)}
        self._parent = [slot.get(tree.parent[b], -1) for b in self._order]
        self._col = [tree.joint_index[b] for b in self._order]
        self._rows = [slot[tree.body_names.index(b)] for b in body_names]

        def table(values):
            return torch.tensor([values[b] for b in self._order], dtype=dtype, device=device)

        self._origin_pos, self._origin_quat, self._axis = (
            table(tree.origin_pos), table(tree.origin_quat), table(tree.axis))
        # The axes again as Python floats (exact copies of the float32 entries) for the joint quaternion: torch 2.7's
        # inductor emits a Triton load of a constant row inside torch.cat that Triton 3.3 refuses to compile.
        self._axis_values = [tuple(float(v) for v in row) for row in self._axis.tolist()]
        self._body_names = list(body_names)
        self.joint_names = list(tree.joint_names)

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    def __call__(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        root_lin_vel: torch.Tensor | None = None,
        root_ang_vel: torch.Tensor | None = None,
        joint_vel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Positions ``(..., B, 3)``, wxyz quaternions ``(..., B, 4)``, linear and angular velocities ``(..., B, 3)``
        of the bodies in the world frame, from the root's world pose and twist and the joints ``(..., J)`` in
        ``joint_names`` order. Velocities left out are zero."""
        root_lin_vel = torch.zeros_like(root_pos) if root_lin_vel is None else root_lin_vel
        root_ang_vel = torch.zeros_like(root_pos) if root_ang_vel is None else root_ang_vel
        joint_vel = torch.zeros_like(joint_pos) if joint_vel is None else joint_vel
        root_quat = root_quat / root_quat.norm(dim=-1, keepdim=True)
        batch = joint_pos.shape[:-1]
        pos, quat, lin, ang = [], [], [], []
        for k, (parent, col) in enumerate(zip(self._parent, self._col)):
            if parent < 0:
                pos.append(root_pos), quat.append(root_quat), lin.append(root_lin_vel), ang.append(root_ang_vel)
                continue
            p_quat = quat[parent]
            offset = quat_apply(p_quat, self._origin_pos[k].expand(*batch, 3))
            q = quat_mul(p_quat, self._origin_quat[k].expand(*batch, 4))
            p = pos[parent] + offset
            v = lin[parent] + torch.linalg.cross(ang[parent], offset)
            w = ang[parent]
            if col >= 0:
                half = (joint_pos[..., col] / 2).unsqueeze(-1)
                s, axis = half.sin(), self._axis_values[k]
                q = quat_mul(q, torch.cat([half.cos(), s * axis[0], s * axis[1], s * axis[2]], dim=-1))
                w = w + quat_apply(q, self._axis[k].expand(*batch, 3)) * joint_vel[..., col : col + 1]
            pos.append(p), quat.append(q), lin.append(v), ang.append(w)
        return tuple(torch.stack([x[r] for r in self._rows], dim=-2) for x in (pos, quat, lin, ang))
