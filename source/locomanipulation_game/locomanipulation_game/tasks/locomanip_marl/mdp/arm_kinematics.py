"""The arms' kinematics in the pelvis frame from the robot's URDF, batched in torch: forward kinematics, the wrists'
Jacobians, a damped-least-squares solve of joint angles for wrist poses, and self-collision by cuRobo's spheres.

Plain torch (no Isaac Lab), so the MuJoCo plant and the deploy can run the same code. Quaternions are w, x, y, z. The
torso joint is held at its angle (torso_q). Spheres and their ignore lists come from CL_Assets'
curobo_assets/h1_2_magpie_curobo.yml (the Magpie gripper is the wrist_yaw_link's spheres); two spheres collide when
their centres are closer than their radii plus twice the buffer.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import yaml


def _rpy(r: float, p: float, y: float) -> torch.Tensor:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return torch.tensor([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                         [-sp, cp * sr, cp * cr]], dtype=torch.float64)


def _axis_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotations about the unit axis (3,) by angle (...)."""
    x, y, z = axis.tolist()
    c, s = torch.cos(angle), torch.sin(angle)
    t = 1.0 - c
    rows = [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]
    return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


def quat_from_matrix(R: torch.Tensor) -> torch.Tensor:
    """(..., 4) w x y z, w >= 0, of rotation matrices (..., 3, 3)."""
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    w = 0.5 * torch.sqrt((1 + m00 + m11 + m22).clamp(min=0))
    x = 0.5 * torch.sqrt((1 + m00 - m11 - m22).clamp(min=0)).copysign(R[..., 2, 1] - R[..., 1, 2])
    y = 0.5 * torch.sqrt((1 - m00 + m11 - m22).clamp(min=0)).copysign(R[..., 0, 2] - R[..., 2, 0])
    z = 0.5 * torch.sqrt((1 - m00 - m11 + m22).clamp(min=0)).copysign(R[..., 1, 0] - R[..., 0, 1])
    q = torch.stack([w, x, y, z], dim=-1)
    return q / q.norm(dim=-1, keepdim=True)


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def rotation_error(R_goal: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle of R_goal R^T (the rotation that takes R to R_goal, world/pelvis-frame)."""
    q = quat_from_matrix(R_goal @ R.transpose(-1, -2))
    v = q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, q[..., :1])
    return torch.where(s > 1e-9, v / s.clamp(min=1e-12) * angle, 2.0 * v)


class ArmKinematics:
    """Both arms of the URDF's robot from the pelvis: fk, jacobian, solve, collides (module doc)."""

    ARMS = ("left", "right")
    JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")

    def __init__(self, urdf: str | Path, curobo_yml: str | Path, device: str = "cpu", torso_q: float = 0.0,
                 dtype: torch.dtype = torch.float32):
        self.device, self.dtype = device, dtype
        root = ET.parse(str(urdf)).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}

        def origin(name):
            o = joints[name].find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
            axis = torch.tensor([float(v) for v in joints[name].find("axis").get("xyz").split()], dtype=torch.float64)
            return torch.tensor(xyz, dtype=torch.float64), _rpy(*rpy), axis / axis.norm()

        # the torso link in the pelvis frame, fixed
        p, R, axis = origin("torso_joint")
        self.torso_R = (R @ _axis_rotation(axis, torch.tensor(torso_q, dtype=torch.float64))).to(device, dtype)
        self.torso_p = p.to(device, dtype)
        # per arm and joint: the joint origin in its parent link, its fixed rotation, its axis, and the axis's cross
        # product matrix K and K^2 (Rodrigues: R(a) = I + sin(a) K + (1 - cos(a)) K^2); (2, 7, ...)
        o = [[origin(f"{arm}_{j}_joint") for j in self.JOINTS] for arm in self.ARMS]
        self.offsets = torch.stack([torch.stack([x[0] for x in arm]) for arm in o]).to(device, dtype)
        self.rots = torch.stack([torch.stack([x[1] for x in arm]) for arm in o]).to(device, dtype)
        axes = torch.stack([torch.stack([x[2] for x in arm]) for arm in o])
        K = torch.zeros(2, 7, 3, 3, dtype=torch.float64)
        K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -axes[..., 2], axes[..., 1], -axes[..., 0]
        K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = axes[..., 2], -axes[..., 1], axes[..., 0]
        self.axes, self.K, self.KK = axes.to(device, dtype), K.to(device, dtype), (K @ K).to(device, dtype)
        # per arm link its mass and centre of mass (in the link), for gravity_torques; the wrist_yaw_link's with
        # everything fixed beyond it (the gripper, its own joints at 0) lumped in
        inertial = {}
        for link in root.findall("link"):
            m, o = link.find("inertial/mass"), link.find("inertial/origin")
            com = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
            inertial[link.get("name")] = (float(m.get("value")) if m is not None else 0.0, torch.tensor(com, dtype=torch.float64))
        children = {}
        for j in joints.values():
            children.setdefault(j.find("parent").get("link"), []).append(j)
        masses, coms = torch.zeros(2, 7, dtype=torch.float64), torch.zeros(2, 7, 3, dtype=torch.float64)
        for a, arm in enumerate(self.ARMS):
            for k, jn in enumerate(self.JOINTS):
                masses[a, k], coms[a, k] = inertial[f"{arm}_{jn}_link"]
            # the wrist's subtree beyond the chain, each link's centre of mass in the wrist frame
            m_sum, mc_sum = masses[a, 6].clone(), masses[a, 6] * coms[a, 6]
            stack = [(f"{arm}_wrist_yaw_link", torch.zeros(3, dtype=torch.float64), torch.eye(3, dtype=torch.float64))]
            while stack:
                parent, p, R = stack.pop()
                for j in children.get(parent, []):
                    o = j.find("origin")
                    xyz = torch.tensor([float(v) for v in o.get("xyz", "0 0 0").split()], dtype=torch.float64) if o is not None else torch.zeros(3, dtype=torch.float64)
                    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
                    child = j.find("child").get("link")
                    pc, Rc = p + R @ xyz, R @ _rpy(*rpy)
                    mass, com = inertial.get(child, (0.0, torch.zeros(3, dtype=torch.float64)))
                    m_sum, mc_sum = m_sum + mass, mc_sum + mass * (pc + Rc @ com)
                    stack.append((child, pc, Rc))
            masses[a, 6], coms[a, 6] = m_sum, mc_sum / m_sum
        self.masses, self.coms = masses.to(device, dtype), coms.to(device, dtype)
        # cuRobo's spheres: per sphere its link (index into links()), centre in the link, radius
        cfg = yaml.safe_load(open(curobo_yml))["robot_cfg"]["kinematics"]
        self.buffer = float(cfg["collision_sphere_buffer"])
        names = self.link_names()
        link, centre, radius = [], [], []
        for name, spheres in cfg["collision_spheres"].items():
            for s in spheres:
                link.append(names.index(name)); centre.append(s["center"]); radius.append(s["radius"])
        self.sphere_link = torch.tensor(link, device=device)
        self.sphere_centre = torch.tensor(centre, device=device, dtype=dtype)
        self.sphere_radius = torch.tensor(radius, device=device, dtype=dtype)
        ignore = {(a, b) for a, bs in cfg["self_collision_ignore"].items() for b in bs}
        sl = [names[i] for i in link]
        pairs = [(i, j) for i in range(len(sl)) for j in range(i + 1, len(sl))
                 if sl[i] != sl[j] and (sl[i], sl[j]) not in ignore and (sl[j], sl[i]) not in ignore]
        self.pairs = torch.tensor(pairs, device=device)

    def link_names(self) -> list[str]:
        """links() order: pelvis, torso, then each arm's 7 links."""
        return ["pelvis", "torso_link"] + [f"{a}_{j}_link" for a in self.ARMS for j in self.JOINTS]

    def links(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q (N, 2, 7) -> each link's position (N, 16, 3) and rotation (N, 16, 3, 3) in the pelvis frame."""
        p, R = self._arm_links(q)
        n = q.shape[0]
        eye = torch.eye(3, device=self.device, dtype=self.dtype).expand(n, 1, 3, 3)
        ps = torch.cat([torch.zeros(n, 1, 3, device=self.device, dtype=self.dtype), self.torso_p.expand(n, 1, 3),
                        p.reshape(n, 14, 3)], dim=1)
        Rs = torch.cat([eye, self.torso_R.expand(n, 1, 3, 3), R.reshape(n, 14, 3, 3)], dim=1)
        return ps, Rs

    def _arm_links(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """q (N, 2, 7) -> the arms' links' positions (N, 2, 7, 3) and rotations (N, 2, 7, 3, 3), pelvis frame."""
        s, c = torch.sin(q)[..., None, None], torch.cos(q)[..., None, None]
        local = self.rots @ (torch.eye(3, device=self.device, dtype=self.dtype) + s * self.K + (1.0 - c) * self.KK)
        p = self.torso_p.expand(q.shape[0], 2, 3)
        R = self.torso_R.expand(q.shape[0], 2, 3, 3)
        ps, Rs = [], []
        for k in range(7):
            p = p + (R @ self.offsets[:, k].unsqueeze(-1)).squeeze(-1)
            R = R @ local[:, :, k]
            ps.append(p); Rs.append(R)
        return torch.stack(ps, dim=2), torch.stack(Rs, dim=2)

    def gravity_torques(self, q: torch.Tensor, gravity_b: torch.Tensor) -> torch.Tensor:
        """(N, 2, 7): the torque gravity puts on each arm joint at q (N, 2, 7), from the URDF's masses (the gripper
        lumped into the wrist), gravity_b (N, 3) the gravity vector (m/s^2) in the pelvis frame. A PD holds q against
        it with its target at q - torque / kp."""
        p, R = self._arm_links(q)
        z = (R @ self.axes.unsqueeze(-1)).squeeze(-1)  # (N, 2, 7, 3) joint axes
        mc = self.masses.unsqueeze(-1) * (p + (R @ self.coms.unsqueeze(-1)).squeeze(-1))  # m x centre of mass
        # per joint, the links from it outward: sum m c - (sum m) p_joint, crossed with gravity
        mc_out = mc.flip(2).cumsum(2).flip(2)
        m_out = self.masses.flip(1).cumsum(1).flip(1).unsqueeze(-1)
        lever = mc_out - m_out * p
        g = gravity_b.to(self.dtype).view(-1, 1, 1, 3).expand_as(lever)
        return (z * torch.cross(lever, g, dim=-1)).sum(-1)

    def _jacobian(self, p: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """(N, 2, 6, 7) of each wrist origin, pelvis frame, from _arm_links() output."""
        z = (R @ self.axes.unsqueeze(-1)).squeeze(-1)  # (N, 2, 7, 3) joint axes
        lin = torch.cross(z, p[:, :, 6:7] - p, dim=-1)
        return torch.cat([lin, z], dim=-1).transpose(-1, -2)

    def solve(self, goal: torch.Tensor, q0: torch.Tensor, low: torch.Tensor, high: torch.Tensor, iters: int = 60,
              damping: float = 0.05, rot_weight: float = 0.5, max_step: float = 0.2) -> torch.Tensor:
        """Joint angles (N, 2, 7) putting each wrist at goal (N, 2, 7: position, quaternion), from q0, inside
        [low, high] (2, 7) or (N, 2, 7): damped least squares on the pose error [position, rot_weight x axis-angle],
        each step capped at max_step per joint and clamped into the bounds."""
        q = q0.clone()
        W = torch.tensor([1, 1, 1, rot_weight, rot_weight, rot_weight], device=self.device, dtype=self.dtype)
        Rg = matrix_from_quat(goal[..., 3:])
        eye = torch.eye(6, device=self.device, dtype=self.dtype) * damping ** 2
        for _ in range(iters):
            p, R = self._arm_links(q)
            err = torch.cat([goal[..., :3] - p[:, :, 6], rotation_error(Rg, R[:, :, 6])], -1) * W
            J = self._jacobian(p, R) * W.view(6, 1)
            step = (J.transpose(-1, -2) @ torch.linalg.solve(J @ J.transpose(-1, -2) + eye, err.unsqueeze(-1)))
            q = torch.maximum(torch.minimum(q + step.squeeze(-1).clamp(-max_step, max_step), high), low)
        return q

    def spheres(self, q: torch.Tensor) -> torch.Tensor:
        """(N, S, 3) sphere centres in the pelvis frame."""
        p, R = self.links(q)
        return p[:, self.sphere_link] + (R[:, self.sphere_link] @ self.sphere_centre.unsqueeze(-1)).squeeze(-1)

    def collides(self, q: torch.Tensor) -> torch.Tensor:
        """(N,) True where any checked sphere pair overlaps (radii plus twice the buffer)."""
        c = self.spheres(q)
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        d = (c[:, i] - c[:, j]).norm(dim=-1)
        return (d < self.sphere_radius[i] + self.sphere_radius[j] + 2 * self.buffer).any(dim=1)

    def path_collides(self, q_from: torch.Tensor, q_to: torch.Tensor, step: float = 0.05,
                      chunk: int = 65536) -> torch.Tensor:
        """(N,) True where the straight joint-space path from q_from to q_to (both (N, 2, 7)), sampled every step rad
        of its largest joint move, collides anywhere after its start."""
        if len(q_from) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        n = max(int(math.ceil(float((q_to - q_from).abs().max()) / step)), 1)
        frac = torch.arange(1, n + 1, device=self.device, dtype=self.dtype).view(n, 1, 1, 1) / n
        path = (q_from + (q_to - q_from) * frac).reshape(-1, 2, 7)  # sample-major: (n * N, 2, 7)
        hit = torch.cat([self.collides(path[i:i + chunk]) for i in range(0, len(path), chunk)])
        return hit.view(n, -1).any(dim=0)
