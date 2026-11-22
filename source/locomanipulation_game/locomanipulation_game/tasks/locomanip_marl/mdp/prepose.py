"""Resets from the states the real H1-2 is handed to the policy in: lmg_deploy's pre-pose, settled or not.

On the robot, marl_deploy_node._prepose ramps every joint from wherever it is to default_joint_pos at the policy's own
gains and holds there until the operator confirms; the confirm is accepted unsettled (any joint more than 0.15 rad off)
8 s after the ramp. On 2026-11-20 (LegsEnergyDefault200k_s43's bag) the robot was confirmed hung on the hoist (errors up
to 0.12 rad, elbows and wrist pitch), and lowered onto its feet with the hoist still carrying part of the weight: ankle
pitches sagged -0.17 / -0.21 rad, the torso 5 deg forward, max error 0.215 rad.

prepose_reset (a reset event after reset_base and reset_joints) redraws a share of each reset's envs from that range:
- every joint offset from the default pose per joint group (joint_ranges), or for a share of them (sag_prob) a
  correlated sag: the ankles dorsiflexed by s * sag_ankle and the knees bent by s * sag_knee, s ~ U(0, 1), the hip
  pitches set so the feet stay flat under the drawn base pitch (the robot lowered onto its feet under load);
- for a share of them (fast_vel_prob) joint velocities up to +-fast_vel (an engage while still swaying);
- the base pitched and rolled (pitch_range, roll_range; the reset's x, y, yaw and velocities kept), placed so its
  lowest sole corner (SOLE_BOX) is clearance above the ground;
- a gantry on some: hung (hung_prob; all the weight, the feet lift_range off the ground) or partial (partial_prob; a
  share partial_share_range of the weight, the feet loaded). The rope is a vertical force at support_point on
  support_body (the hoist's hook above the shoulders: hanging from a point above the centre of mass, the robot swings
  as on the hoist; at the waist it tumbled), share x (m g + k_z (z_anchor - z) - c_z v_z), never pushing, with a soft
  horizontal tether to the anchor; it carries
  its share until release_start (0 with release_now_prob, else U(keep_range)) and lets go over U(release_duration_range)
  s. A share of the hung ones (lower_prob) is lowered first, as the hoist does before it is unhooked: from
  U(lower_start_range) s the anchor comes down over U(lower_duration_range) s by the feet's lift plus a slack of
  U(lower_slack_range) m (the rope's spring then carries less: at k_z 3000 N/m, 0.2 m of slack takes about 600 N off),
  and the rope lets go U(settle_range) s after the lowering ends. The env sets the force once per policy step
  (apply_support) and the fall check ignores an env while the rope is attached (support_share > 0). The rewards are
  unchanged under it.

A fixed base (LocoManip-ArmsOnly-v0's pinned pelvis) keeps its upright pose and gets no gantry: only the joints are
drawn.

At these resets the env also hands the policies the deploy's state (LocoManipMarlEnv._reset_idx): a zero velocity
command until its next resample, and the targets before the first step at the default pose (the pre-pose's hold). The
rest is already the deploy's at every reset: the gait clock at its start, the LSTM state zeroed, the legs' previous
action (q - default) / scale for the measured q, the arms' previous action and residual zero and their IK from the
measured angles.
"""

from __future__ import annotations

import re
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# The H1-2's sole in its ankle_roll_link frame (the collision mesh's flat bottom): x, y extents and its height (m)
SOLE_BOX = ((-0.086, 0.174), (-0.043, 0.043), -0.045)
GRAVITY = 9.81

__all__ = ["SOLE_BOX", "prepose_reset"]


class prepose_reset(ManagerTermBase):
    """The module docstring's reset. The per-env gantry state lives here; the env applies it (apply_support)."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.robot = env.scene[params["asset_cfg"].name]
        self.fixed = bool(self.robot.is_fixed_base)
        n, dev = env.num_envs, env.device
        names = self.robot.joint_names
        # per joint, its group's offset range (joints no group names stay at their reset value)
        self._low = torch.zeros(len(names), device=dev)
        self._high = torch.zeros(len(names), device=dev)
        self._drawn = torch.zeros(len(names), dtype=torch.bool, device=dev)
        for pattern, (low, high) in params["joint_ranges"].items():
            hit = torch.tensor([re.fullmatch(pattern, name) is not None for name in names], device=dev)
            if not hit.any():
                raise ValueError(f"prepose_reset: joint_ranges {pattern!r} matches no joint")
            self._low[hit], self._high[hit] = low, high
            self._drawn |= hit
        self._ankle = self.robot.find_joints(".*_ankle_pitch_joint", preserve_order=True)[0]
        self._knee = self.robot.find_joints(".*_knee_joint", preserve_order=True)[0]
        self._hip = self.robot.find_joints(".*_hip_pitch_joint", preserve_order=True)[0]
        self._feet = self.robot.find_bodies(params["feet_body_names"], preserve_order=True)[0]
        self._support_body = self.robot.find_bodies(params["support_body"])[0][0]
        self._support_point = torch.tensor(params["support_point"], device=dev)
        zeros = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.active = torch.zeros(n, dtype=torch.bool, device=dev)
        self.share0, self.weight = zeros(), zeros()
        self.release_start, self.release_duration = zeros(), zeros()
        self.anchor = torch.zeros(n, 3, device=dev)  # where the hook was at the reset (anchor_at: where it is now)
        self.lower_start, self.lower_duration, self.lower_drop = zeros(), zeros(), zeros()
        self._forces = torch.zeros(n, 1, 3, device=dev)
        self._torques = torch.zeros(n, 1, 3, device=dev)
        self._pushing = False
        self._gains = (params["support_stiffness"], params["support_damping"], params["tether_stiffness"],
                       params["tether_damping"])

    def reset(self, env_ids=None):
        pass  # __call__ redraws the reset envs

    def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, prob: float,
                 joint_ranges: dict, sag_prob: float, sag_ankle: float, sag_knee: float, fast_vel_prob: float,
                 fast_vel: float, pitch_range: tuple, roll_range: tuple, clearance: float, hung_prob: float,
                 partial_prob: float, partial_share_range: tuple, lift_range: tuple, release_now_prob: float,
                 keep_range: tuple, release_duration_range: tuple, lower_prob: float, lower_start_range: tuple,
                 lower_duration_range: tuple, lower_slack_range: tuple, settle_range: tuple, feet_body_names: list,
                 support_body: str,
                 support_point: tuple, support_stiffness: float, support_damping: float, tether_stiffness: float, tether_damping: float):
        robot, dev = self.robot, env.device
        env_ids = torch.as_tensor(env_ids, device=dev)
        self.active[env_ids] = False
        self.share0[env_ids] = 0.0
        self.lower_drop[env_ids] = 0.0
        ids = env_ids[torch.rand(len(env_ids), device=dev) < prob]
        m = len(ids)
        if m == 0:
            return
        self.active[ids] = True
        u = lambda low, high, *shape: low + (high - low) * torch.rand(*shape, device=dev)  # noqa: E731

        # -- the base's tilt (a pinned pelvis stays upright)
        pitch = torch.zeros(m, device=dev) if self.fixed else u(*pitch_range, m)
        roll = torch.zeros(m, device=dev) if self.fixed else u(*roll_range, m)

        # -- the joints: per group, or a sag with the feet flat under the base's pitch
        default = robot.data.default_joint_pos[ids]
        q = robot.data.joint_pos[ids].clone()
        offset = self._low + (self._high - self._low) * torch.rand(m, len(self._low), device=dev)
        q[:, self._drawn] = (default + offset)[:, self._drawn]
        sag = torch.rand(m, device=dev) < sag_prob
        if sag.any():
            s = torch.rand(int(sag.sum()), 1, device=dev)
            rows = sag.nonzero(as_tuple=True)[0]
            ankle, knee = -sag_ankle * s, sag_knee * s
            # hip pitch + knee + ankle pitch (+ the base's pitch) is the sole's pitch: flat when they cancel
            hip = -(ankle + knee) - pitch[rows].unsqueeze(1)
            for joint_ids, value in ((self._ankle, ankle), (self._knee, knee), (self._hip, hip)):
                q[rows.unsqueeze(1), torch.tensor(joint_ids, device=dev)] = default[rows][:, joint_ids] + value
        limits = robot.data.soft_joint_pos_limits[ids]
        q = torch.maximum(torch.minimum(q, limits[..., 1]), limits[..., 0])
        dq = robot.data.joint_vel[ids].clone()
        fast = torch.rand(m, device=dev) < fast_vel_prob
        dq[fast] = u(-fast_vel, fast_vel, int(fast.sum()), dq.shape[1])
        robot.write_joint_state_to_sim(q, dq, env_ids=ids)

        if self.fixed:
            return
        # -- the base: the reset's x, y, yaw and velocities, the drawn tilt, placed on the ground (or hung above it)
        pose = robot.data.root_link_pose_w[ids].clone()
        _, _, yaw = euler_xyz_from_quat(pose[:, 3:7])
        pose[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        robot.write_root_pose_to_sim(pose, env_ids=ids)
        env.sim.forward()  # the links' poses for the new joints and base
        kind = torch.rand(m, device=dev)
        hung = kind < hung_prob
        partial = (kind >= hung_prob) & (kind < hung_prob + partial_prob)
        lift = torch.where(hung, u(*lift_range, m), torch.zeros(m, device=dev))
        pose[:, 2] += env.scene.env_origins[ids, 2] + clearance + lift - self._lowest_sole(ids)
        robot.write_root_pose_to_sim(pose, env_ids=ids)

        # -- the gantry
        self.share0[ids] = torch.where(hung, 1.0, torch.where(partial, u(*partial_share_range, m), 0.0))
        masses = env.physx_body_props()[0][ids.cpu()].sum(dim=1).to(dev)
        self.weight[ids] = masses * GRAVITY
        now = torch.rand(m, device=dev) < release_now_prob
        self.release_start[ids] = torch.where(now, 0.0, u(*keep_range, m))
        self.release_duration[ids] = u(*release_duration_range, m)
        # the hoist lowering some of the hung ones onto their feet, the release after it
        lowered = hung & (torch.rand(m, device=dev) < lower_prob)
        self.lower_start[ids] = u(*lower_start_range, m)
        self.lower_duration[ids] = u(*lower_duration_range, m)
        self.lower_drop[ids] = torch.where(lowered, lift + u(*lower_slack_range, m), 0.0)
        after = self.lower_start[ids] + self.lower_duration[ids] + u(*settle_range, m)
        self.release_start[ids] = torch.where(lowered, after, self.release_start[ids])
        # the anchor where the hook is now: the rope at rest there (the base only moved up since the links' poses)
        hook = self._hook()[0][ids]
        self.anchor[ids] = hook + (pose[:, :3] - robot.data.root_link_pos_w[ids])

    def _lowest_sole(self, ids: torch.Tensor) -> torch.Tensor:
        """The lowest corner of either sole (SOLE_BOX), world z, per env of ids."""
        (x0, x1), (y0, y1), z = SOLE_BOX
        corners = torch.tensor([[x, y, z] for x in (x0, x1) for y in (y0, y1)], device=ids.device)
        pos = self.robot.data.body_link_pos_w[ids][:, self._feet]  # (m, feet, 3)
        quat = self.robot.data.body_link_quat_w[ids][:, self._feet]
        c = corners.view(1, 1, 4, 3).expand(len(ids), len(self._feet), 4, 3)
        q = quat.unsqueeze(2).expand(-1, -1, 4, -1)
        world = pos.unsqueeze(2) + quat_apply(q.reshape(-1, 4), c.reshape(-1, 3)).view(c.shape)
        return world[..., 2].amin(dim=(1, 2))

    def _hook(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(position, velocity) of support_point on support_body, world frame, every env."""
        data, b = self.robot.data, self._support_body
        arm = quat_apply(data.body_link_quat_w[:, b], self._support_point.expand(self.robot.num_instances, 3))
        pos = data.body_link_pos_w[:, b] + arm
        vel = data.body_link_lin_vel_w[:, b] + torch.cross(data.body_link_ang_vel_w[:, b], arm, dim=1)
        return pos, vel

    def anchor_at(self, t: torch.Tensor) -> torch.Tensor:
        """Each env's rope anchor at episode time t (s): where the hook was at the reset, lowered by lower_drop over
        [lower_start, lower_start + lower_duration] (0 for the envs not lowered)."""
        lowered = ((t - self.lower_start) / self.lower_duration.clamp(min=1e-6)).clamp(0.0, 1.0)
        anchor = self.anchor.clone()
        anchor[:, 2] -= self.lower_drop * lowered
        return anchor

    def support_share(self, t: torch.Tensor) -> torch.Tensor:
        """The share of the weight each env's rope carries at episode time t (s): share0 until release_start, then
        down to 0 over release_duration."""
        released = ((t - self.release_start) / self.release_duration.clamp(min=1e-6)).clamp(0.0, 1.0)
        return self.share0 * (1.0 - released)

    def apply_support(self, t: torch.Tensor):
        """Set the rope's force at the hook for this physics substep (t: each env's episode time, s): the force at
        support_body's centre of mass and its moment about it."""
        share = self.support_share(t)
        carrying = share > 0.0
        if not carrying.any():
            if self._pushing:  # the last ropes let go: no force from here
                self._forces.zero_()
                self._torques.zero_()
                self.robot.set_external_force_and_torque(self._forces, self._torques, body_ids=[self._support_body],
                                                         is_global=True)
                self._pushing = False
            return
        k_z, c_z, k_xy, c_xy = self._gains
        pos, vel = self._hook()
        anchor = self.anchor_at(t)
        lift = share * (self.weight + k_z * (anchor[:, 2] - pos[:, 2]) - c_z * vel[:, 2])
        tether = share.unsqueeze(1) * (k_xy * (anchor[:, :2] - pos[:, :2]) - c_xy * vel[:, :2])
        self._forces[:, 0, 2] = torch.where(carrying, lift.clamp(min=0.0), 0.0)
        self._forces[:, 0, :2] = torch.where(carrying.unsqueeze(1), tether, 0.0)
        lever = pos - self.robot.data.body_com_pos_w[:, self._support_body]
        self._torques[:, 0] = torch.cross(lever, self._forces[:, 0], dim=1)
        self.robot.set_external_force_and_torque(self._forces, self._torques, body_ids=[self._support_body],
                                                 is_global=True)
        self._pushing = True
