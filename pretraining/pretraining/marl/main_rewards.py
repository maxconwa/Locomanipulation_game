"""Main's cooperative-task rewards on the world model's predicted state, differentiable (torch only).

Stage 2 through the world model back-propagates main's task reward (``LocoManip-Marl-Pulse-v0``, the
Marl-Direct reward with each agent's share of the other's tracking) through imagined rollouts. This module scores a
predicted step with main's task terms, their weights and kernel widths read from the env's resolved cfg
(``MainRewardCfg.from_env_cfg``), plus stage 1's soft GOLEM e-stop penalties (``marl.rewards.estop_penalties``) on
each agent's own joints:

- the legs' velocity tracking, ``track_velocity_cycle_exp`` (``locomanip_marl/mdp/rewards.py:131-147``):
  ``exp(-|cmd - v|^2 / std^2)`` on the pelvis-frame ``(vx, vy)`` and yaw rate, ``v`` the mean over the last gait cycle
  (``_CycleMean``, rewards.py:102-128; ``cycle_init`` / ``cycle_push`` here), the stds the velocity command's kernel
  stage (``ModalVelocityCommand.lin_vel_std`` / ``ang_vel_std``, commands.py:186-192);
- the arms' wrist tracking (rewards.py:145-175): ``exp(-d^2 / std^2)`` on the wrist position error, ``exp(-theta^2 /
  std^2)`` on its rotation error (fixed std, or the arm command's kernel stage for the ``_annealed`` ones), and the
  capped linear ``min(d, cap)``; errors against the pelvis-frame goal (``pelvis_fixed_goals``: the true target is the
  command on the pelvis, ``ArmTargetsCommand.targets_w``, commands.py:576-583), on the ``*_wrist_yaw_link`` frames
  themselves (``ArmTargetsCommandCfg.body_names``, locomanip_marl_env_cfg.py:67-68, 181), *not* stage 1's tool frame
  (``marl.config.GRASP_OFFSET``);
- ``goal_reached`` (rewards.py:220-223), a one-off bonus on main's hold-on-goal event: not differentiable, taken from
  the caller (``just_reached``), 0 without it.

Every term is ``weight * dt * kernel`` (the reward manager's, ``_prepare_reward_function`` multiplies by dt), summed
per agent with the agent's share of the other's terms as Marl-Direct's cfg weighs them (``arms_*`` on the legs,
``legs_*`` on the arms, marl_direct_env_cfg.py:62-74).

**What this sum is.** It is arm C's training reward: a tracking-only surrogate of main's, not main's reward. It is
deliberately not floored at 0 (main clamps each agent's per-step sum at 0, locomanip_marl_env.py:243, which would zero
the gradient exactly where the arms miss), and it uses stage 1's soft, differentiable e-stop penalty in place of
main's trip cost and ``golem_*`` terms. Main's hold_still, effort, smoothness, contact, limit and self-collision terms
are not modelled (``MainRewardCfg.left_out`` lists those with a nonzero weight). The comparison between arms is scored
by the shared scorecard, not by this sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pretraining.constants import JOINT_NAMES, TRACKED_BODIES
from pretraining.marl.config import AGENTS, EstopPenaltyCfg
from pretraining.marl.rewards import JointLimits, estop_penalties, estop_position_depth, estop_ratios
from pretraining.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul
from pretraining.supermimic.utils.gencoord_helpers import decompose_rep_state

PELVIS = TRACKED_BODIES.index("pelvis")
TORSO = TRACKED_BODIES.index("torso_link")
# main's arm goals' bodies (locomanip_marl_env_cfg.py:67-68), no grasp offset
WRIST_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
WRISTS = tuple(TRACKED_BODIES.index(b) for b in WRIST_BODIES)

# main's reward functions this module models, by name -> kind
_KINDS = {
    "track_velocity_cycle_exp": None,  # "lin_xy" or "ang_z", from its params
    "arm_target_pos_exp": "ee_pos",
    "arm_target_pos_exp_annealed": "ee_pos",
    "arm_target_quat_exp": "ee_quat",
    "arm_target_quat_exp_annealed": "ee_quat",
    "arm_target_pos_capped": "ee_pos_capped",
    "arm_goal_reached": "goal_reached",
}


@dataclass
class WMState:
    """One (predicted) robot state: the root's world pose, its pelvis-frame twist, the joints (``JOINT_NAMES``
    order), and the ``TRACKED_BODIES``' world poses (quaternions w-first) and twists."""

    root_pos: torch.Tensor
    root_quat: torch.Tensor
    lin_vel_b: torch.Tensor
    ang_vel_b: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos: torch.Tensor
    body_quat: torch.Tensor
    body_lin_vel: torch.Tensor
    body_ang_vel: torch.Tensor

    @classmethod
    def from_root(cls, fk, root_pos, root_quat, lin_vel_b, ang_vel_b, joint_pos, joint_vel) -> WMState:
        """From the root and joints through ``fk`` (a ``kinematics.ForwardKinematics`` of ``TRACKED_BODIES``)."""
        if list(fk.body_names) != list(TRACKED_BODIES):
            raise ValueError(f"the forward kinematics' bodies are {fk.body_names}, not TRACKED_BODIES")
        pos, quat, lin, ang = fk(root_pos, root_quat, joint_pos, quat_apply(root_quat, lin_vel_b),
                                 quat_apply(root_quat, ang_vel_b), joint_vel)
        return cls(root_pos, root_quat, lin_vel_b, ang_vel_b, joint_pos, joint_vel, pos, quat, lin, ang)

    @classmethod
    def from_wm(cls, representation, aux_state: torch.Tensor, rep_state: torch.Tensor) -> WMState:
        """From the world model's gencoord state (``marl/representation.py``): ``aux_state`` the root pose,
        ``rep_state`` ``[z, g_b, v_b, w_b, qj, vj]``; the root height is the rep state's, as
        ``GenCoordRepresentation.reconstruct_full_state`` takes it."""
        z, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
        root_pos = torch.cat([aux_state[..., :2], z], -1)
        return cls.from_root(representation.fk, root_pos, aux_state[..., 3:7], v_b, w_b, qj, vj)


def _safe_angle(q: torch.Tensor) -> torch.Tensor:
    """The rotation angle (rad) of quaternions (..., 4), with a finite gradient at the identity."""
    v = (q[..., 1:].square().sum(-1) + 1e-12).sqrt()
    return 2.0 * torch.atan2(v, q[..., 0].abs())


def wrist_poses_in_pelvis(state: WMState) -> tuple[torch.Tensor, torch.Tensor]:
    """Both ``*_wrist_yaw_link`` frames in the pelvis frame: positions (n, 2, 3), quaternions (n, 2, 4) w-first
    (``subtract_frame_transforms`` of the root and the body, observations.py:61-71). No grasp offset."""
    rp, qp = state.root_pos.unsqueeze(-2), state.root_quat.unsqueeze(-2).expand(*state.root_quat.shape[:-1], 2, 4)
    pw, qw = state.body_pos[..., list(WRISTS), :], state.body_quat[..., list(WRISTS), :]
    return quat_apply_inverse(qp, pw - rp), quat_mul(quat_conjugate(qp), qw)


def wrist_errors(state: WMState, arm_command: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per arm, the wrist's position error (m) and rotation error (rad) to the pelvis-frame goal ``arm_command``
    (n, 2, 7, w-first): ``ArmTargetsCommand.errors`` (commands.py:589-604) with pelvis-fixed goals, (n, 2) each."""
    pos, quat = wrist_poses_in_pelvis(state)
    d = pos - arm_command[..., :3]
    pos_error = (d.square().sum(-1) + 1e-12).sqrt()
    rot_error = _safe_angle(quat_mul(quat, quat_conjugate(arm_command[..., 3:])))
    return pos_error, rot_error


def cycle_window(gait_period: float, dt: float) -> int:
    """Steps in one gait cycle, as ``_CycleMean`` (rewards.py:107)."""
    return max(int(round(gait_period / dt)), 1)


def cycle_init(n: int, window: int, device=None) -> dict[str, torch.Tensor]:
    """Empty gait-cycle windows of the pelvis-frame ``(vx, vy, wz)``, as ``_CycleMean.reset``."""
    return {"samples": torch.zeros(n, window, 3, device=device), "count": torch.zeros(n, device=device)}


def cycle_push(cycle: dict[str, torch.Tensor], state: WMState) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Add ``state``'s ``(vx, vy, wz)``; return the new windows and the mean over each so far (n, 3)
    (``_CycleMean._push``: the samples so far, at most a cycle, over their count). Not in place: differentiable."""
    x = torch.cat([state.lin_vel_b[..., :2], state.ang_vel_b[..., 2:3]], -1)
    samples = torch.cat([cycle["samples"][:, 1:], x.unsqueeze(1)], 1)
    count = (cycle["count"] + 1.0).clamp(max=samples.shape[1])
    return {"samples": samples, "count": count}, samples.sum(1) / count.unsqueeze(-1)


@dataclass(frozen=True)
class TermSpec:
    """One of main's reward terms as this module scores it."""

    agent: str
    name: str
    kind: str  # lin_xy, ang_z, ee_pos, ee_quat, ee_pos_capped, goal_reached
    weight: float  # per second, as main's cfg
    std: float = 0.0
    arm: int = 0
    cap: float = 0.0


@dataclass
class MainRewardCfg:
    """Main's modelled terms per agent, the policy step and the e-stop penalties' settings."""

    dt: float
    terms: tuple[TermSpec, ...]
    estop: EstopPenaltyCfg = EstopPenaltyCfg()
    plant: object | None = None  # a plants.Plant for the e-stop torque demand's gains; None: pretraining's
    left_out: tuple[str, ...] = ()  # "<agent>/<term>" of main's terms with a nonzero weight not modelled here
    _limits: dict = field(default_factory=dict, repr=False)

    def limits(self, device) -> JointLimits:
        key = str(device)
        if key not in self._limits:
            self._limits[key] = JointLimits.for_joints(JOINT_NAMES, device, self.plant)
        return self._limits[key]

    @classmethod
    def from_env_cfg(cls, env_cfg, lin_stage: int = 0, ang_stage: int = 0, arm_stage: int = 0,
                     estop: EstopPenaltyCfg | None = None, plant=None) -> MainRewardCfg:
        """Read the terms from main's resolved env cfg (``rewards.legs`` / ``rewards.arms``, after ``sync_derived``)
        at the given kernel stages: the velocity command's linear and yaw stages and the arm command's stage."""
        if not getattr(env_cfg.commands.arm_targets, "pelvis_fixed_goals", False):
            raise ValueError("main_rewards scores pelvis-fixed arm goals only (LocoManip-Marl-Direct-v0 / -Pulse-v0)")
        vel, arm = env_cfg.commands.base_velocity, env_cfg.commands.arm_targets
        lin_std, ang_std = vel.lin_vel_stds[lin_stage], vel.ang_vel_stds[ang_stage]
        terms, left_out = [], []
        for agent in AGENTS:
            for name, term in vars(getattr(env_cfg.rewards, agent)).items():
                if term is None or not hasattr(term, "func") or term.weight == 0.0:
                    continue
                func = getattr(term.func, "__name__", type(term.func).__name__)
                if func not in _KINDS:
                    left_out.append(f"{agent}/{name}")
                    continue
                p = term.params
                if func == "track_velocity_cycle_exp":
                    kind = p["component"]
                    spec = TermSpec(agent, name, kind, term.weight, std=lin_std if kind == "lin_xy" else ang_std)
                elif func == "arm_goal_reached":
                    spec = TermSpec(agent, name, "goal_reached", term.weight)
                elif func == "arm_target_pos_capped":
                    spec = TermSpec(agent, name, "ee_pos_capped", term.weight, arm=p["arm"], cap=p["cap"])
                else:
                    if func.endswith("_annealed"):
                        std = (arm.fine_pos_stds if "pos" in func else arm.rot_stds)[arm_stage]
                    else:
                        std = p["std"]
                    spec = TermSpec(agent, name, _KINDS[func], term.weight, std=std, arm=p["arm"])
                terms.append(spec)
        return cls(dt=env_cfg.sim.dt * env_cfg.decimation, terms=tuple(terms),
                   estop=estop if estop is not None else EstopPenaltyCfg(), plant=plant, left_out=tuple(left_out))


def main_rewards(state: WMState, next_state: WMState, commands: dict[str, torch.Tensor], cfg: MainRewardCfg, *,
                 cycle_mean: torch.Tensor, target: torch.Tensor | None = None,
                 just_reached: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Score one predicted step ``state -> next_state`` under ``commands`` (``velocity`` (n, 3) as the policies see
    it, ``arm`` (n, 2, 7) pelvis-frame goals, w-first).

    ``cycle_mean`` (n, 3) is the gait-cycle mean of ``(vx, vy, wz)`` including ``next_state`` (``cycle_push``;
    required: main scores the cycle mean, rewards.py:140-141, never the instantaneous twist);
    ``target`` (n, 27) the step's clamped joint targets less the encoder bias, for the e-stop torque demand
    (None: no torque term); ``just_reached`` (n,) main's goal-reached event (None: 0).

    Returns ``legs`` and ``arms`` (n,): each agent's ``dt * (sum of weight x kernel + e-stop penalty)``; and per term
    ``<agent>/<term>`` the unweighted kernel (main's term value: its reward-manager value over weight x dt), and
    ``<agent>/estop`` the penalty per second.
    """
    n = next_state.joint_pos.shape[0]
    device, dtype = next_state.joint_pos.device, next_state.joint_pos.dtype
    if cycle_mean is None:
        raise ValueError("main_rewards needs cycle_mean (cycle_push): main scores the gait-cycle mean velocity")
    velocity = commands["velocity"]
    pos_err, rot_err = wrist_errors(next_state, commands["arm"])
    out: dict[str, torch.Tensor] = {}
    totals = {a: torch.zeros(n, device=device, dtype=dtype) for a in AGENTS}
    for t in cfg.terms:
        if t.kind == "lin_xy":
            k = torch.exp(-(velocity[:, :2] - cycle_mean[:, :2]).square().sum(-1) / t.std**2)
        elif t.kind == "ang_z":
            k = torch.exp(-(velocity[:, 2] - cycle_mean[:, 2]).square() / t.std**2)
        elif t.kind == "ee_pos":
            k = torch.exp(-pos_err[:, t.arm].square() / t.std**2)
        elif t.kind == "ee_quat":
            k = torch.exp(-rot_err[:, t.arm].square() / t.std**2)
        elif t.kind == "ee_pos_capped":
            k = pos_err[:, t.arm].clamp(max=t.cap)
        else:  # goal_reached
            k = torch.zeros(n, device=device, dtype=dtype) if just_reached is None else just_reached.to(dtype)
        out[f"{t.agent}/{t.name}"] = k
        totals[t.agent] = totals[t.agent] + t.weight * k
    # stage 1's soft e-stop penalties (marl/wm_env.py _estop): the velocity and position terms on the predicted next
    # joints, the torque demand of the step's target from the state before it
    limits = cfg.limits(device)
    q, dq = state.joint_pos, state.joint_vel
    if target is None:
        vel_ratio, _ = estop_ratios(limits, q, dq, next_state.joint_vel, q)
        tau_ratio = torch.zeros_like(vel_ratio)
    else:
        vel_ratio, tau_ratio = estop_ratios(limits, q, dq, next_state.joint_vel, target)
    depth = estop_position_depth(limits, next_state.joint_pos, cfg.estop.position_margin)
    estop = estop_penalties(cfg.estop, vel_ratio, tau_ratio, depth)
    for i, a in enumerate(AGENTS):
        out[f"{a}/estop"] = estop[:, i]
        out[a] = cfg.dt * (totals[a] + estop[:, i])
    return out
