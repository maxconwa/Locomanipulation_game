"""LocoManip-Marl-Pulse-v0's actions: each agent's latent, decoded by its frozen stage-1 decoder into joint targets.

Each agent acts in its stage-1 actor's 32-wide latent (pretraining/, PULSE as h12_train's stage 2 used it): the legs'
decoder sets the 12 leg joints, the upper body's the torso and the 14 arm joints (the arms agent owns the torso), in
Unitree motor order (pretraining.constants.JOINT_NAMES), mapped to the articulation's joints by name. The decoding
itself is pretraining.marl.pulse.PulseDecoders: z = tanh(u + prior(proprio)), target = default + ACTION_SCALE x
decoder(z, proprio), clipped to pretraining's TARGET_LIMITS within GOLEM's position envelope (joint_limits, which
sync_derived writes). No IK, no residual, no torque bound, no engage blend.

The decoders read one proprio frame per policy step, what the robot would report: gravity and the gyro of the torso
IMU's /lowstate model (env.lowstate_imu: lagged, biased, noisy; the gyro before the actors' low-pass, as h12_train's
deploy reads it), the measured joint angles (angle + encoder offset, main's position noise) less pretraining's default
pose, the joint speeds with main's encoder noise, and both decoders' raw outputs of the step before. The two terms
share it (_PulseShared, one per env): each term's process_actions only hands its latent over, and once both have,
the frame is built and both decode, then the last-action buffer moves on. The order the ActionManager calls them in
cannot matter. A term's reset zeroes its envs' last action (both terms do; it is the same buffer).

What the policies observe of their action (applied_actions) is the latent as the env clamped it (clip_actions). The
env's action delay and encoder offsets act on these terms as on main's (_joint_ids, _processed_actions), and _applied
is the last target (the measured pose after a reset), which the torso IMU's vibration model reads.

pretraining is imported here only, lazily: no other task imports it.
"""

from __future__ import annotations

import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..joint_noise import (JOINT_POS_NOISE, JOINT_VEL_NOISE_FLOOR, JOINT_VEL_NOISE_FLOOR_DEFAULT, joint_vel_noise_std,
                           per_joint)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["PulseLatentAction", "PulseLatentActionCfg"]

_SHARED = "_pulse_shared"  # the env attribute that holds the two terms' _PulseShared


class _PulseShared:
    """Per env: the stage-1 decoders and priors (loaded once), the latents handed over this step, the terms."""

    def __init__(self, env: ManagerBasedEnv, cfg: PulseLatentActionCfg):
        from pretraining.constants import DEFAULT_JOINT_POS, IMU_BODY, JOINT_NAMES
        from pretraining.marl.actors import load_latent_prior
        from pretraining.marl.config import AGENTS
        from pretraining.marl.pulse import PulseDecoders
        from pretraining.marl.runner import load_actors

        self.env, self.agents = env, AGENTS
        self.stage1_run, self.prior = cfg.stage1_run, cfg.prior or None
        robot = env.scene[cfg.asset_name]
        self.robot = robot
        self.joint_ids = robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0]
        if [robot.joint_names[i] for i in self.joint_ids] != list(JOINT_NAMES):
            raise ValueError("the articulation's joints do not cover pretraining's JOINT_NAMES")
        if IMU_BODY != "torso_link":
            raise ValueError(f"the decoders' IMU is on {IMU_BODY}; the env's lowstate IMU is on torso_link's imu_link")
        if self.prior and not os.path.isfile(self.prior):
            raise FileNotFoundError(
                f"no latent prior {self.prior}: fit it (python -m pretraining.scripts.fit_prior {cfg.stage1_run}). "
                "Without one a zero latent is not a standing policy (the decoders' outputs grow through the last "
                "action within a few steps): env.pulse_prior=null is for debugging only")
        actors = load_actors(cfg.stage1_run, env.device)
        priors = load_latent_prior(self.prior, env.device) if self.prior else None
        limits = {n: tuple(b) for n, b in (cfg.joint_limits or {}).items()}
        self.decoders = PulseDecoders(actors, env.num_envs, env.device, priors=priors, limits=limits)
        self.default = torch.tensor([DEFAULT_JOINT_POS[j] for j in JOINT_NAMES], device=env.device)
        floor = per_joint(list(JOINT_NAMES), JOINT_VEL_NOISE_FLOOR, JOINT_VEL_NOISE_FLOOR_DEFAULT)
        self.vel_floor = torch.tensor(floor, device=env.device)
        self.terms: dict[str, PulseLatentAction] = {}
        self.pending: dict[str, torch.Tensor] = {}
        print(f"[INFO] Pulse decoders: {cfg.stage1_run} (latent {self.decoders.latent_dim}), prior "
              f"{self.prior or 'none'}; targets within: " + ", ".join(
                  f"{n} [{lo:.3f}, {hi:.3f}]" for n, lo, hi in zip(
                      JOINT_NAMES, self.decoders.low.tolist(), self.decoders.high.tolist())))

    def sensors(self) -> torch.Tensor:
        """The frame's first 60 columns, as the robot would report them: gravity, gyro, q - default, dq."""
        env, data, ids = self.env, self.robot.data, self.joint_ids
        imu = env.lowstate_imu
        q = data.joint_pos[:, ids]
        offset = getattr(env, "encoder_offset", None)
        if offset is not None:
            q = q + offset[:, ids]
        q = q + (2.0 * torch.rand_like(q) - 1.0) * JOINT_POS_NOISE
        dq = data.joint_vel[:, ids]
        dq = dq + torch.randn_like(dq) * joint_vel_noise_std(dq, self.vel_floor)
        return torch.cat([imu.projected_gravity(), imu.gyro_raw, q - self.default, dq], dim=-1)

    def submit(self, agent: str, latent: torch.Tensor) -> None:
        """Take one agent's latent; once both are in, decode both on one frame and set both terms' targets."""
        if set(self.terms) != set(self.agents):
            # with one term alone nothing would ever decode: its targets would sit at the reset pose
            raise ValueError(f"Pulse action terms for {sorted(self.terms)}; both {self.agents} are needed")
        self.pending[agent] = latent
        if len(self.pending) < len(self.agents):
            return
        targets = self.decoders.step(self.pending, self.sensors())
        self.pending = {}
        for name, target in targets.items():
            self.terms[name].set_target(target)

    def reset(self, env_ids) -> None:
        self.pending = {}
        self.decoders.reset(env_ids)


class PulseLatentAction(ActionTerm):
    """One agent's latent (PulseLatentActionCfg.agent), decoded with the other's into joint targets (the module's
    docstring)."""

    cfg: PulseLatentActionCfg

    def __init__(self, cfg: PulseLatentActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from pretraining.constants import JOINT_NAMES
        from pretraining.marl.config import JOINT_SLICES

        shared = getattr(env, _SHARED, None)
        if shared is None:
            shared = _PulseShared(env, cfg)
            setattr(env, _SHARED, shared)
        elif (shared.stage1_run, shared.prior) != (cfg.stage1_run, cfg.prior or None):
            raise ValueError(f"the Pulse terms disagree: {shared.stage1_run}, {shared.prior} vs {cfg.stage1_run}, "
                             f"{cfg.prior}")
        if cfg.agent in shared.terms:
            raise ValueError(f"a second Pulse action term for {cfg.agent}")
        shared.terms[cfg.agent] = self
        self._shared = shared
        self._joint_names = list(JOINT_NAMES[JOINT_SLICES[cfg.agent]])
        self._joint_ids = self._asset.find_joints(self._joint_names, preserve_order=True)[0]
        self._raw_actions = torch.zeros(self.num_envs, shared.decoders.latent_dim, device=self.device)
        self._applied = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._processed_actions = self._applied.clone()

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def applied_actions(self) -> torch.Tensor:
        """The latent as the env clamped it (what the policy asked for)."""
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._shared.submit(self.cfg.agent, self._raw_actions)

    def set_target(self, target: torch.Tensor) -> None:
        """The decoded joint targets (rad), from _PulseShared once both latents are in."""
        self._applied = target
        self._processed_actions = target.clone()

    def apply_actions(self):
        # the env replaces _processed_actions at each substep by the delayed target less the encoder offsets
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0.0
        self._applied[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._shared.reset(env_ids)


@configclass
class PulseLatentActionCfg(ActionTermCfg):
    class_type: type = PulseLatentAction

    agent: str = MISSING
    """"legs" (the 12 leg joints) or "arms" (the torso and the 14 arm joints): whose stage-1 decoder."""
    stage1_run: str = MISSING
    """The stage-1 run directory (actors.pt, run.json), absolute (MarlPulseEnvCfg.sync_derived resolves it)."""
    prior: str | None = None
    """A fit_prior.py file (absolute), or None / "": no prior."""
    joint_limits: dict[str, tuple[float, float]] | None = None
    """Per joint (rad), intersected with pretraining's TARGET_LIMITS (sync_derived writes GOLEM's position envelope)."""


def resolve(path: str | None, base: str) -> str | None:
    """path made absolute against base; None or "" stays None."""
    if not path:
        return None
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))
