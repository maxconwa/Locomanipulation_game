# Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c),
# src/h12_train/marl/representation.py. Changes: imports; build_world_model's default joint positions are the policy
# joints' (the motion command's joint_ids, JOINT_NAMES order: the Isaac articulation also has the gripper hinges and
# another joint order), as pretraining.supermimic.runners.base_runner.build_world_model reads them.
"""SuperMimic's generalized-coordinate world model with the H1-2's torso IMU.

The actors observe the torso IMU (the ``torso_link`` body's frame, ``constants.IMU_BODY``) in the sim, so the world
model synthesises the same sensor in imagination: ``torso_link`` hangs from the pelvis on
``torso_joint`` (a yaw about the pelvis z axis, no fixed rotation), so the torso frame is
the pelvis frame turned by ``q_torso`` about z, its gravity ``Rz(q)^T g_pelvis`` and its
angular velocity ``Rz(q)^T (w_pelvis + dq_torso z)``. A gyro reads the same angular
velocity anywhere on the rigid torso, so the IMU's offset does not enter. The sim's
actor reads ``joint_pos_rel`` with the env's encoder bias added,
so the synthesised joint positions add the same bias, read back from the sim context.

``ClampedDynamics`` clamps the processed position target to the sim action term's clip
range (golem_safety's, ``constants.TARGET_LIMITS``) inside the world model, as the sim does, so BPTT
sees no gradient for pushing a saturated joint further.
"""

from __future__ import annotations

import torch

from pretraining.supermimic.models import Dynamics
from pretraining.supermimic.representations import GenCoordRepresentation
from pretraining.supermimic.runners.base_runner import BaseRunnerCfg, WorldModel
from pretraining.supermimic.sim_context import SimContext
from pretraining.supermimic.utils.gencoord_helpers import decompose_rep_state
from pretraining.supermimic.utils.helpers import extract_action_scale_offset, unpack_obs_dims

TORSO_JOINT = 12  # torso_joint's column in the policy's (Unitree) joint order


def rotate_into_torso(vec_pelvis: torch.Tensor, q_torso: torch.Tensor) -> torch.Tensor:
    """Express pelvis-frame vectors (..., 3) in the torso frame, the torso yawed by q (rad)."""
    c, s = torch.cos(q_torso), torch.sin(q_torso)
    x, y, z = vec_pelvis.unbind(-1)
    return torch.stack([c * x + s * y, -s * x + c * y, z], dim=-1)


class TorsoImuGenCoordRepresentation(GenCoordRepresentation):
    """GenCoord whose synthesised actor sensor is the torso IMU, not the pelvis's, with the encoder bias."""

    def bind(self, env, sim_context: SimContext) -> None:
        """Bind as GenCoord does and find the encoder bias's sim-context columns."""
        super().bind(env, sim_context)
        columns = sim_context.find("encoder_bias")
        self._encoder_bias = None if columns is None else (columns, sim_context)

    def encoder_bias(self, sim_ctx: torch.Tensor) -> torch.Tensor | float:
        """Return the per-joint encoder bias (rad) the normalised ``sim_ctx`` holds, or 0 without the event.

        The sim's encoders read ``q + bias`` and its action term applies ``target - bias``.
        """
        if self._encoder_bias is None:
            return 0.0
        columns, sim_context = self._encoder_bias
        return sim_context.denormalize(sim_ctx, columns)

    def compute_wm_obs(
        self,
        rep_state: torch.Tensor,
        aux_state: torch.Tensor,
        joint_state: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[gravity_torso, gyro_torso, joint_pos_rel + encoder_bias, joint_vel]`` from the state.

        ``sim_ctx`` is the normalised sim context (``SimContext.read``) the bias is read from.
        """
        del aux_state
        _, g_b, _, w_b, _, _ = decompose_rep_state(rep_state)
        qj, vj = joint_state.chunk(2, dim=-1)
        q, dq = qj[..., TORSO_JOINT], vj[..., TORSO_JOINT]
        w_rel = w_b.clone()
        w_rel[..., 2] = w_rel[..., 2] + dq
        gravity = rotate_into_torso(g_b, q)
        gyro = rotate_into_torso(w_rel, q)
        joint_pos_rel = qj - self._default_joint_pos + self.encoder_bias(sim_ctx)
        return torch.cat([gravity, gyro, joint_pos_rel, vj], dim=-1)


class ClampedDynamics(Dynamics):
    """SuperMimic's dynamics with the processed target clamped to the sim's clip range."""

    def __init__(self, *args, target_low: torch.Tensor, target_high: torch.Tensor, **kwargs):
        """Build the dynamics and hold the per-joint target range (rad, joint order)."""
        super().__init__(*args, **kwargs)
        self.register_buffer("target_low", target_low.clone(), persistent=False)
        self.register_buffer("target_high", target_high.clone(), persistent=False)

    def proc(self, action: torch.Tensor) -> torch.Tensor:
        """Return ``clamp(action * scale + offset, low, high)``, the target the sim applies."""
        return torch.clamp(super().proc(action), self.target_low, self.target_high)


def action_clip(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the sim action term's per-joint target range (low, high), rad, or infinities."""
    term = env.unwrapped.action_manager.get_term("joint_pos")
    if term.cfg.clip is None:
        inf = torch.full((env.num_actions,), float("inf"), device=env.unwrapped.device)
        return -inf, inf
    clip = term._clip[0].to(env.unwrapped.device).float()
    return clip[:, 0], clip[:, 1]


def build_world_model(cfg: BaseRunnerCfg, env, device: str) -> WorldModel:
    """Build supermimic's world model for ``env`` with the torso-IMU representation and clamped targets.

    Mirrors ``supermimic.runners.supermimic_runner.build_world_model`` and
    ``create_representation`` for ``gencoord`` with ``actor_obs='proprio'``.
    """
    obs, _ = env.reset()
    num_full_state, num_goals = unpack_obs_dims(obs)
    num_actions = env.num_actions
    sim_context = SimContext.from_env(env.unwrapped)
    # The policy joints, in the motion command's (JOINT_NAMES) order.
    joint_ids = env.unwrapped.command_manager.get_term("motion").joint_ids
    default_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0][joint_ids].to(device)
    num_joints = (num_full_state - 13) // 2
    rep = TorsoImuGenCoordRepresentation(num_joints, default_joint_pos)
    rep.actor_obs_mode = "proprio"
    rep.register_buffer(
        "_proprio_noise_bounds",
        rep.proprio_noise_bounds(num_actions, device),
        persistent=False,
    )
    rep.dt = env.unwrapped.step_dt
    # Semi-implicit composition of `decimation` substeps under constant acceleration.
    decimation = env.unwrapped.cfg.decimation
    rep.substep_coeff = (decimation + 1) / (2 * decimation)
    rep = rep.to(device)
    rep.bind(env.unwrapped, sim_context)
    scale, offset = extract_action_scale_offset(env)
    low, high = action_clip(env)
    dynamics = ClampedDynamics(
        rep,
        num_actions,
        sim_context.output_dim,
        scale,
        offset,
        hidden_dims=cfg.dynamics_hidden_dims,
        target_low=low,
        target_high=high,
    ).to(device)
    return WorldModel(
        representation=rep,
        dynamics=dynamics,
        sim_context=sim_context,
        num_goals=num_goals,
        num_actions=num_actions,
    )
