# Ported from SuperMimic (HybridRobotics, commit 7e94b07, Apache-2.0), src/supermimic/representations.py. Changes:
# GenCoord only (MaxCoordRepresentation and its imports dropped); the goal-window constants and helpers imported from
# pretraining.supermimic.goal_window; quaternion maths from pretraining.math; bind() reads the soft joint limits of
# the policy joints, in the motion command's order.
"""State representations for the world model.

- GenCoordRepresentation: generalized-coordinate (joint-space) representation
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from pretraining.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv as ManagerBasedRlEnv

    from pretraining.kinematics import ForwardKinematics
    from pretraining.motions import MotionLoader
    from pretraining.supermimic.sim_context import SimContext

from pretraining.supermimic.goal_window import (  # noqa: F401  (re-exported: upstream defines them here)
    GOAL_FRAME_STRIDE,
    GOAL_REL_ORI_DIM,
    GOAL_REL_POS_DIM,
    NUM_GOAL_FRAMES,
    _assemble_goal,
    _compiled_assemble_goal,
    actor_goal_dim,
    assemble_anchor_window,
    extract_actor_goal,
    goal_window_indices,
    motion_anchor_window,
    motion_goal_window,
)
from pretraining.supermimic.utils.gencoord_helpers import (
    compute_gencoord_state,
    decompose_rep_state,
    infer_num_joints_from_state_dim,
    rep_state_dim_from_num_joints,
)
from pretraining.supermimic.utils.helpers import quat_integrate
from pretraining.supermimic.utils.tracking_reward import tracking_reward


def create_representation(
    name: str,
    full_state_dim: int,
    num_actions: int,
    dt: float,
    decimation: int,
    default_joint_pos: torch.Tensor,
    actor_obs_mode: str,
) -> StateRepresentation:
    """Create a state representation wired up for training and reward compute."""
    if name == "gencoord":
        num_joints = infer_num_joints_from_state_dim(full_state_dim)
        rep: StateRepresentation = GenCoordRepresentation(num_joints, default_joint_pos)
    else:
        raise ValueError(f"Unknown representation: {name}. Must be 'gencoord' (MaxCoord is not ported).")
    rep.actor_obs_mode = actor_obs_mode
    if actor_obs_mode == "proprio":
        rep.register_buffer(
            "_proprio_noise_bounds",
            rep.proprio_noise_bounds(num_actions, default_joint_pos.device),
            persistent=False,
        )
    rep.dt = dt
    # Semi-implicit composition of `decimation` substeps under constant accel.
    rep.substep_coeff = (decimation + 1) / (2 * decimation)
    return rep


PROPRIO_NOISE_HALF_WIDTHS: dict[str, float] = {
    "projected_gravity": 0.05,
    "base_ang_vel": 0.2,
    "joint_pos": 0.01,
    "joint_vel": 0.5,
    "actions": 0.0,
}


def term_major_to_temporal(flat: torch.Tensor, term_widths: list[int], history_length: int) -> torch.Tensor:
    """Convert mjlab's term-major flat obs to a ``(B, H, sum(w_i))`` frame-major window.

    Input layout ``[term0_t0..tH-1, term1_t0..tH-1, ...]`` is what the deployed
    actor obs group emits; the decoder consumes frame-major. Applied at the ONNX
    export boundary so deploy and training see the same window.
    """
    parts = []
    off = 0
    for w in term_widths:
        chunk = flat[..., off : off + history_length * w]
        parts.append(chunk.unflatten(-1, (history_length, w)))
        off += history_length * w
    return torch.cat(parts, dim=-1)


def _root_loss_blocks(aux_state: torch.Tensor) -> dict[str, torch.Tensor]:
    """World root position (3) and rotation (9) as squared-error channels.

    The rotation is scaled by sqrt(1/2) so a plain squared difference is
    0.5*||dR||_F^2 = 2(1 - cos theta) = theta^2 + O(theta^4): a nominal in radians.
    """
    rot = matrix_from_quat(aux_state[..., 3:7]).flatten(-2, -1)
    return {"root_pos": aux_state[..., :3], "root_ori": rot * math.sqrt(0.5)}


class StateRepresentation(nn.Module, ABC):
    """Abstract base class for state representations.

    Owns the state-semantics methods (``represent``, ``apply_prediction``,
    etc.) and the world-model reward.

    Subclasses must set these instance attributes in ``__init__``:
      full_state_dim, rep_state_dim, prediction_dim

    And extend this class attribute to cover every key ``loss_blocks`` returns:
      LOSS_WEIGHT_CONFIG: dict[str, str]
    """

    full_state_dim: int
    rep_state_dim: int
    prediction_dim: int
    num_joints: int
    actor_obs_mode: Literal["rep_state", "proprio"] = "rep_state"

    # Each channel group's nominal-error field on ``WorldModelLearnerCfg``. Every
    # representation scores the integrated root pose; subclasses add their own.
    LOSS_WEIGHT_CONFIG: dict[str, str] = {
        "root_pos": "nominal_error_root_pose_pos",
        "root_ori": "nominal_error_root_pose_ori",
    }

    substep_coeff: float = 1.0

    # ------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def represent(self, full_state: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
        """Extract auxiliary state [root_pos(3), root_quat(4)] from full state."""
        ...

    @abstractmethod
    def apply_prediction(
        self,
        aux_state: torch.Tensor,
        curr_rep_state: torch.Tensor,
        prediction: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply prediction to produce next aux_state and next rep_state.

        Returns:
          (next_aux_state, next_rep_state)
        """
        ...

    @abstractmethod
    def reconstruct_full_state(self, aux_state: torch.Tensor, rep_state: torch.Tensor) -> torch.Tensor:
        """Reconstruct full state from aux_state + rep_state (on-demand, for viz)."""
        ...

    def bind(self, env: ManagerBasedRlEnv, sim_context: SimContext) -> None:
        """Resolve env-dependent state needed by the reward and the proprio sensor."""
        # The policy joints' rows, in the motion command's (JOINT_NAMES) order: the Isaac articulation also has the
        # gripper hinges, and its joint order is not the policy's.
        joint_ids = env.command_manager.get_term("motion").joint_ids
        self._soft_joint_pos_limits = env.scene["robot"].data.soft_joint_pos_limits[0][joint_ids]

    @abstractmethod
    def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
        """``[joint_pos, joint_vel]`` in ``qj`` column order."""
        ...

    @abstractmethod
    def compute_wm_obs(
        self,
        rep_state: torch.Tensor,
        aux_state: torch.Tensor,
        joint_state: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> torch.Tensor:
        """Synthesize state-derived proprio from WM-predicted state."""
        ...

    def proprio_obs_dim(self, num_actions: int) -> int:
        """Dim of the deploy-shape sensor proprio."""
        return sum(self.proprio_term_widths(num_actions))

    def proprio_term_widths(self, num_actions: int) -> list[int]:
        """Widths of ``[projected_gravity, base_ang_vel, joint_pos, joint_vel, actions]``,
        the sensor proprio in the order ``_sensor_actor_group`` concatenates it."""
        return [3, 3, self.num_joints, self.num_joints, num_actions]

    def proprio_noise_bounds(self, num_actions: int, device: torch.device | str) -> torch.Tensor:
        """Per-dim uniform-noise half-widths matching the sensor proprio layout."""
        widths = self.proprio_term_widths(num_actions)
        halves = PROPRIO_NOISE_HALF_WIDTHS.values()
        return torch.cat([torch.full((w,), h, device=device) for w, h in zip(widths, halves, strict=True)])

    def actor_obs_dim(self, num_actions: int, num_goals: int, history_length: int) -> int:
        """Dim of the actor input — ``history_length`` sensor frames (proprio or
        rep_state) cat'd with the goal."""
        sensor_or_state = (
            self.rep_state_dim if self.actor_obs_mode == "rep_state" else self.proprio_obs_dim(num_actions)
        )
        return history_length * sensor_or_state + actor_goal_dim(num_goals)

    def add_proprio_noise(self, obs: torch.Tensor) -> torch.Tensor:
        """Add per-dim uniform proprio jitter — the single noise site shared by the
        synthesized sensor and the real-sim / imagination history rings."""
        return obs + ((torch.rand_like(obs) * 2 - 1) * self._proprio_noise_bounds).detach()

    def compute_actor_sensor(
        self,
        rep_state: torch.Tensor,
        sim_ctx: torch.Tensor,
        aux_state: torch.Tensor | None = None,
        joint_state: torch.Tensor | None = None,
        last_action: torch.Tensor | None = None,
        sim_actor_obs: torch.Tensor | None = None,
        noisy: bool = False,
    ) -> torch.Tensor:
        """Sensor-or-state half of the actor input (no goal).

        Returns ``rep_state`` in rep_state mode. In proprio mode, returns
        ``sim_actor_obs`` if provided (mjlab's ``obs["actor"]``) else
        synthesizes via :meth:`compute_wm_obs`; ``noisy=True`` adds per-dim
        uniform jitter scaled by the precomputed ``_proprio_noise_bounds``. ``aux_state``,
        ``joint_state`` and ``last_action`` are required only when synthesizing
        (proprio mode with no ``sim_actor_obs``).
        """
        if self.actor_obs_mode == "rep_state":
            return rep_state
        if sim_actor_obs is not None:
            obs = sim_actor_obs
        else:
            if aux_state is None or joint_state is None or last_action is None:
                raise ValueError(
                    "aux_state, joint_state and last_action are required to synthesize "
                    "proprio obs (no sim_actor_obs provided)"
                )
            obs = self.compute_wm_obs(rep_state, aux_state, joint_state, sim_ctx)
            obs = torch.cat([obs, last_action], dim=-1)
        if noisy:
            obs = self.add_proprio_noise(obs)
        return obs

    def build_actor_obs(self, sensor_history: torch.Tensor, actor_goal: torch.Tensor) -> torch.Tensor:
        """Flatten the ``(B, history_length, sensor_dim)`` window and cat the goal."""
        return torch.cat([sensor_history.flatten(start_dim=1), actor_goal], dim=-1)

    # ------------------------------------------------------------------
    # Default implementations (may be overridden)
    # ------------------------------------------------------------------

    def compute_wm_goal(
        self,
        motion: MotionLoader,
        motion_times: torch.Tensor,
        aux_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build ``(wm_goal, actor_goal)`` from the motion window at ``motion_times``.

        The single goal-build path shared by the WM env, the task goal extractor, and
        the viewer so every site produces an identical layout.
        """
        content = motion_goal_window(motion, motion_times)
        target_pos, target_quat = motion_anchor_window(motion, motion_times)
        root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
        return _compiled_assemble_goal(content, root_pos, root_quat, target_pos, target_quat)

    @abstractmethod
    def world_frame_bodies(
        self, aux_state: torch.Tensor, rep_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """World-frame ``(pos, rot_mat, lin_vel, ang_vel)`` of the tracked bodies."""
        ...

    def compute_wm_reward_from_reference(
        self,
        next_rep_state: torch.Tensor,
        next_aux_state: torch.Tensor,
        next_joint_state: torch.Tensor,
        action: torch.Tensor,
        last_action: torch.Tensor,
        target_body_pos_w: torch.Tensor,
        target_body_quat_w: torch.Tensor,
        target_body_lin_vel_w: torch.Tensor,
        target_body_ang_vel_w: torch.Tensor,
    ) -> torch.Tensor:
        """Per-env scalar reward used for WM-env rollouts.

        ``action``/``last_action`` are raw policy outputs (pre-scale/offset), as the
        replay buffer stores them; :class:`supermimic.models.Dynamics` is the sole
        place the raw→proc transform is applied.
        """
        return tracking_reward(
            next_aux_state,
            *self.world_frame_bodies(next_aux_state, next_rep_state),
            next_joint_state,
            action,
            last_action,
            target_body_pos_w,
            target_body_quat_w,
            target_body_lin_vel_w,
            target_body_ang_vel_w,
            self._soft_joint_pos_limits,
            self.dt,
        )

    @abstractmethod
    def loss_blocks(self, rep_state: torch.Tensor, aux_state: torch.Tensor) -> dict[str, torch.Tensor]:
        """The channel groups the state loss scores, keyed as ``LOSS_WEIGHT_CONFIG``.

        Whatever the blocks leave out is excluded from the loss.
        """
        ...


class GenCoordRepresentation(StateRepresentation):
    """Generalized-coordinate (joint-space) representation.

    Full state: [root_pos(3), root_quat(4), qj(N), root_lin_vel(3), root_ang_vel(3), vj(N)]
    Rep state:  [z(1), g_b(3), v_b(3), w_b(3), qj(N), vj(N)]
    Prediction: [Δv_b(3), Δw_b(3), Δvj(N)] — accel-level deltas in the current
    root frame; poses and derived channels are integrated analytically.
    """

    LOSS_WEIGHT_CONFIG = {
        **StateRepresentation.LOSS_WEIGHT_CONFIG,
        "joint_pos": "nominal_error_joint_pos",
        "joint_vel": "nominal_error_joint_vel",
    }

    def __init__(self, num_joints: int, default_joint_pos: torch.Tensor):
        super().__init__()
        self.num_joints = num_joints
        self.full_state_dim = 13 + 2 * num_joints
        self.rep_state_dim = rep_state_dim_from_num_joints(num_joints)
        self.prediction_dim = 6 + num_joints  # [Δv_b(3), Δw_b(3), Δvj(N)]
        self._default_joint_pos = default_joint_pos
        self.fk: ForwardKinematics  # set by bind

    def bind(self, env: ManagerBasedRlEnv, sim_context: SimContext) -> None:
        super().bind(env, sim_context)
        self.fk = env.command_manager.get_term("motion").motion.fk

    def represent(self, full_state: torch.Tensor) -> torch.Tensor:
        flat = full_state.reshape(-1, self.full_state_dim)
        rep = compute_gencoord_state(flat)
        return rep.reshape(*full_state.shape[:-1], -1)

    def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
        return full_state[..., :7]

    def apply_prediction(
        self,
        aux_state: torch.Tensor,
        curr_rep_state: torch.Tensor,
        prediction: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Semi-implicit joint-space integration of accel-level predictions.

        The network output is ``[Δv_b, Δw_b, Δvj]`` in the CURRENT root
        frame; ``qj`` integrates the joint velocities, the root pose
        integrates the twist, and ``z``/``g_b`` are derived from the
        integrated pose, so every prediction maps to a kinematically
        consistent state. Pose updates weight the delta by ``substep_coeff``
        to match how the sim composes its substeps; velocity states advance
        by the full delta.
        """
        N = self.num_joints
        c = self.substep_coeff
        _, _, v_b, w_b, qj, vj = decompose_rep_state(curr_rep_state)
        dv_b = prediction[..., 0:3]
        dw_b = prediction[..., 3:6]
        dvj = prediction[..., 6 : 6 + N]

        next_v_cur = v_b + dv_b  # next twist, expressed in the current root frame
        next_w_cur = w_b + dw_b
        next_vj = vj + dvj
        next_qj = qj + (vj + c * dvj) * dt

        root_pos = aux_state[..., :3]
        root_quat = aux_state[..., 3:7]
        v_global = quat_apply(root_quat, next_v_cur)
        w_global = quat_apply(root_quat, next_w_cur)
        v_pose = quat_apply(root_quat, v_b + c * dv_b)
        w_pose = quat_apply(root_quat, w_b + c * dw_b)
        next_root_pos = root_pos + v_pose * dt
        next_root_quat = quat_integrate(root_quat, w_pose, dt)

        next_v_b = quat_apply_inverse(next_root_quat, v_global)
        next_w_b = quat_apply_inverse(next_root_quat, w_global)
        world_down = torch.zeros_like(v_global)
        world_down[..., 2] = -1.0
        next_g_b = quat_apply_inverse(next_root_quat, world_down)

        next_aux = torch.cat([next_root_pos, next_root_quat], dim=-1)
        next_rep = torch.cat(
            [next_root_pos[..., 2:3], next_g_b, next_v_b, next_w_b, next_qj, next_vj],
            dim=-1,
        )
        return next_aux, next_rep

    def reconstruct_full_state(self, aux_state: torch.Tensor, rep_state: torch.Tensor) -> torch.Tensor:
        z, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
        root_pos = aux_state[..., :3]
        root_quat = aux_state[..., 3:7]
        # Override root Z with rep_state's z
        root_pos = torch.cat([root_pos[..., :2], z], dim=-1)
        v_global = quat_apply(root_quat, v_b)
        w_global = quat_apply(root_quat, w_b)
        parts = [root_pos, root_quat, qj, v_global, w_global, vj]
        return torch.cat(parts, dim=-1)

    def loss_blocks(self, rep_state: torch.Tensor, aux_state: torch.Tensor) -> dict[str, torch.Tensor]:
        # z, g_b and the root twist all follow from the integrated root pose.
        _, _, _, _, joint_pos, joint_vel = decompose_rep_state(rep_state)
        return {
            **_root_loss_blocks(aux_state),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }

    def world_frame_bodies(
        self, aux_state: torch.Tensor, rep_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, v_b, w_b, qj, vj = decompose_rep_state(rep_state)
        root_pos, root_quat = aux_state[..., :3], aux_state[..., 3:7]
        pos, quat, lin_vel, ang_vel = self.fk(
            root_pos,
            root_quat,
            qj,
            quat_apply(root_quat, v_b),
            quat_apply(root_quat, w_b),
            vj,
        )
        return pos, matrix_from_quat(quat), lin_vel, ang_vel

    def joint_state(self, rep_state: torch.Tensor) -> torch.Tensor:
        _, _, _, _, qj, vj = decompose_rep_state(rep_state)
        return torch.cat([qj, vj], dim=-1)

    def compute_wm_obs(
        self,
        rep_state: torch.Tensor,
        aux_state: torch.Tensor,
        joint_state: torch.Tensor,
        sim_ctx: torch.Tensor,
    ) -> torch.Tensor:
        del aux_state, sim_ctx
        _, g_b, _, w_b, _, _ = decompose_rep_state(rep_state)
        qj, vj = joint_state.chunk(2, dim=-1)
        return torch.cat([g_b, w_b, qj - self._default_joint_pos, vj], dim=-1)
