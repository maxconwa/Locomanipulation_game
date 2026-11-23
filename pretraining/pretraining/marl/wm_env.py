# Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c), src/h12_train/marl/wm_env.py.
# Changes: imports; stage 1 only (CommandWorldModelEnv and fall_probability left out); the termination thresholds read
# as configclass attributes (Isaac Lab); ClipTables (its tool-frame table only: the stage-2 velocity table, sampling
# ranges and collision mask left out), reference_aux_state, legs_goal, arms_goal and ee_window moved here from
# src/h12_train/marl/goals.py (stage 2's module, not ported), otherwise unchanged.
"""The two-agent stage-1 world-model env, on SuperMimic's ``WorldModelEnv``, and the agents' goals.

Both agents observe the same synthesised proprio history (torso IMU, all joints, the
whole last action) and each its own goal; ``step`` returns rewards ``(B, 3)``: legs,
arms, team, each agent's with its estop penalties (``rewards.estop_penalties``) on the
predicted velocity and the PD torque demand of the clamped target. Observations are
``{"legs": {history, goal_*}, "arms": {...}, "critic": x}``.

Stage-1 goals use SuperMimic's goal window (``NUM_GOAL_FRAMES`` reference frames
``GOAL_FRAME_STRIDE`` steps apart, clamped to the clip's last frame):

- legs: per frame the 12 leg joints' targets ``[pos, vel]``, then the reference pelvis
  orientation relative to the robot's (rot6d per frame), as SuperMimic's actor goal;
- arms: per frame the torso's and 14 arm joints' targets ``[pos, vel]``, then per frame
  both tool frames' reference poses in the reference pelvis frame (position + rot6d).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from pretraining.marl.config import ARM_JOINTS, LEG_JOINTS, ImitationRewardCfg
from pretraining.marl.rewards import (
    JointLimits,
    ee_features,
    ee_in_pelvis,
    estop_penalties,
    estop_position_depth,
    estop_ratios,
    imitation_rewards,
)
from pretraining.math import matrix_from_quat
from pretraining.supermimic.envs.world_model_env import WorldModelEnv
from pretraining.supermimic.goal_window import assemble_anchor_window, goal_window_indices, motion_anchor_window

if TYPE_CHECKING:
    from pretraining.motions import MotionLoader

_FK_CHUNK = 4096  # frames per forward-kinematics batch when tabulating


class ClipTables:
    """Per-frame tables of the motion: the tool-frame poses."""

    def __init__(self, motion: MotionLoader):
        """Tabulate both tool frames' poses in the reference pelvis frame over every frame of ``motion``."""
        self.motion = motion
        device = motion.joint_pos.device
        chunks = []
        for start in range(0, motion.time_step_total, _FK_CHUNK):
            frames = torch.arange(start, min(start + _FK_CHUNK, motion.time_step_total), device=device)
            pos, quat, _, _ = motion.body_state(frames)
            chunks.append(ee_features(*ee_in_pelvis(pos, matrix_from_quat(quat))))
        self.ee = torch.cat(chunks)  # (T, 2, 9)


def reference_aux_state(motion: MotionLoader, times: torch.Tensor) -> torch.Tensor:
    """Return the reference root pose ``[pos, quat]`` (7) at ``times``, as stage 1's robot on the clip."""
    return torch.cat([motion.root_pos_w[times], motion.root_quat_w[times]], dim=-1)


def legs_goal(motion: MotionLoader, times: torch.Tensor, aux_state: torch.Tensor) -> torch.Tensor:
    """Return the legs' stage-1 goal at ``times`` for robots at ``aux_state`` (root pose)."""
    idx = goal_window_indices(motion, times)
    content = torch.cat(
        [motion.joint_pos[idx][..., LEG_JOINTS], motion.joint_vel[idx][..., LEG_JOINTS]],
        dim=-1,
    ).flatten(1)
    target_pos, target_quat = motion_anchor_window(motion, times)
    _, rel_ori = assemble_anchor_window(aux_state[..., :3], aux_state[..., 3:7], target_pos, target_quat)
    return torch.cat([content, rel_ori], dim=-1)


def arms_goal(tables: ClipTables, times: torch.Tensor) -> torch.Tensor:
    """Return the arms' stage-1 goal at ``times``: torso and arm targets, tool-frame poses."""
    motion = tables.motion
    idx = goal_window_indices(motion, times)
    content = torch.cat(
        [motion.joint_pos[idx][..., ARM_JOINTS], motion.joint_vel[idx][..., ARM_JOINTS]],
        dim=-1,
    ).flatten(1)
    return torch.cat([content, tables.ee[idx].flatten(1)], dim=-1)


def ee_window(tables: ClipTables, times: torch.Tensor) -> torch.Tensor:
    """Return the reference tool-frame poses over the goal window, flattened."""
    return tables.ee[goal_window_indices(tables.motion, times)].flatten(1)


class _TwoAgentWorldModelEnv(WorldModelEnv):
    """Shared state of the two-agent world-model envs."""

    def __init__(
        self,
        dynamics,
        representation,
        env: Any,
        batch_size: int,
        replay_buffer,
        tables: ClipTables,
        history_length: int,
        termination_softness: float,
        max_step_reward: torch.Tensor,
        joint_limits: JointLimits,
        device: str = "cpu",
    ) -> None:
        """Hold the world model, the replay buffer, the clip tables and the joints' limits.

        ``WorldModelEnv.__init__`` is not called: it reads the motion-relative termination
        thresholds, which stage 2 does not have; this sets the same fields it does.
        """
        if termination_softness <= 0.0:
            raise ValueError(f"termination_softness must be positive, got {termination_softness}")
        self.device = device
        self.dynamics = dynamics
        self.representation = representation
        self.num_envs = batch_size
        self.replay_buffer = replay_buffer
        self.history_length = history_length
        self._num_actions = env.num_actions
        motion_cmd = env.unwrapped.command_manager.get_term("motion")
        self.motion = motion_cmd.motion
        self._anchor_index = motion_cmd.motion_anchor_body_index
        self._termination_softness = termination_softness
        self.tables = tables
        self.max_step_reward = max_step_reward.to(device)  # (2,): legs, arms (own + team)
        self.dt = env.unwrapped.step_dt
        self.joint_limits = joint_limits

    def _estop(self, cfg, rep_state, next_joint, action) -> torch.Tensor:
        """Return each agent's estop penalties (B, 2) for one predicted step.

        The target is the clamped one less the encoder bias, as the sim's actuator applies it.
        """
        q, dq = self.representation.joint_state(rep_state).chunk(2, dim=-1)
        target = self.dynamics.proc(action) - self.representation.encoder_bias(self.sim_ctx)
        q_next, dq_next = next_joint.chunk(2, dim=-1)
        vel, tau = estop_ratios(self.joint_limits, q, dq, dq_next, target)
        depth = estop_position_depth(self.joint_limits, q_next, cfg.position_margin)
        return estop_penalties(cfg, vel, tau, depth)

    def _agent_td(self, goal_key: str, goal: torch.Tensor, history: torch.Tensor, **extra) -> TensorDict:
        """Return one agent's actor input."""
        return TensorDict(
            {"history": history, goal_key: goal, **extra},
            batch_size=[self.num_envs],
            device=self.device,
        )


class ImitationWorldModelEnv(_TwoAgentWorldModelEnv):
    """Stage 1: both agents track the clip; the termination is the tracking task's."""

    def __init__(self, *args, rewards: ImitationRewardCfg, env: Any, **kwargs) -> None:
        """Read the tracking task's pelvis height and tilt thresholds."""
        super().__init__(*args, env=env, **kwargs)
        self.reward_cfg = rewards
        terms = env.cfg.terminations
        self._anchor_pos_threshold = terms.anchor_pos.params["threshold"]
        self._anchor_ori_threshold = terms.anchor_ori.params["threshold"]

    def _wm_step(self, rep_state, aux_state, action, last_action, sim_ctx):
        """Predict the next state, its termination probability and the three rewards."""
        next_aux, next_rep = self.dynamics(rep_state, action, aux_state, sim_ctx)
        next_joint = self.representation.joint_state(next_rep)
        ref_pos, ref_quat, ref_lin, ref_ang = self._ref_body_state
        term_prob = self._compute_term_prob(next_aux, ref_pos, ref_quat)
        pos, rot, lin, ang = self.representation.world_frame_bodies(next_aux, next_rep)
        reward = imitation_rewards(
            self.reward_cfg,
            next_aux,
            pos,
            rot,
            lin,
            ang,
            next_joint,
            action,
            last_action,
            ref_pos,
            ref_quat,
            ref_lin,
            ref_ang,
            self.representation._soft_joint_pos_limits,
            self.dt,
            estop=self._estop(self.reward_cfg.estop, rep_state, next_joint, action),
        )
        return next_aux, next_rep, next_joint, term_prob, reward

    def _build_task_goals(self) -> None:
        """Build both agents' goals and the critic's reference tool-frame window."""
        self._legs_goal = legs_goal(self.motion, self.motion_times, self.aux_state)
        self._arms_goal = arms_goal(self.tables, self.motion_times)
        self._ee_window = ee_window(self.tables, self.motion_times)

    def _critic_obs(self) -> torch.Tensor:
        """Return the full WM state, SuperMimic's whole-body goal, the tool-frame window, the sim context."""
        return torch.cat([self.rep_state, self._wm_goal_obs, self._ee_window, self.sim_ctx], dim=-1)

    def get_observations(self) -> TensorDict:
        """Return both agents' inputs and the critic's."""
        history = self._sensor_history()
        return TensorDict(
            {
                "legs": self._agent_td("goal_motion", self._legs_goal, history),
                "arms": self._agent_td("goal_motion", self._arms_goal, history),
                "critic": self._critic_obs(),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )
