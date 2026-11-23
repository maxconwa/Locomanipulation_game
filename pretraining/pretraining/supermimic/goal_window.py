"""The goal window: the strided future window of reference frames the goal observation and the world model read.

Ported from SuperMimic (HybridRobotics, commit 7e94b07, Apache-2.0): the goal-layout constants and the module-level
goal-window helpers of ``src/supermimic/representations.py``, and ``quat_to_rot6d`` of
``src/supermimic/utils/helpers.py``; quaternion maths from ``pretraining.math`` instead of mjlab's. Importable
without Isaac.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pretraining.math import matrix_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
    from pretraining.motions import MotionLoader

# ---------------------------------------------------------------------------
# Goal layout constants
# ---------------------------------------------------------------------------
# Reference-motion frames exposed in the goal window: a strided future window of
# NUM_GOAL_FRAMES frames spaced GOAL_FRAME_STRIDE motion steps apart. Motion runs
# at the control rate (one frame per env step, ~50 Hz / 0.02 s), so stride 5
# ≈ 0.1 s spacing and a ~0.9 s lookahead with 10 frames.
NUM_GOAL_FRAMES = 10
GOAL_FRAME_STRIDE = 5

# wm_goal  = [content, rel_pos_b(3*N), rel_ori_b(6*N)]
# actor_goal = [content, rel_ori_b(6*N)]
# content = NUM_GOAL_FRAMES joint targets: per frame [joint_pos, joint_vel].
GOAL_REL_POS_DIM = 3 * NUM_GOAL_FRAMES  # rel_pos_b: in wm_goal only
GOAL_REL_ORI_DIM = 6 * NUM_GOAL_FRAMES  # rel_ori_b: in both wm_goal and actor_goal


def actor_goal_dim(num_goals: int) -> int:
    """Number of goal dimensions seen by the actor (wm_goal minus rel_pos_b)."""
    return num_goals - GOAL_REL_POS_DIM


def extract_actor_goal(wm_goal: torch.Tensor) -> torch.Tensor:
    """Extract actor goal from full WM goal by removing rel_pos_b dims."""
    tail = GOAL_REL_POS_DIM + GOAL_REL_ORI_DIM
    return torch.cat([wm_goal[..., :-tail], wm_goal[..., -GOAL_REL_ORI_DIM:]], dim=-1)


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to 6D rotation representation (first two columns of rotation matrix)."""
    mat = matrix_from_quat(quat)  # (..., 3, 3)
    return mat[..., :2].flatten(-2, -1)  # (..., 6)


def _assemble_goal(
    content: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble ``(wm_goal, actor_goal)``; only wm_goal carries the rel_pos dims."""
    rel_pos, rel_ori = assemble_anchor_window(root_pos, root_quat, target_pos, target_quat)
    wm_goal = torch.cat([content, rel_pos, rel_ori], dim=-1)
    actor_goal = torch.cat([content, rel_ori], dim=-1)
    return wm_goal, actor_goal


def goal_window_indices(motion: MotionLoader, motion_times: torch.Tensor) -> torch.Tensor:
    """Clamped windowed frame indices ``[t, t+stride, ..., t+(N-1)*stride]``, ``(E, N)``.

    Future offsets are clamped to the last frame of each env's current clip so the
    window never bleeds into the next concatenated clip — matching how the env
    advances ``motion_times``.
    """
    clip = torch.searchsorted(motion.motion_ends, motion_times, right=True)
    # Past-the-end guard: searchsorted returns numel() at the final frame.
    clip = clip.clamp(max=motion.motion_ends.numel() - 1)
    last_valid = motion.motion_ends[clip] - 1
    offsets = torch.arange(NUM_GOAL_FRAMES, device=motion_times.device) * GOAL_FRAME_STRIDE
    return torch.minimum(motion_times[:, None] + offsets, last_valid[:, None])


def motion_goal_window(motion: MotionLoader, motion_times: torch.Tensor) -> torch.Tensor:
    """Windowed joint targets as the goal content, ``(E, N * 2J)`` — per frame
    ``[joint_pos(J), joint_vel(J)]``.
    """
    idx = goal_window_indices(motion, motion_times)
    frames = torch.cat([motion.joint_pos[idx], motion.joint_vel[idx]], dim=-1)
    return frames.reshape(frames.shape[0], -1)


def motion_anchor_window(motion: MotionLoader, motion_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Motion anchor (the root) world pose ``(pos (E, N, 3), quat (E, N, 4))`` at
    the same windowed indices as :func:`motion_goal_window`.
    """
    idx = goal_window_indices(motion, motion_times)
    return motion.root_pos_w[idx], motion.root_quat_w[idx]


def assemble_anchor_window(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame motion anchor expressed relative to the robot's current root.

    ``root_*``: ``(E, 3)`` / ``(E, 4)``; ``target_*``: ``(E, N, 3)`` / ``(E, N, 4)``.
    Returns flattened ``(rel_pos (E, N*3), rel_ori_rot6d (E, N*6))``. The world-frame
    origin offset cancels in the relative transform, so callers only need the root
    and target anchors expressed in a common frame.
    """
    num_envs, num_frames = target_pos.shape[:2]
    root_pos = root_pos[:, None, :].expand(num_envs, num_frames, 3).reshape(-1, 3)
    root_quat = root_quat[:, None, :].expand(num_envs, num_frames, 4).reshape(-1, 4)
    rel_pos, rel_ori_w = subtract_frame_transforms(
        root_pos, root_quat, target_pos.reshape(-1, 3), target_quat.reshape(-1, 4)
    )
    rel_pos = rel_pos.reshape(num_envs, -1)
    rel_ori = quat_to_rot6d(rel_ori_w).reshape(num_envs, -1)
    return rel_pos, rel_ori


_compiled_assemble_goal = torch.compile(_assemble_goal)
