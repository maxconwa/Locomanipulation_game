# Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c), src/h12_train/marl/actors.py.
# Changes: imports; stage 1 only (LatentPrior, build_latent_prior and CommandActor left out); the goal error message
# names this package.
"""The agents' stage-1 actors: SuperMimic's encoder -> tanh latent -> decoder.

Stage 1 (imitation): one ``MultiEncoderAE`` per agent with a single task, ``motion``,
whose goal is the agent's half of the reference (``wm_env.legs_goal``/``arms_goal``).
"""

from __future__ import annotations

import torch

from pretraining.supermimic.models import MultiEncoderAE
from pretraining.supermimic.tasks.task_spec import TaskSpec


def _unused_extract(*_args) -> torch.Tensor:
    """Raise: the runners build the goals (``wm_env.py``), not the task spec."""
    raise RuntimeError("goals are built by pretraining.marl.wm_env")


def build_imitation_actor(
    obs_dim: int,
    action_dim: int,
    goal_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    history_length: int,
) -> MultiEncoderAE:
    """Return a stage-1 actor reading ``history`` (B, H, obs_dim) and ``goal_motion``."""
    spec = TaskSpec(name="motion", goal_dim=goal_dim, extract_training_goal=_unused_extract)
    return MultiEncoderAE(
        obs_dim=obs_dim,
        action_dim=action_dim,
        task_specs=[spec],
        slot_mix={"motion": 1.0},
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        latent_bc_weight=0.0,
        history_length=history_length,
    )
