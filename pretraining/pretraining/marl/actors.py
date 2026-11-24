# Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c), src/h12_train/marl/actors.py.
# Changes: imports; CommandActor (stage 2) left out; the goal error message names this package; load_latent_prior
# added.
"""The agents' stage-1 actors: SuperMimic's encoder -> tanh latent -> decoder, and PULSE's latent prior.

Stage 1 (imitation): one ``MultiEncoderAE`` per agent with a single task, ``motion``,
whose goal is the agent's half of the reference (``wm_env.legs_goal``/``arms_goal``).

The latent prior (PULSE's learned prior: a frozen network from the proprio history to the stage-1 latent's pre-tanh
mean, fitted by ``scripts/fit_prior.py`` on the stage-1 actors tracking the clips): a task policy on the frozen
decoder outputs a residual ``u`` on it, ``z = tanh(u + prior(history))``, so a zero residual does what the stage-1
policy does in that state (``LocoManip-Marl-Pulse-v0``).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from pretraining.supermimic.models import MultiEncoderAE, build_mlp
from pretraining.supermimic.tasks.task_spec import TaskSpec


class LatentPrior(nn.Module):
    """A latent prior: normalised proprio history (flattened) -> the latent's pre-tanh mean.

    ``mask`` (a buffer, saved with the weights) multiplies the input; ``hide_last_action``
    zeroes each frame's last ``action_dim`` columns, the last action, so the prior reads
    the robot's state only and cannot copy the latent out of the action it produced.
    """

    def __init__(
        self,
        obs_dim: int,
        history_length: int,
        hidden_dims: tuple[int, ...],
        latent_dim: int,
        action_dim: int = 0,
        hide_last_action: bool = False,
    ):
        """Build the MLP and the input mask."""
        super().__init__()
        mask = torch.ones(history_length, obs_dim)
        if hide_last_action:
            mask[:, obs_dim - action_dim :] = 0.0
        self.register_buffer("mask", mask.flatten())
        self.net = build_mlp(history_length * obs_dim, hidden_dims, output_dim=latent_dim)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Return the pre-tanh latent mean for the flattened normalised history."""
        return self.net(history * self.mask)


def build_latent_prior(
    obs_dim: int, history_length: int, hidden_dims: tuple[int, ...], latent_dim: int
) -> LatentPrior:
    """Return a latent prior of the given shape (its mask comes with its weights)."""
    return LatentPrior(obs_dim, history_length, hidden_dims, latent_dim)


def load_latent_prior(path: Path, device) -> dict[str, LatentPrior]:
    """Return the frozen priors of a ``fit_prior.py`` file (``{agent: state dict}``) on ``device``, in eval mode.

    The shapes come from the state dicts: the input width from the mask, the hidden and latent widths from the
    MLP's linear layers (``build_mlp``: Linear, LayerNorm, GELU per hidden layer, then a Linear).
    """
    state = torch.load(Path(path), map_location=device, weights_only=False)
    priors = {}
    for agent, sd in state.items():
        linear = sorted(
            (int(k.split(".")[1]), v.shape) for k, v in sd.items() if k.startswith("net.") and k.endswith(".weight")
            and v.dim() == 2
        )
        widths = [shape[0] for _, shape in linear]
        prior = LatentPrior(sd["mask"].numel(), 1, tuple(widths[:-1]), widths[-1])
        prior.load_state_dict(sd)
        priors[agent] = prior.to(device).eval().requires_grad_(False)
    return priors


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
