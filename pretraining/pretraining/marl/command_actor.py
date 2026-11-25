# Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c), src/h12_train/marl/actors.py
# (CommandActor). Changes: recurrent (an LSTM of 64 with main's cell cap, then an MLP on its output and the input, as
# main's actor, agents/skrl_mappo_cfg.yaml:19-36); it reads main's actor observation and the command, not the
# proprio history; the frozen decoder is applied by the caller (command_runner), so this module is the encoder only;
# the stage-1 latent of the latent-BC term is its own function (stage1_latent).
"""Stage 2's command encoder: main's actor observation and command -> the frozen stage-1 decoder's latent.

``RecurrentCommandActor`` is one agent's encoder. Main's MAPPO actor is an LSTM of 64 whose cell state is squashed
after every step (``c <- tanh(c)``, ``recurrent.py``'s cell cap 1.0: a cell that grew over minutes on the robot
drifted its stance) and an MLP on the LSTM output concatenated with the raw observation; this is the same shape with
a tanh-bounded 32-wide latent ``z`` as its output, which the frozen stage-1 decoder of the agent turns into joint
targets. Recurrent because main's action delay and sensor noise make one frame of proprio a weaker state estimate than
in h12_train's nominal sim (the design's piece 4).

The hidden state is one tensor ``hc`` (n, 2, lstm): ``h`` then ``c``, so it can be stored per env in the replay
buffer and an imagined rollout can start from the state the encoder had in the sim at that step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pretraining.supermimic.models import _CudaGraphNormalization, build_mlp


class RecurrentCommandActor(nn.Module):
    """One agent's command encoder (module docstring)."""

    def __init__(self, obs_dim: int, command_dim: int, latent_dim: int = 32, lstm: int = 64,
                 hidden: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.obs_dim, self.command_dim, self.latent_dim, self.lstm_size = obs_dim, command_dim, latent_dim, lstm
        width = obs_dim + command_dim
        self.normalizer = _CudaGraphNormalization(width)
        self.cell = nn.LSTMCell(width, lstm)
        self.head = build_mlp(lstm + width, hidden, output_dim=latent_dim)

    def initial_state(self, n: int, device=None) -> torch.Tensor:
        """Zero hidden state (n, 2, lstm), as main's actor at an episode start."""
        return torch.zeros(n, 2, self.lstm_size, device=device or self.normalizer._mean.device)

    def forward(self, obs: torch.Tensor, command: torch.Tensor,
                hc: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``z = tanh(MLP([h, x]))`` (n, latent_dim) and the next ``hc``, for observation ``obs`` (n, obs_dim) and
        command ``command`` (n, command_dim); ``hc`` None: zeros."""
        x = self.normalizer(torch.cat([obs, command], dim=-1))
        if hc is None:
            hc = self.initial_state(x.shape[0], x.device)
        h, c = self.cell(x, (hc[:, 0], hc[:, 1]))
        c = torch.tanh(c)  # main's cell cap (s = 1)
        z = torch.tanh(self.head(torch.cat([h, x], dim=-1)))
        return z, torch.stack([h, c], dim=1)

    @torch.no_grad()
    def update_normalization(self, obs: torch.Tensor, command: torch.Tensor) -> None:
        """Update the input normaliser from sim inputs (a no-op in eval mode, so train mode for the call)."""
        mode = self.normalizer.training
        self.normalizer.train()
        self.normalizer.update(torch.cat([obs, command], dim=-1))
        self.normalizer.train(mode)


def stage1_latent(imitation: nn.Module, goal: torch.Tensor) -> torch.Tensor:
    """The frozen stage-1 encoder's latent ``tanh(encoder(normalised goal))`` for a stage-1 goal (h12_train
    CommandActor.forward's ``z_prior``); no gradient."""
    with torch.no_grad():
        return torch.tanh(imitation.encoders["motion"](imitation.goal_normalizers["motion"](goal)))


def latent_bc(z: torch.Tensor, z_stage1: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    """Per row ``weight * mask * mean((z - z_stage1)^2)`` (h12_train's prior term, supermimic's latent BC: the mean
    over the latent's columns)."""
    return weight * mask.to(z.dtype) * (z - z_stage1).square().mean(-1)


def latent_to_action(z: torch.Tensor, clip: float = 5.0) -> torch.Tensor:
    """The Pulse env's action ``u`` for latent ``z`` with no prior (``z = tanh(u)``): ``atanh(clamp(z, +-tanh clip))``,
    clip the env's ``clip_actions``."""
    bound = torch.tanh(torch.tensor(clip, dtype=z.dtype)).item()
    return torch.atanh(z.clamp(-bound, bound))
