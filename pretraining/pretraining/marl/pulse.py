"""The frozen stage-1 decoders as a task policy's action space (PULSE): latent in, joint targets out.

``LocoManip-Marl-Pulse-v0`` (main's ``locomanip_marl/mdp/pulse_actions.py``) gives each agent a latent action ``u``
and decodes it here with that agent's frozen stage-1 actor, as h12_train's stage 2 did
(``marl/actors.py::CommandActor`` at b175f1c, run ``amass_s2_pulse``):

    hist   = actor.obs_normalizer(proprio)                    (one 87-wide frame, history_length 1)
    z      = tanh(u + prior(hist))                            (no prior: tanh(u))
    a      = actor.decoder([z, hist])                         (12 legs, 15 torso + arms; Unitree motor order)
    target = clip(default + ACTION_SCALE * a, TARGET_LIMITS and any tighter limits given)

The proprio frame is the policy contract's ``[gravity 3, gyro 3, q - default 27, dq 27, last action 27]``
(``runner.PROPRIO_DIM``), the last action being both decoders' raw outputs of the step before. Both agents' decoders
read one frame, built before either decodes, and the last-action buffer moves on only after both have: ``step``
takes both latents at once, so the order the env hands them over in cannot matter. Pure torch (no Isaac): the env side
builds the frame's first 60 columns from its sensors.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pretraining.constants import ACTION_SCALE, DEFAULT_JOINT_POS, JOINT_NAMES, TARGET_LIMITS
from pretraining.marl.config import AGENTS, JOINT_SLICES

SENSOR_DIM = 3 + 3 + 2 * len(JOINT_NAMES)  # the frame without the last action
"""Columns of the proprio frame the env builds: gravity, gyro, q - default, dq."""


class PulseDecoders:
    """Both agents' frozen decoders, their priors and the shared last-action buffer, for ``num_envs`` envs.

    ``actors``: ``{agent: MultiEncoderAE}`` (``runner.load_actors``), history length 1. ``priors``: ``{agent:
    LatentPrior}`` (``actors.load_latent_prior``) or None. ``limits``: ``{joint: (low, high)}`` intersected with
    ``TARGET_LIMITS`` (main passes GOLEM's position envelope).
    """

    def __init__(
        self,
        actors: dict[str, nn.Module],
        num_envs: int,
        device,
        priors: dict[str, nn.Module] | None = None,
        limits: dict[str, tuple[float, float]] | None = None,
    ):
        for agent, actor in actors.items():
            if actor.history_length != 1:
                raise ValueError(f"the {agent} stage-1 actor reads {actor.history_length} frames; only 1 is supported")
        if set(actors) != set(AGENTS) or (priors is not None and set(priors) != set(AGENTS)):
            raise ValueError(f"need an actor (and a prior, if any) per agent {AGENTS}")
        self.actors = {a: actors[a].eval().requires_grad_(False) for a in AGENTS}
        self.priors = None if priors is None else {a: priors[a].eval().requires_grad_(False) for a in AGENTS}
        self.latent_dim = self.actors[AGENTS[0]].latent_dim
        self.num_envs, self.device = num_envs, torch.device(device)
        t = lambda values: torch.tensor(values, dtype=torch.float32, device=self.device)  # noqa: E731
        low, high = [], []
        for j in JOINT_NAMES:
            lo, hi = TARGET_LIMITS[j]
            if limits and j in limits:
                lo, hi = max(lo, limits[j][0]), min(hi, limits[j][1])
            if lo > hi:
                raise ValueError(f"{j}: the target limits are empty ({lo:.3f} > {hi:.3f})")
            low.append(lo)
            high.append(hi)
        self.default = t([DEFAULT_JOINT_POS[j] for j in JOINT_NAMES])
        self.scale = t([ACTION_SCALE[j] for j in JOINT_NAMES])
        self.low, self.high = t(low), t(high)
        self.last_action = torch.zeros(num_envs, len(JOINT_NAMES), device=self.device)
        """(num_envs, 27): both decoders' raw outputs of the last step, Unitree motor order; zero after a reset."""

    def proprio(self, sensors: torch.Tensor) -> torch.Tensor:
        """The (num_envs, 87) frame: ``sensors`` (num_envs, SENSOR_DIM) and the last-action buffer."""
        return torch.cat([sensors, self.last_action], dim=-1)

    @torch.no_grad()
    def step(self, latents: dict[str, torch.Tensor], sensors: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode both agents' latents ``{agent: (num_envs, latent_dim)}`` on one proprio frame; return each agent's
        joint targets (rad, its ``JOINT_SLICES`` of ``JOINT_NAMES``) and move the last-action buffer on."""
        frame = self.proprio(sensors).unsqueeze(1)  # built once, before either decodes
        raw, targets = {}, {}
        for agent in AGENTS:
            actor = self.actors[agent]
            hist = actor.obs_normalizer(frame).flatten(1)
            u = latents[agent]
            if self.priors is not None:
                u = u + self.priors[agent](hist)
            a = actor.decoder(torch.cat([torch.tanh(u), hist], dim=-1))
            joints = JOINT_SLICES[agent]
            target = self.default[joints] + self.scale[joints] * a
            raw[agent] = a
            targets[agent] = torch.maximum(torch.minimum(target, self.high[joints]), self.low[joints])
        self.last_action = torch.cat([raw[a] for a in AGENTS], dim=-1)  # after both have decoded
        return targets

    def reset(self, env_ids) -> None:
        """Zero the reset envs' last action (``slice(None)`` or None: every env)."""
        self.last_action[slice(None) if env_ids is None else env_ids] = 0.0
