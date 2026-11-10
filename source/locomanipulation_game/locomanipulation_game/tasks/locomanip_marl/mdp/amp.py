"""Adversarial motion prior (Peng et al. 2021, AMP): a discriminator learns to tell the policy's state transitions
from a reference motion's, and its score is a style reward. The reference is never an input to the policy.

The discriminator lives in the reward term and trains inside the env, so skrl's MAPPO runs unchanged: every step the
term scores each env's (state frame_stride steps ago, this state) and stores it, and every update_every steps (one
rollout) it takes num_updates minibatch steps on the stored transitions against the reference's, frame_stride frames
apart (3: 30 ms, about the reference's own 30 fps; at one 10 ms step a sliding foot moves a few millimetres). LSGAN (targets +1 reference, -1
policy), a gradient penalty on the reference inputs, a logit-weight penalty and weight decay; the reward is
max(0, 1 - (D - 1)^2 / 4), 1 for a reference-like transition, times style_ramp's share: 0 before its first step
count, rising linearly to 1 at its second (the discriminator trains throughout), so the policy learns to walk on
the task reward before the style can make standing still the cheaper way to look like the data. Inputs are standardised by the reference states' mean
and std.

The state (scripts/build_amp_reference.STATE_NAMES builds the reference's): the leg joints and their velocities,
gravity in the pelvis frame, the pelvis's linear and angular velocity in its own frame, and both feet (the ankle-roll
links) relative to the pelvis in its heading frame, their heights above the ground (the height scanner's) and their
velocities in the heading frame: a skating stance foot moves over the ground, a planted one doesn't.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from .commands import ground_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

AMP_STATE_DIM = 47
SOLE = 0.045  # m, the ankle-roll link above the ground standing (build_amp_reference.SOLE)


def amp_state(env: ManagerBasedRLEnv, joint_cfg: SceneEntityCfg, feet_cfg: SceneEntityCfg,
              sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """(num_envs, 47): the reference's state from the sim (build_amp_reference.STATE_NAMES). The ground is the height
    scanner's under the pelvis; where every ray missed, the lower foot's sole."""
    data = env.scene[joint_cfg.name].data
    feet_w = data.body_link_pos_w[:, feet_cfg.body_ids]
    rel = feet_w - data.root_link_pos_w.unsqueeze(1)
    heading = yaw_quat(data.root_link_quat_w).unsqueeze(1).expand(-1, rel.shape[1], -1).reshape(-1, 4)
    feet = quat_apply_inverse(heading, rel.reshape(-1, 3)).reshape(rel.shape[0], -1)
    ground = ground_height(env.scene.sensors[sensor_cfg.name])
    ground = torch.where(torch.isfinite(ground), ground, feet_w[..., 2].min(dim=1).values - SOLE)
    heights = feet_w[..., 2] - ground.unsqueeze(1)
    vel = quat_apply_inverse(heading, data.body_link_lin_vel_w[:, feet_cfg.body_ids].reshape(-1, 3))
    return torch.cat([data.joint_pos[:, joint_cfg.joint_ids], data.joint_vel[:, joint_cfg.joint_ids],
                      data.projected_gravity_b, data.root_link_lin_vel_b, data.root_link_ang_vel_b, feet, heights,
                      vel.reshape(rel.shape[0], -1)], dim=1)


class amp_style(ManagerTermBase):
    """The style reward of an adversarial motion prior on the legs (module doc). params: joint_cfg (the 12 leg joints
    in the reference's order), feet_cfg (the ankle-roll links, left first), sensor_cfg (the height scanner),
    reference (build_amp_reference's npz), and the discriminator's settings."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        ref = np.load(p["reference"])
        states = torch.as_tensor(ref["states"], dtype=torch.float32, device=self.device)
        if states.shape[1] != AMP_STATE_DIM:
            raise ValueError(f"{p['reference']}: states of {states.shape[1]}, the env's are {AMP_STATE_DIM}")
        self._ref_states = states
        self._stride = p.get("frame_stride", 1)
        # pairs (i, i + stride): the frames between are consecutive frames of one clip
        chain = torch.as_tensor(ref["next_valid"], device=self.device)
        ok = chain.clone()
        for k in range(1, self._stride):
            ok[:-k] &= chain[k:]
            ok[-k:] = False
        self._ref_pairs = torch.nonzero(ok).squeeze(1)
        self._mean = states.mean(dim=0)
        self._std = states.std(dim=0).clamp(min=1e-2)
        hidden = p.get("hidden", (1024, 512))
        layers, width = [], 2 * AMP_STATE_DIM
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        self.disc = nn.Sequential(*layers, nn.Linear(width, 1)).to(self.device)
        self.optimizer = torch.optim.Adam(self.disc.parameters(), lr=p.get("lr", 5e-5),
                                          weight_decay=p.get("weight_decay", 1e-4))
        self._update_every = p.get("update_every", 48)
        self._num_updates = p.get("num_updates", 16)
        self._batch = p.get("batch_size", 8192)
        self._gp = p.get("grad_penalty", 5.0)
        self._logit_reg = p.get("logit_reg", 0.05)
        capacity = p.get("buffer_size", 400_000)
        self._buffer = torch.zeros(capacity, 2 * AMP_STATE_DIM, device=self.device)
        self._buffer_next = 0
        self._buffer_full = False
        # each env's last frame_stride states (a ring, slot _slot the oldest) and its steps since the reset
        self._history = torch.zeros(env.num_envs, self._stride, AMP_STATE_DIM, device=self.device)
        self._slot = 0
        self._age = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self._calls = 0
        print(f"[INFO] AMP: {len(states)} reference states, {len(self._ref_pairs)} pairs {self._stride} frames apart"
              f" ({p['reference']}); discriminator {list(hidden)}, {self._num_updates} updates of {self._batch} every"
              f" {self._update_every} steps")

    def reset(self, env_ids=None):
        # an episode's first frame_stride steps have no state of their own that far back
        self._age[slice(None) if env_ids is None else env_ids] = 0

    def _scores(self, pairs: torch.Tensor) -> torch.Tensor:
        a, b = pairs[:, :AMP_STATE_DIM], pairs[:, AMP_STATE_DIM:]
        x = torch.cat([(a - self._mean) / self._std, (b - self._mean) / self._std], dim=1)
        return self.disc(x).squeeze(1)

    def __call__(self, env: ManagerBasedRLEnv, joint_cfg: SceneEntityCfg, feet_cfg: SceneEntityCfg,
                 sensor_cfg: SceneEntityCfg, reference: str, hidden=(1024, 512), lr=5e-5, weight_decay=1e-4,
                 grad_penalty=5.0, logit_reg=0.05, update_every=48, num_updates=16, batch_size=8192,
                 buffer_size=400_000, frame_stride=1, style_ramp=None) -> torch.Tensor:
        # the settings are read once, in __init__
        state = amp_state(env, joint_cfg, feet_cfg, sensor_cfg).clone()
        pairs = torch.cat([self._history[:, self._slot], state], dim=1)
        valid = self._age >= self._stride
        with torch.no_grad():
            score = self._scores(pairs)
        share = 1.0
        if style_ramp is not None:
            start, end = style_ramp
            share = min(max((env.common_step_counter - start) / max(end - start, 1), 0.0), 1.0)
        reward = torch.clamp(1.0 - 0.25 * torch.square(score - 1.0), min=0.0) * valid.float()
        self._store(pairs[valid])
        self._history[:, self._slot] = state
        self._slot = (self._slot + 1) % self._stride
        self._age += 1
        self._calls += 1
        log = getattr(env, "term_log", None)
        if log is not None and valid.any():
            log["AMP/policy_score"] = score[valid].mean()
            log["AMP/style_reward"] = reward[valid].mean()
            log["AMP/style_share"] = torch.tensor(share, device=self.device)
        # evaluation (cfg.training False) scores with the checkpoint's discriminator and leaves it as it is
        if self._calls % self._update_every == 0 and getattr(env.cfg, "training", True):
            self._train(log)
        return reward * share

    def _store(self, pairs: torch.Tensor):
        n, capacity = len(pairs), len(self._buffer)
        if n == 0:
            return
        if n > capacity:
            pairs, n = pairs[-capacity:], capacity
        end = self._buffer_next + n
        if end <= capacity:
            self._buffer[self._buffer_next:end] = pairs
        else:
            first = capacity - self._buffer_next
            self._buffer[self._buffer_next:] = pairs[:first]
            self._buffer[:n - first] = pairs[first:]
        self._buffer_full |= end >= capacity
        self._buffer_next = end % capacity

    def _train(self, log: dict | None):
        size = len(self._buffer) if self._buffer_full else self._buffer_next
        if size < self._batch:
            return
        with torch.inference_mode(False), torch.enable_grad():
            stats = torch.zeros(4, device=self.device)
            for _ in range(self._num_updates):
                policy = self._buffer[torch.randint(0, size, (self._batch,), device=self.device)]
                i = self._ref_pairs[torch.randint(0, len(self._ref_pairs), (self._batch,), device=self.device)]
                expert = torch.cat([self._ref_states[i], self._ref_states[i + self._stride]], dim=1).requires_grad_(True)
                d_expert, d_policy = self._scores(expert), self._scores(policy)
                loss = torch.square(d_expert - 1.0).mean() + torch.square(d_policy + 1.0).mean()
                grad = torch.autograd.grad(d_expert.sum(), expert, create_graph=True)[0]
                penalty = torch.square(grad).sum(dim=1).mean()
                logit = torch.square(self.disc[-1].weight).sum()
                total = loss + 0.5 * self._gp * penalty + self._logit_reg * logit
                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()
                stats += torch.stack([loss.detach(), penalty.detach(), d_expert.detach().mean(), d_policy.detach().mean()])
        if log is not None:
            stats /= self._num_updates
            for key, value in zip(("loss", "grad_penalty", "expert_score", "policy_score_train"), stats):
                log[f"AMP/{key}"] = value

    def checkpoint_state(self) -> dict:
        return {"disc": self.disc.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_checkpoint_state(self, state: dict):
        self.disc.load_state_dict(state["disc"])
        self.optimizer.load_state_dict(state["optimizer"])

