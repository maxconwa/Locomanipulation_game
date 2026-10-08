"""The deployed actors of a MAPPO checkpoint, loaded with torch alone (no skrl): the same computation as
scripts/skrl/export_marl.py, which GOLEM's deploy controller matches to 1e-4.

    actor(obs) = clamp(MLP((clamp((obs - mean) / (std + 1e-8), -5, 5))), -clip_actions, clip_actions)

where mean / std are skrl's RunningStandardScaler statistics saved in the checkpoint and the MLP is the policy's
net_container (ELU between linear layers), i.e. the Gaussian policy's mean action.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, mean, var, layers: nn.Sequential, clip_obs: float = 5.0, clip_actions: float = 10.0):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", torch.sqrt(var.float()))
        self.net = layers
        self.clip_obs, self.clip_actions = clip_obs, clip_actions

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = torch.clamp((obs - self.mean) / (self.std + 1e-8), -self.clip_obs, self.clip_obs)
        return torch.clamp(self.net(z), -self.clip_actions, self.clip_actions)


def _mlp(state: dict, prefix: str) -> nn.Sequential:
    idx = sorted({int(k[len(prefix) + 1:].split(".")[0]) for k in state if k.startswith(prefix + ".")})
    layers = []
    for i, j in enumerate(idx):
        w, b = state[f"{prefix}.{j}.weight"], state[f"{prefix}.{j}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.weight.data.copy_(w)
        lin.bias.data.copy_(b)
        layers.append(lin)
        if i < len(idx) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def load_actors(checkpoint: str, device: str = "cpu", clip_actions: float = 10.0) -> dict[str, Actor]:
    """{"legs": Actor, "arms": Actor} from an skrl MAPPO checkpoint (agent_<N>.pt)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actors = {}
    for agent in ("legs", "arms"):
        pre = ckpt[agent]["observation_preprocessor"]
        actors[agent] = Actor(pre["running_mean"], pre["running_variance"], _mlp(ckpt[agent]["policy"], "net_container"),
                              clip_actions=clip_actions).to(device).eval()
    return actors


def describe(actors: dict[str, Actor]) -> dict:
    return {a: {"obs": int(m.mean.shape[0]), "actions": int(m.net[-1].out_features),
                "params": int(sum(p.numel() for p in m.net.parameters()))} for a, m in actors.items()}
