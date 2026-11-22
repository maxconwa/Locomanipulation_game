"""A synthetic two-agent system with known answers, for the world-model test: linear dynamics with a per-episode
context, linear observations with a context bias and uniform sensor noise, a smooth reward of (s', a, goal), and a
termination when a state coordinate crosses a threshold; episodes also end at random (truncations)."""

import math

import torch

from supermimic.buffer import ReplayBuffer, standard_spec
from supermimic.world_model import AgentDims, GoalFns

S, C = 4, 2
AGENTS = {"p": AgentDims(obs_dim=4, goal_dim=2, action_dim=1, goal_obs_columns=(3,), goal_feature_columns=(0,)),
          "q": AgentDims(obs_dim=2, goal_dim=2, action_dim=1)}
OBS_NOISE = 0.3      # uniform half-width
THRESHOLD = 2.0      # terminate when s'[1] > THRESHOLD
P_TRUNC = 0.01
CLOCK = 0.3


def goal_features(g):
    """raw goal (target, phase) -> (target, sin phase)"""
    return torch.stack([g[:, 0], torch.sin(g[:, 1])], dim=-1)


def goal_advance(g):
    return torch.stack([g[:, 0], g[:, 1] + CLOCK], dim=-1)


GOAL_FNS = {n: GoalFns(goal_features, goal_advance) for n in AGENTS}


class Synthetic:
    def __init__(self, num_envs: int, seed: int = 0):
        gen = torch.Generator().manual_seed(0)  # the system is the same for every seed; the data is not
        self.n = num_envs
        th = 0.5
        rot = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        self.A = torch.block_diag(0.9 * rot, 0.8 * rot)
        self.B = torch.randn(S, 2, generator=gen) * 0.4
        self.Cm = torch.randn(S, C, generator=gen) * 0.2
        self.D = {"p": torch.randn(3, S, generator=gen), "q": torch.randn(2, S, generator=gen)}
        self.E = {"p": torch.randn(3, C, generator=gen), "q": torch.randn(2, C, generator=gen)}
        self.gen = torch.Generator().manual_seed(1000 + seed)
        self.s = torch.zeros(num_envs, S)
        self.xi = torch.zeros(num_envs, C)
        self.goal = {n: torch.zeros(num_envs, 2) for n in AGENTS}
        self.reset(torch.ones(num_envs, dtype=torch.bool))

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen)

    def reset(self, mask):
        n = int(mask.sum())
        self.s[mask] = torch.randn(n, S, generator=self.gen) * 0.5
        self.xi[mask] = self._rand(n, C) * 2 - 1
        for name in AGENTS:
            self.goal[name][mask] = torch.stack([self._rand(n) * 2 - 1, self._rand(n) * 6.28], -1)

    def clean_obs(self, name, s, xi, goal):
        o = s @ self.D[name].T + xi @ self.E[name].T
        if name == "p":
            o = torch.cat([o, goal_features(goal)[:, :1]], -1)
        return o

    def obs(self, name):
        o = self.clean_obs(name, self.s, self.xi, self.goal[name])
        noise = (self._rand(*o.shape) * 2 - 1) * OBS_NOISE
        if name == "p":
            noise[:, 3] = 0.0
        return o + noise

    def dynamics(self, s, a, xi):
        return s @ self.A.T + a @ self.B.T + xi @ self.Cm.T

    @staticmethod
    def reward(name, s_next, a, g_feat):
        i = 0 if name == "p" else 2
        return -(s_next[:, i] - g_feat[:, 0]).square() - 0.1 * a.square().sum(-1) + 0.3 * g_feat[:, 1] * s_next[:, 3]

    def step(self, a):
        """One step; returns (rewards, terminated, truncated) and auto-resets done envs."""
        s_next = self.dynamics(self.s, a, self.xi)
        goal_next = {n: goal_advance(g) for n, g in self.goal.items()}
        rewards = {n: self.reward(n, s_next, a, goal_features(goal_next[n])) for n in AGENTS}
        terminated = s_next[:, 1] > THRESHOLD
        truncated = self._rand(self.n) < P_TRUNC
        self.s, self.goal = s_next, goal_next
        self.reset(terminated | truncated)
        return rewards, terminated, truncated


def collect(env: Synthetic, steps: int, capacity: int | None = None, action_std: float = 1.0) -> ReplayBuffer:
    spec = standard_spec(S, C, 2, {n: d.obs_dim for n, d in AGENTS.items()}, {n: 2 for n in AGENTS}, {})
    buf = ReplayBuffer(spec, env.n, capacity or steps)
    for _ in range(steps):
        step = {"state": env.s.clone(), "ctx": env.xi.clone()}
        for n in AGENTS:
            step[f"obs/{n}"] = env.obs(n)
            step[f"goal/{n}"] = env.goal[n].clone()
        a = torch.randn(env.n, 2, generator=env.gen) * action_std
        step["action"] = a
        rewards, term, trunc = env.step(a)
        step.update({f"reward/{n}": r for n, r in rewards.items()}, terminated=term, truncated=trunc)
        buf.add(step)
    return buf
