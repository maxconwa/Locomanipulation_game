"""The differentiable world model (paper Sec. 3.2): deterministic dynamics conditioned on the episode's sim context xi,
an observation head and a reward head per agent, and one termination head, all fit to the replay buffer by Eq. 1 on
k-step autoregressive rollouts seeded at a logged state and driven by the logged actions.

    s_{t+1} = f(s_t, a_t, xi)            a = every agent's action, concatenated in agent order
    o_i     = g_i(s, xi, phi_i(goal_i))   the noise-free observation of agent i
    r_i     = r_i(s_{t+1}, a_t, phi_i(goal_i,t+1))     without the trip cost when there are trip heads
    p_i     = p_i(s_{t+1}, a_t, phi_i(goal_i,t+1))     the probability that a GOLEM e-stop trip of agent i starts
    d       = d(s_{t+1}, [phi_1(goal_1,t+1), ..., phi_n(goal_n,t+1)])

phi_i = goal_fns[i].features, the caller's map from agent i's raw goal state to its goal features. Goals are exogenous
(commands, clocks, targets): the model never predicts them; the caller's goal_fns[i].advance steps them.

Two deliberate departures from the paper's r(s, a, g) and o = g(s, xi):
- The reward of the transition (s, a) -> s' reads the successor s' and the successor's goal, because the env computes
  its reward from the post-step state (tracking errors, contact forces, the applied joint targets); with the previous
  targets in the state, action-rate terms are visible too. A reward of (s, a) would have to learn the dynamics again.
- The observation head also reads the agent's goal features, because the actors' observations carry goal terms (the
  velocity command, the gait clock, wrist-target errors) that no function of (s, xi) can produce. Observation columns
  that are plain copies of goal features (AgentDims.goal_obs_columns <- goal_feature_columns) are copied exactly, not
  learned, and carry no loss.

Three more of ours, each from the 2026-11-20 diagnosis on LocoManip-LegsOnly-v0 (256 envs x 1000 steps of an untrained
policy, fit offline; per-term errors in normalised units):
- f predicts the change of the state in units of the change's own running std (cfg.normalize_delta), so that slow
  dimensions are not drowned by the network's output noise (with the state's std the model lost to "no change" on
  pelvis height 0.12 vs 0.04, gravity 0.31 vs 0.04, the episode clock 0.27 vs 0.01);
- the heads read the rolled-out state detached (cfg.detach_head_inputs): Eq. 1's reward error, 2-3x the reward's std
  from the sparse trip cost, otherwise dominated the global-norm clip of the shared step and the dynamics stalled;
- the -1 GOLEM e-stop trip cost (1.2% of steps, 90% of the reward's variance) is modelled apart, as a Bernoulli head
  with a fixed cost (cfg.trip_penalty), and imagined states and rewards are clamped to the range of real data
  (cfg.range_margin), so a policy cannot climb the model's extrapolation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SuperMimicCfg

__all__ = ["AgentDims", "GoalFns", "RunningNorm", "RunningRange", "WorldModel", "mlp", "train_world_model"]

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU, "tanh": nn.Tanh}


@dataclass(frozen=True)
class AgentDims:
    obs_dim: int      # O_i, the actor's observation width
    goal_dim: int     # G_i, the goal-feature width (features(goal_state)), not the raw goal state's
    action_dim: int
    goal_obs_columns: tuple[int, ...] = ()      # observation columns that are copies of goal features ...
    goal_feature_columns: tuple[int, ...] = ()  # ... of these feature columns (same length, pairwise)


@dataclass
class GoalFns:
    """Pure-torch maps of one agent's raw goal state (B, raw): its features (B, G_i) and its state one step later."""
    features: Callable[[torch.Tensor], torch.Tensor]
    advance: Callable[[torch.Tensor], torch.Tensor]


def mlp(inputs: int, hidden, outputs: int, activation: str = "elu") -> nn.Sequential:
    layers, width = [], inputs
    for size in hidden:
        layers += [nn.Linear(width, size), ACTIVATIONS[activation]()]
        width = size
    return nn.Sequential(*layers, nn.Linear(width, outputs))


class RunningNorm(nn.Module):
    """Running mean and (population) variance of a (..., dim) quantity, merged batch by batch (Chan et al.), updated
    only from real data: imagined data would let the model's own drift move its input scaling. normalize divides by
    max(std, std_floor) so that near-constant inputs stay finite."""

    def __init__(self, dim: int, std_floor: float = 1e-2):
        super().__init__()
        self.dim, self.std_floor = dim, std_floor
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, self.dim).to(self.mean.dtype)
        n = x.shape[0]
        if n == 0:
            return
        mean_b, var_b = x.mean(0), x.var(0, unbiased=False)
        total = self.count + n
        delta = mean_b - self.mean
        w_old, w_new = (self.count / total).to(self.mean.dtype), (n / total).to(self.mean.dtype)
        self.mean += delta * w_new
        self.var.copy_(self.var * w_old + var_b * w_new + delta.square() * w_old * w_new)
        self.count.copy_(total)

    @property
    def std(self) -> torch.Tensor:
        return self.var.sqrt().clamp_min(self.std_floor)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def skrl_state(self) -> dict[str, torch.Tensor]:
        """As skrl's RunningStandardScaler state_dict (the checkpoint's observation_preprocessor, export_marl.py)."""
        return {"running_mean": self.mean.clone(), "running_variance": self.var.clone(),
                "current_count": self.count.to(self.mean.dtype).clone().reshape(())}


class RunningRange(nn.Module):
    """Per-dimension min and max of a (..., dim) quantity over the real data seen (dim 0: a scalar per sample).
    clamp(x, margin) clamps x to [low - margin * width, high + margin * width] (width = high - low); before any data,
    or with a negative margin, x is returned unchanged."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        shape = (dim,) if dim else ()
        self.register_buffer("low", torch.full(shape, math.inf))
        self.register_buffer("high", torch.full(shape, -math.inf))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, self.dim) if self.dim else x.detach().reshape(-1)
        if x.shape[0] == 0:
            return
        self.low.copy_(torch.minimum(self.low, x.min(0).values.to(self.low)))
        self.high.copy_(torch.maximum(self.high, x.max(0).values.to(self.high)))

    def clamp(self, x: torch.Tensor, margin: float) -> torch.Tensor:
        if margin < 0 or not bool(torch.isfinite(self.low).all()):
            return x
        width = (self.high - self.low) * margin
        return torch.maximum(torch.minimum(x, self.high + width), self.low - width)


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask > 0, x, torch.zeros_like(x)).sum() / mask.sum().clamp_min(1.0)


def _base_bce(labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = _masked_mean(labels, mask).clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def _flat_apply(fn, x: torch.Tensor) -> torch.Tensor:
    """fn on (N, raw) applied to (..., raw)."""
    y = fn(x.reshape(-1, x.shape[-1]))
    return y.reshape(*x.shape[:-1], y.shape[-1])


class WorldModel(nn.Module):
    """Dynamics, observation, reward and termination networks with their input normalisers. All public methods take
    and return raw (unnormalised) quantities and are differentiable in their inputs."""

    def __init__(self, state_dim: int, ctx_dim: int, agents: dict[str, AgentDims], cfg: SuperMimicCfg):
        super().__init__()
        self.cfg, self.state_dim, self.ctx_dim = cfg, state_dim, ctx_dim
        self.agents = dict(agents)
        self.agent_names = list(agents)
        self.action_slices, start = {}, 0
        for name, dims in agents.items():
            self.action_slices[name] = slice(start, start + dims.action_dim)
            start += dims.action_dim
            if len(dims.goal_obs_columns) != len(dims.goal_feature_columns):
                raise ValueError(f"{name}: goal_obs_columns and goal_feature_columns differ in length")
        self.action_dim = start
        self.goal_dim_total = sum(d.goal_dim for d in agents.values())
        floor, act = cfg.norm_std_floor, cfg.activation
        S, C, A = state_dim, ctx_dim, self.action_dim

        self.state_norm = RunningNorm(S, floor)
        self.delta_norm = RunningNorm(S, cfg.delta_std_floor)
        self.state_range = RunningRange(S)
        self.reward_ranges = nn.ModuleDict({n: RunningRange(0) for n in agents})
        self.ctx_norm = RunningNorm(C, floor)
        self.action_norm = RunningNorm(A, floor)
        self.obs_norms = nn.ModuleDict({n: RunningNorm(d.obs_dim, floor) for n, d in agents.items()})
        self.goal_norms = nn.ModuleDict({n: RunningNorm(d.goal_dim, floor) for n, d in agents.items()})

        self.dynamics = mlp(S + A + C, cfg.dynamics_hidden, S, act)
        nn.init.zeros_(self.dynamics[-1].weight)  # start as the identity map s' = s
        nn.init.zeros_(self.dynamics[-1].bias)
        self.obs_heads = nn.ModuleDict(
            {n: mlp(S + C + d.goal_dim, cfg.obs_hidden, d.obs_dim, act) for n, d in agents.items()})
        self.reward_heads = nn.ModuleDict(
            {n: mlp(S + A + d.goal_dim, cfg.reward_hidden, 1, act) for n, d in agents.items()})
        self.trip_heads = nn.ModuleDict(
            {n: mlp(S + A + d.goal_dim, cfg.termination_hidden, 1, act) for n, d in agents.items()} if cfg.trip_penalty else {})
        self.termination_head = mlp(S + self.goal_dim_total, cfg.termination_hidden, 1, act)
        for n, d in agents.items():
            learned = torch.ones(d.obs_dim)
            learned[list(d.goal_obs_columns)] = 0.0
            self.register_buffer(f"_learned_obs_{n}", learned, persistent=False)

    # ---- the model -----------------------------------------------------------------------------------------------
    def step(self, s: torch.Tensor, a: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """s_{t+1} (B, S) from s_t (B, S), every agent's action a_t (B, A) and the sim context xi (B, C); clamped to
        the real data's range (cfg.range_margin)."""
        sn = self.state_norm.normalize(s)
        x = torch.cat([sn, self.action_norm.normalize(a), self.ctx_norm.normalize(xi)], dim=-1)
        if self.cfg.normalize_delta:
            s_next = s + self.delta_norm.denormalize(self.dynamics(x))
        else:
            s_next = self.state_norm.denormalize(sn + self.dynamics(x))
        return self.state_range.clamp(s_next, self.cfg.range_margin)

    def _observe_n(self, agent: str, s, xi, g_feat) -> torch.Tensor:
        x = torch.cat([self.state_norm.normalize(s), self.ctx_norm.normalize(xi),
                       self.goal_norms[agent].normalize(g_feat)], dim=-1)
        return self.obs_heads[agent](x)

    def observe(self, agent: str, s: torch.Tensor, xi: torch.Tensor, g_feat: torch.Tensor) -> torch.Tensor:
        """Agent's noise-free observation (B, O_i) in state s with goal features g_feat (B, G_i)."""
        o = self.obs_norms[agent].denormalize(self._observe_n(agent, s, xi, g_feat))
        dims = self.agents[agent]
        if dims.goal_obs_columns:
            o = o.clone()
            o[..., list(dims.goal_obs_columns)] = g_feat[..., list(dims.goal_feature_columns)]
        return o

    def _reward_input(self, agent: str, s_next, a, g_next_feat) -> torch.Tensor:
        return torch.cat([self.state_norm.normalize(s_next), self.action_norm.normalize(a),
                          self.goal_norms[agent].normalize(g_next_feat)], dim=-1)

    def reward(self, agent: str, s_next: torch.Tensor, a: torch.Tensor, g_next_feat: torch.Tensor) -> torch.Tensor:
        """Agent's reward (B,) of the transition into s_next under action a (all agents), without the fall penalty
        (and without the trip cost when there are trip heads), clamped to the real rewards' range."""
        r = self.reward_heads[agent](self._reward_input(agent, s_next, a, g_next_feat)).squeeze(-1)
        return self.reward_ranges[agent].clamp(r * self.cfg.reward_scale, self.cfg.range_margin)

    def trip_logit(self, agent: str, s_next: torch.Tensor, a: torch.Tensor, g_next_feat: torch.Tensor) -> torch.Tensor:
        """Logit (B,) that a GOLEM e-stop trip of agent's starts on the transition into s_next (needs trip heads)."""
        return self.trip_heads[agent](self._reward_input(agent, s_next, a, g_next_feat)).squeeze(-1)

    def expected_reward(self, agent: str, s_next: torch.Tensor, a: torch.Tensor,
                        g_next_feat: torch.Tensor) -> torch.Tensor:
        """The reward plus the trip cost times the trip probability (just the reward without trip heads): the
        logged task reward's expectation, still without the fall penalty."""
        r = self.reward(agent, s_next, a, g_next_feat)
        if self.trip_heads:
            r = r + self.cfg.trip_penalty * torch.sigmoid(self.trip_logit(agent, s_next, a, g_next_feat))
        return r

    def _goal_all_n(self, g_feat_all: torch.Tensor) -> torch.Tensor:
        parts, start = [], 0
        for name in self.agent_names:
            width = self.agents[name].goal_dim
            parts.append(self.goal_norms[name].normalize(g_feat_all[..., start:start + width]))
            start += width
        return torch.cat(parts, dim=-1)

    def terminal_logit(self, s_next: torch.Tensor, g_feat_all: torch.Tensor) -> torch.Tensor:
        """Logit (B,) that s_next ends the episode; g_feat_all = every agent's goal features, in agent order."""
        x = torch.cat([self.state_norm.normalize(s_next), self._goal_all_n(g_feat_all)], dim=-1)
        return self.termination_head(x).squeeze(-1)

    def terminal_prob(self, s_next: torch.Tensor, g_feat_all: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.terminal_logit(s_next, g_feat_all))

    # ---- fitting -------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def update_norms(self, batch: dict[str, torch.Tensor], goal_fns: dict[str, GoalFns]) -> None:
        """Update the normalisers and ranges from real data: batch holds any of state, ctx, action, obs/<agent>,
        goal/<agent> (raw goal states; their features are taken here), each (..., dim), and reward/<agent> (with
        trip/<agent> when there are trip heads: the range is of the reward without the trip cost), each (...,)."""
        for key, norm in (("state", self.state_norm), ("ctx", self.ctx_norm), ("action", self.action_norm)):
            if key in batch:
                norm.update(batch[key])
        if "state" in batch:
            self.state_range.update(batch["state"])
        for name in self.agent_names:
            if f"obs/{name}" in batch:
                self.obs_norms[name].update(batch[f"obs/{name}"])
            if f"goal/{name}" in batch:
                self.goal_norms[name].update(_flat_apply(goal_fns[name].features, batch[f"goal/{name}"]))
            if f"reward/{name}" in batch:
                self.reward_ranges[name].update(self.trip_free(name, batch[f"reward/{name}"], batch))

    def trip_free(self, agent: str, reward: torch.Tensor, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """The logged reward less the trip cost where a trip started (unchanged without trip heads)."""
        if not self.trip_heads:
            return reward
        return reward - self.cfg.trip_penalty * data[f"trip/{agent}"].to(reward)

    @torch.no_grad()
    def update_delta_norm(self, window: dict[str, torch.Tensor]) -> None:
        """The state-change normaliser from a batch of real windows: s_{j+1} - s_j where entry j + 1 is the
        successor of entry j (next_valid)."""
        s, nv = window["state"], window["next_valid"] > 0
        self.delta_norm.update((s[:, 1:] - s[:, :-1])[nv])

    def loss(self, window: dict[str, torch.Tensor], goal_fns: dict[str, GoalFns]):
        """Eq. 1 on a batch of buffer windows (ReplayBuffer.sample_windows): roll k steps from the logged s_0 on the
        logged actions, s^_{j+1} = f(s^_j, a_j, xi_j), and score every step against the log:
          state        ||norm(s^_{j+1}) - norm(s_{j+1})||^2 / S                where next_valid_j
          observation  ||norm(o^_i(s^_{j+1})) - norm(o_i,j+1)||^2 / O_i      where next_valid_j, plus o^_i(s_0) vs o_0
          reward       ((r^_i(s^_{j+1}, a_j, phi_i(goal_{j+1})) - r_i,j) / reward_scale)^2       where alive_j
                       (r_i,j less trip_penalty * trip_i,j with trip heads; the prediction unclamped)
          trip         BCE(p^_i(s^_{j+1}, a_j, phi_i(goal_{j+1})), trip_i,j)    where alive_j (with trip heads)
          termination  BCE(d^(s^_{j+1}, phi(goal_{j+1})), terminated_j)  where alive_j and not a pure time-out
        The heads read s^_{j+1} detached when cfg.detach_head_inputs (the state term alone trains the dynamics).
        On a done step (alive, terminated or truncated) entry j + 1 is the next episode's first state, so reward and
        termination read the predicted successor and the goal advanced from step j (as imagination does), which is
        how the fall transition's reward and d = 1 are learned. Logged targets are zeroed where masked, so data of
        the next episode cannot reach the loss even as inf or nan (and the rollout after a done runs on zeros).
        Each term is averaged over its mask.
        Returns (total loss, metrics as detached 0-d tensors)."""
        cfg, w = self.cfg, self.cfg.loss_weights
        alive, nv = window["alive"], window["next_valid"]
        B, k = alive.shape
        # entry j belongs to the window's first episode iff j = 0 or next_valid_{j-1} (= alive_j); nothing else of
        # the log is read: zeroed here, so the next episode's data cannot reach the loss even as inf or nan
        entry = torch.cat([torch.ones_like(nv[:, :1]), nv], dim=1)

        def keep(x, mask):
            return torch.where(mask.reshape(*mask.shape, *([1] * (x.dim() - 2))) > 0, x, torch.zeros_like(x))

        s, ctx, a = keep(window["state"], entry), keep(window["ctx"], entry), keep(window["action"], alive)
        terminated, truncated = window["terminated"].bool(), window["truncated"].bool()
        done = (terminated | truncated).unsqueeze(-1)
        term_mask = alive * (terminated | ~truncated).float()   # a step both terminated and truncated is a fall

        feats, succ_feats = {}, {}
        for name in self.agent_names:
            goal = keep(window[f"goal/{name}"], entry)
            feats[name] = _flat_apply(goal_fns[name].features, goal)                            # (B, k + 1, G)
            advanced = _flat_apply(goal_fns[name].features, _flat_apply(goal_fns[name].advance, goal[:, :k]))
            succ_feats[name] = torch.where(done, advanced, feats[name][:, 1:])                 # (B, k, G)
        succ_all = torch.cat([succ_feats[n] for n in self.agent_names], dim=-1)

        s_target = self.state_norm.normalize(s[:, 1:])
        obs_target, obs_cols = {}, {}
        for name in self.agent_names:
            obs_target[name] = self.obs_norms[name].normalize(keep(window[f"obs/{name}"], entry))   # (B, k + 1, O)
            learned = getattr(self, f"_learned_obs_{name}")
            obs_cols[name] = learned / learned.sum().clamp_min(1.0)
        r_target = {n: keep(self.trip_free(n, window[f"reward/{n}"], window), alive) for n in self.agent_names}
        trip_target = {n: keep(window[f"trip/{n}"].float(), alive) for n in self.trip_heads}

        state_err, obs_err, r_hat = [], {n: [] for n in self.agent_names}, {n: [] for n in self.agent_names}
        trip_logits = {n: [] for n in self.trip_heads}
        logits = []
        for name in self.agent_names:
            o0 = self._observe_n(name, s[:, 0], ctx[:, 0], feats[name][:, 0])
            obs_err[name].append(((o0 - obs_target[name][:, 0]).square() * obs_cols[name]).sum(-1))
        s_hat = s[:, 0]
        for j in range(k):
            xi = ctx[:, j]
            s_hat = self.step(s_hat, a[:, j], xi)
            state_err.append((self.state_norm.normalize(s_hat) - s_target[:, j]).square().mean(-1))
            s_in = s_hat.detach() if cfg.detach_head_inputs else s_hat
            for name in self.agent_names:
                o_hat = self._observe_n(name, s_in, xi, feats[name][:, j + 1])
                obs_err[name].append(((o_hat - obs_target[name][:, j + 1]).square() * obs_cols[name]).sum(-1))
                # unclamped: the fit sees the head's own error
                x_r = self._reward_input(name, s_in, a[:, j], succ_feats[name][:, j])
                r_hat[name].append(self.reward_heads[name](x_r).squeeze(-1) * cfg.reward_scale)
                if name in self.trip_heads:
                    trip_logits[name].append(self.trip_heads[name](x_r).squeeze(-1))
            logits.append(self.terminal_logit(s_in, succ_all[:, j]))

        state_err = torch.stack(state_err, 1)                                                     # (B, k)
        state_loss = _masked_mean(state_err, nv)
        obs_mask = torch.cat([torch.ones_like(nv[:, :1]), nv], dim=1)
        obs_loss = sum(_masked_mean(torch.stack(obs_err[n], 1), obs_mask) for n in self.agent_names)
        reward_loss, reward_sq = 0.0, 0.0
        for name in self.agent_names:
            err = torch.stack(r_hat[name], 1) - r_target[name]
            reward_loss = reward_loss + _masked_mean((err / cfg.reward_scale).square(), alive)
            reward_sq = reward_sq + _masked_mean(err.square(), alive).detach()
        logits = torch.stack(logits, 1)
        labels = terminated.float()
        term_loss = _masked_mean(F.binary_cross_entropy_with_logits(logits, labels, reduction="none"), term_mask)
        trip_loss = torch.zeros((), device=s.device, dtype=s.dtype)
        trip_stats = []
        for name in self.trip_heads:
            tl = torch.stack(trip_logits[name], 1)
            trip_loss = trip_loss + _masked_mean(
                F.binary_cross_entropy_with_logits(tl, trip_target[name], reduction="none"), alive)
            trip_stats.append((tl.detach(), trip_target[name]))
        total = (w["state"] * state_loss + w["obs"] * obs_loss + w["reward"] * reward_loss
                 + w["termination"] * term_loss + w.get("trip", 1.0) * trip_loss)

        with torch.no_grad():
            pred = (logits > 0).float()
            metrics = {
                "wm/loss": total.detach(),
                "wm/state_loss": state_loss.detach(),
                "wm/obs_loss": obs_loss.detach(),
                "wm/reward_loss": reward_loss.detach(),
                "wm/termination_loss": term_loss.detach(),
                "wm/state_rmse_1step": _masked_mean(state_err[:, 0], nv[:, 0]).sqrt(),
                f"wm/state_rmse_{k}step": _masked_mean(state_err[:, -1], nv[:, -1]).sqrt(),
                "wm/reward_rmse": (reward_sq / len(self.agent_names)).sqrt(),
                "wm/termination_accuracy": _masked_mean((pred == labels).float(), term_mask),
                "wm/termination_positive_rate": _masked_mean(labels, term_mask),
                "wm/termination_predicted_rate": _masked_mean(pred, term_mask),
                # the BCE of always predicting the batch's positive rate: the head has learned something below it
                "wm/termination_base_bce": _base_bce(labels, term_mask),
            }
            if self.trip_heads:
                metrics["wm/trip_loss"] = trip_loss.detach()
                metrics["wm/trip_base_bce"] = sum(_base_bce(t, alive) for _, t in trip_stats)
                metrics["wm/trip_positive_rate"] = sum(_masked_mean(t, alive) for _, t in trip_stats) / len(trip_stats)
                metrics["wm/trip_predicted_prob"] = sum(_masked_mean(torch.sigmoid(lg), alive)
                                                        for lg, _ in trip_stats) / len(trip_stats)
        return total, metrics


def train_world_model(wm: WorldModel, optimizer: torch.optim.Optimizer, buffer, goal_fns: dict[str, GoalFns],
                      cfg: SuperMimicCfg | None = None) -> dict[str, float]:
    """One world-model update: cfg.wm_batches gradient steps on cfg.wm_batch_size windows of cfg.window steps each,
    global-norm clipped. Returns the metrics averaged over the steps (plus the last pre-clip grad norm)."""
    cfg = cfg or wm.cfg
    wm.train()
    sums: dict[str, torch.Tensor] = {}
    grad_norm = torch.zeros(())
    for _ in range(cfg.wm_batches):
        window = buffer.sample_windows(cfg.wm_batch_size, cfg.window)
        wm.update_delta_norm(window)
        loss, metrics = wm.loss(window, goal_fns)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(wm.parameters(), cfg.grad_clip)
        optimizer.step()
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
    out = {key: float(value) / cfg.wm_batches for key, value in sums.items()}
    out["wm/grad_norm"] = float(grad_norm)
    return out
