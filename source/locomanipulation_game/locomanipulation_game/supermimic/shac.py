"""Policy improvement through the frozen world model (paper Sec. 3.3): SHAC-style backpropagation through time over
H imagined steps with a bootstrapping critic per trained agent, and the collection-time acting that goes with it.

Imagination starts from replay entries (state, sim context, goal states, the actors' stored LSTM states). Each step,
every trained agent observes the model's noise-free observation plus re-drawn sensor noise, its LSTM actor acts with
its mean action (deterministic, clamped at +-action_clip as the env clamps it), untrained agents act with zeros, the
model steps, the goals advance, and the model's reward and termination heads score the step. Each agent maximises
    J_i = E[ sum_{t<H} gamma^t c_t (r_i,t+1 + fall_penalty d_t+1) + gamma^H c_H V_i(s_H, g_H, xi) ],
c_0 = 1, c_t+1 = c_t (1 - d_t+1): the paper's Eq. 2 with the episode's fall cost made explicit (the logged reward
excludes it; the termination head supplies its probability). r is the model's expected reward (WorldModel.
expected_reward: the trip cost times the trip head's probability included). Each actor's gradient is that of its own
J_i only.

The action clamp is a plain clamp, not a straight-through: beyond the clip the env ignores the action, so the true
gradient there is zero, and a straight-through would reward pushing the mean further out where nothing changes. The
env's clip (10) sits far outside the actions the policies use, so saturation is not expected to stall learning.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .buffer import unflatten_lstm
from .config import SuperMimicCfg, set_lr
from .world_model import GoalFns, RunningNorm, mlp

__all__ = ["Critic", "Imagination", "ShacTrainer", "actor_step", "OBS_EPSILON", "OBS_CLIP"]

# skrl's RunningStandardScaler, as export_marl.py bakes it into the deployed actor
OBS_EPSILON, OBS_CLIP = 1e-8, 5.0


def actor_step(policy: nn.Module, obs_mean: torch.Tensor, obs_var: torch.Tensor, obs: torch.Tensor,
               h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One differentiable step of an LSTM actor (recurrent.LSTMGaussianPolicy's acting branch, behind the export's
    observation scaling): z = clip((obs - mean) / (sqrt(var) + 1e-8), -5, 5), LSTM step from (h, c) (num_layers, B,
    hidden), the cell state capped (c <- cap tanh(c / cap)) when cell_cap > 0, then the MLP on the LSTM output (and z
    with skip_observations). Returns (mean action (B, A), h, c)."""
    z = torch.clamp((obs - obs_mean) / (torch.sqrt(obs_var) + OBS_EPSILON), -OBS_CLIP, OBS_CLIP)
    out, (h, c) = policy.lstm(z.unsqueeze(1), (h, c))
    if policy.cell_cap > 0.0:
        c = policy.cell_cap * torch.tanh(c / policy.cell_cap)
    features = out.squeeze(1)
    if policy.skip_observations:
        features = torch.cat([features, z], dim=-1)
    return policy.net(features), h, c


class Critic(nn.Module):
    """V_i(s, phi_i(g_i), xi) on the world model's normalisers (shared, read-only here)."""

    def __init__(self, state_norm: RunningNorm, goal_norm: RunningNorm, ctx_norm: RunningNorm, hidden=(256, 128),
                 activation: str = "elu"):
        super().__init__()
        # held outside the module tree: the critic's state_dict and optimizer must not own the world model's stats
        self._norms = (state_norm, goal_norm, ctx_norm)
        self.net = mlp(state_norm.dim + goal_norm.dim + ctx_norm.dim, hidden, 1, activation)

    def forward(self, s: torch.Tensor, g_feat: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        ns, ng, nc = self._norms
        return self.net(torch.cat([ns.normalize(s), ng.normalize(g_feat), nc.normalize(xi)], dim=-1)).squeeze(-1)


@dataclass
class Imagination:
    """One imagined batch. J: per trained agent the objective (0-d, differentiable in that agent's actor).
    Detached, for the critic: states (H + 1, B, S), feats[agent] (H + 1, B, G_i), ctx (B, C), rewards[agent] (H, B)
    effective (fall penalty included), done (H, B) = d_t+1, survival (H + 1, B) = c_t."""
    J: dict[str, torch.Tensor]
    states: torch.Tensor
    feats: dict[str, torch.Tensor]
    ctx: torch.Tensor
    rewards: dict[str, torch.Tensor]
    done: torch.Tensor
    survival: torch.Tensor
    action_abs: torch.Tensor


@contextlib.contextmanager
def _frozen(*modules: nn.Module):
    """requires_grad off for every parameter of modules (restored after): the actors' backward pass then neither
    builds nor fills gradients of the world model or the critics."""
    params = [p for m in modules for p in m.parameters()]
    flags = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


class ShacTrainer:
    """Actors (one per trained agent, the keys of `actors`) and their critics, improved through world_model.

    world_model: a WorldModel (or anything with its agents / agent_names / action_slices / step / observe / reward /
    terminal_logit / state_norm / ctx_norm / goal_norms; expected_reward is used instead of reward where it has one). goal_fns[agent]: GoalFns for every agent. obs_norms[agent]:
    the actor's observation normaliser (mean / var; the caller updates it from real observations only, and saves it
    as the checkpoint's observation_preprocessor). obs_noise[agent]: (O_i,) half-widths of the uniform sensor noise
    re-added to imagined observations. action_clip: the env's clamp of raw actions (None = no clamp)."""

    def __init__(self, world_model, actors: dict[str, nn.Module], goal_fns: dict[str, GoalFns],
                 obs_norms: dict[str, RunningNorm], obs_noise: dict[str, torch.Tensor], cfg: SuperMimicCfg,
                 action_clip: float | None = None):
        self.wm, self.cfg, self.goal_fns, self.action_clip = world_model, cfg, goal_fns, action_clip
        self._reward = getattr(world_model, "expected_reward", world_model.reward)
        self.agent_names = list(world_model.agent_names)
        unknown = set(actors) - set(self.agent_names)
        if unknown:
            raise ValueError(f"actors for agents the world model does not have: {sorted(unknown)}")
        self.trained = [n for n in self.agent_names if n in actors]
        self.actors, self.obs_norms = dict(actors), dict(obs_norms)
        self.obs_noise = {n: torch.as_tensor(obs_noise[n]) for n in self.trained}
        ref = next(iter(world_model.parameters()), None)
        self.device = ref.device if ref is not None else torch.device("cpu")
        self.dtype = ref.dtype if ref is not None else torch.float32
        self.critics = {n: Critic(world_model.state_norm, world_model.goal_norms[n], world_model.ctx_norm,
                                  cfg.critic_hidden, cfg.activation).to(self.device, self.dtype)
                        for n in self.trained}
        self.actor_params = {n: [p for name, p in actors[n].named_parameters() if name != "log_std_parameter"]
                             for n in self.trained}
        self.actor_opts = {n: torch.optim.Adam(self.actor_params[n], lr=cfg.lr, betas=tuple(cfg.actor_betas))
                           for n in self.trained}
        self.critic_opts = {n: torch.optim.Adam(self.critics[n].parameters(), lr=cfg.lr) for n in self.trained}

    def set_lr(self, factor: float) -> None:
        """Actors' and critics' learning rate = cfg.lr * factor (config.lr_factor)."""
        for opt in list(self.actor_opts.values()) + list(self.critic_opts.values()):
            set_lr(opt, self.cfg.lr * factor)

    # ---- acting in the real env ----------------------------------------------------------------------------------
    def initial_lstm(self, num_envs: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        out = {}
        for n in self.trained:
            lstm = self.actors[n].lstm
            zeros = torch.zeros(lstm.num_layers, num_envs, lstm.hidden_size, device=self.device, dtype=self.dtype)
            out[n] = (zeros, zeros.clone())
        return out

    @staticmethod
    def reset_lstm(lstm_state: dict, done: torch.Tensor) -> dict:
        """The LSTM states with the envs where done (N,) zeroed (for the step after an episode ended)."""
        keep = (~done.view(1, -1, 1).bool()).float()
        return {n: (h * keep, c * keep) for n, (h, c) in lstm_state.items()}

    def collection_noise(self, num_envs: int) -> torch.Tensor:
        """Each env's collection noise std (num_envs,): cfg.action_noise_std, or with cfg.action_noise_std_max above
        it, log-uniform in [action_noise_std, action_noise_std_max]."""
        low, high = self.cfg.action_noise_std, self.cfg.action_noise_std_max
        if high <= low or low <= 0.0:
            return torch.full((num_envs,), low, device=self.device, dtype=self.dtype)
        u = torch.rand(num_envs, device=self.device, dtype=self.dtype)
        return torch.exp(math.log(low) + u * (math.log(high) - math.log(low)))

    @torch.no_grad()
    def act(self, observations: dict[str, torch.Tensor], lstm_state: dict, explore: bool = True,
            noise_std: torch.Tensor | None = None):
        """Collection-time actions: each trained agent's mean action plus N(0, std^2) when explore (std = noise_std
        (N,) per env, else cfg.action_noise_std), clamped at +-action_clip; zeros for the untrained agents. Returns
        (the concatenated executed action (N, A), per-agent actions, the next LSTM state). Store lstm_state (the
        input) with this step in the buffer."""
        num_envs = next(iter(observations.values())).shape[0]
        std = self.cfg.action_noise_std if noise_std is None else noise_std.unsqueeze(-1)
        parts, per_agent, next_state = [], {}, {}
        for n in self.agent_names:
            width = self.wm.agents[n].action_dim
            if n in self.trained:
                h, c = lstm_state[n]
                mean, h, c = actor_step(self.actors[n], self.obs_norms[n].mean, self.obs_norms[n].var,
                                        observations[n], h, c)
                next_state[n] = (h, c)
                a = mean + std * torch.randn_like(mean) if explore else mean
                if self.action_clip is not None:
                    a = a.clamp(-self.action_clip, self.action_clip)
            else:
                a = torch.zeros(num_envs, width, device=self.device, dtype=self.dtype)
            per_agent[n] = a
            parts.append(a)
        return torch.cat(parts, dim=-1), per_agent, next_state

    # ---- imagination ---------------------------------------------------------------------------------------------
    def imagine(self, starts: dict[str, torch.Tensor], horizon: int | None = None, noise: bool = True) -> Imagination:
        """Roll the world model H steps from buffer entries (ReplayBuffer.sample_starts: state, ctx, goal/<agent>,
        h/<agent>, c/<agent>). The objectives J are differentiable in the actors; the world model and critics are
        frozen for the pass. noise=False drops the re-added sensor noise (for tests)."""
        cfg, wm = self.cfg, self.wm
        H = cfg.horizon if horizon is None else horizon
        s, xi = starts["state"], starts["ctx"]
        B = s.shape[0]
        goals = {n: starts[f"goal/{n}"] for n in self.agent_names}
        lstm = {}
        for n in self.trained:
            layers = self.actors[n].lstm.num_layers
            lstm[n] = (unflatten_lstm(starts[f"h/{n}"], layers), unflatten_lstm(starts[f"c/{n}"], layers))
        feats = {n: self.goal_fns[n].features(goals[n]) for n in self.agent_names}
        survival = torch.ones(B, device=s.device, dtype=s.dtype)
        discount = 1.0
        J = {n: torch.zeros((), device=s.device, dtype=s.dtype) for n in self.trained}
        states, feat_hist = [s.detach()], {n: [feats[n].detach()] for n in self.trained}
        rewards, dones, survivals, action_abs = {n: [] for n in self.trained}, [], [survival.detach()], []
        for actor in self.actors.values():
            actor.train()  # cuDNN's LSTM backward runs only in training mode (no dropout: same outputs)
        with _frozen(wm, *self.critics.values()):
            for _ in range(H):
                parts = []
                for n in self.agent_names:
                    if n in self.trained:
                        o = wm.observe(n, s, xi, feats[n])
                        if noise:
                            o = o + (2.0 * torch.rand_like(o) - 1.0) * self.obs_noise[n].to(o)
                        h, c = lstm[n]
                        a, h, c = actor_step(self.actors[n], self.obs_norms[n].mean, self.obs_norms[n].var, o, h, c)
                        lstm[n] = (h, c)
                        if self.action_clip is not None:
                            a = a.clamp(-self.action_clip, self.action_clip)
                        action_abs.append(a.detach().abs().mean())
                    else:
                        a = torch.zeros(B, wm.agents[n].action_dim, device=s.device, dtype=s.dtype)
                    parts.append(a)
                action = torch.cat(parts, dim=-1)
                s = wm.step(s, action, xi)
                goals = {n: self.goal_fns[n].advance(goals[n]) for n in self.agent_names}
                feats = {n: self.goal_fns[n].features(goals[n]) for n in self.agent_names}
                d = torch.sigmoid(wm.terminal_logit(s, torch.cat([feats[n] for n in self.agent_names], dim=-1)))
                for n in self.trained:
                    r = self._reward(n, s, action, feats[n]) + cfg.fall_penalty * d
                    J[n] = J[n] + discount * (survival * r).mean()
                    rewards[n].append(r.detach())
                    feat_hist[n].append(feats[n].detach())
                survival = survival * (1.0 - d)
                discount *= cfg.gamma
                states.append(s.detach())
                dones.append(d.detach())
                survivals.append(survival.detach())
            for n in self.trained:
                J[n] = J[n] + discount * (survival * self.value(n, s, feats[n], xi)).mean()
        return Imagination(
            J=J, states=torch.stack(states), feats={n: torch.stack(v) for n, v in feat_hist.items()}, ctx=xi.detach(),
            rewards={n: torch.stack(v) for n, v in rewards.items()}, done=torch.stack(dones),
            survival=torch.stack(survivals),
            action_abs=torch.stack(action_abs).mean() if action_abs else torch.zeros(()))

    def actor_gradients(self, imag: Imagination, weight: float = 1.0) -> dict[str, list[torch.Tensor]]:
        """d(-weight J_i) / d(actor i's parameters), for each trained agent, from its own objective only."""
        grads = {}
        for i, n in enumerate(self.trained):
            last = i == len(self.trained) - 1
            g = torch.autograd.grad(-weight * imag.J[n], self.actor_params[n], retain_graph=not last,
                                    allow_unused=True)
            grads[n] = [torch.zeros_like(p) if gi is None else gi for p, gi in zip(self.actor_params[n], g)]
        return grads

    # ---- critic --------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def critic_targets(self, imag: Imagination, agent: str) -> torch.Tensor:
        """Eq. 3's lambda-returns (H, B) with the current critic: R_H = V_H,
        R_t = r_t+1 + gamma (1 - d_t+1) ((1 - lambda) V_t+1 + lambda R_t+1)."""
        cfg = self.cfg
        H1, B = imag.states.shape[:2]
        values = self.value(agent, imag.states.reshape(H1 * B, -1), imag.feats[agent].reshape(H1 * B, -1),
                            imag.ctx.repeat(H1, 1)).reshape(H1, B)
        targets = torch.empty_like(imag.rewards[agent])
        ret = values[-1]
        for t in range(H1 - 2, -1, -1):
            ret = imag.rewards[agent][t] + cfg.gamma * (1.0 - imag.done[t]) * (
                (1.0 - cfg.gae_lambda) * values[t + 1] + cfg.gae_lambda * ret)
            targets[t] = ret
        low, high = self.value_bounds(agent)
        return targets.clamp(low, high)

    def value_bounds(self, agent: str) -> tuple[float, float]:
        """The range a return can have, from the real rewards' range (the world model's reward_ranges, widened by
        cfg.range_margin, plus the trip cost) and one fall: [min(r_low, 0) / (1 - gamma) + fall_penalty,
        max(r_high, 0) / (1 - gamma)]. (-inf, inf) without ranges or with range_margin < 0."""
        cfg = self.cfg
        ranges = getattr(self.wm, "reward_ranges", None)
        if cfg.range_margin < 0 or ranges is None or not bool(torch.isfinite(ranges[agent].low)):
            return -math.inf, math.inf
        lo, hi = float(ranges[agent].low), float(ranges[agent].high)
        width = (hi - lo) * cfg.range_margin
        lo, hi = lo - width + min(cfg.trip_penalty, 0.0), hi + width + max(cfg.trip_penalty, 0.0)
        horizon = 1.0 / (1.0 - cfg.gamma)
        return min(lo, 0.0) * horizon + min(cfg.fall_penalty, 0.0), max(hi, 0.0) * horizon

    def value(self, agent: str, s: torch.Tensor, g_feat: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """The critic's value clamped to value_bounds: where the critic extrapolates past any possible return the
        actor gets no gradient from it, and the critic's own targets cannot ratchet it further (the 2026-11-20 runs'
        critic loss of 1e6: the bootstrap fed on its own overestimates)."""
        low, high = self.value_bounds(agent)
        return self.critics[agent](s, g_feat, xi).clamp(low, high)

    def fit_critic(self, imag: Imagination, agent: str) -> float:
        """critic_epochs passes over critic_minibatches shuffled minibatches of every (t, b) pair, MSE to the fixed
        targets, global-norm clipped. Returns the mean loss of the last epoch."""
        cfg, critic, opt = self.cfg, self.critics[agent], self.critic_opts[agent]
        targets = self.critic_targets(imag, agent)
        H, B = targets.shape
        s = imag.states[:H].reshape(H * B, -1)
        g = imag.feats[agent][:H].reshape(H * B, -1)
        xi = imag.ctx.repeat(H, 1)
        y = targets.reshape(H * B)
        total = H * B
        size = (total + cfg.critic_minibatches - 1) // cfg.critic_minibatches
        last = []
        for epoch in range(cfg.critic_epochs):
            perm = torch.randperm(total, device=y.device)
            losses = []
            for start in range(0, total, size):
                idx = perm[start:start + size]
                loss = (critic(s[idx], g[idx], xi[idx]) - y[idx]).square().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg.grad_clip)
                opt.step()
                losses.append(loss.detach())
            if epoch == cfg.critic_epochs - 1:
                last = losses
        return float(torch.stack(last).mean())

    # ---- one policy update ---------------------------------------------------------------------------------------
    def _apply_actor_step(self, agent: str, grads: list[torch.Tensor]) -> float:
        cfg = self.cfg
        params = self.actor_params[agent]
        for p, g in zip(params, grads):
            p.grad = g
        norm = nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        nn.utils.clip_grad_value_(params, cfg.actor_grad_clip_elem)
        self.actor_opts[agent].step()
        self.actor_opts[agent].zero_grad(set_to_none=True)
        return float(norm)

    def update(self, buffer, step_actors: bool = True) -> dict[str, float]:
        """Sample cfg.imagine_batch starts, imagine cfg.horizon steps (in cfg.imagine_chunks sequential passes whose
        gradients add up to the full batch's), step every trained actor (unless step_actors is False: the critics'
        warm-up), then fit every critic to that rollout."""
        cfg = self.cfg
        starts = buffer.sample_starts(cfg.imagine_batch)
        chunks = max(1, cfg.imagine_chunks)
        sizes = [len(x) for x in torch.arange(cfg.imagine_batch).chunk(chunks)]
        grads, pieces, offset = None, [], 0
        for size in sizes:
            part = {k: v[offset:offset + size] for k, v in starts.items()}
            offset += size
            if step_actors:
                imag = self.imagine(part)
                g = self.actor_gradients(imag, weight=size / cfg.imagine_batch)
                grads = g if grads is None else {n: [a + b for a, b in zip(grads[n], g[n])] for n in g}
            else:
                with torch.no_grad():
                    imag = self.imagine(part)
            imag.J = {n: j.detach() * size / cfg.imagine_batch for n, j in imag.J.items()}
            pieces.append(imag)
        imag = pieces[0] if len(pieces) == 1 else _concat(pieces)
        metrics = {}
        for n in self.trained:
            metrics[f"shac/{n}/imagined_return"] = float(imag.J[n])
            if step_actors:
                metrics[f"shac/{n}/actor_grad_norm"] = self._apply_actor_step(n, grads[n])
        for n in self.trained:
            metrics[f"shac/{n}/critic_loss"] = self.fit_critic(imag, n)
        metrics["shac/survival_H"] = float(imag.survival[-1].mean())
        metrics["shac/termination_prob"] = float(imag.done.mean())
        metrics["shac/action_abs"] = float(imag.action_abs)
        return metrics


def _concat(pieces: list[Imagination]) -> Imagination:
    """Batch-concatenate detached chunks; J (already weighted by chunk share) is summed."""
    return Imagination(
        J={n: sum(p.J[n] for p in pieces) for n in pieces[0].J},
        states=torch.cat([p.states for p in pieces], 1),
        feats={n: torch.cat([p.feats[n] for p in pieces], 1) for n in pieces[0].feats},
        ctx=torch.cat([p.ctx for p in pieces], 0),
        rewards={n: torch.cat([p.rewards[n] for p in pieces], 1) for n in pieces[0].rewards},
        done=torch.cat([p.done for p in pieces], 1),
        survival=torch.cat([p.survival for p in pieces], 1),
        action_abs=torch.stack([p.action_abs for p in pieces]).mean())
