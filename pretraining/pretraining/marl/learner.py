# Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c), src/h12_train/marl/learner.py.
# Changes: imports (torch._functorch.config imported explicitly: torch 2.7 does not load it with torch).
"""Two actors trained by BPTT through one world model, each on its own return.

SuperMimic's ``ActorCriticPolicyLearner`` for two agents: both actors act in one
imagined rollout; the legs' return sums the legs' and the team reward, the arms' the
arms' and the team reward, each discounted, masked by survival and bootstrapped by its
own critic (the returns differ, so the critics do). Each actor's gradient is of its
own return only (two backward passes through the rollout). The critics are fitted on
GAE returns of the rollout, as SuperMimic's.
"""

from __future__ import annotations

import torch
import torch._functorch.config
import torch.nn as nn
from tensordict import TensorDict

from pretraining.marl.config import AGENTS
from pretraining.supermimic.learners.learner_utils import (
    CapturedUpdate,
    _optimize_step,
    build_optimizers,
    graph_state,
)
from pretraining.supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg, gae_returns

# Each actor's gradient takes its own backward pass through the rollout, so the first
# keeps the graph (retain_graph), which compiled backward functions with donated
# buffers refuse.
torch._functorch.config.donated_buffer = False


def trainable(module: nn.Module) -> list[nn.Parameter]:
    """Return the module's parameters that require gradients."""
    return [p for p in module.parameters() if p.requires_grad]


class TwoAgentLearner:
    """BPTT actor updates and GAE critic fits for the legs and arms agents."""

    def __init__(
        self,
        actors: dict[str, nn.Module],
        critics: dict[str, nn.Module],
        wm_env,
        cfg: ActorCriticPolicyLearnerCfg,
        device: str = "cpu",
    ):
        """Build an optimiser per actor (its trainable parameters) and per critic."""
        self.actors = actors
        self.critics = critics
        self.wm_env = wm_env
        self.cfg = cfg
        self.params = {a: trainable(m) for a, m in actors.items()}
        self.actor_optimizers = {
            a: build_optimizers([_ParamHolder(self.params[a], m)], cfg.lr, betas=(0.9, cfg.beta2), device=device)
            for a, m in actors.items()
        }
        self.critic_optimizers = {a: build_optimizers([c], cfg.lr, device=device) for a, c in critics.items()}
        self._v_max = wm_env.max_step_reward / (1 - cfg.gamma)  # (2,)
        on_cuda = torch.device(device).type == "cuda"
        self._rollout = {a: torch.compile(m) if on_cuda else m for a, m in actors.items()}
        self._captured_update = CapturedUpdate(self._update_from, self._graph_state)

    def update(self) -> dict[str, torch.Tensor]:
        """Imagine ``num_batches`` rollouts from replay-buffer states and update everything."""
        for m in self.actors.values():
            m.train()
        for module in (self.wm_env.dynamics, self.wm_env.representation):
            module.eval()
            module.requires_grad_(False)
        starts = [self.wm_env.sample_initial_states() for _ in range(self.cfg.num_batches)]
        return self._captured_update(torch.stack(starts))

    def _update_from(self, starts: TensorDict) -> dict[str, torch.Tensor]:
        """Return the per-batch logs of every rollout in ``starts``, averaged."""
        totals: dict[str, torch.Tensor] = {}
        for start in starts.unbind(0):
            for k, v in self._train_batch(start).items():
                totals[k] = totals[k] + v if k in totals else v
        # Policy/return_<agent> is the mean bootstrapped imagined return of a rollout: the
        # discounted, survival-masked sum of the dt-scaled own + team rewards over the
        # horizon plus the discounted critic value of its last state.
        return {f"Policy/{k}": v / self.cfg.num_batches for k, v in totals.items()}

    def _train_batch(self, start: TensorDict) -> dict[str, torch.Tensor]:
        """Imagine one rollout from ``start`` and step each actor on its own return."""
        cfg = self.cfg
        obs = self.wm_env.reset(start)
        ret = {a: 0.0 for a in AGENTS}
        aux = {a: 0.0 for a in AGENTS}
        survival = 1.0
        rollout: list[TensorDict] = []
        for t in range(cfg.horizon):
            actions = torch.cat([self._rollout[a](obs[a]) for a in AGENTS], dim=-1)
            next_obs, reward, term_prob, _ = self.wm_env.step(actions)
            own = {
                "legs": reward[..., 0] + reward[..., 2],
                "arms": reward[..., 1] + reward[..., 2],
            }
            step = {
                "critic": obs["critic"],
                "done": term_prob,
                **{f"reward_{a}": own[a] for a in AGENTS},
            }
            rollout.append(TensorDict(step, batch_size=term_prob.shape).detach())
            for a in AGENTS:
                ret[a] = ret[a] + cfg.gamma**t * survival * own[a]
                aux[a] = aux[a] + self.actors[a].auxiliary_loss()
            if cfg.survival_masking:
                survival = survival * (1 - term_prob)
            obs = next_obs
        logs: dict[str, torch.Tensor] = {}
        losses = {}
        for i, a in enumerate(AGENTS):
            ret[a] = ret[a] + self._bootstrap(a, i, obs, survival, term_prob)
            losses[a] = (-ret[a] + aux[a]).mean()
            logs[f"return_{a}"] = ret[a].mean().detach()
        grads = {}
        for i, a in enumerate(AGENTS):
            # allow_unused=False: a trainable parameter off its agent's graph (a mis-wired
            # encoder) raises instead of silently getting a zero gradient.
            grads[a] = torch.autograd.grad(losses[a], self.params[a], retain_graph=i < len(AGENTS) - 1)
        for a in AGENTS:
            for p, g in zip(self.params[a], grads[a], strict=True):
                p.grad = g
            logs[f"grad_norm_{a}"] = torch.nn.utils.clip_grad_norm_(self.params[a], max_norm=1.0)
            if cfg.grad_clip_value > 0:
                torch.nn.utils.clip_grad_value_(self.params[a], cfg.grad_clip_value)
            for opt in self.actor_optimizers[a]:
                opt.step()
                opt.zero_grad(set_to_none=True)
            stats = getattr(self.actors[a], "auxiliary_stats", lambda: {})()
            logs.update({f"{k}_{a}": v for k, v in stats.items()})
        logs.update(self._fit_critics(rollout, obs))
        return logs

    def _bootstrap(self, agent: str, i: int, obs: TensorDict, survival, term_prob) -> torch.Tensor:
        """Return the agent's critic value of the rollout's last state, discounted."""
        critic = self.critics[agent]
        critic.requires_grad_(False)
        value = critic(obs["critic"]).clamp(max=self._v_max[i]).squeeze(-1)
        alive = survival if self.cfg.survival_masking else 1.0 - term_prob
        return self.cfg.gamma**self.cfg.horizon * value * alive

    def _fit_critics(self, rollout: list[TensorDict], last_obs: TensorDict) -> dict[str, torch.Tensor]:
        """Fit each agent's critic to the GAE returns of its own reward."""
        steps = torch.stack(rollout)
        logs = {}
        for i, a in enumerate(AGENTS):
            critic = self.critics[a]
            with torch.no_grad():
                values = critic(steps["critic"])
                last_value = critic(last_obs["critic"])
            returns = gae_returns(
                values,
                last_value,
                steps[f"reward_{a}"].unsqueeze(-1),
                steps["done"].unsqueeze(-1),
                self.cfg.gamma,
                self.cfg.lam,
            ).clamp(max=self._v_max[i])
            obs, returns = steps["critic"].flatten(0, 1), returns.flatten(0, 1)
            perm = torch.randperm(len(returns) // 4 * 4, device=returns.device)
            critic.requires_grad_(True)
            losses = []
            for _ in range(5):
                for idx in perm.view(4, -1):
                    loss = (returns[idx] - critic(obs[idx])).pow(2).mean()
                    _optimize_step(self.critic_optimizers[a], loss, critic.parameters(), max_norm=1.0)
                    losses.append(loss.detach())
            logs[f"value_function_{a}"] = torch.stack(losses).mean()
        return logs

    def _graph_state(self) -> list[torch.Tensor]:
        """What the captured update reads that Python can rebind."""
        wm = self.wm_env
        state = graph_state([wm.dynamics, wm.representation], [])
        for a in AGENTS:
            state += graph_state([self.actors[a]], self.actor_optimizers[a])
            state += graph_state([self.critics[a]], self.critic_optimizers[a])
        return state


class _ParamHolder(nn.Module):
    """Expose a subset of a module's parameters, with its ``nn.Sequential`` structure, to
    ``build_optimizers`` (Muon on hidden Linear weights, Adam on the rest)."""

    def __init__(self, params: list[nn.Parameter], module: nn.Module):
        """Keep the trainable Sequentials of ``module`` and the loose trainable parameters."""
        super().__init__()
        wanted = {id(p) for p in params}
        seqs = [
            s
            for s in module.modules()
            if isinstance(s, nn.Sequential)
            and all(id(p) in wanted for p in s.parameters())
            and any(True for _ in s.parameters())
        ]
        self.seqs = nn.ModuleList(seqs)
        covered = {id(p) for s in seqs for p in s.parameters()}
        self.rest = nn.ParameterList([p for p in params if id(p) not in covered])
