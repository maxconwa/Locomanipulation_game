"""ShacTrainer's actor gradients against central finite differences of its own objective, on a small hand-written
differentiable world model (float64), with two trained agents whose actions both drive the dynamics: each actor's
gradient is that of its own J_i. Then an update leaves the world model untouched (no .grad, same values), and splitting
the imagined batch into chunks gives the same step as one pass."""

import copy
import math

import torch
import torch.nn as nn

from actors import StandInLSTMPolicy
from supermimic.buffer import flatten_lstm
from supermimic.config import SuperMimicCfg
from supermimic.shac import ShacTrainer
from supermimic.world_model import AgentDims, GoalFns, RunningNorm

S, C = 3, 2
AGENTS = {"u": AgentDims(obs_dim=3, goal_dim=1, action_dim=2), "v": AgentDims(obs_dim=2, goal_dim=1, action_dim=1)}
GOAL_FNS = {n: GoalFns(features=lambda g: g * 0.5, advance=lambda g: g + 0.1) for n in AGENTS}


def _norm(dim, gen):
    norm = RunningNorm(dim).double()
    norm.update(torch.randn(50, dim, generator=gen, dtype=torch.float64) * 1.5 + 0.3)
    return norm


class StubWorldModel(nn.Module):
    """Linear-quadratic dynamics, linear observations, quadratic rewards, a linear termination logit."""

    def __init__(self, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        r = lambda *shape: nn.Parameter(torch.randn(*shape, generator=gen, dtype=torch.float64) * 0.3)  # noqa: E731
        self.agents, self.agent_names = AGENTS, list(AGENTS)
        self.A, self.B, self.Q, self.Cx = r(S, S), r(S, 3), r(S, S), r(S, C)
        self.D = nn.ParameterDict({"u": r(3, S), "v": r(2, S)})
        self.w = nn.ParameterDict({"u": r(S), "v": r(S)})
        self.term = r(S + 2)
        self.state_norm, self.ctx_norm = _norm(S, gen), _norm(C, gen)
        self.goal_norms = nn.ModuleDict({n: _norm(1, gen) for n in AGENTS})

    def step(self, s, a, xi):
        return s + 0.2 * (s @ self.A.T + a @ self.B.T + (s * s) @ self.Q.T * 0.1 + xi @ self.Cx.T)

    def observe(self, agent, s, xi, g):
        return s @ self.D[agent].T + g

    def reward(self, agent, s_next, a, g):
        return -(s_next @ self.w[agent] - g[:, 0]).square() - 0.1 * a.square().sum(-1)

    def terminal_logit(self, s_next, g_all):
        return torch.cat([s_next, g_all], -1) @ self.term - 2.0


def make_trainer(seed=0, **cfg_kw):
    torch.manual_seed(seed)
    cfg = SuperMimicCfg(horizon=4, gamma=0.9, critic_hidden=(16,), imagine_batch=16, critic_epochs=2,
                        critic_minibatches=2, **cfg_kw)
    wm = StubWorldModel(seed)
    actors = {"u": StandInLSTMPolicy(3, 2, hidden_size=5, layers=(8,), skip_observations=True, cell_cap=1.5).double(),
              "v": StandInLSTMPolicy(2, 1, hidden_size=4, layers=(6,)).double()}
    gen = torch.Generator().manual_seed(seed + 1)
    obs_norms = {n: _norm(AGENTS[n].obs_dim, gen) for n in AGENTS}
    noise = {n: torch.zeros(AGENTS[n].obs_dim, dtype=torch.float64) for n in AGENTS}
    return ShacTrainer(wm, actors, GOAL_FNS, obs_norms, noise, cfg, action_clip=10.0)


def make_starts(trainer, batch=16, seed=5):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float64)  # noqa: E731
    starts = {"state": r(batch, S), "ctx": r(batch, C)}
    for n, actor in trainer.actors.items():
        starts[f"goal/{n}"] = r(batch, 1)
        starts[f"h/{n}"] = flatten_lstm(r(actor.num_layers, batch, actor.hidden_size) * 0.5)
        starts[f"c/{n}"] = flatten_lstm(r(actor.num_layers, batch, actor.hidden_size) * 0.5)
    return starts


def test_actor_gradient_matches_finite_differences():
    trainer = make_trainer()
    starts = make_starts(trainer)
    imag = trainer.imagine(starts, noise=False)
    assert 0.0 < float(imag.done.mean()) < 1.0  # the survival weight is exercised
    grads = trainer.actor_gradients(imag)
    eps, worst = 1e-5, 0.0
    for agent in ("u", "v"):
        params = trainer.actor_params[agent]
        for pi in (0, 1, 2, len(params) - 2, len(params) - 1):     # LSTM weights and biases, the MLP's last layer
            p = params[pi]
            for flat in torch.randperm(p.numel(), generator=torch.Generator().manual_seed(pi))[:3].tolist():
                with torch.no_grad():
                    orig = p.view(-1)[flat].item()
                    p.view(-1)[flat] = orig + eps
                    j_plus = float(trainer.imagine(starts, noise=False).J[agent])
                    p.view(-1)[flat] = orig - eps
                    j_minus = float(trainer.imagine(starts, noise=False).J[agent])
                    p.view(-1)[flat] = orig
                fd = (j_plus - j_minus) / (2 * eps)
                analytic = -float(grads[agent][pi].view(-1)[flat])  # actor_gradients returns d(-J)/d(theta)
                err = abs(fd - analytic) / max(1e-6, abs(fd), abs(analytic))
                worst = max(worst, err)
                assert abs(fd - analytic) < 1e-5 * abs(analytic) + 1e-9, (agent, pi, flat, fd, analytic)
    print(f"\nworst relative error, analytic vs finite differences: {worst:.2e}")


class FixedStarts:
    def __init__(self, starts):
        self.starts = starts

    def sample_starts(self, batch):
        return {k: v[:batch] for k, v in self.starts.items()}


def test_update_freezes_world_model_and_chunks_agree():
    one = make_trainer(seed=1)
    two = make_trainer(seed=1, imagine_chunks=2)
    starts = make_starts(one, seed=9)
    wm_before = copy.deepcopy(one.wm.state_dict())
    critic_before = copy.deepcopy(one.critics["u"].state_dict())
    actor_before = copy.deepcopy(one.actors["u"].state_dict())
    m1 = one.update(FixedStarts(starts))
    m2 = two.update(FixedStarts(starts))
    for name, p in one.wm.named_parameters():
        assert p.grad is None, name
        assert torch.equal(p, wm_before[name]), name
        assert p.requires_grad
    assert any(not torch.equal(p, critic_before[k]) for k, p in one.critics["u"].state_dict().items())
    assert any(not torch.equal(p, actor_before[k]) for k, p in one.actors["u"].state_dict().items())
    assert torch.equal(one.actors["u"].log_std_parameter, actor_before["log_std_parameter"])
    for agent in ("u", "v"):
        assert abs(m1[f"shac/{agent}/imagined_return"] - m2[f"shac/{agent}/imagined_return"]) < 1e-10
        assert abs(m1[f"shac/{agent}/actor_grad_norm"] - m2[f"shac/{agent}/actor_grad_norm"]) < 1e-10
        for p1, p2 in zip(one.actors[agent].parameters(), two.actors[agent].parameters()):
            assert torch.allclose(p1, p2, atol=1e-12)


def test_values_and_critic_targets_are_clamped_to_possible_returns():
    """With reward ranges on the world model, the critic's value (in J and in its targets) is clamped to
    [min(r_low, 0) / (1 - gamma) + fall_penalty, max(r_high, 0) / (1 - gamma)]."""
    from supermimic.world_model import RunningRange
    trainer = make_trainer(range_margin=0.0, fall_penalty=-5.0)
    trainer.wm.reward_ranges = nn.ModuleDict({n: RunningRange(0).double() for n in AGENTS})
    for n in AGENTS:
        trainer.wm.reward_ranges[n].update(torch.tensor([-0.5, 2.0], dtype=torch.float64))
    low, high = trainer.value_bounds("u")
    assert abs(low - (-0.5 / 0.1 - 5.0)) < 1e-9 and abs(high - 2.0 / 0.1) < 1e-9
    with torch.no_grad():
        trainer.critics["u"].net[-1].bias.fill_(1e4)
    imag = trainer.imagine(make_starts(trainer))
    targets = trainer.critic_targets(imag, "u")
    assert float(targets.max()) <= high + 1e-9
    v = trainer.value("u", imag.states[-1], imag.feats["u"][-1], imag.ctx)
    assert torch.all(v == high)


def test_collection_noise_per_env():
    """action_noise_std_max 0: every env's std is action_noise_std; above it, log-uniform in the range."""
    std = make_trainer(action_noise_std=0.1, action_noise_std_max=0.0).collection_noise(500)
    assert torch.all(std == 0.1)
    std = make_trainer(action_noise_std=0.1, action_noise_std_max=0.5).collection_noise(4000)
    assert float(std.min()) >= 0.1 and float(std.max()) <= 0.5
    assert abs(float(std.log().mean()) - 0.5 * (math.log(0.1) + math.log(0.5))) < 0.03
