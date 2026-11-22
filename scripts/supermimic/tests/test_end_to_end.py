"""The whole loop on a toy: a 1-D point mass with a per-episode actuator gain (the sim context) is driven to a goal
position. World model, critic and LSTM actor all start untrained; each iteration collects a few real steps with
action noise, takes one world-model update and one SHAC update, as scripts/supermimic/train.py does with Isaac. The
true return of the deterministic policy must improve far beyond the untrained one's."""

import time

import torch

from actors import StandInLSTMPolicy
from supermimic import (AgentDims, GoalFns, ReplayBuffer, RunningNorm, ShacTrainer, SuperMimicCfg, WorldModel,
                        flatten_lstm, standard_spec, train_world_model)

EPISODE, LIMIT, CLIP = 50, 2.5, 3.0
AGENTS = {"mass": AgentDims(obs_dim=3, goal_dim=1, action_dim=1, goal_obs_columns=(2,), goal_feature_columns=(0,))}
GOAL_FNS = {"mass": GoalFns(features=lambda g: g, advance=lambda g: g)}
OBS_NOISE = torch.tensor([0.02, 0.05, 0.0])


class PointMass:
    """state (x, v), ctx (gain,), goal (target,); reward -(x' - g)^2 - 0.01 a^2; ends at |x'| > LIMIT or EPISODE."""

    def __init__(self, n, seed):
        self.n, self.gen = n, torch.Generator().manual_seed(seed)
        self.s, self.xi, self.g = torch.zeros(n, 2), torch.zeros(n, 1), torch.zeros(n, 1)
        self.t = torch.zeros(n, dtype=torch.long)
        self.reset(torch.ones(n, dtype=torch.bool))

    def reset(self, mask):
        k = int(mask.sum())
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(k, 1, generator=self.gen)  # noqa: E731
        self.s[mask] = torch.cat([u(-1.0, 1.0), torch.zeros(k, 1)], -1)
        self.xi[mask], self.g[mask], self.t[mask] = u(0.6, 1.4), u(-1.5, 1.5), 0

    def obs(self):
        o = torch.cat([self.s, self.g], -1)
        return o + (2 * torch.rand(o.shape, generator=self.gen) - 1) * OBS_NOISE

    def step(self, a):
        a = a.clamp(-CLIP, CLIP)
        v = 0.85 * self.s[:, 1:] + 0.3 * self.xi * a
        x = self.s[:, :1] + 0.2 * v
        self.s = torch.cat([x, v], -1)
        reward = -(x - self.g).square().squeeze(-1) - 0.01 * a.square().squeeze(-1)
        self.t += 1
        terminated = x.squeeze(-1).abs() > LIMIT
        truncated = (self.t >= EPISODE) & ~terminated
        self.reset(terminated | truncated)
        return reward, terminated, truncated


@torch.no_grad()
def true_return(trainer, cfg, n=256):
    """Mean undiscounted return of the deterministic policy over one episode from fixed starts (a fall ends it with
    the fall penalty)."""
    env = PointMass(n, seed=999)
    lstm = trainer.initial_lstm(n)
    total, alive = torch.zeros(n), torch.ones(n, dtype=torch.bool)
    for _ in range(EPISODE):
        action, _, lstm = trainer.act({"mass": env.obs()}, lstm, explore=False)
        reward, terminated, truncated = env.step(action)
        total += alive * (reward + cfg.fall_penalty * terminated)
        alive &= ~(terminated | truncated)
    return float(total.mean())


def test_supermimic_loop_learns_the_point_mass():
    torch.manual_seed(0)
    cfg = SuperMimicCfg(window=8, horizon=16, gamma=0.97, wm_batches=2, wm_batch_size=256, imagine_batch=256,
                        dynamics_hidden=(128, 128), obs_hidden=(64,), reward_hidden=(64, 64),
                        termination_hidden=(64,), critic_hidden=(64, 64), lr=1e-3, buffer_steps=200, collect_steps=4,
                        action_noise_std=0.3, critic_epochs=5, critic_minibatches=4)
    n = 64
    env = PointMass(n, seed=0)
    wm = WorldModel(2, 1, AGENTS, cfg)
    actor = StandInLSTMPolicy(3, 1, hidden_size=16, layers=(32,), skip_observations=True, cell_cap=2.0)
    obs_norm = RunningNorm(3)
    trainer = ShacTrainer(wm, {"mass": actor}, GOAL_FNS, {"mass": obs_norm}, {"mass": OBS_NOISE}, cfg,
                          action_clip=CLIP)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    spec = standard_spec(2, 1, 1, {"mass": 3}, {"mass": 1}, {"mass": (actor.num_layers, actor.hidden_size)})
    buffer = ReplayBuffer(spec, n, cfg.buffer_steps)

    initial = true_return(trainer, cfg)
    lstm = trainer.initial_lstm(n)
    history, start = [], time.time()
    for it in range(160):
        for _ in range(cfg.collect_steps if len(buffer) > cfg.window else cfg.window + 1):
            obs = env.obs()
            obs_norm.update(obs)
            step = {"state": env.s.clone(), "ctx": env.xi.clone(), "obs/mass": obs, "goal/mass": env.g.clone(),
                    "h/mass": flatten_lstm(lstm["mass"][0]), "c/mass": flatten_lstm(lstm["mass"][1])}
            action, _, lstm = trainer.act({"mass": obs}, lstm, explore=True)
            reward, terminated, truncated = env.step(action)
            step.update({"action": action, "reward/mass": reward, "terminated": terminated, "truncated": truncated})
            wm.update_norms(step, GOAL_FNS)
            buffer.add(step)
            lstm = trainer.reset_lstm(lstm, terminated | truncated)
        wm_metrics = train_world_model(wm, wm_opt, buffer, GOAL_FNS, cfg)
        metrics = trainer.update(buffer)
        if it % 40 == 39:
            history.append((it + 1, true_return(trainer, cfg), wm_metrics["wm/state_rmse_8step"],
                            metrics["shac/mass/imagined_return"]))
    final = true_return(trainer, cfg)
    print(f"\ntrue return: untrained {initial:.2f} -> trained {final:.2f} ({time.time() - start:.1f} s); "
          f"(iteration, return, wm 8-step rmse, imagined J): {[tuple(round(x, 3) for x in h) for h in history]}")
    assert final > initial + 0.6 * abs(initial)
    assert final > -15.0  # seeds 0-3 reach -4.5 to -8.8; driving every start to its goal costs about -5
