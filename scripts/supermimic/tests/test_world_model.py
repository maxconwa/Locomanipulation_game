"""WorldModel on the synthetic system (synthetic.py), whose dynamics, observations, rewards and terminations are known:
a small model fit for a few hundred steps predicts 8 steps ahead, the reward and the termination, and its observation
head recovers the noise-free observation from noisy targets. Data of the next episode inside a window (masked) cannot
reach the loss: corrupted to inf / nan, the fit is unchanged."""

import torch

from supermimic.config import SuperMimicCfg
from supermimic.world_model import WorldModel, train_world_model
from synthetic import AGENTS, C, GOAL_FNS, OBS_NOISE, S, Synthetic, collect

K = 8


def small_cfg(**kw) -> SuperMimicCfg:
    cfg = SuperMimicCfg(window=K, wm_batches=1, wm_batch_size=256, dynamics_hidden=(128, 128), obs_hidden=(64, 64),
                        reward_hidden=(64, 64), termination_hidden=(64, 64), lr=2e-3, reward_scale=2.0)
    for key, value in kw.items():
        setattr(cfg, key, value)
    return cfg


class Corrupting:
    """A buffer whose windows have every masked entry (data of the next episode) replaced by nan or a huge value."""

    def __init__(self, buffer):
        self.buffer = buffer

    def sample_windows(self, batch, k):
        w = self.buffer.sample_windows(batch, k)
        entry = torch.cat([torch.ones_like(w["next_valid"][:, :1]), w["next_valid"]], 1) == 0
        alive = w["alive"] == 0
        for name, x in w.items():
            if x.dtype == torch.bool or name in ("alive", "next_valid"):
                continue
            mask = alive if x.shape[1] == k else entry
            value = float("nan") if name.startswith(("obs", "reward")) else 1e30
            x[mask] = value
        return w


def fit(buffer, steps=800, seed=0):
    torch.manual_seed(seed)
    cfg = small_cfg()
    wm = WorldModel(S, C, AGENTS, cfg)
    data = buffer.buffer if isinstance(buffer, Corrupting) else buffer
    wm.update_norms({k: v for k, v in data.data.items() if not k.startswith(("reward", "term", "trunc"))}, GOAL_FNS)
    opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    history = []
    for it in range(steps):
        if it == int(steps * 0.7):
            for group in opt.param_groups:
                group["lr"] = cfg.lr * 0.2
        metrics = train_world_model(wm, opt, buffer, GOAL_FNS, cfg)
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values()), metrics
        history.append(metrics)
    return wm, history


def evaluate(wm, env, steps=2000):
    """k-step state, reward and observation errors against the true system on fresh data."""
    torch.manual_seed(123)
    buf = collect(Synthetic(64, seed=7), 60)
    w = buf.sample_windows(steps, K)
    with torch.no_grad():
        _, metrics = wm.loss(w, GOAL_FNS)
        s, xi = w["state"][:, 0], w["ctx"][:, 0]
        feats = GOAL_FNS["p"].features(w["goal/p"][:, 0])
        clean = env.clean_obs("p", s, xi, w["goal/p"][:, 0])
        o_hat = wm.observe("p", s, xi, feats)
        obs_rmse = (o_hat - clean).square().mean().sqrt().item()
        # exact copy of the goal column
        assert torch.allclose(o_hat[:, 3], feats[:, 0])
        # k-step rollout in raw units against the truth, on windows without a done
        ok = w["next_valid"].prod(1) > 0
        s_hat, s_true = s[ok], s[ok]
        for j in range(K):
            s_hat = wm.step(s_hat, w["action"][ok, j], xi[ok])
            s_true = env.dynamics(s_true, w["action"][ok, j], xi[ok])
        raw_rmse = (s_hat - s_true).square().mean().sqrt().item()
    return metrics, obs_rmse, raw_rmse


def test_world_model_fits_the_synthetic_system():
    env = Synthetic(64)
    buffer = collect(env, 300)
    wm, history = fit(buffer)
    metrics, obs_rmse, raw_rmse = evaluate(wm, env)
    print(f"\nfirst {history[0]['wm/loss']:.3f} last {history[-1]['wm/loss']:.3f}; held-out: "
          f"state 1-step {float(metrics['wm/state_rmse_1step']):.4f} 8-step {float(metrics['wm/state_rmse_8step']):.4f}"
          f" (normalised), raw 8-step {raw_rmse:.4f}, reward rmse {float(metrics['wm/reward_rmse']):.4f}, obs rmse vs"
          f" clean {obs_rmse:.4f} (noise std {OBS_NOISE / 3 ** 0.5:.4f}), termination acc"
          f" {float(metrics['wm/termination_accuracy']):.4f}"
          f" (positives {float(metrics['wm/termination_positive_rate']):.3f})")
    assert history[-1]["wm/loss"] < 0.1 * history[0]["wm/loss"]
    assert float(metrics["wm/state_rmse_8step"]) < 0.08
    assert float(metrics["wm/reward_rmse"]) < 0.3           # the reward std is about 3.5
    assert obs_rmse < 0.5 * OBS_NOISE / 3 ** 0.5           # well under the sensor noise: the conditional mean
    assert float(metrics["wm/termination_accuracy"]) > 0.97


def test_masked_data_cannot_reach_the_loss():
    env = Synthetic(64)
    buffer = collect(env, 300)
    wm, history = fit(Corrupting(buffer), steps=400)
    metrics, obs_rmse, _ = evaluate(wm, env)
    assert float(metrics["wm/state_rmse_8step"]) < 0.15
    assert obs_rmse < OBS_NOISE / 3 ** 0.5
    for p in wm.parameters():
        assert torch.isfinite(p).all()
