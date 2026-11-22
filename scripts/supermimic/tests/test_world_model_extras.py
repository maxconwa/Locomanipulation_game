"""The world model's additions to the paper's (world_model.py's docstring): heads detached from the dynamics, the
state change in its own units, a trip head with a fixed cost, and the clamps to the real data's range; and the
critic's value clamp (shac.py). On the synthetic system (synthetic.py), with a trip of agent p whenever s'[0] > 1."""

import torch

from supermimic.buffer import ReplayBuffer, standard_spec
from supermimic.config import SuperMimicCfg
from supermimic.world_model import RunningRange, WorldModel, train_world_model
from synthetic import AGENTS, C, GOAL_FNS, S, Synthetic

TRIP_COST, TRIP_AT = -1.0, 1.0


def collect_with_trips(env: Synthetic, steps: int) -> ReplayBuffer:
    """synthetic.collect with trip/<agent> fields: p trips when s'[0] > TRIP_AT, charged TRIP_COST in its reward."""
    spec = standard_spec(S, C, 2, {n: d.obs_dim for n, d in AGENTS.items()}, {n: 2 for n in AGENTS}, {}, trips=True)
    buf = ReplayBuffer(spec, env.n, steps)
    for _ in range(steps):
        step = {"state": env.s.clone(), "ctx": env.xi.clone()}
        for n in AGENTS:
            step[f"obs/{n}"] = env.obs(n)
            step[f"goal/{n}"] = env.goal[n].clone()
        a = torch.randn(env.n, 2, generator=env.gen)
        trip = (env.dynamics(env.s, a, env.xi)[:, 0] > TRIP_AT).float()
        rewards, term, trunc = env.step(a)
        rewards["p"] = rewards["p"] + TRIP_COST * trip
        step.update({f"reward/{n}": r for n, r in rewards.items()}, action=a, terminated=term, truncated=trunc)
        step.update({"trip/p": trip, "trip/q": torch.zeros(env.n)})
        buf.add(step)
    return buf


def small_cfg(**kw) -> SuperMimicCfg:
    return SuperMimicCfg(window=8, wm_batches=1, wm_batch_size=256, dynamics_hidden=(128, 128), obs_hidden=(64, 64),
                         reward_hidden=(64, 64), termination_hidden=(64, 64), lr=2e-3, reward_scale=2.0, **kw)


def test_trip_head_and_trip_free_reward():
    torch.manual_seed(0)
    buffer = collect_with_trips(Synthetic(64), 300)
    rate = float(buffer.data["trip/p"].mean())
    assert 0.02 < rate < 0.4, rate
    cfg = small_cfg(trip_penalty=TRIP_COST)
    wm = WorldModel(S, C, AGENTS, cfg)
    wm.update_norms({k: v for k, v in buffer.data.items() if not k.startswith(("term", "trunc"))}, GOAL_FNS)
    # the reward range is the trip-free reward's: no -1 spikes in it
    assert float(wm.reward_ranges["p"].low) > float(buffer.data["reward/p"].min())
    opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    for _ in range(600):
        metrics = train_world_model(wm, opt, buffer, GOAL_FNS, cfg)
    torch.manual_seed(1)
    with torch.no_grad():
        _, m = wm.loss(buffer.sample_windows(2000, 8), GOAL_FNS)
    assert float(m["wm/trip_loss"]) < 0.5 * float(m["wm/trip_base_bce"]), m   # a deterministic trip: well learned
    # expected_reward = reward + cost * p
    w = buffer.sample_windows(64, 1)
    s1, a, g = w["state"][:, 1], w["action"][:, 0], GOAL_FNS["p"].features(w["goal/p"][:, 1])
    with torch.no_grad():
        p = torch.sigmoid(wm.trip_logit("p", s1, a, g))
        assert torch.allclose(wm.expected_reward("p", s1, a, g), wm.reward("p", s1, a, g) + TRIP_COST * p)
    assert metrics["wm/state_rmse_1step"] < 0.2


def test_heads_do_not_train_the_dynamics_when_detached():
    torch.manual_seed(0)
    buffer = collect_with_trips(Synthetic(16), 40)
    for detach in (True, False):
        cfg = small_cfg(trip_penalty=TRIP_COST, detach_head_inputs=detach,
                        loss_weights={"state": 0.0, "obs": 1.0, "reward": 1.0, "termination": 1.0, "trip": 1.0})
        wm = WorldModel(S, C, AGENTS, cfg)
        wm.update_norms({k: v for k, v in buffer.data.items() if not k.startswith(("term", "trunc"))}, GOAL_FNS)
        torch.nn.init.normal_(wm.dynamics[-1].weight, std=0.1)  # a non-zero last layer, so gradients reach the rest
        loss, _ = wm.loss(buffer.sample_windows(64, 8), GOAL_FNS)
        loss.backward()
        grad = sum(float(p.grad.abs().sum()) for p in wm.dynamics.parameters() if p.grad is not None)
        assert (grad == 0.0) == detach, (detach, grad)


def test_state_change_units_and_range_clamp():
    torch.manual_seed(0)
    buffer = collect_with_trips(Synthetic(32), 60)
    wm = WorldModel(S, C, AGENTS, small_cfg())
    wm.update_norms({"state": buffer.data["state"]}, GOAL_FNS)
    wm.update_delta_norm(buffer.sample_windows(512, 4, fields=["state"]))
    # the untrained network (zero last layer) predicts the mean change
    s, a, xi = buffer.data["state"][5], buffer.data["action"][5], buffer.data["ctx"][5]
    assert torch.allclose(wm.step(s, a, xi), s + wm.delta_norm.mean, atol=1e-6)
    # predictions far outside the data are clamped to its range widened by range_margin
    with torch.no_grad():
        wm.dynamics[-1].bias.fill_(1e6)
    out = wm.step(s, a, xi)
    width = wm.state_range.high - wm.state_range.low
    assert torch.allclose(out, (wm.state_range.high + wm.cfg.range_margin * width).expand_as(out))
    r = RunningRange(0)
    assert r.clamp(torch.tensor([5.0]), 0.1).item() == 5.0   # no data yet: unchanged
    r.update(torch.tensor([0.0, 1.0]))
    assert torch.allclose(r.clamp(torch.tensor([5.0, -5.0, 0.5]), 0.1), torch.tensor([1.1, -0.1, 0.5]))
