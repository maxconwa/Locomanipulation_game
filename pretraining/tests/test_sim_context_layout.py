"""The sim-context layout a fine-tune or resume takes from its source run (CPU; stand-in envs, no app)."""

import json
from types import SimpleNamespace

import pytest
import torch

from pretraining.supermimic import sim_context as sc
from pretraining.supermimic.sim_context import SimContext, layout_from_bundle

N_JOINTS = 27


def _base_bundle():
    """A main-plant base run's sim context: foot friction 0.3-1.2, COM +-(0.025, 0.05, 0.05), encoder bias +-0.01."""
    ranges = [(0.3, 1.2), (-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)] + [(-0.01, 0.01)] * N_JOINTS
    ctx = SimContext([SimpleNamespace(name=n, dim=d) for n, d in (("foot_friction", 1), ("base_com", 3),
                                                                   ("encoder_bias", N_JOINTS))], ranges)
    return ctx, ranges


def test_layout_round_trips_and_is_inferred_from_an_old_bundle():
    ctx, ranges = _base_bundle()
    layout = ctx.layout()
    assert [(e["name"], e["dim"]) for e in layout] == [("foot_friction", 1), ("base_com", 3), ("encoder_bias", 27)]
    flat = [v for e in layout for r in e["ranges"] for v in r]
    assert flat == pytest.approx([v for r in ranges for v in r], abs=1e-6)
    assert layout_from_bundle(ctx.state_dict(), N_JOINTS) == layout
    json.dumps(layout)  # run.json holds it
    with pytest.raises(ValueError, match="matches no combination"):
        layout_from_bundle({"mean": torch.zeros(2), "scale": torch.ones(2)}, N_JOINTS)


class _LiveBias:
    """A stand-in encoder_bias spec: main's offsets drawn wider than the base's +-0.01."""

    name, dim = "encoder_bias", 0
    available = classmethod(sc.SimContextSpec.available.__func__)

    def bind(self, env):
        self.dim = N_JOINTS
        return [(-0.04, 0.04)] * N_JOINTS

    def read(self, env):
        return env.encoder_bias


def test_fine_tune_env_keeps_the_source_columns(monkeypatch, capsys):
    """The fine-tune env has no foot_friction or base_com event: their 4 columns stay, read the source's mean
    (normalised 0); encoder_bias stays live with the source's normalisation (0.02 rad reads 2.0)."""
    monkeypatch.setitem(sc._CANDIDATES, "encoder_bias", _LiveBias)
    ctx, _ = _base_bundle()
    env = SimpleNamespace(
        cfg=SimpleNamespace(events=SimpleNamespace(foot_friction=None, base_com=None, encoder_bias=object())),
        num_envs=2, device="cpu", encoder_bias=torch.full((2, N_JOINTS), 0.02),
    )
    ft = SimContext.from_env(env, ctx.layout())
    assert ft.output_dim == 31 and [s.name for s in ft.specs] == ["foot_friction", "base_com", "encoder_bias"]
    torch.testing.assert_close(ft.mean, ctx.mean)
    torch.testing.assert_close(ft.scale, ctx.scale)
    out = ft.read(env)
    torch.testing.assert_close(out[:, :4], torch.zeros(2, 4))
    torch.testing.assert_close(out[:, 4:], torch.full((2, N_JOINTS), 2.0))
    out = capsys.readouterr().out
    assert "appended none" in out and "foot_friction, base_com held at the source run's mean" in out


def test_a_live_spec_of_another_width_is_refused(monkeypatch):
    class _Narrow(_LiveBias):
        def bind(self, env):
            self.dim = 12
            return [(-0.01, 0.01)] * 12

    monkeypatch.setitem(sc._CANDIDATES, "encoder_bias", _Narrow)
    ctx, _ = _base_bundle()
    env = SimpleNamespace(cfg=SimpleNamespace(events=SimpleNamespace(encoder_bias=object())), num_envs=1,
                          device="cpu")
    with pytest.raises(ValueError, match="encoder_bias: 12 columns here, 27 in the source run"):
        SimContext.from_env(env, ctx.layout())


def test_train_reads_the_source_layout(tmp_path):
    from pretraining.scripts import train

    ctx, _ = _base_bundle()
    (tmp_path / "run.json").write_text(json.dumps({"motion_files": []}))
    torch.save({"sim_context": ctx.state_dict()}, tmp_path / "world_model.pt")
    assert train.source_sim_context_layout(tmp_path) == ctx.layout()  # an old run: inferred from its world model
    (tmp_path / "run.json").write_text(json.dumps({"sim_context": [{"name": "x", "dim": 1, "ranges": [[0, 1]]}]}))
    assert train.source_sim_context_layout(tmp_path)[0]["name"] == "x"  # recorded: taken as is


# -- the fine-tune's widened layout (--fine-tune-from: source columns, then the env's new ones) ----------------------


def _fake(name, dim, rng):
    """A stand-in spec class: available when the env names it in ``env.live``, ``dim`` columns over ``rng``, reading
    ``env.raw[name]``."""

    class _Fake:
        def __init__(self):
            self.name, self.dim = name, dim

        @staticmethod
        def available(env):
            return name in env.live

        def bind(self, env):
            return list(rng) if isinstance(rng, list) else [rng] * dim

        def read(self, env):
            return env.raw[name]

    return _Fake


def _fakes(monkeypatch, env_ranges):
    """Replace every spec by a ``_fake`` with the env's ranges, keeping ``_CANDIDATES``'s order."""
    monkeypatch.setattr(sc, "_CANDIDATES", {n: _fake(n, *env_ranges.get(n, (1, (0.0, 1.0)))) for n in sc._CANDIDATES})


# main's fine-tune env: friction 0.1-1.25 (feet), main's encoder offsets (legs +-0.02, ankle pitch +-0.04, rest 0),
# the delay, gains and strength per joint, torso mass; no COM event
_OFFSETS = [0.02] * 4 + [0.04, 0.02] + [0.02] * 4 + [0.04, 0.02] + [0.0] * 15
_MAIN_RANGES = {"foot_friction": (1, (0.1, 1.25)), "encoder_bias": (N_JOINTS, [(-w, w) for w in _OFFSETS]),
                "action_delay": (1, (0.0, 10.0)),
                "actuator_gain": (N_JOINTS, (0.8, 1.2)), "actuator_strength": (N_JOINTS, (0.85, 1.15)),
                "torso_mass": (1, (-2.0, 4.0))}
_NEW = ["action_delay", "actuator_gain", "actuator_strength", "torso_mass"]


def _main_env(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    off = torch.tensor(_OFFSETS)
    raw = {"foot_friction": 0.1 + 1.15 * torch.rand(n, 1, generator=g),
           "encoder_bias": (2 * torch.rand(n, N_JOINTS, generator=g) - 1) * off,
           "action_delay": torch.randint(0, 11, (n, 1), generator=g).float(),
           "actuator_gain": 0.8 + 0.4 * torch.rand(n, N_JOINTS, generator=g),
           "actuator_strength": 0.85 + 0.3 * torch.rand(n, N_JOINTS, generator=g),
           "torso_mass": -2 + 6 * torch.rand(n, 1, generator=g)}
    live = {"foot_friction", "encoder_bias", *_NEW}
    return SimpleNamespace(live=live, raw=raw, num_envs=n, device="cpu", cfg=SimpleNamespace(events=SimpleNamespace()))


def _fine_tune_ctx(monkeypatch, capsys=None):
    ctx, _ = _base_bundle()
    _fakes(monkeypatch, _MAIN_RANGES)
    env = _main_env()
    return ctx, SimContext.from_env(env, ctx.layout(), fine_tune=True), env


def test_fine_tune_layout_is_the_source_then_the_new_specs(monkeypatch, capsys):
    ctx, ft, env = _fine_tune_ctx(monkeypatch)
    out = capsys.readouterr().out
    assert [s.name for s in ft.specs] == ["foot_friction", "base_com", "encoder_bias", *_NEW]
    assert ft.output_dim == 31 + 1 + 27 + 27 + 1
    assert ("appended action_delay, actuator_gain, actuator_strength, torso_mass" in out
            and "base_com held at the source run's mean" in out and "renormalised encoder_bias" in out)
    torch.testing.assert_close(ft.mean[:31], ctx.mean)
    # friction (another centre) and COM (held) keep the source's normalisation; the offsets take main's half-widths
    # where wider (the arms' 0 keep the source's 0.01)
    torch.testing.assert_close(ft.scale[:4], ctx.scale[:4])
    torch.testing.assert_close(ft.scale[4:31], torch.tensor([max(w, 0.01) for w in _OFFSETS]))
    # the fixed constants: delay 5 +- 5 substeps, gains 1 +- 0.2, strength 1 +- 0.15, torso mass 1 +- 3 kg
    torch.testing.assert_close(ft.mean[31:], torch.tensor([5.0] + [1.0] * 54 + [1.0]))
    torch.testing.assert_close(ft.scale[31:], torch.tensor([5.0] + [0.2] * 27 + [0.15] * 27 + [3.0]))
    assert ft.read(env)[:, 4:].abs().max() <= 1.0 + 1e-6
    # a resume of the fine-tune keeps its layout exactly, nothing appended or renormalised
    again = SimContext.from_env(env, ft.layout())
    assert [s.name for s in again.specs] == [s.name for s in ft.specs]
    torch.testing.assert_close(again.mean, ft.mean)
    torch.testing.assert_close(again.scale, ft.scale)
    assert "appended none" in capsys.readouterr().out
    # the base layout without --fine-tune-from's flag (a resume of an older run): kept as it was
    assert SimContext.from_env(env, ctx.layout()).output_dim == 31


def test_widening_keeps_the_loaded_function(monkeypatch, capsys):
    """Random source dynamics and critic; the fine-tune's context: the same raw friction, COM and offsets, random
    values in the appended columns. The widened networks give the source's outputs exactly."""
    from pretraining.marl.runner import _pad_optimizer_state, widen_sim_context
    from pretraining.supermimic.models import build_mlp

    ctx, ft, env = _fine_tune_ctx(monkeypatch)
    torch.manual_seed(3)
    prefix, n = 40, env.num_envs
    src_dyn, src_critic = build_mlp(prefix + 31, (64, 32), 9), build_mlp(prefix + 31, (16,), 1)
    state = {"world_model": {"dynamics": {f"mlp.{k}": v for k, v in src_dyn.state_dict().items()},
                             "sim_context": ctx.state_dict()},
             "critics": {"legs": src_critic.state_dict()}}
    widen_sim_context(state, ft)
    dyn, critic = build_mlp(prefix + ft.output_dim, (64, 32), 9), build_mlp(prefix + ft.output_dim, (16,), 1)
    dyn.load_state_dict({k[4:]: v for k, v in state["world_model"]["dynamics"].items()})
    critic.load_state_dict(state["critics"]["legs"])
    torch.testing.assert_close(state["world_model"]["sim_context"]["scale"], ft.scale)

    head = torch.randn(n, prefix)
    raw_src = torch.cat([env.raw["foot_friction"], torch.zeros(n, 3), env.raw["encoder_bias"]], -1)
    x_src = torch.cat([head, (raw_src - ctx.mean) / ctx.scale], -1)
    for seed in range(3):  # whatever the appended columns hold
        x_ft = torch.cat([head, ft.read(env)], -1)
        x_ft[:, prefix + 31:] = torch.randn(n, ft.output_dim - 31) * 3
        torch.testing.assert_close(dyn(x_ft), src_dyn(x_src), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(critic(x_ft), src_critic(x_src), rtol=1e-5, atol=1e-6)

    # the optimisers' moments of the widened layer: zero-padded
    opt = torch.optim.Adam(critic.parameters())
    small = build_mlp(prefix + 31, (16,), 1)
    opt_small = torch.optim.Adam(small.parameters())
    small(torch.randn(2, prefix + 31)).sum().backward()
    opt_small.step()
    opt.load_state_dict(opt_small.state_dict())
    _pad_optimizer_state(opt)
    first = opt.state[critic[0].weight]["exp_avg"]
    assert first.shape == critic[0].weight.shape and first[:, prefix + 31:].abs().max() == 0
    torch.testing.assert_close(first[:, :prefix + 31], opt_small.state[small[0].weight]["exp_avg"])


def test_encoder_bias_renormalisation_keeps_the_function_for_the_same_offset():
    from pretraining.supermimic.sim_context import widen_input_weight

    torch.manual_seed(0)
    src_mean, src_scale = torch.zeros(N_JOINTS), torch.full((N_JOINTS,), 0.01)
    tgt_scale = torch.tensor([max(w, 0.01) for w in _OFFSETS])
    w = torch.randn(8, 5 + N_JOINTS)
    w2 = widen_input_weight(w, src_mean, src_scale, src_mean, tgt_scale)
    assert w2.shape == w.shape
    offset = (2 * torch.rand(4, N_JOINTS) - 1) * torch.tensor(_OFFSETS)
    head = torch.randn(4, 5)
    torch.testing.assert_close(torch.cat([head, offset / tgt_scale], -1) @ w2.T,
                               torch.cat([head, offset / src_scale], -1) @ w.T)
    with pytest.raises(ValueError, match="does not start with the source"):
        widen_input_weight(w, src_mean, src_scale, src_mean + 0.1, tgt_scale)


def test_stage2_context_reads_main_and_refuses_what_it_cannot():
    from pretraining.marl.command_runner import _Spec
    from pretraining.marl.marl_source import MarlEnvSource

    n = 3
    delays = torch.tensor([0, 5, 10])
    u = SimpleNamespace(cfg=SimpleNamespace(events=SimpleNamespace(), action_delay_range=(0, 10)), num_envs=n,
                        device="cpu", _delay_steps=delays, encoder_offset=torch.full((n, 30), 0.02))
    source = object.__new__(MarlEnvSource)
    source.u, source.num_envs, source.device, source.joint_ids = u, n, "cpu", list(range(N_JOINTS))
    ctx = SimContext([_Spec("base_com", 3), _Spec("encoder_bias", N_JOINTS), _Spec("action_delay", 1),
                      _Spec("actuator_gain", N_JOINTS), _Spec("torso_mass", 1)],
                     [(-0.05, 0.05)] * 3 + [(-0.02, 0.02)] * N_JOINTS + [(0.0, 10.0)] + [(0.8, 1.2)] * N_JOINTS
                     + [(-2.0, 4.0)])
    out = source.sim_ctx(ctx)
    torch.testing.assert_close(out[:, :3], torch.zeros(n, 3))  # main has no COM event: its true offset, 0
    torch.testing.assert_close(out[:, 3:30], torch.ones(n, N_JOINTS))
    torch.testing.assert_close(out[:, 30], torch.tensor([-1.0, 0.0, 1.0]))  # main's own delay
    # no actuator_scales or add_torso_mass event: nominal gains (1) and torso mass (+0 kg)
    torch.testing.assert_close(out[:, 31:], torch.cat([torch.zeros(n, N_JOINTS), torch.full((n, 1), -1 / 3)], -1))
    with pytest.raises(ValueError, match="foot_friction.*no nominal value"):
        source.sim_ctx(SimContext([_Spec("foot_friction", 1)], [(0.3, 1.2)]))
    with pytest.raises(ValueError, match="wind: the stage-2 source cannot read it"):
        source.sim_ctx(SimContext([_Spec("wind", 1)], [(0.0, 1.0)]))
    u.cfg.events.base_com = object()
    with pytest.raises(ValueError, match="base_com: the env draws a torso COM"):
        source.sim_ctx(ctx)
