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
    assert "foot_friction, base_com held at the source run's mean" in capsys.readouterr().out


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
