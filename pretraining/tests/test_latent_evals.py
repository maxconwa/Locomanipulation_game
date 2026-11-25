"""Latent policies in the evals (evals/isaac/latent.py; evals/mujoco/latent.py and
pretraining/scripts/export_decoders.py): what --latent_run names, the command encoders' loading and their action,
and the Isaac legs scorecard on a stand-in command-encoder run (``sim``, in its own subprocesses).

The CPU cases need no app.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evals.isaac import env as E  # noqa: E402
from evals.isaac import latent as L  # noqa: E402

PULSE = "LocoManip-Marl-Pulse-v0"


def _skrl_run(root: Path, task: str | None, steps=(100, 2000, 300)) -> Path:
    run = root / "2026-11-24_00-00-00_FairB_s42"
    (run / "params").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    if task:
        (run / "params" / "task.txt").write_text(task + "\n")
    for s in steps:
        (run / "checkpoints" / f"agent_{s}.pt").write_bytes(b"")
    return run


def _commands_run(root: Path, meta: dict, bundle: dict | None = None) -> Path:
    run = root / "FairC_s42"
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(meta))
    torch.save(bundle if bundle is not None else {}, run / "commands.pt")
    return run


def test_latent_run_names_a_pulse_skrl_run_or_checkpoint(tmp_path):
    run = _skrl_run(tmp_path, PULSE)
    r = L.resolve_latent_run(run)
    assert (r.kind, r.task, r.step, r.checkpoint.name) == ("skrl", PULSE, 2000, "agent_2000.pt")  # the newest
    r = L.resolve_latent_run(run / "checkpoints" / "agent_300.pt")
    assert (r.kind, r.run_dir, r.step) == ("skrl", run.resolve(), 300)
    assert L.hydra_overrides(r) == []


def test_latent_run_refuses_a_joint_space_run(tmp_path):
    for task in ("LocoManip-Marl-Torso-v0", None):
        root = tmp_path / str(task)
        run = _skrl_run(root, task)
        with pytest.raises(SystemExit, match="--checkpoint"):
            L.resolve_latent_run(run)
    with pytest.raises(SystemExit, match="no such run"):
        L.resolve_latent_run(tmp_path / "nothing")


def test_latent_run_names_a_commands_run_on_its_stage1_without_prior(tmp_path):
    run = _commands_run(tmp_path, {"stage1": "data/pretraining/runs/main_s1_finetune", "plant": "main",
                                   "task": PULSE, "num_envs": 4}, {"steps_per_env": 10})
    r = L.resolve_latent_run(run)
    assert r.kind == "commands" and r.task == PULSE and r.checkpoint == run.resolve() / "commands.pt"
    assert r.stage1 == (REPO / "data/pretraining/runs/main_s1_finetune").resolve()
    assert r.step == 40  # steps_per_env x the run's env count
    assert L.hydra_overrides(r) == [f"env.pulse_stage1_run={r.stage1}", "env.pulse_prior=null"]
    assert L.LatentPolicy(r).trainer == "commands"


def test_latent_run_refuses_a_commands_run_without_stage1_or_of_another_task(tmp_path):
    with pytest.raises(SystemExit, match="stage1"):
        L.resolve_latent_run(_commands_run(tmp_path / "a", {"plant": "main"}))
    with pytest.raises(SystemExit, match="evaluated in"):
        L.resolve_latent_run(_commands_run(tmp_path / "b", {"stage1": "/x", "task": "LocoManip-Marl-Torso-v0"}))


def test_latent_from_z_inverts_the_tanh_inside_the_envs_clamp():
    u = torch.linspace(-4.9, 4.9, 99)
    assert torch.allclose(L.latent_from_z(torch.tanh(u)), u, atol=1e-3)
    edge = L.latent_from_z(torch.tensor([-1.0, 1.0]))
    assert torch.isfinite(edge).all() and edge.abs().max() <= L.LATENT_CLIP + 1e-4


class _FakeActor(nn.Module):
    """A small RecurrentCommandActor stand-in: forward(obs, command, hc (n, 2, h)) -> (tanh, hc)."""

    def __init__(self, obs_dim: int, command_dim: int, h: int = 4):
        super().__init__()
        self.obs_dim, self.h = obs_dim, h
        self.cell = nn.LSTMCell(obs_dim + command_dim, h)
        self.head = nn.Linear(h, 3)

    def initial_state(self, n, device=None):
        return torch.zeros(n, 2, self.h, device=device)

    def forward(self, obs, command, hc=None):
        hc = self.initial_state(obs.shape[0]) if hc is None else hc
        h, c = self.cell(torch.cat([obs, command], -1), (hc[:, 0], hc[:, 1]))
        return torch.tanh(self.head(h)), torch.stack([h, c], 1)


def test_load_command_actors_checks_the_envs_observation_width():
    obs = {"legs": 5, "arms": 7}
    loader = lambda path, device: {a: _FakeActor(obs[a], 3) for a in L.AGENTS}  # noqa: E731
    actors = L.load_command_actors("c.pt", obs, "cpu", loader=loader)
    assert set(actors) == set(L.AGENTS) and not any(p.requires_grad for p in actors["arms"].parameters())
    with pytest.raises(SystemExit, match="arms command encoder reads 7 observation values, this env gives 8"):
        L.load_command_actors("c.pt", {"legs": 5, "arms": 8}, "cpu", loader=loader)


def test_the_real_encoders_round_trip_through_commands_pt(tmp_path):
    """command_runner's own commands.pt layout loads through the evals' loader, with the env's widths."""
    from pretraining.marl.command_actor import RecurrentCommandActor
    from pretraining.marl.command_runner import COMMAND_DIMS

    obs = {"legs": 20, "arms": 30}
    enc = {a: RecurrentCommandActor(obs[a], COMMAND_DIMS[a], 32, 64, (16,)) for a in L.AGENTS}
    torch.save({"encoders": {a: m.state_dict() for a, m in enc.items()}, "obs_dims": obs,
                "command_dims": dict(COMMAND_DIMS), "latent_dim": 32, "lstm": 64, "command_hidden_dims": [16],
                "stage1": "data/pretraining/runs/x", "steps_per_env": 3}, tmp_path / "commands.pt")
    actors = L.load_command_actors(tmp_path / "commands.pt", obs, "cpu")
    x, c = torch.randn(2, 20), torch.randn(2, 3)
    assert torch.equal(actors["legs"](x, c)[0], enc["legs"].eval()(x, c)[0])
    assert L.initial_hc(actors["arms"], 5, "cpu").shape == (5, 2, 64)


class _Term:
    def __init__(self, command):
        self.command = command


class _Commands:
    def __init__(self, n):
        self.terms = {"base_velocity": _Term(torch.randn(n, 3)), "arm_targets": _Term(torch.randn(n, 14))}

    def get_term(self, name):
        return self.terms[name]


def test_command_inputs_are_the_encoders_training_commands():
    """The velocity command as seen; the wrist goals xyz + xyzw (the env's command is w-first)."""
    n = 3
    base = type("Base", (), {"num_envs": n, "command_manager": _Commands(n)})()
    cmds = L.command_inputs(base)
    arm = base.command_manager.terms["arm_targets"].command.view(n, 2, 7)
    assert torch.equal(cmds["legs"], base.command_manager.terms["base_velocity"].command)
    assert cmds["arms"].shape == (n, 14)
    assert torch.equal(cmds["arms"].view(n, 2, 7)[..., 3:6], arm[..., 4:7])  # x y z of the quaternion first
    assert torch.equal(cmds["arms"].view(n, 2, 7)[..., 6], arm[..., 3])      # w last


def test_a_commands_policy_acts_atanh_z_and_resets_only_the_done_envs():
    n = 4
    base = type("Base", (), {"num_envs": n, "command_manager": _Commands(n)})()
    cmds = L.command_inputs(base)
    obs = {"legs": torch.randn(n, 5), "arms": torch.randn(n, 7)}
    torch.manual_seed(0)
    policy = L.LatentPolicy(L.LatentRun("commands", Path("."), Path("c.pt"), PULSE, None))
    policy.base = base
    policy.actors = {a: _FakeActor(obs[a].shape[1], cmds[a].shape[1]).eval() for a in L.AGENTS}
    policy.hc = {a: L.initial_hc(policy.actors[a], n, "cpu") for a in L.AGENTS}
    with torch.no_grad():
        actions = policy.act(obs)
        z, _ = policy.actors["legs"](obs["legs"], cmds["legs"])
    assert torch.allclose(actions["legs"], torch.atanh(z))
    from pretraining.marl.command_actor import latent_to_action
    assert torch.allclose(L.latent_from_z(z), latent_to_action(z))  # as the encoders trained
    assert policy.hc["arms"].abs().sum() > 0
    policy.reset(torch.tensor([False, True, False, True]))
    hc = policy.hc["arms"]
    assert hc[[1, 3]].abs().sum() == 0 and hc[[0, 2]].abs().sum() > 0


def test_two_agent_runs_are_evaluated_in_their_own_task(tmp_path):
    """A run that trained both agents keeps its task given the other body's policy; a one-agent run, or one that did
    not record its task, runs in Marl-Direct as it always has (the joint-space evals unchanged)."""
    for task, expected in ((E.MARL_DIRECT, E.MARL_DIRECT), ("LocoManip-Marl-Torso-v0", "LocoManip-Marl-Torso-v0"),
                           (PULSE, PULSE), ("LocoManip-LegsOnly-v0", E.MARL_DIRECT),
                           ("LocoManip-ArmsOnly-v0", E.MARL_DIRECT), (None, E.MARL_DIRECT)):
        run = tmp_path / str(task)
        (run / "params").mkdir(parents=True)
        if task:
            (run / "params" / "task.txt").write_text(task)
        assert E.two_agent_task(run) == expected


# ---- sim: the legs scorecard on a stand-in command-encoder run ---------------------------------------------------

COMMANDS_STANDIN_SCRIPT = textwrap.dedent("""
    import json, sys
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    import torch
    from isaaclab_tasks.utils import load_cfg_from_registry

    import locomanipulation_game.tasks  # noqa: F401
    from pretraining.marl.command_actor import RecurrentCommandActor
    from pretraining.marl.command_runner import COMMAND_DIMS
    from pretraining.marl.main_obs import MainActorObs

    stage1, out = sys.argv[1], sys.argv[2]
    cfg = load_cfg_from_registry("LocoManip-Marl-Pulse-v0", "env_cfg_entry_point")
    widths = MainActorObs(cfg, "cpu").widths()
    torch.manual_seed(0)
    enc = {a: RecurrentCommandActor(widths[a], COMMAND_DIMS[a], 32, 64, (64,)) for a in ("legs", "arms")}
    torch.save({"encoders": {a: m.state_dict() for a, m in enc.items()}, "obs_dims": widths,
                "command_dims": dict(COMMAND_DIMS), "latent_dim": 32, "lstm": 64, "command_hidden_dims": [64],
                "stage1": stage1, "plant": "main", "steps_per_env": 1}, out + "/commands.pt")
    json.dump({"stage1": stage1, "plant": "main", "task": "LocoManip-Marl-Pulse-v0", "num_envs": 2},
              open(out + "/run.json", "w"))
    print("STANDIN OK", flush=True)
    import os
    os._exit(0)
""")
SCORECARD_SCRIPT = "import runpy, sys; sys.argv = ['get_scorecard', *sys.argv[1:]]; runpy.run_module(" \
                   "'evals.isaac.legs.get_scorecard', run_name='__main__')"
# what the legs scorecard writes for an skrl run (res.update in get_scorecard.main, and scorecard.write's)
SKRL_LEGS_KEYS = {"backend", "body", "run", "step", "checkpoint", "task", "trainer", "overrides", "stochastic", "full",
                  "wrist", "other_policy", "conditions", "conditions_hash", "code_commit", "rows", "targets_met",
                  "targets", "coupled_rows"}


@pytest.mark.sim
def test_scorecard_runs_a_c_policy_from_a_standin_run(tmp_path):
    """A train_commands stand-in (untrained encoders on a stand-in stage-1 run of main's plant) through the legs
    scorecard with --latent_run: 2 envs, 1 segment; the json has an skrl run's keys, and says it was a latent run."""
    from test_pulse import _run, _stage1_run

    stage1 = _stage1_run(tmp_path)
    run = tmp_path / "FairC_standin"
    run.mkdir()
    assert "STANDIN OK" in _run(COMMANDS_STANDIN_SCRIPT, stage1, run)
    out = tmp_path / "scorecard"
    _run(SCORECARD_SCRIPT, "--latent_run", run, "--num_envs", 2, "--segments_per_env", 1, "--out", out)
    res = json.loads(Path(str(out) + ".json").read_text())
    assert SKRL_LEGS_KEYS <= set(res)
    assert res["latent"] == "commands" and res["task"] == PULSE and res["trainer"] == "commands"
    assert res["conditions"]["arms"] == "policy" and res["step"] == 2
    assert res["conditions"]["num_envs"] == 2


# ---- export_decoders (Task 13) ------------------------------------------------------------------------------------

def test_exported_decoders_and_prior_equal_the_eager_ones(tmp_path, monkeypatch):
    """export_decoders on a stand-in stage-1 run of main's plant: each TorchScript decoder equals the eager decoder
    (PulseDecoders' step before the tanh and the target: normalised frame, [z, hist]) on 100 random inputs to 1e-5,
    the prior likewise; plant.json is the plant's."""
    from test_pulse import _actors

    from pretraining.marl.actors import LatentPrior
    from pretraining.marl.runner import PROPRIO_DIM
    from pretraining.plants import MAIN
    from pretraining.scripts import export_decoders as X

    monkeypatch.setattr(X, "REPO", tmp_path)
    run = tmp_path / "data" / "stage1"
    run.mkdir(parents=True)
    import dataclasses

    from test_pulse import HIDDEN

    from pretraining.marl.runner import TwoAgentRunnerCfg
    from pretraining.marl.runs import config_from_dict
    runner = dataclasses.asdict(config_from_dict(TwoAgentRunnerCfg, {"actor_hidden_dims": list(HIDDEN)}))
    (run / "run.json").write_text(json.dumps({"stage": "imitation", "plant": "main", "runner": runner}, default=str))
    actors = _actors()
    torch.save({a: m.state_dict() for a, m in actors.items()}, run / "actors.pt")
    priors = {a: LatentPrior(PROPRIO_DIM, 1, (16,), 32) for a in L.AGENTS}
    torch.save({a: p.state_dict() for a, p in priors.items()}, run / "prior.pt")
    (run / "prior.json").write_text(json.dumps({"plant": "main"}))
    out = tmp_path / "logs" / "decoders"
    info = X.export(run, out)
    assert info["plant"] == "main" and info["prior"] == str(run / "prior.pt")
    assert info["default_joint_pos"] == [MAIN.default_joint_pos[j] for j in info["joint_names"]]
    torch.manual_seed(1)
    frame, z = torch.randn(100, PROPRIO_DIM), torch.tanh(torch.randn(100, 32))
    for agent, name in (("legs", "legs_decoder.pt"), ("arms", "upper_decoder.pt")):
        actor = actors[agent].eval()
        with torch.no_grad():
            eager = actor.decoder(torch.cat([z, actor.obs_normalizer(frame.unsqueeze(1)).flatten(1)], -1))
            scripted = torch.jit.load(str(out / name))(z, frame)
        assert torch.allclose(scripted, eager, atol=1e-5)
    with torch.no_grad():
        prior = torch.jit.load(str(out / "prior.pt"))(frame)
        for i, agent in enumerate(L.AGENTS):
            hist = actors[agent].obs_normalizer(frame.unsqueeze(1)).flatten(1)
            assert torch.allclose(prior[i], priors[agent].eval()(hist), atol=1e-5)
    assert X.export(run, tmp_path / "logs" / "noprior", prior=None)["prior"] is None

    # a train_commands run's encoders: scripted equal to eager, the state carried as (n, 2, lstm)
    from pretraining.marl.command_actor import RecurrentCommandActor
    from pretraining.marl.command_runner import COMMAND_DIMS
    obs = {"legs": 20, "arms": 30}
    enc = {a: RecurrentCommandActor(obs[a], COMMAND_DIMS[a], 32, 64, (16,)).eval() for a in L.AGENTS}
    cruns = tmp_path / "data" / "commands"
    cruns.mkdir()
    torch.save({"encoders": {a: m.state_dict() for a, m in enc.items()}, "obs_dims": obs,
                "command_dims": dict(COMMAND_DIMS), "latent_dim": 32, "lstm": 64, "command_hidden_dims": [16],
                "stage1": "data/stage1", "steps_per_env": 3}, cruns / "commands.pt")
    info = X.export(run, tmp_path / "logs" / "c", prior=None, commands=cruns)
    assert info["commands"]["obs_dims"] == obs and info["commands"]["lstm"] == 64
    x, c = torch.randn(100, 30), torch.randn(100, COMMAND_DIMS["arms"])
    hc = torch.randn(100, 2, 64)
    with torch.no_grad():
        z_eager, hc_eager = enc["arms"](x, c, hc)
        z_scripted, hc_scripted = torch.jit.load(str(tmp_path / "logs" / "c" / "arms_commands.pt"))(x, c, hc)
    assert torch.allclose(z_scripted, z_eager, atol=1e-5) and torch.allclose(hc_scripted, hc_eager, atol=1e-5)


def test_export_decoders_writes_only_under_data_or_logs(tmp_path):
    from pretraining.scripts import export_decoders as X

    with pytest.raises(SystemExit, match="AMASS-derived"):
        X.check_out(tmp_path / "decoders")
    assert X.check_out(X.REPO / "logs" / "evals" / "x" / "decoders") == (X.REPO / "logs/evals/x/decoders").resolve()


# ---- sim: the Isaac proprio frame, recorded for the MuJoCo frame's test (evals/mujoco/test_latent.py) ---------------

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pulse_frame.json"
FRAME_FIXTURE_SCRIPT = textwrap.dedent("""
    import json, sys
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_apply_inverse
    from isaaclab_tasks.utils import load_cfg_from_registry

    import locomanipulation_game.tasks  # noqa: F401
    from locomanipulation_game.tasks.locomanip_marl.mdp import pulse_actions
    from pretraining.constants import JOINT_NAMES

    cfg = load_cfg_from_registry("LocoManip-Marl-Pulse-v0", "env_cfg_entry_point")
    cfg.scene.num_envs = 1
    cfg.pulse_stage1_run, cfg.pulse_prior = sys.argv[1], ""
    # no sensor noise, bias, lag or vibration and no encoder offsets: the frame is the state's own
    imu = cfg.lowstate_imu
    imu.quat_bias_deg = {k: (0.0, 0.0) for k in ("roll", "pitch", "yaw")}
    imu.quat_noise_deg, imu.quat_lag_s, imu.gyro_bias, imu.gyro_noise = 0.0, (0.0, 0.0), (0.0, 0.0, 0.0), 0.0
    imu.vibration_per_target_rate = 0.0
    cfg.encoder_offset_rad = None
    pulse_actions.JOINT_POS_NOISE = 0.0
    pulse_actions.joint_vel_noise_std = lambda dq, floor: torch.zeros_like(dq)
    env = gym.make("LocoManip-Marl-Pulse-v0", cfg=cfg)
    base = env.unwrapped
    base.reset()
    zero = {a: torch.zeros(1, 32, device=base.device) for a in ("legs", "arms")}
    with torch.inference_mode():
        for _ in range(3):  # a few steps from the reset pose: the robot near its default pose, moving a little
            base.step(zero)
        shared = base._pulse_shared
        frame = shared.decoders.proprio(shared.sensors())[0]
        d, ids = shared.robot.data, shared.joint_ids
        out = {"root_quat_wxyz": d.root_quat_w[0].tolist(), "root_ang_vel_w": d.root_ang_vel_w[0].tolist(),
               "root_lin_vel_w": d.root_lin_vel_w[0].tolist(), "joint_names": list(JOINT_NAMES),
               "joint_pos": d.joint_pos[0, ids].tolist(), "joint_vel": d.joint_vel[0, ids].tolist(),
               "default_joint_pos": shared.default.tolist(), "frame": frame.tolist()}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print("FRAME OK", flush=True)
    import os
    os._exit(0)
""")


@pytest.mark.sim
def test_record_the_isaac_proprio_frame_fixture(tmp_path):
    """The Pulse env's 87-wide decoder frame (pulse_actions._PulseShared.sensors and the last action) a few steps
    after a reset, noise, biases and lag off, with the state it came from: pretraining/tests/fixtures/pulse_frame.json
    (numbers only), which evals/mujoco/test_latent.py rebuilds in MuJoCo."""
    from test_pulse import _run, _stage1_run

    stage1 = _stage1_run(tmp_path)
    FIXTURE.parent.mkdir(exist_ok=True)
    assert "FRAME OK" in _run(FRAME_FIXTURE_SCRIPT, stage1, FIXTURE)
    rec = json.loads(FIXTURE.read_text())
    assert len(rec["frame"]) == 87 and len(rec["joint_pos"]) == 27
