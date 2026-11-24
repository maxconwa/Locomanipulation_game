"""PULSE on the cooperative task: the latent prior and fit_prior.py ported from h12_train, the decoders
(``marl/pulse.py``) and main's ``LocoManip-Marl-Pulse-v0`` (``locomanip_marl/marl_pulse_env_cfg.py``,
``mdp/pulse_actions.py``).

The CPU cases need no app. The ``sim`` ones: the env cfgs (the Pulse task's settings, and the guards' leaving the
existing tasks' cfgs as they were) in the session's app; ``fit_prior.py`` on stand-in clips and the Pulse env on its
output, each in its own subprocess (each launches its own app).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import yaml

from pretraining.constants import ACTION_SCALE, DEFAULT_JOINT_POS, JOINT_NAMES, TARGET_LIMITS
from pretraining.marl.actors import LatentPrior, build_imitation_actor, load_latent_prior
from pretraining.marl.config import ACTION_DIMS, AGENTS, JOINT_SLICES
from pretraining.marl.pulse import SENSOR_DIM, PulseDecoders
from pretraining.marl.runner import IMITATION_GOAL_DIMS, PROPRIO_DIM

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "source" / "locomanipulation_game" / "locomanipulation_game" / "tasks" / "locomanip_marl"
SUBPROCESS_TIMEOUT_S = 900
HIDDEN = (64, 32)
N = 6


def _actors(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    actors = {a: build_imitation_actor(PROPRIO_DIM, ACTION_DIMS[a], IMITATION_GOAL_DIMS[a], 32, HIDDEN, 1) for a in AGENTS}
    for actor in actors.values():  # a normaliser that is not the identity, as a trained one
        actor.obs_normalizer._mean.uniform_(-0.5, 0.5)
        actor.obs_normalizer._var.uniform_(0.5, 2.0)
        actor.obs_normalizer._std.copy_(actor.obs_normalizer._var.sqrt())
    return actors


def _expected(actors, latents, frame, priors=None):
    """The decoding written out (h12_train's CommandActor.latent / decode), on one (N, 87) frame."""
    out, raw = {}, {}
    default = torch.tensor([DEFAULT_JOINT_POS[j] for j in JOINT_NAMES])
    scale = torch.tensor([ACTION_SCALE[j] for j in JOINT_NAMES])
    low = torch.tensor([TARGET_LIMITS[j][0] for j in JOINT_NAMES])
    high = torch.tensor([TARGET_LIMITS[j][1] for j in JOINT_NAMES])
    with torch.no_grad():
        for a in AGENTS:
            hist = actors[a].obs_normalizer(frame.unsqueeze(1)).flatten(1)
            u = latents[a] + (priors[a](hist) if priors else 0.0)
            raw[a] = actors[a].decoder(torch.cat([torch.tanh(u), hist], -1))
            s = JOINT_SLICES[a]
            out[a] = (default[s] + scale[s] * raw[a]).clamp(low[s], high[s])
    return out, torch.cat([raw[a] for a in AGENTS], -1)


def _latents(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {a: torch.randn(N, 32, generator=g) for a in AGENTS}


def test_decoders_read_one_frame_and_move_the_last_action_on_after_both():
    actors = _actors()
    dec = PulseDecoders(actors, N, "cpu")
    sensors = torch.randn(N, SENSOR_DIM)
    targets = dec.step(_latents(1), sensors)
    frame0 = torch.cat([sensors, torch.zeros(N, 27)], -1)  # zero last action at the start
    want, raw = _expected(actors, _latents(1), frame0)
    for a in AGENTS:
        assert targets[a].shape == (N, ACTION_DIMS[a])
        torch.testing.assert_close(targets[a], want[a])
    torch.testing.assert_close(dec.last_action, raw)  # both agents' raw outputs, Unitree motor order
    # the next step: both decoders read the last action of the step before (neither the other's new one)
    sensors2 = torch.randn(N, SENSOR_DIM)
    targets2 = dec.step(_latents(2), sensors2)
    want2, raw2 = _expected(actors, _latents(2), torch.cat([sensors2, raw], -1))
    for a in AGENTS:
        torch.testing.assert_close(targets2[a], want2[a])
    torch.testing.assert_close(dec.last_action, raw2)


def test_agent_order_does_not_matter():
    actors = _actors()
    sensors = torch.randn(N, SENSOR_DIM)
    forward, backward = PulseDecoders(actors, N, "cpu"), PulseDecoders(actors, N, "cpu")
    lat = _latents(3)
    a = forward.step({"legs": lat["legs"], "arms": lat["arms"]}, sensors)
    b = backward.step({"arms": lat["arms"], "legs": lat["legs"]}, sensors)
    for agent in AGENTS:
        torch.testing.assert_close(a[agent], b[agent])
    torch.testing.assert_close(forward.last_action, backward.last_action)


def test_reset_zeroes_only_the_reset_envs_last_action():
    dec = PulseDecoders(_actors(), N, "cpu")
    dec.step(_latents(4), torch.randn(N, SENSOR_DIM))
    before = dec.last_action.clone()
    dec.reset(torch.tensor([1, 4]))
    assert dec.last_action[[1, 4]].abs().sum() == 0
    keep = [0, 2, 3, 5]
    torch.testing.assert_close(dec.last_action[keep], before[keep])
    dec.reset(None)
    assert dec.last_action.abs().sum() == 0


def test_targets_stay_within_target_limits_and_the_given_limits():
    actors = _actors()
    tight = {"left_knee_joint": (0.5, 0.7), "torso_joint": (-0.01, 0.01), "left_hip_yaw_joint": (-10.0, 10.0)}
    dec = PulseDecoders(actors, N, "cpu", limits=tight)
    for seed in range(5):
        targets = dec.step({a: 50.0 * v for a, v in _latents(seed).items()}, 10.0 * torch.randn(N, SENSOR_DIM))
        full = torch.cat([targets[a] for a in AGENTS], -1)
        for i, j in enumerate(JOINT_NAMES):
            lo, hi = TARGET_LIMITS[j]
            if j in tight:
                lo, hi = max(lo, tight[j][0]), min(hi, tight[j][1])
            assert (full[:, i] >= lo - 1e-6).all() and (full[:, i] <= hi + 1e-6).all(), j
    with pytest.raises(ValueError, match="empty"):
        PulseDecoders(actors, N, "cpu", limits={"left_knee_joint": (3.0, 4.0)})


def test_the_prior_adds_to_the_latent_before_the_tanh():
    actors = _actors()
    priors = {a: LatentPrior(PROPRIO_DIM, 1, (16,), 32) for a in AGENTS}
    sensors = torch.randn(N, SENSOR_DIM)
    dec = PulseDecoders(actors, N, "cpu", priors=priors)
    targets = dec.step(_latents(5), sensors)
    want, _ = _expected(actors, _latents(5), torch.cat([sensors, torch.zeros(N, 27)], -1), priors)
    for a in AGENTS:
        torch.testing.assert_close(targets[a], want[a])


def test_decoders_refuse_a_longer_history_or_a_missing_agent():
    actors = _actors()
    longer = dict(actors, arms=build_imitation_actor(PROPRIO_DIM, 15, IMITATION_GOAL_DIMS["arms"], 32, HIDDEN, 2))
    with pytest.raises(ValueError, match="2 frames"):
        PulseDecoders(longer, N, "cpu")
    with pytest.raises(ValueError, match="per agent"):
        PulseDecoders({"legs": actors["legs"]}, N, "cpu")


def test_latent_prior_mask_and_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    hidden = LatentPrior(PROPRIO_DIM, 1, (16, 8), 32, action_dim=27, hide_last_action=True)
    assert hidden.mask[:60].eq(1).all() and hidden.mask[60:].eq(0).all()
    x = torch.randn(3, PROPRIO_DIM)
    y = x.clone()
    y[:, 60:] = torch.randn(3, 27)
    torch.testing.assert_close(hidden(x), hidden(y))  # the last action cannot reach it
    priors = {"legs": hidden, "arms": LatentPrior(PROPRIO_DIM, 1, (24,), 32)}
    torch.save({a: p.state_dict() for a, p in priors.items()}, tmp_path / "prior.pt")
    loaded = load_latent_prior(tmp_path / "prior.pt", "cpu")
    for a in AGENTS:
        torch.testing.assert_close(loaded[a](x), priors[a](x))
        assert not any(p.requires_grad for p in loaded[a].parameters()) and not loaded[a].training


def test_fit_prior_arguments_and_report():
    from pretraining.scripts.fit_prior import build_parser, explained

    args = build_parser().parse_args(["data/pretraining/runs/x"])
    assert args.out is None and args.num_envs == 4096 and args.iterations == 6000 and args.hidden_dims == [512, 256]
    assert not args.hide_last_action
    r = explained([1.0] * 100 + [0.1] * 500, [2.0] * 600)
    assert r["mse"] == pytest.approx(0.1) and r["latent_variance"] == pytest.approx(2.0)
    assert r["explained"] == pytest.approx(0.95)


def test_pulse_agent_cfg_is_the_mappo_yaml_at_discount_0_99():
    spec = importlib.util.spec_from_file_location("_agents", TASKS / "agents" / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # by path: the package's __init__ needs Isaac
    cfg = module.skrl_mappo_pulse_cfg()
    base = yaml.safe_load((TASKS / "agents" / "skrl_mappo_cfg.yaml").read_text())
    assert cfg["agent"].pop("discount_factor") == 0.99
    base["agent"].pop("discount_factor")
    assert cfg == base


# --- with the app -----------------------------------------------------------------------------------------------------


@pytest.mark.sim
def test_existing_tasks_cfgs_keep_their_action_limits_and_arms_settings(sim_app):
    """The cfg guards (sync_derived's loop over the action terms, ArmsOnly's arm_pos check) leave the existing tasks
    as they were: both joint-space terms get GOLEM's envelope (one dict, as before), ArmsOnly's settings apply."""
    from locomanipulation_game.tasks.locomanip_marl.arms_only_env_cfg import ArmsOnlyEnvCfg
    from locomanipulation_game.tasks.locomanip_marl.golem_safety import position_envelope
    from locomanipulation_game.tasks.locomanip_marl.legs_only_env_cfg import LegsOnlyEnvCfg
    from locomanipulation_game.tasks.locomanip_marl.marl_direct_env_cfg import MarlDirectEnvCfg
    from locomanipulation_game.tasks.locomanip_marl.mdp import IKPostureArmActionCfg

    for cls in (MarlDirectEnvCfg, LegsOnlyEnvCfg, ArmsOnlyEnvCfg):
        cfg = cls()
        envelope = {n: list(b) for n, b in position_envelope(cfg.golem_estop).items()}
        assert cfg.actions.joint_pos.joint_limits == envelope, cls.__name__
        assert cfg.actions.arm_pos.joint_limits is cfg.actions.joint_pos.joint_limits, cls.__name__
        assert cfg.agent_action_terms == {"legs": ["joint_pos"], "arms": ["arm_pos"]}
    for cls in (MarlDirectEnvCfg, ArmsOnlyEnvCfg):
        cfg = cls()
        assert isinstance(cfg.actions.arm_pos, IKPostureArmActionCfg)
        assert cfg.actions.arm_pos.residual_integrate == cfg.arms_integrate
        assert cfg.policy_initial_std == {"arms": 0.1} and cfg.rewards.arms.left_ee_pos_k20.weight == 1.0
        assert not hasattr(cfg.events, "joint_ranges")


@pytest.mark.sim
def test_pulse_cfg(sim_app):
    from isaaclab.actuators import IdealPDActuatorCfg

    from locomanipulation_game.tasks.locomanip_marl.golem_safety import position_envelope
    from locomanipulation_game.tasks.locomanip_marl.locomanip_marl_env_cfg import ACTION_DELAY_RANGE
    from locomanipulation_game.tasks.locomanip_marl.marl_pulse_env_cfg import (
        PULSE_REMOVED_TERMS,
        PULSE_ZEROED_TERMS,
        MarlPulseEnvCfg,
    )
    from pretraining import constants

    cfg = MarlPulseEnvCfg()
    assert cfg.decimation == 10 and cfg.sim.dt == 0.002 and cfg.sim.render_interval == 10
    assert cfg.agent_action_terms == {"legs": ["legs_latent"], "arms": ["arms_latent"]}
    assert [n for n, t in vars(cfg.actions).items() if t is not None] == ["legs_latent", "arms_latent"]
    run = REPO / "data" / "pretraining" / "runs" / "isaac_s1_golem_aware"
    for term in (cfg.actions.legs_latent, cfg.actions.arms_latent):
        assert term.stage1_run == str(run) and term.prior == str(run / "prior.pt")
        assert term.joint_limits == {n: list(b) for n, b in position_envelope(cfg.golem_estop).items()}
    cfg.pulse_prior = None
    cfg.sync_derived()
    assert cfg.actions.legs_latent.prior is None and cfg.actions.arms_latent.prior is None
    cfg.pulse_prior, cfg.pulse_stage1_run = "/abs/prior.pt", "data/other"
    cfg.sync_derived()
    assert cfg.actions.arms_latent.prior == "/abs/prior.pt" and cfg.actions.arms_latent.stage1_run == str(REPO / "data/other")
    # the policies
    assert cfg.clip_actions == 5.0
    assert cfg.policy_output_init_scale == {"legs": 0.01, "arms": 0.01}
    assert cfg.policy_initial_std == {"legs": 0.1, "arms": 0.1}
    assert cfg.policy_log_std_bounds == {a: (math.log(0.03), math.log(0.5)) for a in ("legs", "arms")}
    # the observations' applied action and the rewards
    obs = cfg.observations
    assert obs.legs.actions.params["action_name"] == "legs_latent" == obs.critic.legs_actions.params["action_name"]
    assert obs.arms.actions.params["action_name"] == "arms_latent" == obs.critic.arms_actions.params["action_name"]
    for name in PULSE_ZEROED_TERMS:
        assert getattr(cfg.rewards.legs, name).weight == 0.0, name
    for name in PULSE_REMOVED_TERMS:
        assert getattr(cfg.rewards.legs, name) is None, name
    assert cfg.rewards.legs.action_rate.params["action_name"] == "legs_latent" and cfg.rewards.legs.action_rate.weight < 0
    assert cfg.rewards.arms.action_rate.params["action_name"] == "arms_latent" and cfg.rewards.arms.action_rate.weight < 0
    for name in ("track_lin_vel_xy", "track_ang_vel_z", "golem_position", "knee_valgus", "arms_goal_reached"):
        assert getattr(cfg.rewards.legs, name).weight != 0.0, name
    for name in ("alive", "hold_still", "target_rate", "goal_reached", "legs_track_lin_vel_xy", "left_ee_pos_k20"):
        assert getattr(cfg.rewards.arms, name).weight != 0.0, name
    assert cfg.rewards.arms.goal_reached.weight == pytest.approx(cfg.goal_bonus / 0.02)
    # main's randomisation and sensing
    assert cfg.events.prepose is not None and cfg.events.actuator_scales is not None
    assert cfg.action_delay_range == ACTION_DELAY_RANGE and cfg.encoder_offset_rad and cfg.gyro_lowpass_hz > 0
    assert cfg.warm_start_deadline_steps == 1
    # the robot: explicit PD at pretraining's values, its default pose, its joint ranges at start-up
    robot = cfg.scene.robot
    assert not robot.spawn.articulation_props.fix_root_link
    assert robot.init_state.pos[2] == constants.DEFAULT_ROOT_HEIGHT
    assert {j: robot.init_state.joint_pos[j] for j in JOINT_NAMES} == dict(constants.DEFAULT_JOINT_POS)
    seen = []
    for name, act in robot.actuators.items():
        if name == "grippers":
            continue
        assert isinstance(act, IdealPDActuatorCfg), name
        for j in act.joint_names_expr:
            assert (act.stiffness[j], act.damping[j], act.effort_limit[j], act.effort_limit_sim[j], act.armature[j]) == (
                constants.KP[j], constants.KD[j], constants.EFFORT[j], constants.EFFORT[j], constants.ARMATURE[j])
            seen.append(j)
    assert sorted(seen) == sorted(JOINT_NAMES)
    assert cfg.events.joint_ranges.mode == "startup"
    assert cfg.events.joint_ranges.params["ranges"] == dict(constants.JOINT_RANGES)


def _run(script: str, *args) -> str:
    """Run ``script`` (python source) in a fresh interpreter from the repository root; kill it on timeout."""
    env = dict(os.environ, WANDB_MODE="disabled", PRETRAINING_CUDA_GRAPHS="0")
    proc = subprocess.Popen([sys.executable, "-c", script, *map(str, args)], cwd=REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=SUBPROCESS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        pytest.fail(f"timed out after {SUBPROCESS_TIMEOUT_S} s:\n{out[-4000:]}")
    assert proc.returncode == 0, out[-6000:]
    return out


def _stage1_run(tmp_path) -> Path:
    """A stage-1 run directory on stand-in clips: untrained small actors to the policy contract, small runner sizes."""
    from pretraining.marl.runner import TwoAgentRunnerCfg
    from pretraining.marl.runs import config_from_dict
    from standins import write_clips

    files = write_clips(tmp_path / "motions", n=3, frames=120)
    runner = config_from_dict(TwoAgentRunnerCfg, {
        "buffer_size": 50, "actor_hidden_dims": list(HIDDEN), "critic_hidden_dims": [32],
        "dynamics_hidden_dims": [64, 64], "wm": {"num_batches": 1, "batch_size": 64, "horizon": 2},
        "policy": {"batch_size": 32, "horizon": 4}})
    run = tmp_path / "stage1"
    run.mkdir()
    meta = {"stage": "imitation", "runner": dataclasses.asdict(runner), "motion_files": [str(f) for f in files]}
    (run / "run.json").write_text(json.dumps(meta, default=str))
    torch.save({a: m.state_dict() for a, m in _actors().items()}, run / "actors.pt")
    return run


PULSE_ENV_SCRIPT = textwrap.dedent("""
    import sys
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
    import gymnasium as gym
    import torch
    from isaaclab.actuators import IdealPDActuator
    from isaaclab_tasks.utils import load_cfg_from_registry

    import locomanipulation_game.tasks  # noqa: F401
    from pretraining import constants
    from pretraining.marl.config import AGENTS, JOINT_SLICES

    TASK = "LocoManip-Marl-Pulse-v0"
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = 4
    cfg.pulse_stage1_run, cfg.pulse_prior = sys.argv[1], sys.argv[2]
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        env.reset()
        assert env.cfg.action_spaces == {"legs": 32, "arms": 32}, env.cfg.action_spaces
        assert abs(env.step_dt - 0.02) < 1e-9
        robot = env.scene["robot"]
        shared = env._pulse_shared
        assert shared.decoders.priors is not None
        names = list(constants.JOINT_NAMES)
        ids = robot.find_joints(names, preserve_order=True)[0]
        terms = {a: env.action_manager.get_term(f"{a}_latent") for a in AGENTS}
        for a, term in terms.items():
            assert [robot.joint_names[i] for i in term._joint_ids] == names[JOINT_SLICES[a]], a
        assert env._legs_term is terms["legs"]
        # pretraining's robot: default pose, joint ranges (the start-up event), nominal gains (GOLEM's torque check)
        default = torch.tensor([constants.DEFAULT_JOINT_POS[j] for j in names], device=env.device)
        assert torch.allclose(robot.data.default_joint_pos[:, ids], default)
        ranges = torch.tensor([constants.JOINT_RANGES[j] for j in names], device=env.device)
        assert torch.allclose(robot.data.joint_pos_limits[:, ids], ranges, atol=1e-5)
        kp = torch.tensor([constants.KP[j] for j in names], device=env.device)
        assert torch.allclose(robot.data.default_joint_stiffness[:, ids], kp)
        # actuator_scales drew per-env gains on every explicit PD group
        pd = [act for act in robot.actuators.values() if isinstance(act, IdealPDActuator)]
        assert sum(act.num_joints for act in pd) == 27
        nominal = env._actuator_nominal
        assert any(not torch.allclose(act.stiffness, nominal[n][0]) for n, act in robot.actuators.items() if n in nominal)
        # the delay histories are the latent terms' joints
        assert [h.shape[-1] for h in env._target_history] == [12, 15]

        # one frame per step, read before either decodes; the last action moves on after both
        frames = []
        sensors = shared.sensors
        shared.sensors = lambda: frames.append(sensors()) or frames[-1]
        before = shared.decoders.last_action.clone()
        actions = {a: 0.3 * torch.randn(4, 32, device=env.device) for a in AGENTS}
        env.step(actions)
        assert len(frames) == 1
        frame = torch.cat([frames[0], before], -1)
        with torch.no_grad():
            raw = []
            for a in AGENTS:
                actor = shared.decoders.actors[a]
                hist = actor.obs_normalizer(frame.unsqueeze(1)).flatten(1)
                z = torch.tanh(actions[a].clamp(-5, 5) + shared.decoders.priors[a](hist))
                raw.append(actor.decoder(torch.cat([z, hist], -1)))
                s = JOINT_SLICES[a]
                want = (shared.decoders.default[s] + shared.decoders.scale[s] * raw[-1]).clamp(
                    shared.decoders.low[s], shared.decoders.high[s])
                assert torch.allclose(terms[a]._applied, want, atol=1e-5), a
        assert torch.allclose(shared.decoders.last_action, torch.cat(raw, -1), atol=1e-5)
        # the order the terms are called in does not matter
        fixed = frames[0].clone()
        shared.sensors = lambda: fixed
        saved = shared.decoders.last_action.clone()
        u = {a: torch.randn(4, 32, device=env.device) for a in AGENTS}
        out = {}
        for order in (AGENTS, AGENTS[::-1]):
            shared.decoders.last_action = saved.clone()
            for a in order:
                terms[a].process_actions(u[a])
            out[order] = {a: terms[a]._applied.clone() for a in AGENTS}
        for a in AGENTS:
            assert torch.equal(out[AGENTS][a], out[AGENTS[::-1]][a]), a
        shared.sensors = sensors
        # a few steps without NaN, then a reset zeroes the reset env's last action
        for _ in range(5):
            env.step({a: 0.3 * torch.randn(4, 32, device=env.device) for a in AGENTS})
        assert torch.isfinite(robot.data.joint_pos).all()
        assert shared.decoders.last_action.abs().sum() > 0
        env._reset_idx(torch.tensor([2], device=env.device))
        assert shared.decoders.last_action[2].abs().sum() == 0
        assert shared.decoders.last_action[[0, 1, 3]].abs().sum() > 0
        print("PULSE ENV OK")
    finally:
        env.close()
        app.close()
""")

FIT_PRIOR_SCRIPT = "import runpy, sys; sys.argv = ['fit_prior', *sys.argv[1:]]; runpy.run_module(" \
                   "'pretraining.scripts.fit_prior', run_name='__main__')"


@pytest.mark.sim
def test_fit_prior_on_standins_then_the_pulse_env_decodes_with_it(tmp_path):
    """fit_prior.py (3 iterations, 8 envs) writes prior.pt beside the stage-1 actors and prior.json with each agent's
    explained variance; the Pulse env then builds on that run and prior and decodes as PulseDecoders does."""
    run = _stage1_run(tmp_path)
    out = _run(FIT_PRIOR_SCRIPT, run, "--num-envs", 8, "--iterations", 3, "--hidden-dims", 16)
    assert "[fit_prior] 0:" in out
    report = json.loads((run / "prior.json").read_text())
    assert report["iterations"] == 3 and report["hidden_dims"] == [16]
    for a in AGENTS:
        assert set(report[a]) == {"mse", "latent_variance", "explained"}
    priors = load_latent_prior(run / "prior.pt", "cpu")
    assert set(priors) == set(AGENTS) and priors["legs"].net[0].in_features == PROPRIO_DIM
    out = _run(PULSE_ENV_SCRIPT, run, run / "prior.pt")
    assert "PULSE ENV OK" in out
