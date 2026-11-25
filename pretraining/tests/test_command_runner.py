"""Stage 2 through the world model: the recurrent command encoder, CommandRunner, train_commands (Tasks 10, 11).

The CPU cases run the runner against a stand-in data source (random states near main's stance, main's observation
widths, a stand-in of the Pulse env's resolved cfg from ``test_main_rewards_obs``) and a tiny stand-in stage-1 run on
stand-in clips. The ``sim`` case runs it against the real LocoManip-Marl-Pulse-v0.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pytest
import torch

from pretraining.constants import JOINT_NAMES
from pretraining.marl.command_actor import RecurrentCommandActor, latent_to_action
from pretraining.marl.command_law import VelocityLaw
from pretraining.marl.command_runner import (
    COMMAND_DIMS,
    CommandRunner,
    CommandRunnerCfg,
    load_commands,
)
from pretraining.marl.config import ACTION_DIMS, AGENTS
from pretraining.marl.main_obs import MainActorObs
from pretraining.marl.actors import build_imitation_actor
from pretraining.marl.representation import TorsoImuGenCoordRepresentation
from pretraining.marl.runner import IMITATION_GOAL_DIMS, PROPRIO_DIM
from pretraining.plants import MAIN, PRETRAIN
from pretraining.supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
from pretraining.supermimic.learners.wm_learner import WorldModelLearnerCfg
from pretraining.supermimic.models import Dynamics
from pretraining.supermimic.sim_context import SimContext

HIDDEN = (32, 16)
N = 6


# -- the encoder ---------------------------------------------------------------------------------------------------


def test_recurrent_actor_shapes_and_tanh():
    torch.manual_seed(0)
    actor = RecurrentCommandActor(99, 3)
    assert actor.cell.hidden_size == 64 and actor.latent_dim == 32
    obs, cmd = 10.0 * torch.randn(5, 99), torch.randn(5, 3)
    z, hc = actor(obs, cmd)  # no state: zeros
    assert z.shape == (5, 32) and hc.shape == (5, 2, 64)
    assert z.abs().max() < 1.0
    assert hc[:, 1].abs().max() <= 1.0  # main's cell cap
    z2, hc2 = actor(obs, cmd, hc)
    assert not torch.allclose(z, z2)  # it is recurrent: the same input in another state
    z0, _ = actor(obs, cmd, actor.initial_state(5))
    assert torch.allclose(z, z0)
    z2.sum().backward()
    assert all(p.grad is not None for p in actor.parameters())
    # the env's action: z = tanh(u) without a prior, u inside clip_actions
    u = latent_to_action(torch.tensor([0.0, 0.5, 1.0, -1.0]))
    assert torch.allclose(torch.tanh(u[:2]), torch.tensor([0.0, 0.5]))
    assert u.abs().max() <= 5.0 + 1e-4


# -- stand-ins -----------------------------------------------------------------------------------------------------


def _pulse_cfg_standin():
    from test_main_rewards_obs import pulse_cfg_standin

    cfg = pulse_cfg_standin()
    cfg.decimation = MAIN.decimation  # main's plant: 100 Hz
    return cfg


class StandinSource:
    """A data source with the Pulse env's interface (marl_source.MarlEnvSource): states near main's stance that
    drift, main's observation widths, the commands held, an episode ending every so often."""

    def __init__(self, n: int = N, plant=MAIN, seed: int = 0):
        self.device, self.num_envs, self.plant, self.step_dt = "cpu", n, plant, plant.step_dt
        self.env_cfg = _pulse_cfg_standin()
        self.obs_dims = MainActorObs(self.env_cfg, "cpu").widths()
        low, high = zip(*(plant.target_limits[j] for j in JOINT_NAMES))
        self.target_low, self.target_high = torch.tensor(low), torch.tensor(high)
        self.velocity_law = VelocityLaw(((-0.5, 1.0), (-0.3, 0.3), (-1.0, 1.0)), 0.1, 0.1, 0.1, (0.5, 1.0),
                                        (0.2, 0.4), 0.5)
        self.arm_law = None
        self.gen = torch.Generator().manual_seed(seed)
        self.default = torch.tensor([plant.default_joint_pos[j] for j in JOINT_NAMES])
        self.last_action = torch.zeros(n, 27)
        self.time_outs = torch.zeros(n, dtype=torch.bool)
        self.t = torch.zeros(n)
        self.steps = 0

    def _r(self, *s):
        return torch.randn(*s, generator=self.gen)

    def _state(self):
        n = self.num_envs
        pos = torch.tensor([0.0, 0.0, self.plant.default_root_height]) + 0.01 * self._r(n, 3)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4) + 0.01 * self._r(n, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        return torch.cat([pos, quat, self.default + 0.05 * self._r(n, 27), 0.2 * self._r(n, 3), 0.2 * self._r(n, 3),
                          0.3 * self._r(n, 27)], -1)

    def _obs(self):
        return {a: self._r(self.num_envs, d) for a, d in self.obs_dims.items()}

    def _commands(self):
        n = self.num_envs
        arm = torch.cat([torch.tensor([0.3, 0.2, 0.2]).expand(n, 2, 3).clone(),
                         torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 2, 4)], -1)
        vel = self.velocity_law.sample(n, self.gen)
        return {"velocity": vel, "velocity_target": vel.clone(), "arm": arm}

    @property
    def clock(self):
        return self.t * self.step_dt

    def sim_ctx(self, sim_context):
        return torch.zeros(self.num_envs, sim_context.output_dim)

    def reset(self):
        self.t.zero_()
        return self._state(), self._obs(), self._commands()

    def step(self, latents):
        assert set(latents) == set(AGENTS) and all(z.abs().max() < 1.0 for z in latents.values())
        self.steps += 1
        self.last_action = 0.3 * self._r(self.num_envs, 27)
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        dones[self.steps % self.num_envs] = self.steps % 4 == 0
        self.time_outs = dones & (torch.arange(self.num_envs) % 2 == 0)
        self.t = torch.where(dones, torch.zeros_like(self.t), self.t + 1)
        return self._state(), self._obs(), self._commands(), dones


# a main-plant fine-tune's sim context (--fine-tune-from a base run): the base's columns, then main's new ones
_FINE_TUNED = ([("foot_friction", 1), ("base_com", 3), ("encoder_bias", 27), ("action_delay", 1), ("actuator_gain", 27),
                ("actuator_strength", 27), ("torso_mass", 1)],
               [(0.3, 1.2)] + [(-0.05, 0.05)] * 3 + [(-0.02, 0.02)] * 27 + [(0.0, 10.0)] + [(0.8, 1.2)] * 27
               + [(0.85, 1.15)] * 27 + [(-2.0, 4.0)])


def make_stage1_run(run: Path, plant: str = "main", clips: bool = True, seed: int = 0,
                    fine_tuned: bool = False) -> Path:
    """A stage-1 run directory: untrained small actors to the policy contract, a small world model of ``plant``,
    stand-in clips. ``fine_tuned``: its world model takes a fine-tune's 87 sim-context columns, recorded in
    run.json."""
    from standins import write_clips

    torch.manual_seed(seed)
    run.mkdir(parents=True)
    files = write_clips(run.parent / "motions", n=2, frames=60) if clips else []
    meta = {"stage": "imitation", "plant": plant, "motion_files": [str(f) for f in files],
            "runner": {"latent_dim": 32, "actor_hidden_dims": list(HIDDEN), "history_length": 1,
                       "dynamics_hidden_dims": [32, 32]}}
    (run / "run.json").write_text(json.dumps(meta))
    actors = {a: build_imitation_actor(PROPRIO_DIM, ACTION_DIMS[a], IMITATION_GOAL_DIMS[a], 32, HIDDEN, 1)
              for a in AGENTS}
    torch.save({a: m.state_dict() for a, m in actors.items()}, run / "actors.pt")
    p = MAIN if plant == "main" else PRETRAIN
    default = torch.tensor([p.default_joint_pos[j] for j in JOINT_NAMES])
    scale = torch.tensor([p.action_scale[j] for j in JOINT_NAMES])
    rep = TorsoImuGenCoordRepresentation(27, default)
    rep.dt = p.step_dt

    class _S:
        def __init__(self, name, dim):
            self.name, self.dim = name, dim

    if fine_tuned:
        ctx = SimContext([_S(n, d) for n, d in _FINE_TUNED[0]], _FINE_TUNED[1])
        meta["sim_context"] = ctx.layout()
        (run / "run.json").write_text(json.dumps(meta))
    else:
        ctx = SimContext([_S("foot_friction", 1), _S("base_com", 3), _S("encoder_bias", 27)],
                         [(0.3, 1.2)] + [(-0.05, 0.05)] * 3 + [(-0.01, 0.01)] * 27)
    dynamics = Dynamics(rep, 27, ctx.output_dim, scale.view(1, 27), default.view(1, 27), hidden_dims=(32, 32))
    torch.save({"dynamics": dynamics.state_dict(), "sim_context": ctx.state_dict()}, run / "world_model.pt")
    return run


def small_cfg(**kw) -> CommandRunnerCfg:
    kw = {"max_iterations": 2, "max_steps": 0, "save_interval": 1, **kw}
    return CommandRunnerCfg(
        rollout=5, replay_ratio=1.0, buffer_size=16, lstm=16, command_hidden_dims=(32,), critic_hidden_dims=(16,),
        wm=WorldModelLearnerCfg(num_batches=2, batch_size=8, horizon=3),
        policy=ActorCriticPolicyLearnerCfg(batch_size=8, horizon=4, gamma=0.995, lam=0.975), **kw)


# -- the runner ----------------------------------------------------------------------------------------------------


def test_command_runner_refuses_plant_mismatch(tmp_path):
    """Review Focus 2: a stage-1 run of one plant decoded on the other is refused, naming both, before anything is
    loaded (this run has no actors.pt or world_model.pt)."""
    run = tmp_path / "stage1"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"plant": "pretrain"}))
    with pytest.raises(ValueError, match="pretrain plant.*main plant"):
        CommandRunner(StandinSource(), run, small_cfg(), run_dir=tmp_path / "out")
    # without a plant in run.json: pretraining's
    (run / "run.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="pretrain plant"):
        CommandRunner(StandinSource(), run, small_cfg())


def test_command_runner_refuses_a_world_model_of_another_plant(tmp_path):
    run = make_stage1_run(tmp_path / "stage1", plant="pretrain")
    meta = json.loads((run / "run.json").read_text())
    meta["plant"] = "main"  # run.json says main, the world model's action space is pretraining's
    (run / "run.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="world model's action scale or offset"):
        CommandRunner(StandinSource(), run, small_cfg())


def _params(module):
    return [p.detach().clone() for p in module.parameters()]


def _changed(before, module):
    return any(not torch.equal(a, b) for a, b in zip(before, module.parameters()))


def test_command_runner_two_iterations_on_cpu(tmp_path):
    """Two iterations on the stand-in source: the encoders and the world model move, the stage-1 actors do not, the
    step count is two rollouts, the files Task 12 reads are written, and a resume continues the count."""
    run = make_stage1_run(tmp_path / "stage1")
    runner = CommandRunner(StandinSource(), run, small_cfg(), run_dir=tmp_path / "out")
    assert runner.clips is not None and runner.updates_per_iteration == 1
    enc = {a: _params(runner.encoders[a]) for a in AGENTS}
    dec = {a: _params(runner.stage1[a]) for a in AGENTS}
    wm = _params(runner.dynamics)
    out = runner.run({"stage1": str(run), "plant": "main"})
    assert runner.steps_per_env == 2 * runner.cfg.rollout
    for a in AGENTS:
        assert _changed(enc[a], runner.encoders[a]), a
        assert not _changed(dec[a], runner.stage1[a]), a
    assert _changed(wm, runner.dynamics)
    for f in ("run.json", "progress.csv", "commands.pt", "world_model.pt", "checkpoint.pt", "checkpoint.json"):
        assert (out / f).exists(), f
    bundle = torch.load(out / "commands.pt", weights_only=False)
    assert bundle["plant"] == "main" and bundle["stage1"].endswith("stage1") and bundle["steps_per_env"] == 10
    enc_loaded = load_commands(out / "commands.pt", "cpu")
    for a in AGENTS:
        assert enc_loaded[a].command_dim == COMMAND_DIMS[a]
        for p, q in zip(enc_loaded[a].parameters(), runner.encoders[a].parameters()):
            assert torch.equal(p, q)
    rows = (out / "progress.csv").read_text().splitlines()
    assert len(rows) == 3 and "peak_gpu_gb" in rows[0] and "Policy/return_legs" in rows[0]
    meta = json.loads((out / "run.json").read_text())
    assert meta["plant"] == "main" and meta["cfg"]["clip_command_prob"] == 0.25
    # resume: the count carries on to the new cap
    again = CommandRunner(StandinSource(seed=1), run, small_cfg(max_iterations=3), run_dir=out)
    again.load_checkpoint(out / "checkpoint.pt")
    assert again.iteration == 2 and again.steps_per_env == 10
    again.run(resume=True)
    assert again.steps_per_env == 15
    assert json.loads((out / "checkpoint.json").read_text())["steps_per_env"] == 15


def test_command_runner_takes_a_fine_tuned_world_model(tmp_path):
    """A stage-1 fine-tune's world model (87 sim-context columns, run.json's layout): built at its width with its
    normalisation, the source asked for every column each step, and it trains."""
    run = make_stage1_run(tmp_path / "stage1", fine_tuned=True)
    asked = []

    class _Source(StandinSource):
        def sim_ctx(self, sim_context):
            asked.append([s.name for s in sim_context.specs])
            return super().sim_ctx(sim_context)

    runner = CommandRunner(_Source(), run, small_cfg(max_iterations=1), run_dir=tmp_path / "out")
    assert runner.sim_context.output_dim == 87 and runner.sim_context.find("encoder_bias") == slice(4, 31)
    torch.testing.assert_close(runner.sim_context.scale[31:32], torch.tensor([5.0]))
    runner.run({"stage1": str(run), "plant": "main"})
    assert len(asked) == runner.cfg.rollout and asked[0] == [n for n, _ in _FINE_TUNED[0]]
    meta = json.loads((run / "run.json").read_text())
    meta["sim_context"] = meta["sim_context"][:-1]
    (run / "run.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="run.json's sim context is 86 wide, the world model's 87"):
        CommandRunner(StandinSource(), run, small_cfg(), run_dir=tmp_path / "out2")


def test_command_runner_stops_at_max_steps(tmp_path):
    run = make_stage1_run(tmp_path / "stage1")
    runner = CommandRunner(StandinSource(), run, small_cfg(max_steps=12, max_iterations=100, clip_command_prob=0.0),
                           run_dir=tmp_path / "out")
    assert runner.clips is None
    runner.run({})
    assert runner.steps_per_env == 15  # three rollouts of 5: the cap is checked after each iteration


def test_clip_segments_pull_toward_the_stage1_latent(tmp_path):
    """Every row a clip segment: the latent-BC term is on and its value is the weighted mean squared distance."""
    run = make_stage1_run(tmp_path / "stage1")
    runner = CommandRunner(StandinSource(), run, small_cfg(clip_command_prob=1.0), run_dir=tmp_path / "out")
    runner.collect()
    logs = runner.update_encoders()
    for a in AGENTS:
        assert logs[f"Policy/latent_bc_{a}"] > 0.0
    runner.cfg.prior_weight = 0.0
    logs = runner.update_encoders()
    for a in AGENTS:
        assert logs[f"Policy/latent_bc_{a}"] == 0.0


def test_decoders_read_main_joint_noise_and_encoder_offsets_in_imagination(tmp_path):
    """compute_wm_obs adds no noise; the decoders' frame gets the rollout's encoder offsets (the actor observation's
    own draw) and main's joint noise (pulse_actions.py:96-102), gravity and gyro untouched."""
    run = make_stage1_run(tmp_path / "stage1")
    runner = CommandRunner(StandinSource(), run, small_cfg(), run_dir=tmp_path / "out")
    runner.obs_model.reset(4000, runner.gen)
    clean = torch.zeros(4000, 60)
    clean[:, 6 + 27 :] = 2.0  # dq: speed-scaled noise
    noisy = runner.decoder_sensors(clean)
    assert torch.equal(noisy[:, :6], clean[:, :6])
    dq_noise = noisy[:, 33:] - clean[:, 33:]
    q_noise = noisy[:, 6:33] - runner.obs_model.encoder_offset
    assert q_noise.abs().max() <= 0.003 + 1e-6 and q_noise.std() > 0.001
    assert runner.obs_model.encoder_offset.abs().max() > 0.0
    assert dq_noise.std() > 0.5  # 0.6 x |dq| = 1.2 rad/s at 2 rad/s
    # the clean model output itself carries none
    assert runner.rep._proprio_noise_bounds.abs().max() > 0.0  # it exists, but only add_proprio_noise reads it


@pytest.mark.sim
def test_command_runner_two_iterations(sim_app, tmp_path):
    """The runner on LocoManip-Marl-Pulse-v0 (8 envs, no prior) with a stand-in stage-1 run at plant main: two
    iterations, the encoders move, the decoders do not, steps_per_env is two rollouts."""
    import gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry

    import locomanipulation_game.tasks  # noqa: F401
    from pretraining.marl.marl_source import MarlEnvSource

    run = make_stage1_run(tmp_path / "stage1")
    task = "LocoManip-Marl-Pulse-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = 8
    cfg.pulse_stage1_run, cfg.pulse_prior = str(run), ""
    env = gym.make(task, cfg=cfg)
    try:
        source = MarlEnvSource(env)
        assert source.plant is MAIN and source.obs_dims == MainActorObs(env.unwrapped.cfg, "cpu").widths()
        runner = CommandRunner(source, run, small_cfg(), run_dir=tmp_path / "out")
        enc = {a: _params(runner.encoders[a]) for a in AGENTS}
        dec = {a: _params(runner.stage1[a]) for a in AGENTS}
        runner.run({"stage1": str(run), "plant": "main"})
        assert runner.steps_per_env == 2 * runner.cfg.rollout
        for a in AGENTS:
            assert _changed(enc[a], runner.encoders[a]), a
            assert not _changed(dec[a], runner.stage1[a]), a
        assert torch.isfinite(runner.buffer._storage["state"]).all()
    finally:
        env.close()


@pytest.mark.sim
def test_pulse_source_supplies_a_fine_tuned_world_models_context(sim_app, tmp_path):
    """Main's Pulse env supplies every column of a fine-tune's layout from its own draws: the delay is
    env._delay_steps, gains and strength reproduce the actuators' kp and effort limit, the torso mass and friction lie
    in main's ranges, the encoder offsets are env.encoder_offset, the COM 0; and the runner trains on it."""
    import gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry

    import locomanipulation_game.tasks  # noqa: F401
    from pretraining.marl.marl_source import MarlEnvSource

    run = make_stage1_run(tmp_path / "stage1", fine_tuned=True)
    task = "LocoManip-Marl-Pulse-v0"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = 16
    cfg.pulse_stage1_run, cfg.pulse_prior = str(run), ""
    env = gym.make(task, cfg=cfg)
    try:
        source = MarlEnvSource(env)
        source.reset()
        runner = CommandRunner(source, run, small_cfg(max_iterations=1), run_dir=tmp_path / "out")
        ctx = runner.sim_context
        raw = source.sim_ctx(ctx) * ctx.scale + ctx.mean
        u = env.unwrapped
        col = {s.name: ctx.find(s.name) for s in ctx.specs}
        torch.testing.assert_close(raw[:, col["action_delay"]].squeeze(-1), u._delay_steps.float())
        torch.testing.assert_close(raw[:, col["base_com"]], torch.zeros(16, 3, device=u.device))
        torch.testing.assert_close(raw[:, col["encoder_bias"]], u.encoder_offset[:, source.joint_ids])
        gain, strength = raw[:, col["actuator_gain"]], raw[:, col["actuator_strength"]]
        robot = u.scene["robot"]
        for name, (kp, _, limit) in u._actuator_nominal.items():
            act = robot.actuators[name]
            idx = [JOINT_NAMES.index(j) for j in act.joint_names if j in JOINT_NAMES]
            if len(idx) != act.num_joints:
                continue
            torch.testing.assert_close(act.stiffness, kp * gain[:, idx] * strength[:, idx])
            torch.testing.assert_close(act.effort_limit, limit * strength[:, idx])
        mass, friction = raw[:, col["torso_mass"]], raw[:, col["foot_friction"]]
        assert mass.min() >= -2.0 - 1e-4 and mass.max() <= 4.0 + 1e-4
        assert friction.min() >= 0.1 - 1e-5 and friction.max() <= 1.25 + 1e-5
        runner.run({"stage1": str(run), "plant": "main"})
        assert torch.isfinite(runner.buffer._storage["sim_ctx"]).all()
    finally:
        env.close()


# -- train_commands ------------------------------------------------------------------------------------------------


def _args(**kw):
    from pretraining.scripts.train_commands import build_parser

    argv = []
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return build_parser().parse_args(argv)


def test_train_commands_plan_and_resume(tmp_path, monkeypatch):
    from pretraining.scripts.train_commands import UsageError, plan

    monkeypatch.setenv("PRETRAINING_RUNS_DIR", str(tmp_path / "runs"))
    stage1 = make_stage1_run(tmp_path / "stage1")
    p = plan(_args(stage1=stage1, run_name="c1", num_envs=64, seed=7, max_seconds=60))
    assert p["mode"] == "new" and p["run_dir"] == (tmp_path / "runs" / "c1").resolve()
    assert p["num_envs"] == 64 and p["cfg"].seed == 7 and p["plant"] == "main"
    assert p["task"] == "LocoManip-Marl-Pulse-v0" and p["cfg"].max_steps == 40032
    assert p["cfg"].wm.horizon == 16 and p["cfg"].policy.horizon == 64
    assert p["cfg"].clip_command_prob == 0.25 and p["cfg"].prior_weight == 0.1
    p = plan(_args(stage1=stage1, run_name="c2", wm_batch_size=256, policy_batch_size=32, policy_horizon=32,
                   max_steps=4800))
    assert (p["cfg"].wm.batch_size, p["cfg"].policy.batch_size, p["cfg"].policy.horizon) == (256, 32, 32)
    assert p["cfg"].max_steps == 4800 and p["num_envs"] == 4096
    # an existing directory is refused
    (tmp_path / "runs" / "c1").mkdir(parents=True)
    with pytest.raises(UsageError, match="exists"):
        plan(_args(stage1=stage1, run_name="c1"))
    with pytest.raises(UsageError, match="--stage1"):
        plan(_args(run_name="c3"))
    # resume reads run.json
    run_dir = tmp_path / "runs" / "c1"
    cfg = dataclasses.asdict(small_cfg(max_steps=500))
    (run_dir / "run.json").write_text(json.dumps({"stage": "commands", "stage1": str(stage1), "plant": "main",
                                                  "task": "LocoManip-Marl-Pulse-v0", "num_envs": 32, "seed": 3,
                                                  "cfg": cfg}))
    with pytest.raises(UsageError, match="checkpoint"):
        plan(_args(resume=run_dir))
    (run_dir / "checkpoint.pt").write_bytes(b"")
    (run_dir / "checkpoint.json").write_text(json.dumps({"iteration": 4, "steps_per_env": 25, "wall_s": 9.0}))
    p = plan(_args(resume=run_dir, max_seconds=30))
    assert p["mode"] == "resume" and p["run_dir"] == run_dir.resolve() and p["num_envs"] == 32
    assert p["stage1"] == stage1 and p["cfg"].rollout == 5 and p["cfg"].max_steps == 500
    assert p["cfg"].wm.horizon == 3 and p["max_seconds"] == 30
    p = plan(_args(resume=run_dir, max_steps=900))
    assert p["cfg"].max_steps == 900
    with pytest.raises(UsageError, match="cannot be given with --resume"):
        plan(_args(resume=run_dir, num_envs=8))
    assert isinstance(_args(), argparse.Namespace)
