"""Stage 2 through the world model: the recurrent command encoder, CommandRunner, train_commands (Tasks 10, 11).

The CPU cases run the runner against a stand-in data source (random states near main's stance, main's observation
widths, a stand-in of the Pulse env's resolved cfg from ``test_main_rewards_obs``) and a tiny stand-in stage-1 run on
stand-in clips. The ``sim`` case runs it against the real LocoManip-Marl-Pulse-v0.
"""

from __future__ import annotations

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

    def sim_ctx(self, dim):
        return torch.zeros(self.num_envs, dim)

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


def make_stage1_run(run: Path, plant: str = "main", clips: bool = True, seed: int = 0) -> Path:
    """A stage-1 run directory: untrained small actors to the policy contract, a small world model of ``plant``,
    stand-in clips."""
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
    dynamics = Dynamics(rep, 27, 31, scale.view(1, 27), default.view(1, 27), hidden_dims=(32, 32))

    class _S:
        def __init__(self, name, dim):
            self.name, self.dim = name, dim

    ctx = SimContext([_S("foot_friction", 1), _S("base_com", 3), _S("encoder_bias", 27)],
                     [(0.3, 1.2)] + [(-0.05, 0.05)] * 3 + [(-0.01, 0.01)] * 27)
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
