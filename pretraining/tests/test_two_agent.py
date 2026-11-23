"""The two-agent stage-1 layer ported from h12_train: split, actors, goals, rewards, estop penalties, run files, and
the imitation runner on the stand-in env.

The CPU cases carry over h12_train's ``tests/test_two_agent.py`` and ``tests/test_runs.py`` (branch supermimic,
commit b175f1c) where they apply to stage 1.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from pretraining.constants import ESTOP_LIMITS, JOINT_NAMES, TRACKED_BODIES
from pretraining.kinematics import URDF, ForwardKinematics, UrdfTree
from pretraining.math import matrix_from_quat
from pretraining.motions import MotionLoader
from pretraining.supermimic.utils.helpers import matrix_to_quat
from standins import write_clips

pytestmark = pytest.mark.filterwarnings("ignore:TensorFloat32 tensor cores:UserWarning")  # inductor's TF32 hint

DT = 0.02
REPO = Path(__file__).resolve().parents[2]
AMASS_S1 = REPO / "data" / "runs" / "amass_s1"


@pytest.fixture(scope="module")
def motion(tmp_path_factory):
    fk = ForwardKinematics(UrdfTree.from_file(URDF, JOINT_NAMES), TRACKED_BODIES, "cpu")
    clips = tmp_path_factory.mktemp("clips")
    write_clips(clips, n=2, frames=120)
    return MotionLoader(clips, fk, "cpu")


def _bodies(n: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Random tracked-body positions (n, 14, 3) and rotations (n, 14, 3, 3)."""
    pos = torch.randn(n, len(TRACKED_BODIES), 3)
    quat = torch.nn.functional.normalize(torch.randn(n, len(TRACKED_BODIES), 4), dim=-1)
    return pos, matrix_from_quat(quat)


# ---------------------------------------------------------------------------
# The split, the actors and the goals
# ---------------------------------------------------------------------------


def test_every_joint_is_driven_by_exactly_one_agent():
    from pretraining.marl.config import ACTION_DIMS, AGENTS, JOINT_SLICES

    owned = [n for a in AGENTS for n in JOINT_NAMES[JOINT_SLICES[a]]]
    assert sorted(owned) == sorted(JOINT_NAMES) and len(owned) == len(set(owned)) == 27
    legs, arms = JOINT_NAMES[JOINT_SLICES["legs"]], JOINT_NAMES[JOINT_SLICES["arms"]]
    assert all(("hip" in n or "knee" in n or "ankle" in n) for n in legs)
    assert arms[0] == "torso_joint" and all(("shoulder" in n or "elbow" in n or "wrist" in n) for n in arms[1:])
    assert ACTION_DIMS == {"legs": 12, "arms": 15}


def test_actor_shapes_and_latent_is_tanh_32():
    from pretraining.marl.actors import build_imitation_actor
    from pretraining.marl.runner import TwoAgentRunnerCfg

    cfg = TwoAgentRunnerCfg()
    assert (cfg.latent_dim, cfg.actor_hidden_dims, cfg.critic_hidden_dims, cfg.history_length) == (
        32, (2048, 1024, 512, 256), (256, 128), 1)
    assert cfg.actor_obs == "proprio"
    actor = build_imitation_actor(87, 12, 300, cfg.latent_dim, cfg.actor_hidden_dims, cfg.history_length)
    linears = [m for m in actor.encoders["motion"] if isinstance(m, torch.nn.Linear)]
    assert [m.out_features for m in linears] == [2048, 1024, 512, 256, 32] and linears[0].in_features == 300
    dec = [m for m in actor.decoder if isinstance(m, torch.nn.Linear)]
    assert dec[0].in_features == 32 + 87 and dec[-1].out_features == 12
    td = TensorDict({"history": torch.randn(5, 1, 87), "goal_motion": 100 * torch.randn(5, 300)}, batch_size=[5])
    actor.eval()
    assert actor(td).shape == (5, 12)
    # the decoder reads tanh(encoder(goal)): a saturating goal still gives a latent in (-1, 1)
    seen = {}
    actor.decoder[0].register_forward_hook(lambda m, inp, out: seen.update(x=inp[0]))
    actor(td)
    z = seen["x"][:, :32]
    goal_n = actor.goal_normalizers["motion"](td["goal_motion"])
    torch.testing.assert_close(z, torch.tanh(actor.encoders["motion"](goal_n)))
    assert z.abs().max() < 1.0


def test_goal_dims_split_between_agents_as_h12_train(motion):
    from pretraining.marl.runner import IMITATION_GOAL_DIMS, imitation_goal_dims
    from pretraining.marl.wm_env import ClipTables, arms_goal, legs_goal, reference_aux_state
    from pretraining.supermimic.goal_window import goal_window_indices

    tables = ClipTables(motion)
    runner = SimpleNamespace(device="cpu", motion=motion, tables=tables)
    # SuperMimic's window of 10 frames: legs 12 joints x [pos, vel] + rot6d; arms 15 x [pos, vel] + 2 tool frames x 9
    assert imitation_goal_dims(runner) == IMITATION_GOAL_DIMS == {"legs": 10 * 30, "arms": 10 * 48}
    t = torch.tensor([3, 40, 150])
    idx = goal_window_indices(motion, t)
    w = idx.shape[1]
    lg = legs_goal(motion, t, reference_aux_state(motion, t))
    lc = lg[:, : w * 24].unflatten(-1, (w, 24))
    torch.testing.assert_close(lc[..., :12], motion.joint_pos[idx][..., :12])
    torch.testing.assert_close(lc[..., 12:], motion.joint_vel[idx][..., :12])
    # frame 0's orientation relative to the robot is the identity when the robot is on the reference
    rel = lg[:, w * 24 :].unflatten(-1, (w, 6))
    torch.testing.assert_close(rel[:, 0], torch.tensor([1.0, 0, 0, 1, 0, 0]).expand(3, 6), atol=1e-5, rtol=0)
    ag = arms_goal(tables, t)
    ac = ag[:, : w * 30].unflatten(-1, (w, 30))
    torch.testing.assert_close(ac[..., :15], motion.joint_pos[idx][..., 12:])
    torch.testing.assert_close(ac[..., 15:], motion.joint_vel[idx][..., 12:])
    torch.testing.assert_close(ag[:, w * 30 :], tables.ee[idx].flatten(1))


def test_tool_frames_are_the_grasp_offset_on_the_wrists_in_the_pelvis_frame():
    from pretraining.marl.config import GRASP_OFFSET, PELVIS, WRISTS
    from pretraining.marl.rewards import ee_in_pelvis

    pos, rot = _bodies(3)
    p, r = ee_in_pelvis(pos, rot)
    for k, w in enumerate(WRISTS):
        world = pos[:, w] + rot[:, w] @ torch.tensor(GRASP_OFFSET)
        expect = (rot[:, PELVIS].transpose(-1, -2) @ (world - pos[:, PELVIS]).unsqueeze(-1))[..., 0]
        torch.testing.assert_close(p[:, k], expect, atol=1e-5, rtol=0)
        torch.testing.assert_close(r[:, k], rot[:, PELVIS].transpose(-1, -2) @ rot[:, w], atol=1e-5, rtol=0)


def test_torso_frame_rotation():
    """A pelvis-frame vector turned by the torso yaw reads as the inverse yaw in the torso."""
    from pretraining.marl.representation import rotate_into_torso

    out = rotate_into_torso(torch.tensor([[1.0, 0.0, 0.5]]), torch.tensor([math.pi / 2]))
    torch.testing.assert_close(out, torch.tensor([[0.0, -1.0, 0.5]]), atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Rewards and estop penalties
# ---------------------------------------------------------------------------


def test_imitation_rewards_are_maximal_on_the_reference():
    """Predicted bodies equal to the reference earn every kernel's weight."""
    from pretraining.marl.config import PELVIS, ImitationRewardCfg
    from pretraining.marl.rewards import imitation_rewards, max_step_rewards

    cfg = ImitationRewardCfg()
    pos, rot = _bodies()
    quat = matrix_to_quat(rot)
    vel, ang = torch.randn_like(pos), torch.randn_like(pos)
    aux = torch.cat([pos[:, PELVIS], quat[:, PELVIS]], dim=-1)
    joint, act = torch.zeros(4, 54), torch.zeros(4, 27)
    limits = torch.tensor([[-1.0, 1.0]]).repeat(27, 1)
    r = imitation_rewards(cfg, aux, pos, rot, vel, ang, joint, act, act, pos, quat, vel, ang, limits, DT)
    ceiling = max_step_rewards(cfg, DT)
    # h12_train's ceiling: team 0.5 + 0.5 + 4; legs 5 kernels' weights; arms 3.5
    torch.testing.assert_close(ceiling, DT * torch.tensor([5.0 + 5.0, 3.5 + 5.0]))
    torch.testing.assert_close(r[:, 0] + r[:, 2], ceiling[0].expand(4), atol=1e-4, rtol=0)
    torch.testing.assert_close(r[:, 1] + r[:, 2], ceiling[1].expand(4), atol=1e-4, rtol=0)


def test_estop_penalty_values():
    """h12_train's hand-computed cases: a knee at its estop velocity costs the legs velocity * 0.1; an elbow's torque
    demand at 0.9 of its estop costs nothing, at 1.08 costs torque * 0.18; a hip yaw at its estop position limit costs
    ``position``, half way into the band half of it, outside nothing; the defaults have no position term."""
    from pretraining.marl.config import EstopPenaltyCfg
    from pretraining.marl.rewards import JointLimits, estop_penalties, estop_position_depth, estop_ratios

    limits = JointLimits.for_joints(JOINT_NAMES, "cpu")
    cfg = EstopPenaltyCfg()
    assert (cfg.margin, cfg.velocity, cfg.torque, cfg.position, cfg.position_margin) == (0.9, -5.0, -5.0, 0.0, 0.05)
    q, dq_next = torch.zeros(1, 27), torch.zeros(1, 27)
    knee, elbow = JOINT_NAMES.index("left_knee_joint"), JOINT_NAMES.index("left_elbow_joint")
    dq_next[0, knee] = ESTOP_LIMITS["left_knee_joint"].dq
    target = torch.zeros(1, 27)
    target[0, elbow] = 0.9 * ESTOP_LIMITS["left_elbow_joint"].tau / limits.kp[elbow]
    vel, tau = estop_ratios(limits, q, torch.zeros(1, 27), dq_next, target)
    assert float(vel[0, knee]) == pytest.approx(1.0)
    assert float(tau[0, elbow]) == pytest.approx(0.9)
    pen = estop_penalties(cfg, vel, tau)
    assert float(pen[0, 0]) == pytest.approx(cfg.velocity * 0.1, rel=1e-5)
    assert float(pen[0, 1]) == pytest.approx(0.0, abs=1e-6)
    target[0, elbow] *= 1.2
    _, tau = estop_ratios(limits, q, torch.zeros(1, 27), dq_next, target)
    pen = estop_penalties(cfg, vel, tau)
    assert float(pen[0, 1]) == pytest.approx(cfg.torque * 0.18, rel=1e-4)
    # the damping term: kd dq enters the demand with its sign
    dq = torch.zeros(1, 27)
    dq[0, elbow] = 1.0
    _, tau_d = estop_ratios(limits, q, dq, dq_next, torch.zeros(1, 27))
    assert float(tau_d[0, elbow]) == pytest.approx(float(limits.kd[elbow]) / ESTOP_LIMITS["left_elbow_joint"].tau)

    pos_cfg = EstopPenaltyCfg(position=-5.0, position_margin=0.05)
    hip = JOINT_NAMES.index("left_hip_yaw_joint")
    zero = torch.zeros(1, 27)
    for room, cost in ((0.0, -5.0), (0.025, -2.5), (0.06, 0.0)):
        q = torch.zeros(1, 27)
        q[0, hip] = ESTOP_LIMITS["left_hip_yaw_joint"].q_high - room
        pen = estop_penalties(pos_cfg, zero, zero, estop_position_depth(limits, q, pos_cfg.position_margin))
        assert float(pen[0, 0]) == pytest.approx(cost, abs=1e-4)
        assert float(pen[0, 1]) == pytest.approx(0.0, abs=1e-6)
    off = estop_penalties(EstopPenaltyCfg(), zero, zero, estop_position_depth(limits, q, 0.05))
    assert float(off.abs().sum()) == 0.0


def test_joint_limits_are_the_constants_in_joint_order():
    from pretraining.constants import EFFORT, KD, KP
    from pretraining.marl.rewards import JointLimits

    lim = JointLimits.for_joints(JOINT_NAMES, "cpu")
    for i, j in enumerate(JOINT_NAMES):
        e = ESTOP_LIMITS[j]
        assert (float(lim.kp[i]), float(lim.kd[i]), float(lim.effort[i])) == pytest.approx((KP[j], KD[j], EFFORT[j]))
        assert (float(lim.dq[i]), float(lim.tau[i]), float(lim.q_low[i]), float(lim.q_high[i])) == pytest.approx(
            (e.dq, e.tau, e.q_low, e.q_high))


# ---------------------------------------------------------------------------
# Run files
# ---------------------------------------------------------------------------


def _rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def test_progress_csv_widens_its_header_when_keys_appear(tmp_path):
    from pretraining.marl.runs import ProgressCsv

    p = ProgressCsv(tmp_path / "progress.csv")
    p.append({"iteration": 0, "loss": 1.0})
    p.append({"iteration": 1, "loss": 0.5, "Episode_Termination/fall_height": 3})
    p.append({"iteration": 2, "loss": 0.4})
    header, rows = _rows(tmp_path / "progress.csv")
    assert header == ["iteration", "loss", "Episode_Termination/fall_height"]
    assert [r["Episode_Termination/fall_height"] for r in rows] == ["", "3", ""]
    assert (tmp_path / "progress.csv").read_text().count("iteration") == 1


def test_progress_csv_resume_drops_rows_after_the_checkpoint(tmp_path):
    from pretraining.marl.runs import ProgressCsv

    p = ProgressCsv(tmp_path / "progress.csv")
    for i in range(6):
        p.append({"iteration": i, "loss": float(i)})
    q = ProgressCsv(tmp_path / "progress.csv", keep_through_iteration=3)
    q.append({"iteration": 4, "loss": 40.0})
    header, rows = _rows(tmp_path / "progress.csv")
    assert header == ["iteration", "loss"]
    assert [int(r["iteration"]) for r in rows] == [0, 1, 2, 3, 4]
    assert rows[-1]["loss"] == "40.0"


def test_reward_cfg_round_trips_through_run_json():
    from pretraining.marl.config import EstopPenaltyCfg, ImitationRewardCfg, Kernel
    from pretraining.marl.runs import config_from_dict

    cfg = ImitationRewardCfg(legs_root_pos=Kernel(3.0, 0.3), estop=EstopPenaltyCfg(position=-1.0))
    assert config_from_dict(ImitationRewardCfg, json.loads(json.dumps(dataclasses.asdict(cfg)))) == cfg


def test_runner_cfg_round_trips_through_run_json():
    from pretraining.marl.runner import TwoAgentRunnerCfg
    from pretraining.marl.runs import config_from_dict
    from pretraining.supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg

    cfg = TwoAgentRunnerCfg(actor_hidden_dims=(8, 4), seed=7, policy=ActorCriticPolicyLearnerCfg(batch_size=3))
    assert config_from_dict(TwoAgentRunnerCfg, json.loads(json.dumps(dataclasses.asdict(cfg)))) == cfg


@pytest.mark.skipif(not (AMASS_S1 / "run.json").exists(), reason="no data/runs/amass_s1 on this machine")
def test_h12_train_run_json_rebuilds_our_cfgs():
    """amass_s1's recorded settings rebuild the ported cfgs; its values come through, its reward cfg is the default."""
    from pretraining.marl.config import ImitationRewardCfg
    from pretraining.marl.runner import TwoAgentRunnerCfg
    from pretraining.marl.runs import config_from_dict

    meta = json.loads((AMASS_S1 / "run.json").read_text())
    cfg = config_from_dict(TwoAgentRunnerCfg, meta["runner"])
    assert cfg.wm.batch_size == meta["runner"]["wm"]["batch_size"] and cfg.policy.horizon == 32
    assert cfg.actor_hidden_dims == (2048, 1024, 512, 256) and cfg.return_sampler.memory == 256
    assert config_from_dict(ImitationRewardCfg, meta["rewards"]) == ImitationRewardCfg()


def test_run_json_paths_are_repository_relative_and_resolve_back(tmp_path, monkeypatch):
    from pretraining.marl.runs import REPO_ROOT, default_runs_dir, from_repo_path, to_repo_path

    assert REPO_ROOT == REPO
    inside = REPO_ROOT / "data" / "motions" / "h12" / "walk.npz"
    assert to_repo_path(inside) == "data/motions/h12/walk.npz"
    assert from_repo_path(to_repo_path(inside)) == inside
    assert to_repo_path(tmp_path) == str(tmp_path.resolve())
    assert from_repo_path(str(tmp_path)) == tmp_path
    monkeypatch.delenv("PRETRAINING_RUNS_DIR", raising=False)
    assert default_runs_dir() == str(REPO / "data" / "pretraining" / "runs")
    monkeypatch.setenv("PRETRAINING_RUNS_DIR", str(tmp_path))
    assert default_runs_dir() == str(tmp_path)


def test_keep_last_and_atomic_writes(tmp_path):
    from pretraining.marl.runs import atomic_save, atomic_write_text, keep_last, read_checkpoint_info

    for i in range(5):
        atomic_save({"i": i}, tmp_path / f"it_{i:08d}.pt")
    keep_last(tmp_path, "it_*.pt", 2)
    assert sorted(p.name for p in tmp_path.glob("it_*.pt")) == ["it_00000003.pt", "it_00000004.pt"]
    atomic_write_text(tmp_path / "checkpoint.json", json.dumps({"iteration": 4, "wall_s": 1.5}))
    assert read_checkpoint_info(tmp_path) == {"iteration": 4, "wall_s": 1.5}
    assert not list(tmp_path.glob(".*.tmp"))


# ---------------------------------------------------------------------------
# Loading actors
# ---------------------------------------------------------------------------


def _write_run(run_dir: Path, actors: dict, **runner) -> None:
    from pretraining.marl.runner import TwoAgentRunnerCfg

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.replace(TwoAgentRunnerCfg(), **runner)
    (run_dir / "run.json").write_text(json.dumps({"stage": "imitation", "runner": dataclasses.asdict(cfg)}))
    torch.save({a: m.state_dict() for a, m in actors.items()}, run_dir / "actors.pt")


def test_load_actors_refuses_other_shapes(tmp_path):
    from pretraining.marl.actors import build_imitation_actor
    from pretraining.marl.runner import IMITATION_GOAL_DIMS, load_actors

    small = {"legs": (12, IMITATION_GOAL_DIMS["legs"]), "arms": (15, IMITATION_GOAL_DIMS["arms"])}
    actors = {a: build_imitation_actor(87, n, g, 32, (64, 32), 1) for a, (n, g) in small.items()}
    _write_run(tmp_path / "ok", actors, actor_hidden_dims=(64, 32))
    loaded = load_actors(tmp_path / "ok", "cpu")
    for a in actors:
        assert not loaded[a].training
        for k, v in actors[a].state_dict().items():
            assert torch.equal(loaded[a].state_dict()[k], v), (a, k)
    # an arms actor whose goal is 10 wider than the contract's: refused, naming both shapes
    wrong = dict(actors, arms=build_imitation_actor(87, 15, IMITATION_GOAL_DIMS["arms"] + 10, 32, (64, 32), 1))
    _write_run(tmp_path / "wide", wrong, actor_hidden_dims=(64, 32))
    g = IMITATION_GOAL_DIMS["arms"]
    with pytest.raises(ValueError, match=rf"arms goal_normalizers\.motion\._mean.*\(1, {g + 10}\).*\(1, {g}\)"):
        load_actors(tmp_path / "wide", "cpu")
    # hidden sizes other than run.json's
    _write_run(tmp_path / "hidden", actors)
    with pytest.raises(ValueError, match=r"legs encoders\.motion\.0\.weight.*\(64, 300\).*\(2048, 300\)"):
        load_actors(tmp_path / "hidden", "cpu")


@pytest.mark.skipif(not (AMASS_S1 / "actors.pt").exists(), reason="no data/runs/amass_s1 on this machine")
def test_actors_pt_from_h12_train_loads():
    from pretraining.marl.config import AGENTS
    from pretraining.marl.runner import IMITATION_GOAL_DIMS, load_actors

    actors = load_actors(AMASS_S1, "cpu")
    saved = torch.load(AMASS_S1 / "actors.pt", map_location="cpu", weights_only=False)
    out = {}
    for a in AGENTS:
        assert actors[a].state_dict().keys() == saved[a].keys()
        assert int(actors[a].obs_normalizer.count) > 0  # the trained normalisers came with it
        td = TensorDict(
            {"history": torch.zeros(1, 1, 87), "goal_motion": torch.zeros(1, IMITATION_GOAL_DIMS[a])}, batch_size=[1]
        )
        with torch.no_grad():
            out[a] = actors[a](td)[0]
        assert torch.isfinite(out[a]).all()
    assert out["legs"].shape == (12,) and out["arms"].shape == (15,)


# ---------------------------------------------------------------------------
# The imitation runner on the 16-env stand-in env
# ---------------------------------------------------------------------------

TASK = "LocoManip-Pretrain-Track-v0"
NUM_ENVS = 16


def _small_cfg(runs_dir, **kw):
    """A small runner cfg that updates from the second iteration on and logs every iteration."""
    from pretraining.marl.runner import TwoAgentRunnerCfg
    from pretraining.supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
    from pretraining.supermimic.learners.wm_learner import WorldModelLearnerCfg

    return TwoAgentRunnerCfg(
        runs_dir=str(runs_dir),
        buffer_size=50,
        replay_ratio=64,
        log_interval=1,
        save_interval=1000,
        actor_hidden_dims=(64, 64),
        critic_hidden_dims=(32,),
        dynamics_hidden_dims=(64, 64),
        wm=WorldModelLearnerCfg(num_batches=1, batch_size=16, horizon=2),
        policy=ActorCriticPolicyLearnerCfg(batch_size=32, horizon=4),
        **kw,
    )


@pytest.fixture(scope="module")
def stage1(sim_app, tmp_path_factory):
    """An imitation runner after 3 iterations on the stand-in env, and its run directory."""
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    from isaaclab_tasks.utils import load_cfg_from_registry

    import pretraining.tasks  # noqa: F401  (registers the gym ids)
    from pretraining.marl.runner import ImitationRunner
    from pretraining.supermimic.runners.base_runner import EnvAdapter

    # A failed CUDA-graph capture would break the CUDA RNG for the whole session (learners/learner_utils.py).
    os.environ["PRETRAINING_CUDA_GRAPHS"] = "0"
    os.environ["WANDB_MODE"] = "disabled"
    sim_utils.create_new_stage()
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = NUM_ENVS
    clips = tmp_path_factory.mktemp("sim_clips")
    files = write_clips(clips, n=3, frames=120)
    cfg.commands.motion.motion_file = str(clips)
    cfg.commands.motion.sampling_mode = "external"  # the runner's "return" sampler sets the distribution
    env = gym.make(TASK, cfg=cfg).unwrapped
    adapter = EnvAdapter(env)
    adapter.close = lambda: None  # run() closes its env; the tests below keep using this one
    runs = tmp_path_factory.mktemp("runs")
    run_dir = runs / "s1"
    runner = ImitationRunner(
        adapter, _small_cfg(runs, run_name="s1", max_iterations=3), run_dir=run_dir,
        motion_files=[str(f) for f in files], device=str(env.device),
    )
    log_dir = runner.run()
    yield runner, log_dir, files
    env.close()


@pytest.mark.sim
def test_imitation_runner_writes_h12_train_run_files(stage1):
    runner, log_dir, files = stage1
    for name in ("run.json", "progress.csv", "world_model.pt", "actors.pt", "critics.pt", "checkpoint.pt",
                 "checkpoint.json"):
        assert (log_dir / name).exists(), name
    meta = json.loads((log_dir / "run.json").read_text())
    assert meta["stage"] == "imitation" and meta["num_envs"] == NUM_ENVS
    assert meta["motion_files"] == [str(f.resolve()) for f in files]  # outside the repository: absolute
    assert meta["runner"]["wm"]["horizon"] == 2 and meta["rewards"]["estop"]["margin"] == 0.9
    assert json.loads((log_dir / "checkpoint.json").read_text())["iteration"] == 2
    actors = torch.load(log_dir / "actors.pt", map_location="cpu", weights_only=False)
    assert set(actors) == {"legs", "arms"} and "decoder.0.weight" in actors["legs"]
    assert set(torch.load(log_dir / "world_model.pt", map_location="cpu")) == {"dynamics", "sim_context"}
    state = torch.load(log_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert {"iteration", "wall_s", "world_model", "actors", "critics", "optimizers", "return_sampler"} <= set(state)


@pytest.mark.sim
def test_progress_csv_has_h12_train_columns_and_finite_losses(stage1):
    runner, log_dir, _ = stage1
    header, rows = _rows(log_dir / "progress.csv")
    assert [int(r["iteration"]) for r in rows] == [0, 1, 2]
    trained = [r for r in rows if r.get("Loss/state")]
    assert trained and all(math.isfinite(float(r["Loss/state"])) for r in trained)
    for r in rows:
        for k, v in r.items():
            if k.startswith(("Policy/", "Loss/", "Safety/")) and v not in ("", None):
                assert math.isfinite(float(v)), k
    assert any(r.get("Policy/return_legs") for r in rows) and any(r.get("Policy/return_arms") for r in rows)
    for a in ("legs", "arms"):
        for k in (f"{a}_force_saturated", f"{a}_q_over_estop", f"{a}_q_within_0.05", f"{a}_dq_over_0.9",
                  f"{a}_dq_over_1.0", f"{a}_tau_demand_over_0.9", f"{a}_tau_demand_over_1.0"):
            assert f"Safety/{k}" in header, k
    assert "Metrics/motion/error_body_pos" in header
    if (AMASS_S1 / "progress.csv").exists():
        h12_header, _ = _rows(AMASS_S1 / "progress.csv")
        missing = [k for k in h12_header if k not in header]
        assert missing == [], missing


@pytest.mark.sim
def test_goal_dims_and_proprio_width_on_the_env(stage1):
    from pretraining.marl.runner import IMITATION_GOAL_DIMS, imitation_goal_dims

    runner, _, _ = stage1
    assert imitation_goal_dims(runner) == IMITATION_GOAL_DIMS
    assert runner.obs_dim == 3 + 3 + 27 + 27 + 27


@pytest.mark.sim
def test_actors_read_the_torso_imu_and_the_world_model_synthesises_it(stage1):
    """The representation reproduces the sim's actor frame: torso IMU, biased joint positions, velocities."""
    from pretraining.tasks.events import encoder_bias

    runner, _, _ = stage1
    env = runner.env.unwrapped
    with torch.inference_mode():
        env.reset()
        zero = torch.zeros(NUM_ENVS, 27, device=env.device)
        for _ in range(5):
            runner.env.step(zero + 0.3 * torch.randn_like(zero))
        obs = runner.env.get_observations()
        sim_ctx = runner._sim_context.read(env)
    state = obs["full_state"]
    rep = runner._representation
    rep_state = rep.represent(state)
    synth = rep.compute_wm_obs(rep_state, rep.extract_aux_state(state), rep.joint_state(rep_state), sim_ctx)
    assert encoder_bias(env).abs().max() > 1e-3  # the bias event is on, so the joint check is not vacuous
    torch.testing.assert_close(synth[:, :6], obs["actor"][:, :6], atol=2e-3, rtol=0)
    torch.testing.assert_close(synth[:, 6:33], obs["actor"][:, 6:33], atol=1e-4, rtol=0)
    torch.testing.assert_close(synth[:, 33:60], obs["actor"][:, 33:60], atol=1e-4, rtol=0)


@pytest.mark.sim
def test_world_model_clamps_targets_as_the_sim_does(stage1):
    runner, _, _ = stage1
    term = runner.env.unwrapped.action_manager.get_term("joint_pos")
    action = 40.0 * torch.randn(NUM_ENVS, 27, device=runner.device)
    with torch.inference_mode():
        term.process_actions(action)
        sim_target = term.processed_actions.clone()
    a = action.clone().requires_grad_(True)
    wm_target = runner._dynamics.proc(a)
    torch.testing.assert_close(wm_target.detach(), sim_target, atol=1e-6, rtol=0)
    saturated = (sim_target <= term._clip[:, :, 0] + 1e-6) | (sim_target >= term._clip[:, :, 1] - 1e-6)
    assert saturated.any() and not saturated.all()
    (grad,) = torch.autograd.grad(wm_target.sum(), a)
    assert torch.all(grad[saturated] == 0) and torch.all(grad[~saturated] > 0)


@pytest.mark.sim
def test_a_clip_replayed_in_the_world_model_frame_earns_the_maximum_stage1_rewards(stage1):
    """Bodies built from the reference by the representation earn every kernel and do not fall; one agent's joint
    off the reference lowers only that agent's terms."""
    from pretraining.marl.config import ImitationRewardCfg
    from pretraining.marl.rewards import imitation_rewards, max_step_rewards

    runner, _, _ = stage1
    m = runner.motion
    dev = runner.device
    frames = torch.arange(0, m.time_step_total - 1, 7, device=dev)
    full = torch.cat([m.root_pos_w[frames], m.root_quat_w[frames], m.joint_pos[frames], m.root_lin_vel_w[frames],
                      m.root_ang_vel_w[frames], m.joint_vel[frames]], dim=-1)
    rep = runner._representation
    rep_state, aux = rep.represent(full), rep.extract_aux_state(full)
    tp, tq, tl, ta = m.body_state(frames)
    act = torch.zeros(len(frames), 27, device=dev)
    limits = torch.tensor([[-10.0, 10.0]], device=dev).repeat(27, 1)
    cfg = ImitationRewardCfg()

    def rewards(rs, ax):
        return imitation_rewards(cfg, ax, *rep.world_frame_bodies(ax, rs), rep.joint_state(rs), act, act, tp, tq, tl,
                                 ta, limits, DT)

    r = rewards(rep_state, aux)
    ceiling = max_step_rewards(cfg, DT).to(dev)
    torch.testing.assert_close(r[:, 0] + r[:, 2], ceiling[0].expand(len(frames)), atol=1e-3, rtol=0)
    torch.testing.assert_close(r[:, 1] + r[:, 2], ceiling[1].expand(len(frames)), atol=1e-3, rtol=0)
    assert runner.wm_env._compute_term_prob(aux, tp, tq).max() < 1e-6
    for joint, hit, kept in (("left_elbow_joint", 1, 0), ("left_knee_joint", 0, 1)):
        off = full.clone()
        off[:, 7 + JOINT_NAMES.index(joint)] += 0.5
        r_off = rewards(rep.represent(off), rep.extract_aux_state(off))
        assert (r_off[:, hit] < r[:, hit] - 1e-3).all(), joint
        torch.testing.assert_close(r_off[:, kept], r[:, kept], atol=1e-5, rtol=0)


@pytest.mark.sim
def test_stage1_sim_terminations_equal_the_world_models(stage1):
    from pretraining.marl.wm_env import reference_aux_state

    runner, _, _ = stage1
    terms = runner.env.unwrapped.cfg.terminations
    assert runner.wm_env._anchor_pos_threshold == terms.anchor_pos.params["threshold"]
    assert runner.wm_env._anchor_ori_threshold == terms.anchor_ori.params["threshold"]
    t = torch.tensor([5], device=runner.device)
    tp, tq, _, _ = runner.motion.body_state(t)
    aux = reference_aux_state(runner.motion, t)
    aux[:, 2] -= runner.wm_env._anchor_pos_threshold
    assert float(runner.wm_env._compute_term_prob(aux, tp, tq)) == pytest.approx(0.5, abs=1e-4)


@pytest.mark.sim
def test_policy_update_leaves_the_world_model_bit_identical_and_moves_both_actors(stage1):
    from pretraining.marl.config import AGENTS

    runner, _, _ = stage1
    dyn = {k: v.clone() for k, v in runner._dynamics.state_dict().items()}
    act = {a: [p.detach().clone() for p in runner.actors[a].parameters()] for a in AGENTS}
    runner.learner.update()
    torch.cuda.synchronize()
    for k, v in runner._dynamics.state_dict().items():
        assert torch.equal(v, dyn[k]), k
    for a in AGENTS:
        assert any(not torch.equal(p, q) for p, q in zip(runner.actors[a].parameters(), act[a], strict=True)), a


@pytest.mark.sim
def test_every_stage1_actor_parameter_gets_a_gradient(stage1):
    from pretraining.marl.config import AGENTS
    from pretraining.marl.learner import trainable

    runner, _, _ = stage1
    wm = runner.wm_env
    for m in runner.actors.values():
        m.train()
    obs = wm.reset(wm.sample_initial_states())
    ret = {a: 0.0 for a in AGENTS}
    for _ in range(4):
        actions = torch.cat([runner.actors[a](obs[a]) for a in AGENTS], dim=-1)
        obs, r, _, _ = wm.step(actions)
        ret["legs"] = ret["legs"] + r[:, 0] + r[:, 2]
        ret["arms"] = ret["arms"] + r[:, 1] + r[:, 2]
    for i, a in enumerate(AGENTS):
        loss = -ret[a].mean() + runner.actors[a].auxiliary_loss().mean()
        grads = torch.autograd.grad(loss, trainable(runner.actors[a]), retain_graph=i == 0, allow_unused=True)
        assert all(g is not None for g in grads), a


@pytest.mark.sim
def test_a_world_model_of_another_action_space_is_refused(stage1):
    runner, _, _ = stage1
    bundle = runner._wm.bundle()
    bundle["dynamics"] = dict(bundle["dynamics"])
    bundle["dynamics"]["action_scale"] = torch.full_like(bundle["dynamics"]["action_scale"], 0.5)
    with pytest.raises(ValueError, match="another action space"):
        runner._load_world_model(bundle)


def _reusable(env):
    """Return ``env`` (an EnvAdapter) ready for another runner in this process: the first runner stepped it under
    ``torch.inference_mode`` (as h12_train's does), which leaves Isaac Lab's buffers inference tensors that a reset
    outside inference mode cannot write, so its resets run in inference mode too (the runner only reads their
    observations' widths). A training process builds one runner per env and never needs this."""
    reset = env.reset

    def reset_in_inference_mode():
        with torch.inference_mode():
            return reset()

    env.reset = reset_in_inference_mode
    return env


@pytest.mark.sim
def test_a_new_run_refuses_an_existing_run_directory(stage1):
    from pretraining.marl.runner import ImitationRunner

    runner, log_dir, files = stage1
    again = ImitationRunner(
        _reusable(runner.env), _small_cfg(log_dir.parent, max_iterations=2), run_dir=log_dir,
        motion_files=[str(f) for f in files], device=runner.device,
    )
    with pytest.raises(FileExistsError):
        again.run()


@pytest.mark.sim
def test_resume_restores_everything_and_continues_the_curves(stage1):
    from pretraining.marl.config import AGENTS
    from pretraining.marl.runner import ImitationRunner, resume_info
    from pretraining.marl.runs import read_checkpoint_info

    runner, log_dir, files = stage1
    info = read_checkpoint_info(log_dir)
    meta, start = resume_info(log_dir)
    assert start == info["iteration"] + 1 and meta["stage"] == "imitation"
    saved = torch.load(log_dir / "checkpoint.pt", map_location=runner.device, weights_only=False)
    resumed = ImitationRunner(
        _reusable(runner.env), _small_cfg(log_dir.parent, max_iterations=info["iteration"] + 5, resume_warmup=1),
        run_dir=log_dir, motion_files=[str(f) for f in files], device=runner.device,
    )
    resumed.load_checkpoint(log_dir / "checkpoint.pt")
    assert resumed.start_iteration == start and resumed._wall_offset == info["wall_s"]
    for a in AGENTS:
        for k, v in resumed.actors[a].state_dict().items():
            assert torch.equal(v, saved["actors"][a][k]), (a, k)
    for k, v in resumed._dynamics.state_dict().items():
        assert torch.equal(v, saved["world_model"]["dynamics"][k]), k
    for name, opts in resumed._optimizers().items():
        for opt, sd in zip(opts, saved["optimizers"][name], strict=True):
            assert len(opt.state) == len(sd["state"]) > 0, name
    resumed.run(resume=True)
    lines = (log_dir / "progress.csv").read_text().splitlines()
    assert sum(line.startswith("iteration,") for line in lines) == 1
    _, rows = _rows(log_dir / "progress.csv")
    its = [int(r["iteration"]) for r in rows]
    assert its == sorted(set(its)) and its[-1] == info["iteration"] + 4
    walls = [float(r["wall_s"]) for r in rows]
    assert walls == sorted(walls)
    assert read_checkpoint_info(log_dir)["iteration"] == info["iteration"] + 4
    assert "resumed" in json.loads((log_dir / "run.json").read_text())
