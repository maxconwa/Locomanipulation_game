"""Smoke tests of the scripts on the stand-in env: the cross-check, the replay rollout and training."""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os

import pytest
import torch

from standins import write_clips

pytestmark = pytest.mark.filterwarnings("ignore:TensorFloat32 tensor cores:UserWarning")  # inductor's TF32 hint


def _untrained_run(run_dir):
    """An h12_train-style run directory with untrained (small) actors to the policy contract."""
    from pretraining.marl.actors import build_imitation_actor
    from pretraining.marl.config import ACTION_DIMS, AGENTS
    from pretraining.marl.runner import IMITATION_GOAL_DIMS, PROPRIO_DIM, TwoAgentRunnerCfg

    cfg = dataclasses.replace(TwoAgentRunnerCfg(), actor_hidden_dims=(64, 32))
    actors = {
        a: build_imitation_actor(PROPRIO_DIM, ACTION_DIMS[a], IMITATION_GOAL_DIMS[a], 32, (64, 32), 1) for a in AGENTS
    }
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"stage": "imitation", "runner": dataclasses.asdict(cfg)}))
    torch.save({a: m.state_dict() for a, m in actors.items()}, run_dir / "actors.pt")


@pytest.mark.sim
def test_crosscheck_runs_on_standins(sim_app, tmp_path):
    import isaaclab.sim as sim_utils

    from pretraining.scripts import crosscheck, rollout

    os.environ["PRETRAINING_CUDA_GRAPHS"] = "0"
    sim_utils.create_new_stage()  # a fresh stage: an earlier sim test may have left its prims
    motions = tmp_path / "motions"
    write_clips(motions, n=5, frames=120)
    # one shorter clip, last in the motion: the batch steps on after it ends, past the end of the motion
    (short,) = write_clips(tmp_path / "short", n=1, frames=60, seed=3)
    short.rename(motions / "standin_short.npz")
    _untrained_run(tmp_path / "run")
    out = tmp_path / "out"
    args = crosscheck.parse_args(
        ["--run", str(tmp_path / "run"), "--motion-dir", str(motions), "--clips", "3", "--seed", "1",
         "--include", "standin_004", "standin_002", "standin_short", "--out", str(out)]
    )
    summary = crosscheck.run(args)

    with open(out / "crosscheck.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    clips = (out / "clips.txt").read_text().split()
    assert clips == [r["clip"] for r in rows] and len(set(clips)) == 3 and {"standin_004", "standin_short"} <= set(clips)
    for r in rows:
        # a 120-frame clip is 119 scored steps at 50 Hz, unless the robot fell or a termination ended it first
        steps, frames = int(r["steps"]), 60 if r["clip"] == "standin_short" else 120
        assert 1 <= steps <= frames - 1
        assert float(r["clip_s"]) == pytest.approx((frames - 1) * 0.02)
        assert r["ended_by"] in {"clip_end", "time_cap", "fall", "anchor_pos", "anchor_ori"}
        assert (r["ended_by"] == "clip_end") == (steps == frames - 1)
        assert (r["fell_at_s"] != "") == (r["ended_by"] == "fall")
        assert float(r["time_upright_s"]) == pytest.approx(steps * 0.02)
        numeric = [k for k in r if k.endswith(("_m", "_rad")) or k.startswith("Safety/")]
        assert {"body_pos_err_m", "pelvis_pos_err_m", "joint_err_rad"} <= set(numeric)
        assert any(k.startswith("Safety/legs_") for k in numeric) and any(k.startswith("Safety/arms_") for k in numeric)
        for k in numeric:
            assert math.isfinite(float(r[k])), (r["clip"], k)
        for k in numeric:
            if k.startswith("Safety/"):
                assert 0.0 <= float(r[k]) <= 1.0
    assert "3 clips" in summary

    # the same clips read back from the list
    again = rollout.choose_clips(motions, 3, seed=7, include=[], clips_file=out / "clips.txt")
    assert [p.stem for p in again] == clips


class _ReplayActor(torch.nn.Module):
    """Commands the clip's next joint pose for one agent's joints: ``(q_ref[t + 1] - default) / scale``."""

    def __init__(self, cmd, joints: slice):
        super().__init__()
        from pretraining.constants import ACTION_SCALE, DEFAULT_JOINT_POS, JOINT_NAMES

        self.cmd, self.joints = cmd, joints
        dev = cmd.device
        self.default = torch.tensor([DEFAULT_JOINT_POS[j] for j in JOINT_NAMES], device=dev)[joints]
        self.scale = torch.tensor([ACTION_SCALE[j] for j in JOINT_NAMES], device=dev)[joints]

    def forward(self, td):
        frames = torch.clamp(self.cmd.time_steps + 1, max=self.cmd.motion.time_step_total - 1)
        return (self.cmd.motion.joint_pos[frames][:, self.joints] - self.default) / self.scale


REPLAY_JOINT_ERR = 0.15  # rad; measured 0.05-0.11 over 0.2 s on the stand-ins (PD lag, the robot off balance)


@pytest.mark.sim
def test_rollout_scores_a_reference_replay_near_zero(sim_app, tmp_path):
    """Commanding the reference's joint poses tracks them: errors near zero after the first step and a small mean
    joint error over the first 0.2 s (a frame offset or a frame-of-reference error would show here). Only 0.2 s: the
    stand-ins' pelvis path is not one the joints alone can produce, and the robot leaves it (anchor termination
    after 12-31 steps)."""
    import isaaclab.sim as sim_utils

    import pretraining.tasks  # noqa: F401
    from pretraining.marl.config import JOINT_SLICES
    from pretraining.scripts import rollout

    files = write_clips(tmp_path / "motions", n=3, frames=120)
    results = {}
    for seconds in (0.02, 0.2):
        sim_utils.create_new_stage()
        env = rollout.make_env(files, len(files), "cuda:0", seed=0)
        cmd = env.unwrapped.command_manager.get_term("motion")
        actors = {a: _ReplayActor(cmd, JOINT_SLICES[a]) for a in ("legs", "arms")}
        try:
            results[seconds] = rollout.rollout(env, actors, 1, seconds)
        finally:
            env.close()
    for r in results[0.02]:
        assert r["steps"] == 1 and r["ended_by"] == "time_cap"
        assert r["body_pos_err_m"] < 0.02 and r["pelvis_pos_err_m"] < 0.02 and r["joint_err_rad"] < 0.02
    for r in results[0.2]:
        assert r["steps"] == 10 and r["ended_by"] == "time_cap"
        assert r["joint_err_rad"] < REPLAY_JOINT_ERR


# Training (scripts/train.py), run as a subprocess: each launch starts its own Isaac app.

TRAIN_TIMEOUT_S = 600
TRAIN_ENVS = 64
SMALL_CONFIG = """\
num_envs: 4096  # more than the tests use: --num-envs overrides it
runner:
  buffer_size: 50
  actor_obs: proprio
  log_interval: 1
  save_interval: 100000
  checkpoint_interval_s: 100000.0
  resume_warmup: 2
  replay_ratio: 4
  actor_hidden_dims: [64, 64]
  critic_hidden_dims: [32]
  dynamics_hidden_dims: [64, 64]
  wm: {num_batches: 1, batch_size: 64, horizon: 2}
  policy: {batch_size: 32, horizon: 4}
rewards:
  estop: {velocity: -5.0}
"""


def _train(tmp_path, *args, ok=True):
    """Run ``python -m pretraining.scripts.train *args`` with the runs root in ``tmp_path``; kill its process group
    on timeout. Returns the completed process; ``ok`` asserts exit 0, ``ok=False`` a non-zero exit."""
    import signal
    import subprocess
    import sys
    from pathlib import Path

    env = dict(os.environ, PRETRAINING_RUNS_DIR=str(tmp_path / "runs"), PRETRAINING_CUDA_GRAPHS="0",
               WANDB_MODE="disabled")
    cmd = [sys.executable, "-m", "pretraining.scripts.train", *map(str, args)]
    proc = subprocess.Popen(cmd, cwd=Path(__file__).resolve().parents[1], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=TRAIN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        pytest.fail(f"train.py timed out after {TRAIN_TIMEOUT_S} s:\n{out[-4000:]}")
    proc.stdout_text = out
    if ok:
        assert proc.returncode == 0, out[-4000:]
    else:
        assert proc.returncode != 0, out[-4000:]
    return proc


def _train_setup(tmp_path):
    """Stand-in clips and the small config file; returns (clip files, config path)."""
    files = write_clips(tmp_path / "motions", n=3, frames=120)
    config = tmp_path / "small.yaml"
    config.write_text(SMALL_CONFIG)
    return files, config


def _iterations(run_dir):
    with open(run_dir / "progress.csv") as fh:
        return [int(r["iteration"]) for r in csv.DictReader(fh)]


@pytest.mark.sim
def test_smoke_training_writes_a_run_refuses_its_directory_and_fine_tunes(tmp_path):
    """A new run (64 envs, fewer than the config's num_envs and than the batch sizes of base.yaml would need) writes
    the run files and run.json (env count, config content, motion files, seed); a second start on the same name is
    refused; a config with an unknown key is refused naming it; a fine-tune starts a new run from the first one's
    checkpoint with its own rewards, continuing the iteration count (as h12_train's amass_s1_golem_aware)."""
    import yaml

    files, config = _train_setup(tmp_path)
    common = ["--config", config, "--motion-dir", tmp_path / "motions", "--num-envs", TRAIN_ENVS]
    proc = _train(tmp_path, *common, "--run-name", "a", "--max-iterations", 30, "--seed", 5)
    run = tmp_path / "runs" / "a"
    assert str(run) in proc.stdout_text
    assert "s_per_iteration" in proc.stdout_text and "peak_gpu_gb" in proc.stdout_text
    for name in ("run.json", "progress.csv", "actors.pt", "critics.pt", "world_model.pt", "checkpoint.pt"):
        assert (run / name).exists(), name
    meta = json.loads((run / "run.json").read_text())
    assert meta["num_envs"] == TRAIN_ENVS
    assert meta["config"] == yaml.safe_load(SMALL_CONFIG)
    assert meta["motion_files"] == [str(f.resolve()) for f in files]
    assert meta["runner"]["seed"] == 5 and meta["runner"]["wm"]["batch_size"] == 64
    assert meta["rewards"]["estop"]["position"] == 0.0
    assert _iterations(run) == list(range(30))
    with open(run / "progress.csv") as fh:
        losses = [float(r["Loss/state"]) for r in csv.DictReader(fh) if r.get("Loss/state")]
    assert losses and all(math.isfinite(v) for v in losses)

    # the same run name again: refused before the app starts, naming the directory; nothing in it changed
    before = (run / "run.json").read_text()
    proc = _train(tmp_path, *common, "--run-name", "a", "--max-iterations", 30, ok=False)
    assert str(run) in proc.stdout_text and "exists" in proc.stdout_text
    assert (run / "run.json").read_text() == before

    bad = tmp_path / "bad.yaml"
    bad.write_text(SMALL_CONFIG.replace("  wm: {", "  wm: {batch_sise: 3, "))
    proc = _train(tmp_path, "--config", bad, "--motion-dir", tmp_path / "motions", "--run-name", "bad", ok=False)
    assert "wm.batch_sise" in proc.stdout_text and not (tmp_path / "runs" / "bad").exists()

    # fine-tune: 5 iterations of a new run from a's checkpoint with the estop position term on
    tuned_cfg = tmp_path / "tuned.yaml"
    tuned_cfg.write_text(SMALL_CONFIG.replace("estop: {velocity: -5.0}", "estop: {velocity: -5.0, position: -20.0}")
                         .replace("log_interval: 1", "log_interval: 1\n  max_iterations: 7"))
    _train(tmp_path, "--config", tuned_cfg, "--motion-dir", tmp_path / "motions", "--num-envs", TRAIN_ENVS,
           "--run-name", "b", "--fine-tune-from", run, "--max-iterations", 5)
    tuned = tmp_path / "runs" / "b"
    meta = json.loads((tuned / "run.json").read_text())
    assert meta["fine_tuned_from"] == f"{run} at iteration 29"
    assert meta["resumed"][0]["iteration"] == 30
    assert meta["rewards"]["estop"]["position"] == -20.0 and meta["config"]["runner"]["max_iterations"] == 7  # --max-iterations 5 overrides it
    assert meta["runner"]["max_iterations"] == 35
    assert _iterations(tuned) == list(range(35))  # a's curves, then the fine-tune's
    assert json.loads((tuned / "checkpoint.json").read_text())["iteration"] == 34
    assert _iterations(run) == list(range(30)) and (run / "run.json").read_text() == before  # the source untouched


@pytest.mark.sim
def test_resume_after_a_wall_time_stop_continues_the_iteration_count(tmp_path):
    """A run stopped by --max-seconds saves and exits 0; --resume continues it to --max-iterations, the count
    carrying on, with only --max-iterations / --max-seconds taken from the command line."""
    _, config = _train_setup(tmp_path)
    proc = _train(tmp_path, "--config", config, "--motion-dir", tmp_path / "motions", "--num-envs", TRAIN_ENVS,
                  "--run-name", "r", "--max-iterations", 100000, "--max-seconds", 40)
    run = tmp_path / "runs" / "r"
    stopped = json.loads((run / "checkpoint.json").read_text())["iteration"]
    out = proc.stdout_text[-2000:]
    # stopped by the clock, far short of --max-iterations (the stand-in run does tens of iterations per second at
    # most), with a checkpoint at the last logged iteration
    assert 1 <= stopped < 10000, f"stopped at {stopped}:\n{out}"
    assert (run / "checkpoint.pt").exists() and _iterations(run)[-1] == stopped, out
    proc = _train(tmp_path, "--resume", run, "--max-iterations", stopped + 11)
    assert str(run) in proc.stdout_text
    its = _iterations(run)
    assert its == list(range(stopped + 11)) and its[-1] == stopped + 10
    assert json.loads((run / "checkpoint.json").read_text())["iteration"] == stopped + 10
    meta = json.loads((run / "run.json").read_text())
    assert [r["iteration"] for r in meta["resumed"]] == [stopped + 1] and meta["resumed"][0]["wall_s"] > 40
    assert meta["num_envs"] == TRAIN_ENVS

    # any other option with --resume is refused: the run's settings are its run.json's
    proc = _train(tmp_path, "--resume", run, "--num-envs", 8, ok=False)
    assert "--num-envs" in proc.stdout_text


def test_a_fine_tune_on_other_clips_is_refused(tmp_path, monkeypatch):
    """plan() (no app) refuses a fine-tune whose clips are not the source run's, naming the counts and position."""
    from pretraining.scripts import train

    monkeypatch.setenv("PRETRAINING_RUNS_DIR", str(tmp_path / "runs"))
    files, config = _train_setup(tmp_path)
    source = tmp_path / "src"
    source.mkdir()
    (source / "run.json").write_text(json.dumps({"motion_files": [str(f) for f in files[:2]]}))
    (source / "checkpoint.json").write_text(json.dumps({"iteration": 9, "wall_s": 1.0}))
    (source / "checkpoint.pt").write_bytes(b"")
    argv = ["--config", str(config), "--run-name", "ft", "--fine-tune-from", str(source)]
    with pytest.raises(train.UsageError, match="gives 3 clips, .* trained on 2; they first differ at position 2"):
        train.plan(train.build_parser().parse_args([*argv, "--motion-dir", str(tmp_path / "motions")]))
    other = tmp_path / "other"
    other.mkdir()
    (other / "standin_000.npz").symlink_to(files[0])
    (other / "standin_009.npz").symlink_to(files[2])
    with pytest.raises(train.UsageError, match="differ at position 1"):
        train.plan(train.build_parser().parse_args([*argv, "--motion-dir", str(other)]))
    # the source's own clips: accepted; --max-iterations counts the fine-tune's iterations
    (source / "run.json").write_text(json.dumps({"motion_files": [str(f) for f in files]}))
    ok = train.plan(train.build_parser().parse_args([*argv, "--motion-dir", str(tmp_path / "motions"),
                                                     "--max-iterations", "4"]))
    assert ok["mode"] == "fine_tune" and ok["start"] == 10 and ok["cfg"].max_iterations == 14
