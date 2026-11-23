"""Smoke tests of the scripts on the stand-in env: the cross-check."""

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
