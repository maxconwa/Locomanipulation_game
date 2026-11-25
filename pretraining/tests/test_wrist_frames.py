"""``scripts/wrist_frames.py``: the clip replay's wrist-frame check. The CPU cases test its measures; the ``sim`` one
runs it on a stand-in clip in its own process (it launches its own app)."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from pretraining.math import quat_from_angle_axis
from pretraining.scripts.wrist_frames import coverage, in_pelvis, percentiles, tool_frame_rotation_deg

REPO = Path(__file__).resolve().parents[2]
TIMEOUT_S = 900


def _random_quat(n: int, gen: torch.Generator) -> torch.Tensor:
    q = torch.randn(n, 4, generator=gen)
    return q / q.norm(dim=-1, keepdim=True)


def test_coverage_takes_both_tolerances_and_the_double_cover():
    goal_pos = torch.tensor([[0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.3, 0.2, 0.1]])
    z = torch.tensor([0.0, 0.0, 1.0])
    goal_quat = torch.stack([quat_from_angle_axis(torch.tensor(a), z) for a in (0.0, 0.0, math.radians(90))])
    # one clip pose 5 cm off, rotated 20 deg, written as -q
    pos = torch.tensor([[0.35, 0.2, 0.1]])
    quat = -quat_from_angle_axis(torch.tensor(math.radians(20)), z).unsqueeze(0)
    assert coverage(goal_pos, goal_quat, pos, quat).tolist() == [True, True, False]  # 70 deg off the third
    assert not coverage(goal_pos[:1], goal_quat[:1], pos + torch.tensor([0.06, 0.0, 0.0]), quat).item()  # 11 cm off


def test_goal_frame_is_the_tool_frame_rotated_by_nothing():
    gen = torch.Generator().manual_seed(0)
    n = 16
    pelvis_pos, pelvis_quat = torch.randn(n, 3, generator=gen), _random_quat(n, gen)
    wrist_pos, wrist_quat = torch.randn(n, 2, 3, generator=gen), _random_quat(2 * n, gen).view(n, 2, 4)
    rot, offset = tool_frame_rotation_deg(pelvis_pos, pelvis_quat, wrist_pos, wrist_quat)
    assert rot.shape == (n, 2) and rot.max() < 0.05  # deg: float32 round-off only
    assert len(offset) == 3 and offset[0] > 0.1  # a translation along the wrist's x
    pos, quat = in_pelvis(pelvis_pos, pelvis_quat, pelvis_pos.unsqueeze(1), pelvis_quat.unsqueeze(1))
    torch.testing.assert_close(pos, torch.zeros_like(pos), atol=1e-5, rtol=0)
    assert torch.allclose(quat[..., 0].abs(), torch.ones(n, 1), atol=1e-5)


def test_percentiles():
    p = percentiles(torch.arange(101, dtype=torch.float32))
    assert p == {"p50": 50.0, "p95": 95.0, "max": 100.0}


@pytest.mark.sim
def test_wrist_frames_on_standin_clip(tmp_path):
    from standins import write_clips

    clip = write_clips(tmp_path / "motions", n=1, frames=60)[0]
    out = tmp_path / "out"
    env = dict(os.environ)
    proc = subprocess.Popen(
        [sys.executable, "-m", "pretraining.scripts.wrist_frames", "--motion-dir", str(tmp_path / "motions"),
         "--clips", "1", "--stride", "3", "--out", str(out)],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        log, _ = proc.communicate(timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        log, _ = proc.communicate()
        pytest.fail(f"timed out after {TIMEOUT_S} s:\n{log[-4000:]}")
    assert proc.returncode == 0, log[-6000:]
    text = (out / "wrist_frames.json").read_text()
    result = json.loads(text)
    assert result["clips"] == 1 and result["frames"] == 20
    # Isaac's USD and the URDF's FK agree on the wrist links in the pelvis frame
    assert result["fk_pos_err_m"]["max"] < 1e-3, result["fk_pos_err_m"]
    assert result["fk_rot_err_deg"]["max"] < 0.5, result["fk_rot_err_deg"]
    assert result["goal_vs_tool_frame"]["rotation_deg_max"] < 0.05
    cov = result["goal_coverage"]
    assert 0.0 <= cov["both"] <= 1.0 and cov["table_poses"] > 0
    assert "yaw-only" in cov["coverage_note"]
    # aggregates only: the clip's name is nowhere in the output
    assert clip.stem not in text
