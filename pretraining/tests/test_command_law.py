"""Main's command law and the segment process (``marl/command_law.py``), on the CPU, no simulator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pretraining.marl.command_law import ArmGoalLaw, CommandProcess, VelocityLaw

STANDING_HEIGHT = 0.993


def _velocity_cfg():
    """MarlCommandsCfg.base_velocity's sampling fields (locomanip_marl_env_cfg.py:159-176) as a stand-in."""
    ranges = SimpleNamespace(lin_vel_x=(-0.55, 0.55), lin_vel_y=(-0.54, 0.54), ang_vel_z=(-0.97, 0.97))
    return SimpleNamespace(ranges=ranges, rel_standing_envs=0.25, pure_turn_prob=0.2, pure_lateral_prob=0.2,
                           pure_turn_speed=(0.2, 0.97), pure_lateral_speed=(0.15, 0.54), command_filter_tau=0.5)


def _table_file(tmp_path, rows: int = 4000):
    """A stand-in arm-target table file in ArmTargetsCommand's format: per arm (rows, 7) pelvis-frame poses, w-first."""
    gen = torch.Generator().manual_seed(3)
    tables = []
    for side in (1.0, -1.0):
        pos = torch.rand(rows, 3, generator=gen) * torch.tensor([0.4, 0.3, 1.0]) + torch.tensor([0.1, 0.0, -0.15])
        pos[:, 1] *= side
        quat = torch.randn(rows, 4, generator=gen)
        quat = quat / quat.norm(dim=-1, keepdim=True)
        quat = torch.where(quat[:, :1] < 0, -quat, quat)
        tables.append(torch.cat([pos, quat], -1))
    rest = [torch.tensor([0.234, side * 0.21, 0.095, 1.0, 0.0, 0.0, 0.0]) for side in (1.0, -1.0)]
    path = tmp_path / "tables.pt"
    torch.save({"settings": {}, "body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"], "tables": tables,
                "default_poses": rest}, path)
    return path, tables


def test_velocity_law_shares():
    law = VelocityLaw.from_cfg(_velocity_cfg())
    gen = torch.Generator().manual_seed(0)
    v = law.sample(100_000, gen)
    assert v.shape == (100_000, 3)
    zero = (v == 0)
    standing = zero.all(-1)
    turn = zero[:, 0] & zero[:, 1] & ~zero[:, 2]
    lateral = zero[:, 0] & ~zero[:, 1] & zero[:, 2]
    assert abs(standing.float().mean().item() - 0.25) < 0.01
    assert abs(turn.float().mean().item() - 0.15) < 0.01
    assert abs(lateral.float().mean().item() - 0.15) < 0.01
    assert (v[:, 0].abs() <= 0.55).all() and (v[:, 1].abs() <= 0.54).all() and (v[:, 2].abs() <= 0.97).all()
    # the pure turns and walks sideways at their speed ranges, either sign
    assert (v[turn, 2].abs() >= 0.2).all() and (v[lateral, 1].abs() >= 0.15).all()
    assert (v[turn, 2] > 0).float().mean().item() == pytest.approx(0.5, abs=0.02)


def test_arm_law_draws_table_rows(tmp_path):
    path, tables = _table_file(tmp_path)
    law = ArmGoalLaw.from_table(path, STANDING_HEIGHT, z_range=(1.0, 1.5))
    gen = torch.Generator().manual_seed(1)
    goals = law.sample(2000, gen)
    assert goals.shape == (2000, 2, 7)
    assert (goals[..., 3] >= 0).all()
    assert torch.allclose(goals[..., 3:].norm(dim=-1), torch.ones(2000, 2), atol=1e-5)
    for arm in range(2):
        # each pose is a row of its arm's table, lowered (z only) when the goal was below what the table reaches
        d = (goals[:, arm, None, [0, 1, 3, 4, 5, 6]] - tables[arm][None, :, [0, 1, 3, 4, 5, 6]]).abs().amax(-1)
        assert (d.min(-1).values < 1e-6).all()
    # the two arms' rows near one goal height: within twice the spread of each other
    assert ((goals[:, 0, 2] - goals[:, 1, 2]).abs() <= 2 * law.height_spread + 1e-6).all()


def test_arm_law_offtable_and_rest_goals(tmp_path):
    path, _ = _table_file(tmp_path)
    law = ArmGoalLaw.from_table(path, STANDING_HEIGHT, z_range=(1.0, 1.5), rest_prob=1.0)
    goals = law.sample(10, torch.Generator().manual_seed(2))
    assert torch.allclose(goals, law.rest.expand(10, 2, 7))
    law = ArmGoalLaw.from_table(path, STANDING_HEIGHT, z_range=(1.0, 1.5), offtable_prob=1.0,
                                offtable_scales=(0.5, 0.5))
    goals = law.sample(500, torch.Generator().manual_seed(2))
    # halfway from the rest pose: the positions within half the table's reach of it, unit quaternions, qw >= 0
    assert (goals[..., 3] >= 0).all()
    assert torch.allclose(goals[..., 3:].norm(dim=-1), torch.ones(500, 2), atol=1e-5)


def test_segment_length_mean(tmp_path):
    path, _ = _table_file(tmp_path)
    dt = 0.01
    process = CommandProcess(dt, mean_hold_s=6.0, clip_prob=0.25, velocity_law=VelocityLaw.from_cfg(_velocity_cfg()),
                             arm_law=ArmGoalLaw.from_table(path, STANDING_HEIGHT, z_range=(1.0, 1.5)))
    n, steps = 4000, 3000
    state = process.start({"velocity": torch.zeros(n, 3), "arm": torch.zeros(n, 2, 7)})
    gen = torch.Generator().manual_seed(4)
    ends, clips = 0, 0
    for _ in range(steps):
        state = process.step(state, gen)
        ends += int(state["new_velocity"].sum())
        clips += int((state["new_velocity"] & state["clip_velocity"]).sum())
    mean_s = n * steps * dt / ends
    assert abs(mean_s - 6.0) < 0.2
    assert abs(clips / ends - 0.25) < 0.02


def test_start_holds_recorded_command(tmp_path):
    path, _ = _table_file(tmp_path)
    process = CommandProcess(0.02, velocity_law=VelocityLaw.from_cfg(_velocity_cfg()),
                             arm_law=ArmGoalLaw.from_table(path, STANDING_HEIGHT, z_range=(1.0, 1.5)))
    gen = torch.Generator().manual_seed(5)
    velocity = torch.randn(8, 3)
    arm = torch.randn(8, 2, 7)
    target = torch.randn(8, 3)
    state = process.start({"velocity": velocity, "arm": arm, "velocity_target": target})
    assert torch.equal(state["velocity"], velocity) and torch.equal(state["arm"], arm)
    assert torch.equal(state["velocity_target"], target)
    assert not state["clip_velocity"].any() and not state["clip_arm"].any()
    # not aliased: the recorded tensors stay as they were
    velocity_before = velocity.clone()
    state["velocity"] += 1.0
    assert torch.equal(velocity, velocity_before)
    # a row whose segment did not end keeps its target and arm command; its seen velocity moves toward the target
    state = process.start({"velocity": velocity, "arm": arm, "velocity_target": target})
    nxt = process.step(state, gen)
    kept = ~nxt["new_velocity"]
    assert torch.equal(nxt["velocity_target"][kept], target[kept])
    alpha = process.filter_alpha
    assert torch.allclose(nxt["velocity"][kept], (velocity + alpha * (target - velocity))[kept])
    kept_arm = ~nxt["new_arm"]
    assert torch.equal(nxt["arm"][kept_arm], arm[kept_arm])
    # without a recorded target the command is its own target
    state = process.start({"velocity": velocity, "arm": arm})
    assert torch.equal(state["velocity_target"], velocity)
