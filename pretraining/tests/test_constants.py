"""The constants are h12_train's (robot/h12_constants.py, tasks.py at b175f1c) in main's joint order."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from pretraining import constants

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT = REPO_ROOT / "data" / "runs" / "amass_s2_golem" / "export" / "export.yaml"


@pytest.mark.sim  # main's h1_2.py imports isaaclab, which needs the app's pxr
def test_joint_order_is_mains(sim_app):
    from locomanipulation_game.assets.h1_2 import ALL_JOINTS_NAMES

    assert list(constants.JOINT_NAMES) == ALL_JOINTS_NAMES


def test_agent_slices_split_the_joints():
    assert len(constants.JOINT_NAMES[constants.LEG_JOINTS]) == 12
    assert constants.JOINT_NAMES[constants.UPPER_JOINTS][0] == "torso_joint"
    assert len(constants.JOINT_NAMES[constants.UPPER_JOINTS]) == 15


def test_tables_cover_every_joint():
    for table in (constants.DEFAULT_JOINT_POS, constants.KP, constants.KD, constants.EFFORT, constants.ARMATURE,
                  constants.ACTION_SCALE, constants.TARGET_LIMITS, constants.ESTOP_LIMITS):
        assert set(table) == set(constants.JOINT_NAMES)


def test_action_scale_is_quarter_effort_over_kp_capped():
    for j in constants.JOINT_NAMES:
        assert constants.ACTION_SCALE[j] == pytest.approx(
            min(0.25, 0.25 * constants.EFFORT[j] / constants.KP[j]), abs=1e-6)


def test_tracked_bodies():
    assert len(constants.TRACKED_BODIES) == 14
    assert constants.ANCHOR_BODY == "pelvis" and constants.ANCHOR_BODY in constants.TRACKED_BODIES
    assert constants.SIM_DT * constants.DECIMATION == pytest.approx(0.02)


@pytest.mark.skipif(not EXPORT.exists(), reason=f"{EXPORT} absent")
def test_values_match_the_h12_train_export():
    cfg = yaml.safe_load(EXPORT.read_text())
    assert cfg["joint_names"] == list(constants.JOINT_NAMES)
    np.testing.assert_allclose(cfg["kp"], [constants.KP[j] for j in constants.JOINT_NAMES])
    np.testing.assert_allclose(cfg["kd"], [constants.KD[j] for j in constants.JOINT_NAMES])
    np.testing.assert_allclose(cfg["effort_limits"], [constants.EFFORT[j] for j in constants.JOINT_NAMES])
    np.testing.assert_allclose(cfg["default_joint_pos"], [constants.DEFAULT_JOINT_POS[j] for j in constants.JOINT_NAMES])
    np.testing.assert_allclose(cfg["action_scale"], [constants.ACTION_SCALE[j] for j in constants.JOINT_NAMES], atol=1e-6)
    for j, (lo, hi) in cfg["target_limits"].items():
        assert constants.TARGET_LIMITS[j] == pytest.approx((lo, hi), abs=1e-6)
    for j, lim in cfg["estop_limits"].items():
        if j == "source":
            continue
        e = constants.ESTOP_LIMITS[j]
        assert (e.q_low, e.q_high, e.dq, e.tau) == pytest.approx((*lim["q"], lim["dq"], lim["tau"]), abs=1e-6)


@pytest.mark.sim
def test_robot_cfg_uses_the_constants(sim_app):
    from pretraining.robot import H1_2_PRETRAIN_CFG, MASSLESS_FRAME_FIX

    cfg = H1_2_PRETRAIN_CFG
    assert cfg.spawn.articulation_props.enabled_self_collisions
    for name, act in cfg.actuators.items():
        if name == "grippers":
            continue
        for j in act.joint_names_expr:
            assert (act.stiffness[j], act.damping[j], act.effort_limit_sim[j], act.armature[j]) == (
                constants.KP[j], constants.KD[j], constants.EFFORT[j], constants.ARMATURE[j])
        assert act.friction == 0.0
    assert sorted(j for a in cfg.actuators.values() for j in a.joint_names_expr if j in constants.KP) == sorted(
        constants.JOINT_NAMES)
    assert cfg.init_state.joint_pos == {**constants.DEFAULT_JOINT_POS, **cfg.init_state.joint_pos}
    assert MASSLESS_FRAME_FIX.mode == "startup"
