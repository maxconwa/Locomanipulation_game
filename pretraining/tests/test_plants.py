"""The plants: PRETRAIN is constants', MAIN is main's articulation, and main's explicit PD is damping-stable."""

import re
import subprocess
import sys

import pytest

from pretraining import constants
from pretraining.plants import MAIN, PRETRAIN, get_plant


def test_pretrain_plant_is_constants():
    p = PRETRAIN
    assert (p.sim_dt, p.decimation, p.explicit, p.step_dt) == (constants.SIM_DT, constants.DECIMATION, False, 0.02)
    assert p.kp == constants.KP and p.kd == constants.KD and p.effort == constants.EFFORT
    assert p.armature == constants.ARMATURE and p.velocity_limit == constants._VELOCITY_LIMITS
    assert p.action_scale == constants.ACTION_SCALE and p.default_joint_pos == constants.DEFAULT_JOINT_POS
    assert p.default_root_height == constants.DEFAULT_ROOT_HEIGHT and p.target_limits == constants.TARGET_LIMITS


def test_get_plant_refuses_unknown_names():
    assert get_plant("main") is MAIN and get_plant("pretrain") is PRETRAIN
    with pytest.raises(ValueError, match="main, pretrain"):
        get_plant("golem")


def test_main_action_scales():
    expected = {"torso_joint": 0.1, "left_knee_joint": 0.1875, "right_hip_pitch_joint": 0.2,
                "left_hip_yaw_joint": 0.2, "right_ankle_pitch_joint": 0.1875, "left_wrist_roll_joint": 0.2375,
                "right_wrist_yaw_joint": 0.2375, "left_shoulder_pitch_joint": 0.04, "right_elbow_joint": 0.05625}
    for j, v in expected.items():
        assert MAIN.action_scale[j] == pytest.approx(v, abs=1e-12), j


@pytest.mark.sim  # main's h1_2.py imports isaaclab, which needs the app's pxr
def test_main_plant_matches_main_articulation(sim_app):
    from isaaclab.actuators import IdealPDActuatorCfg
    from isaaclab.utils.string import resolve_matching_names_values
    from locomanipulation_game.assets.h1_2 import H1_2_MAGPIE_CFG, STANCE_PELVIS_HEIGHT
    from locomanipulation_game.tasks.locomanip_marl.locomanip_marl_env_cfg import LocoManipMarlEnvCfg

    names = list(constants.JOINT_NAMES)
    seen = set()
    for act in H1_2_MAGPIE_CFG.actuators.values():
        joints = [j for j in names if any(re.fullmatch(p, j) for p in act.joint_names_expr)]
        if not joints:
            continue  # the grippers
        assert isinstance(act, IdealPDActuatorCfg)
        seen.update(joints)
        for attr, table in (("stiffness", MAIN.kp), ("damping", MAIN.kd), ("effort_limit_sim", MAIN.effort),
                            ("velocity_limit_sim", MAIN.velocity_limit), ("armature", MAIN.armature)):
            value = getattr(act, attr)
            if isinstance(value, dict):
                _, resolved, vals = resolve_matching_names_values(value, joints)
                got = dict(zip(resolved, vals))
            else:
                got = {j: value for j in joints}
            assert got == {j: table[j] for j in joints}, (attr, joints)
    assert seen == set(names)
    pose = {p: v for p, v in H1_2_MAGPIE_CFG.init_state.joint_pos.items() if "hinge" not in p}  # not policy joints
    _, resolved, vals = resolve_matching_names_values(pose, names)
    assert dict(zip(resolved, vals)) == MAIN.default_joint_pos
    assert MAIN.default_root_height == STANCE_PELVIS_HEIGHT == H1_2_MAGPIE_CFG.init_state.pos[2]
    main_env = LocoManipMarlEnvCfg()
    assert (MAIN.sim_dt, MAIN.decimation) == (main_env.sim.dt, main_env.decimation)
    assert MAIN.step_dt == pytest.approx(0.01, abs=1e-15)


@pytest.mark.sim
def test_main_plant_damping_eigenvalues_below_one():
    # its own process: an Articulation made and cleared in the session's app hangs the next test's simulation
    for scale in ("1.0", "1.2"):
        run = subprocess.run([sys.executable, "-m", "pretraining.scripts.pd_stability", "--plant", "main",
                              "--kd-scale", scale], capture_output=True, text=True, timeout=600)
        summary = [line for line in run.stdout.splitlines() if line.startswith(("plant", "  ", "max"))]
        assert run.returncode == 0, (scale, summary or run.stderr[-2000:])
