"""The H1-2 articulation of the pretraining: main's Magpie USD with h12_train's actuators.

Implicit PD at ``constants``' kp and kd (h12_train's, from ``robot/h12_constants.py`` at b175f1c), its effort limits
and per-joint armature, no joint friction, self-collision on, soft joint limits at 0.9 as h12_train's. The gripper
hinges, not policy joints, are held at main's rest pose by stiff implicit drives (main's ``h1_2.py`` grippers group).
"""

import copy
import re

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from locomanipulation_game.assets.h1_2 import _SPAWN_CFG, GRIPPER_JOINTS
from locomanipulation_game.tasks.locomanip_marl.locomanip_marl_env_cfg import MarlEventCfg

from pretraining import constants


def _group(pattern: str) -> ImplicitActuatorCfg:
    """One joint group's implicit PD, its values listed per joint by exact name."""
    joints = [j for j in constants.JOINT_NAMES if re.fullmatch(pattern, j)]
    return ImplicitActuatorCfg(
        joint_names_expr=joints,
        stiffness={j: constants.KP[j] for j in joints},
        damping={j: constants.KD[j] for j in joints},
        effort_limit_sim={j: constants.EFFORT[j] for j in joints},
        armature={j: constants.ARMATURE[j] for j in joints},
        friction=0.0,
    )


_ACTUATORS = {name: _group(pattern) for name, pattern in constants.ACTUATOR_GROUPS.items()}
# main's grippers group: stiff implicit drives hold each hinge at its default in place of the missing linkage.
_ACTUATORS["grippers"] = ImplicitActuatorCfg(
    joint_names_expr=GRIPPER_JOINTS,
    effort_limit_sim=10.0,
    velocity_limit_sim=3.14,
    stiffness=100.0,
    damping=5.0,
    armature=0.01,
)

H1_2_PRETRAIN_CFG = ArticulationCfg(
    spawn=copy.deepcopy(_SPAWN_CFG),  # main's: H1_2_MAGPIE_USD, self-collision on
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, constants.DEFAULT_ROOT_HEIGHT),
        joint_pos={**constants.DEFAULT_JOINT_POS, "[lr]g_.*_hinge_.*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=_ACTUATORS,
)

# main's start-up event (locomanip_marl MarlEventCfg.massless_frames): the four frames without an inertial in the
# URDF, which the USD import gives 1 kg each, set to 0.01 kg.
MASSLESS_FRAME_FIX = copy.deepcopy(MarlEventCfg().massless_frames)
