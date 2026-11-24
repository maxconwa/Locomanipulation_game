"""The H1-2 articulation of a plant (``plants``): main's Magpie USD with the plant's actuators.

``PRETRAIN``: implicit PD at h12_train's kp and kd (``constants``, from ``robot/h12_constants.py`` at b175f1c), its
effort limits and per-joint armature, no joint friction. ``MAIN``: main's ``h1_2.py`` explicit ``IdealPDActuatorCfg``
on the plant's gains, effort and velocity limits and armature. Either way self-collision is on, soft joint limits at
0.9 as h12_train's (and main's), and the gripper hinges, not policy joints, are held at main's rest pose by stiff
implicit drives (main's ``h1_2.py`` grippers group).
"""

import copy
import re

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from locomanipulation_game.assets.h1_2 import _SPAWN_CFG, GRIPPER_JOINTS
from locomanipulation_game.tasks.locomanip_marl.locomanip_marl_env_cfg import MarlEventCfg

from pretraining import constants
from pretraining.plants import PRETRAIN, Plant


def _group(plant: Plant, pattern: str) -> ImplicitActuatorCfg | IdealPDActuatorCfg:
    """One joint group's PD, its values listed per joint by exact name."""
    joints = [j for j in constants.JOINT_NAMES if re.fullmatch(pattern, j)]

    def per(table):
        return {j: table[j] for j in joints}

    if plant.explicit:
        # as main's h1_2.py _pd: the effort and velocity limits apply to the computed torque and the PhysX joint
        effort, velocity = per(plant.effort), per(plant.velocity_limit)
        return IdealPDActuatorCfg(joint_names_expr=joints, effort_limit=effort, effort_limit_sim=effort,
                                  velocity_limit=velocity, velocity_limit_sim=velocity, stiffness=per(plant.kp),
                                  damping=per(plant.kd), armature=per(plant.armature))
    return ImplicitActuatorCfg(
        joint_names_expr=joints,
        stiffness=per(plant.kp),
        damping=per(plant.kd),
        effort_limit_sim=per(plant.effort),
        armature=per(plant.armature),
        friction=0.0,
    )


def robot_cfg(plant: Plant) -> ArticulationCfg:
    """The articulation of ``plant``, standing at its default pose and root height."""
    actuators = {name: _group(plant, pattern) for name, pattern in constants.ACTUATOR_GROUPS.items()}
    # main's grippers group: stiff implicit drives hold each hinge at its default in place of the missing linkage.
    actuators["grippers"] = ImplicitActuatorCfg(
        joint_names_expr=GRIPPER_JOINTS,
        effort_limit_sim=10.0,
        velocity_limit_sim=3.14,
        stiffness=100.0,
        damping=5.0,
        armature=0.01,
    )
    return ArticulationCfg(
        spawn=copy.deepcopy(_SPAWN_CFG),  # main's: H1_2_MAGPIE_USD, self-collision on
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, plant.default_root_height),
            joint_pos={**plant.default_joint_pos, "[lr]g_.*_hinge_.*": 0.0},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators=actuators,
    )


H1_2_PRETRAIN_CFG = robot_cfg(PRETRAIN)

# main's start-up event (locomanip_marl MarlEventCfg.massless_frames): the four frames without an inertial in the
# URDF, which the USD import gives 1 kg each, set to 0.01 kg.
MASSLESS_FRAME_FIX = copy.deepcopy(MarlEventCfg().massless_frames)
