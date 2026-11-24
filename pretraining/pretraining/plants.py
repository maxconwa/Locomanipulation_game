"""The plants a stage-1 policy can be trained on: the simulated motor PD and its rates, per joint.

``PRETRAIN`` is the pretraining's own (``constants``: h12_train's gains, implicit PD at 0.005 s x 4, 50 Hz).
``MAIN`` is the main task's (``locomanipulation_game.assets.h1_2`` and ``locomanip_marl_env_cfg``: sport mode's gains
as explicit ``IdealPDActuatorCfg`` at 0.002 s x 5, 100 Hz, armature 0.01 everywhere), transcribed so this module
stays importable without Isaac; ``tests/test_plants.py`` checks the transcription against main's articulation.
Both clip position targets at the pretraining's ``TARGET_LIMITS`` (golem_safety's, which the robot applies whichever
policy runs).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pretraining import constants


@dataclass(frozen=True)
class Plant:
    """One plant: physics step ``sim_dt`` s, ``decimation`` physics steps per policy step, explicit (torque computed
    in Isaac Lab every physics step) or implicit (PhysX's drive) PD; per-joint dicts over ``constants.JOINT_NAMES``."""

    name: str
    sim_dt: float
    decimation: int
    explicit: bool
    kp: dict[str, float]
    kd: dict[str, float]
    effort: dict[str, float]
    armature: dict[str, float]
    velocity_limit: dict[str, float]
    action_scale: dict[str, float]
    default_joint_pos: dict[str, float]
    default_root_height: float
    target_limits: dict[str, tuple[float, float]]

    @property
    def step_dt(self) -> float:
        """s per policy step."""
        return self.sim_dt * self.decimation


def _per_joint(table: dict[str, float]) -> dict[str, float]:
    """Expand a {pattern: value} table (the first fully matching pattern applies) to {joint: value}."""
    return {j: next(v for p, v in table.items() if re.fullmatch(p, j)) for j in constants.JOINT_NAMES}


PRETRAIN = Plant(
    name="pretrain",
    sim_dt=constants.SIM_DT,
    decimation=constants.DECIMATION,
    explicit=False,
    kp=dict(constants.KP),
    kd=dict(constants.KD),
    effort=dict(constants.EFFORT),
    armature=dict(constants.ARMATURE),
    # golem_safety's; the implicit articulation leaves PhysX's velocity limits as they were (robot_cfg does not set them)
    velocity_limit=dict(constants._VELOCITY_LIMITS),
    action_scale=dict(constants.ACTION_SCALE),
    default_joint_pos=dict(constants.DEFAULT_JOINT_POS),
    default_root_height=constants.DEFAULT_ROOT_HEIGHT,
    target_limits=dict(constants.TARGET_LIMITS),
)

# main's h1_2.py _ACTUATORS (sport mode's /lowcmd gains; torso kp 500 as golem_safety's kp_max, wrist kd 1.0).
_MAIN_KP = _per_joint({
    ".*_hip_.*_joint": 250.0, ".*_knee_joint": 400.0, ".*_ankle_pitch_joint": 80.0, ".*_ankle_roll_joint": 40.0,
    "torso_joint": 500.0, ".*_shoulder_pitch_joint": 250.0, ".*_shoulder_roll_joint": 400.0,
    ".*_shoulder_yaw_joint": 80.0, ".*_elbow_joint": 80.0, ".*_wrist_.*_joint": 20.0,
})
_MAIN_KD = _per_joint({
    ".*_hip_.*_joint": 6.5, ".*_knee_joint": 10.0, ".*_ankle_pitch_joint": 1.5, ".*_ankle_roll_joint": 0.4,
    "torso_joint": 6.0, ".*_shoulder_pitch_joint": 6.0, ".*_shoulder_roll_joint": 10.0, ".*_shoulder_yaw_joint": 2.0,
    ".*_elbow_joint": 2.0, ".*_wrist_.*_joint": 1.0,
})
_MAIN_EFFORT = _per_joint({
    ".*_hip_.*_joint": 200.0, ".*_knee_joint": 300.0, ".*_ankle_pitch_joint": 60.0, ".*_ankle_roll_joint": 40.0,
    "torso_joint": 200.0, ".*_shoulder_(pitch|roll)_joint": 40.0, ".*_shoulder_yaw_joint": 18.0,
    ".*_elbow_joint": 18.0, ".*_wrist_.*_joint": 19.0,
})
_MAIN_VELOCITY = _per_joint({
    ".*_hip_.*_joint": 23.0, ".*_knee_joint": 14.0, ".*_ankle_.*_joint": 9.0, "torso_joint": 23.0,
    ".*_shoulder_(pitch|roll)_joint": 9.0, ".*_shoulder_yaw_joint": 20.0, ".*_elbow_joint": 20.0,
    ".*_wrist_.*_joint": 31.4,
})
# main's _INIT_STATE: knees bent at the stance the sport gait was fitted to, arms down.
_MAIN_DEFAULT_JOINT_POS = _per_joint({
    ".*_hip_pitch_joint": -0.375, ".*_knee_joint": 0.697, ".*_ankle_pitch_joint": -0.322, ".*": 0.0,
})
_MAIN_STANCE_PELVIS_HEIGHT = 0.9798  # m, h1_2.py STANCE_PELVIS_HEIGHT

MAIN = Plant(
    name="main",
    sim_dt=0.002,
    decimation=5,
    explicit=True,
    kp=_MAIN_KP,
    kd=_MAIN_KD,
    effort=_MAIN_EFFORT,
    armature={j: 0.01 for j in constants.JOINT_NAMES},
    velocity_limit=_MAIN_VELOCITY,
    # the pretraining's rule (constants.ACTION_SCALE) on main's gains: a unit action asks for at most a quarter of
    # the effort limit
    action_scale={j: min(0.25, 0.25 * _MAIN_EFFORT[j] / _MAIN_KP[j]) for j in constants.JOINT_NAMES},
    default_joint_pos=_MAIN_DEFAULT_JOINT_POS,
    default_root_height=_MAIN_STANCE_PELVIS_HEIGHT,
    target_limits=dict(constants.TARGET_LIMITS),
)

PLANTS: dict[str, Plant] = {p.name: p for p in (PRETRAIN, MAIN)}


def get_plant(name: str) -> Plant:
    """The plant called ``name``; ValueError listing the known names otherwise."""
    if name not in PLANTS:
        raise ValueError(f"unknown plant {name!r}; known plants: {', '.join(sorted(PLANTS))}")
    return PLANTS[name]
