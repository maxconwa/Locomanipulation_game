"""Observation groups shared by both players (legs and adversary).

`actions` is mdp.last_action, the agent's own action vector: 12 wide for the
legs, 14 for the adversary. FrozenPolicyAction replaces the `actions` term when
a group feeds a frozen policy, so keep that name.
No height_scan: the actor is blind, like ALMI's lower body. The scanner stays in
the scene for the base_height and feet_swing_height rewards.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from locomanipulation_game.assets.h1_2 import ALL_JOINTS_NAMES

from .. import mdp

HISTORY_LENGTH = 1

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            history_length=HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        base_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=HISTORY_LENGTH, flatten_history_dim=True)
        gait_phase = ObsTerm(
            func=mdp.gait_phase_sin, params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()