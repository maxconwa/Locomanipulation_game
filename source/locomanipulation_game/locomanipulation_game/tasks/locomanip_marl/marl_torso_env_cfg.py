"""LocoManip-Marl-Torso-v0: LocoManip-Marl-Direct-v0 with the arms agent driving the torso joint too, as GOLEM's
controller does (target = default + TORSO_ACTION_SCALE x action, the torso's action first of the arms'; 15 wide). The
pure-RL arm of the fair comparison: the PULSE task's upper-body decoder owns the torso, so this one does as well.

Everything else is Marl-Direct's; arms_drive_torso (LocoManipMarlEnvCfg.sync_arms_drive_torso) adds the torso term,
its applied action to the arms' and the critic's observations and its rate to the arms' reward at their action_rate
weight.
"""

from isaaclab.utils import configclass

from .marl_direct_env_cfg import MarlDirectEnvCfg


@configclass
class MarlTorsoEnvCfg(MarlDirectEnvCfg):
    arms_drive_torso: bool = True
