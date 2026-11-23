"""The joint position action with mjlab's encoder-bias shift.

Ported from mjlab (HybridRobotics' mjlab_research, commit 6ea3e57, Apache-2.0),
``src/mjlab/envs/mdp/actions/actions.py`` (``JointPositionAction.apply_actions``): the target sent to the joints is
the processed (clipped) target minus the encoder bias, so a policy reading biased encoders commands in the same
biased coordinates. ``processed_actions`` stays unshifted.
"""

from __future__ import annotations

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

from pretraining.constants import JOINT_NAMES
from pretraining.tasks.events import encoder_bias


class BiasedJointPositionAction(JointPositionAction):
    def __init__(self, cfg: BiasedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        if list(self._joint_names) != list(JOINT_NAMES):
            raise ValueError(f"the action's joints {self._joint_names} are not JOINT_NAMES (env.encoder_bias's order)")

    def apply_actions(self):
        target = self.processed_actions - encoder_bias(self._env)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)


@configclass
class BiasedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = BiasedJointPositionAction
