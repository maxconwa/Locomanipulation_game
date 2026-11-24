"""The joint targets reaching the motors late, as main's env does (``LocoManipMarlEnv``, ``action_delay_range``).

Each env draws an integer delay of ``delay_range`` physics substeps at every reset; substep k of a policy step then
applies the target of ceil((d - k) / decimation) policy steps ago (main's rule), shifted by the encoder bias as
``BiasedJointPositionAction``. ``processed_actions`` stays the step's own target (what the policy asked for). After a
reset the history holds the pose the env starts in, read at its first step (the motion command writes that pose after
the action manager's reset).
"""

from __future__ import annotations

import torch
from isaaclab.utils import configclass

from pretraining.tasks.actions import BiasedJointPositionAction, BiasedJointPositionActionCfg
from pretraining.tasks.events import encoder_bias


class TargetDelay(BiasedJointPositionAction):
    cfg: TargetDelayCfg

    def __init__(self, cfg: TargetDelayCfg, env):
        super().__init__(cfg, env)
        low, high = cfg.delay_range
        self._decimation = env.cfg.decimation
        self.delay_steps = torch.randint(low, high + 1, (self.num_envs,), device=self.device)
        depth = -(-high // self._decimation)  # a delay of d substeps reaches back ceil(d / decimation) steps
        # targets of the last depth + 1 policy steps, newest first, in the policy's (biased) coordinates
        self._history = torch.zeros(self.num_envs, depth + 1, self.action_dim, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._substep = 0
        self._rows = torch.arange(self.num_envs, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        if self._fresh.any():
            # the pose the env starts in, as its biased encoders read it: applied, it holds the joints there
            measured = self._asset.data.joint_pos[:, self._joint_ids] + encoder_bias(self._env)
            self._history[self._fresh] = measured[self._fresh].unsqueeze(1)
            self._fresh[:] = False
        self._history[:] = self._history.roll(1, dims=1)
        self._history[:, 0] = self.processed_actions
        self._substep = 0

    def apply_actions(self):
        back = (self.delay_steps - self._substep).clamp(min=0)
        back = torch.div(back + self._decimation - 1, self._decimation, rounding_mode="floor")
        target = self._history[self._rows, back] - encoder_bias(self._env)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)
        self._substep += 1

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        n = self.delay_steps[env_ids].numel()
        low, high = self.cfg.delay_range
        self.delay_steps[env_ids] = torch.randint(low, high + 1, (n,), device=self.device)
        self._fresh[env_ids] = True


@configclass
class TargetDelayCfg(BiasedJointPositionActionCfg):
    class_type: type = TargetDelay
    delay_range: tuple[int, int] = (0, 0)
    """Physics substeps each env's targets apply late, drawn uniformly (both ends included) at each reset."""
