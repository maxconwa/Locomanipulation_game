"""Gaussian action distribution with a bounded std, for the legs actor.

rsl_rl's scalar std has no bounds. In a zero-reward round, entropy is the only
steady gradient on it, so it drifts up (0.47 -> 14.6 in game 5) until the
actions blow up the critic.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from rsl_rl.modules import GaussianDistribution


class ClampedGaussianDistribution(GaussianDistribution):
    """GaussianDistribution whose learnable std is kept inside [min_std, max_std].

    Same parameter name (std_param / log_std_param) as the parent, so checkpoints
    from the plain GaussianDistribution load with strict=True.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "scalar",
        min_std: float = 0.05,
        max_std: float = 1.0,
    ) -> None:
        super().__init__(output_dim, init_std=init_std, std_type=std_type)
        self.min_std = min_std
        self.max_std = max_std

    def update(self, mlp_output: torch.Tensor) -> None:
        # Project the parameter itself, not just the value used. Clamping only the
        # value would zero its gradient outside the range, so a std pushed past
        # max_std would stay pinned there for good.
        with torch.no_grad():
            if self.std_type == "scalar":
                self.std_param.clamp_(self.min_std, self.max_std)
            else:
                self.log_std_param.clamp_(math.log(self.min_std), math.log(self.max_std))
        super().update(mlp_output)


@configclass
class ClampedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Cfg for ClampedGaussianDistribution; every field is passed to its __init__."""

    class_name: str = f"{__name__}:ClampedGaussianDistribution"
    min_std: float = 0.05
    max_std: float = 1.0
