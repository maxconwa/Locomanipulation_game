"""Explicit PD with MuJoCo-style passive joint dynamics: the motor torque kp (q* - q) - kd qd, clipped at the effort
limit, and on the joint a passive torque -d qd - f tanh(qd / v0) besides it (MuJoCo's joint damping and frictionloss,
the latter smoothed over v0).

GOLEM's Robocasa MJCF (CL_Assets robosuite_assets h1_2 robot.xml) gives every H1-2 joint damping 10 N m s/rad,
frictionloss 0.2 N m and armature 0.1 kg m^2; the training plant had none of the first two and armature 0.01, so in
MuJoCo the legs were damped 2.5-4x more than their own PD (hip kd 2.5, knee 4) and Mid360 barely stepped. The passive
terms are per env and joint, drawn by mdp.randomize_joint_passive_dynamics (armature goes to PhysX directly).

applied_effort stays the motor's torque (what the GOLEM e-stop monitor, the torque bounds and joint_effort read, as
the robot's tau_est), and the joint gets applied_effort plus the passive torque, as MuJoCo adds its passive forces to
the actuator's.
"""

from __future__ import annotations

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class PassivePDActuator(IdealPDActuator):
    cfg: PassivePDActuatorCfg

    def __init__(self, cfg: PassivePDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        shape = (self._num_envs, self.num_joints)
        self.passive_damping = torch.full(shape, float(cfg.passive_damping), device=self._device)
        self.passive_friction = torch.full(shape, float(cfg.passive_friction), device=self._device)
        self.passive_effort = torch.zeros(shape, device=self._device)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action = super().compute(control_action, joint_pos, joint_vel)
        self.passive_effort = -self.passive_damping * joint_vel - self.passive_friction * torch.tanh(
            joint_vel / self.cfg.friction_velocity
        )
        control_action.joint_efforts = self.applied_effort + self.passive_effort
        return control_action


@configclass
class PassivePDActuatorCfg(IdealPDActuatorCfg):
    class_type: type = PassivePDActuator

    passive_damping: float = 0.0
    """N m s/rad: the joint's viscous damping (MuJoCo damping), every env until randomized."""
    passive_friction: float = 0.0
    """N m: the joint's dry friction (MuJoCo frictionloss), every env until randomized."""
    friction_velocity: float = 0.05
    """rad/s: the tanh smoothing of the dry friction (MuJoCo solves it as a constraint; at 500 Hz explicit steps a
    sign() would chatter)."""
