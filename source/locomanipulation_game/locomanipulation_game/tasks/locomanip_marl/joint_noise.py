"""The H1-2's joint encoders as /lowstate's motor_state reports them, the joint observations every task's actors read.

Measured on the robot (2026-11-23 bags of LegsFastEnv60k_s42, ~/bags/fastenv60k_2026-11-23/): the velocity noise is
dq minus the derivative of q over 20 ms, the position noise q minus its 20 ms mean.
- Standing still (standing2, 18 min): velocity noise 0.0004-0.0011 rad/s on the hips and knees, 0.010-0.014 on the
  ankles (their speeds come through the parallel linkage), 0.009 on the arms; position noise 1e-5 rad.
- Walking (lmg_..._1440 / _1445, ~25 s each at the 0.70 s stride, legs' |dq| 0.55 rad/s mean): velocity noise 0.33-0.72
  on the hips and knees, 0.15-0.40 on the ankles, 0.05 on the arms; position noise 0.0012 (at most 0.0025). Noise-free
  MuJoCo walking gives 0.05 on the same measure, so this is the encoders', not the gait's own high frequencies.
So the velocity noise grows with the joint's speed: a Gaussian of std floor + JOINT_VEL_NOISE_PER_SPEED x |dq| (about
0.6 x the walking legs' 0.55 rad/s), the floor per joint group above the still robot's. Until 2026-11-23 every run
trained on uniform +-1.5 rad/s (std 0.87) on every joint and +-0.01 rad on the positions, 2.5x the walking robot's
velocity noise and 100-1000x the standing one's: MuJoCo with that noise jittered its targets ~0.017 rad per step
standing, the robot 0.001-0.002 (the engage grid's noise ablation, 2026-11-23).

Encoder offsets: each leg joint's encoder reads its angle plus an offset drawn per reset (ENCODER_OFFSET_RAD), as a
motor zeroed slightly off does: the policy sees angle + offset, and the motor, holding its measured angle at the
target, puts the joint at target - offset (LocoManipMarlEnv applies both). The ankle pitches draw twice the others: the
2026-11-20 and 11-23 bags' soles disagree by ~3 deg in pitch by kinematics, a suspected right ankle pitch offset.

The MuJoCo evals (evals/mujoco/sim.py) load this file by path, outside Isaac Lab: the numbers and per_joint then, not
the noise cfg.
"""

from __future__ import annotations

import re

import torch

try:
    from isaaclab.utils import configclass
    from isaaclab.utils.noise import NoiseCfg
except ImportError:  # loaded by path outside Isaac Lab (the MuJoCo evals): the numbers only
    NoiseCfg = None

# rad/s: the velocity noise's floor per joint (first matching pattern; JOINT_VEL_NOISE_FLOOR_DEFAULT otherwise),
# about 2x the still robot's
JOINT_VEL_NOISE_FLOOR = {r".*_ankle_.*_joint": 0.02, r".*_(shoulder|elbow|wrist)_.*_joint": 0.015}
JOINT_VEL_NOISE_FLOOR_DEFAULT = 0.002
# rad/s of velocity noise std per rad/s of the joint's speed
JOINT_VEL_NOISE_PER_SPEED = 0.6
# rad: the position noise, uniform +- (the walking robot's worst 0.0025)
JOINT_POS_NOISE = 0.003
# rad: each leg joint's encoder offset, uniform +- per reset (first matching pattern); joints not named draw none
ENCODER_OFFSET_RAD = {r".*_ankle_pitch_joint": 0.04, r".*_(hip_.*|knee|ankle_roll)_joint": 0.02}


def per_joint(names: list[str], table: dict[str, float], default: float) -> list[float]:
    """Each name's value: the first pattern of table it matches in full, else default."""
    return [next((v for p, v in table.items() if re.fullmatch(p, n)), default) for n in names]


def joint_vel_noise_std(dq, floor, per_speed: float = JOINT_VEL_NOISE_PER_SPEED):
    """The velocity noise's std at speeds dq (a tensor or an array): floor + per_speed * |dq|, per column."""
    return floor + per_speed * abs(dq)


def speed_scaled_gaussian_noise(data: torch.Tensor, cfg: SpeedScaledNoiseCfg) -> torch.Tensor:
    """data plus Gaussian noise of std floor + per_speed * |data|, per column."""
    if not isinstance(cfg.floor, torch.Tensor) or cfg.floor.device != data.device:
        cfg.floor = torch.as_tensor(cfg.floor, dtype=data.dtype, device=data.device)
    return data + torch.randn_like(data) * joint_vel_noise_std(data, cfg.floor, cfg.per_speed)


if NoiseCfg is not None:

    @configclass
    class SpeedScaledNoiseCfg(NoiseCfg):
        """Additive Gaussian noise whose std grows with the observed value: floor (one per column, or one for all)
        plus per_speed times its magnitude (the joint encoders' velocities, module docstring)."""

        func = speed_scaled_gaussian_noise
        floor: torch.Tensor | list[float] | float = JOINT_VEL_NOISE_FLOOR_DEFAULT
        per_speed: float = JOINT_VEL_NOISE_PER_SPEED

    def joint_vel_noise(names: list[str], scale: float = 1.0) -> SpeedScaledNoiseCfg:
        """The encoders' velocity noise on the joints names, in that order, times scale."""
        floor = per_joint(names, JOINT_VEL_NOISE_FLOOR, JOINT_VEL_NOISE_FLOOR_DEFAULT)
        return SpeedScaledNoiseCfg(floor=[scale * f for f in floor], per_speed=scale * JOINT_VEL_NOISE_PER_SPEED)
