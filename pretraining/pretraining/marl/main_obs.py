"""Main's actor observations synthesised from the world model's predicted state (torch only).

``MainActorObs`` builds each agent's actor vector of main's cooperative task (``LocoManip-Marl-Pulse-v0``) term for
term in its resolved cfg's order (``MarlObservationsCfg``, locomanip_marl_env_cfg.py:239-275, less the terms the task
drops: Pulse / Direct give the legs no arm terms and the arms no IMU, command or gait terms), from a ``WMState``
(``marl/main_rewards.py``), the commands, the agents' last latents and the gait clock:

- ``lowstate_gyro`` (mdp/imu.py:14-16, lowstate_imu.py:109-125): the ``torso_link`` angular velocity in its own frame
  (``imu_link`` is fixed to it unrotated), plus the per-reset gyro bias (``LowstateImuCfg.gyro_bias``) and the gyro
  noise, through main's 8 Hz Butterworth low-pass (imu_filter.py:19-61, state carried across calls). Main samples it
  at every physics substep; here once per policy step, held over it (the filter's exact zero-order hold over
  ``decimation`` substeps of a constant input), the white noise at ``gyro_noise / sqrt(decimation)``, the mean of
  ``decimation`` independent samples. Left out: the vibration (needs the leg targets' rate) and the float64 filter
  state (float here, for the gradient);
- ``lowstate_projected_gravity`` (lowstate_imu.py:130-141): gravity in the ``torso_link`` frame rotated by the
  per-reset bias rotation (``quat_bias_deg``) and a fresh noise rotation (``quat_noise_deg``, roll and pitch). Left
  out: the quaternion's lag (``quat_lag_s``, 0-20 ms);
- ``generated_commands``: the velocity command as the policies see it (``commands["velocity"]``, low-passed);
  ``arm_goal_active``: ``commands["arm_goal"]``, 1 without it; ``arm_targets_in_root_xyzw`` (observations.py:28-31):
  the pelvis-frame goals, xyzw;
- ``joint_pos_rel_measured`` (observations.py:51-57): ``q - default + encoder offset`` (``encoder_offset_rad``,
  uniform per reset, locomanip_marl_env.py:273-276) plus the term's uniform noise; ``joint_vel_rel`` plus the
  encoders' speed-scaled Gaussian noise (joint_noise.py:43-50);
- ``gait_phase_sin`` / ``_cos`` (mdp/legs_rewards.py:28-46): the clock ``t`` (s, the caller's, advanced by dt per
  step as ``episode_length_buf * step_dt``), both legs at phase 0 while the command's norm is under ``zero_cmd_eps``;
- ``applied_action``: the agent's last latent (``PulseLatentAction.applied_actions``, pulse_actions.py:157-159);
- ``body_pose_in_root_xyzw`` / ``arm_target_error_in_root`` (observations.py:60-101): the ``*_wrist_yaw_link`` poses
  in the pelvis frame (qw >= 0, xyzw) and their errors to the goals (target - wrist, axis-angle of
  ``q_target q_wrist^-1``), each with its term's uniform noise.

The terms' own noise is applied as the group's ``enable_corruption`` says (``corrupt`` overrides); the IMU's bias and
noise always, as main's ``LowstateImu`` (its cfg zeroes them). Both drawn from ``reset``'s generator.
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import torch

from pretraining.constants import JOINT_NAMES, TRACKED_BODIES
from pretraining.marl.main_rewards import TORSO, WMState, wrist_poses_in_pelvis
from pretraining.math import quat_apply_inverse, quat_conjugate, quat_from_euler_xyz, quat_mul

_TASK_DIR = (Path(__file__).resolve().parents[3] / "source" / "locomanipulation_game" / "locomanipulation_game"
             / "tasks" / "locomanip_marl")


def _load(name: str):
    """One of main's torch-only modules by path: the package's __init__ imports Isaac Lab."""
    spec = importlib.util.spec_from_file_location(f"_main_{name}", _TASK_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_imu_filter = _load("imu_filter")
_joint_noise = _load("joint_noise")


def _pattern_value(table: dict, name: str, default: float = 0.0) -> float:
    return next((v for p, v in table.items() if re.fullmatch(p, name)), default)


def _xyzw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat_wxyz[..., 1:], quat_wxyz[..., :1]], -1)


def _unique(q: torch.Tensor) -> torch.Tensor:
    return torch.where(q[..., :1] < 0, -q, q)


def axis_angle(q: torch.Tensor) -> torch.Tensor:
    """isaaclab.utils.math.axis_angle_from_quat (w taken >= 0) with a finite gradient at the identity."""
    q = _unique(q)
    v = q[..., 1:]
    norm = (v.square().sum(-1, keepdim=True) + 1e-12).sqrt()
    half = torch.atan2(norm, q[..., :1])
    return v / norm * 2.0 * half


def _func_name(term) -> str:
    return getattr(term.func, "__name__", type(term.func).__name__)


class MainActorObs:
    """Main's actor observations of a batch of imagined envs (module docstring). Stateful: ``reset`` draws the
    per-env offsets and biases and restarts the gyro low-pass, each call advances it one policy step."""

    _BUILDERS = ("lowstate_gyro", "lowstate_projected_gravity", "generated_commands", "arm_goal_active",
                 "arm_targets_in_root_xyzw", "joint_pos_rel_measured", "joint_vel_rel", "gait_phase_sin",
                 "gait_phase_cos", "applied_action", "body_pose_in_root_xyzw", "arm_target_error_in_root")

    def __init__(self, env_cfg, device, corrupt: bool | None = None):
        self.device = torch.device(device)
        self.agents = tuple(env_cfg.possible_agents)
        self.decimation = int(env_cfg.decimation)
        self.dt = env_cfg.sim.dt * env_cfg.decimation
        self.gait_period, self.zero_cmd_eps = env_cfg.gait_period, env_cfg.zero_cmd_eps
        self.groups: dict[str, list[tuple[str, str, object]]] = {}
        self.corrupt = {}
        for agent in self.agents:
            group = getattr(env_cfg.observations, agent)
            terms = []
            for name, term in vars(group).items():
                if term is None or not hasattr(term, "func"):
                    continue
                kind = _func_name(term)
                if kind not in self._BUILDERS:
                    raise ValueError(f"{agent}/{name}: main's observation {kind} is not modelled")
                self._check_params(agent, name, kind, term.params)
                terms.append((name, kind, term))
            self.groups[agent] = terms
            self.corrupt[agent] = bool(getattr(group, "enable_corruption", True)) if corrupt is None else corrupt
        robot = env_cfg.scene.robot.init_state.joint_pos
        self.default_joint_pos = torch.tensor([_pattern_value(robot, j) for j in JOINT_NAMES], device=self.device)
        offsets = env_cfg.encoder_offset_rad or {}
        self.encoder_offset_range = torch.tensor(_joint_noise.per_joint(list(JOINT_NAMES), offsets, 0.0),
                                                 device=self.device)
        self.imu = env_cfg.lowstate_imu
        # imu_filter.discrete_matrices: the exact ZOH of the 2nd-order Butterworth over one policy step
        ad, bd = _imu_filter.discrete_matrices(env_cfg.gyro_lowpass_hz, self.dt)
        self._ad, self._bd = ad.float().to(self.device), bd.float().to(self.device)
        self.gen = None

    @staticmethod
    def _check_params(agent: str, name: str, kind: str, params: dict):
        asset_cfg = params.get("asset_cfg")
        if asset_cfg is None:
            return
        if kind in ("joint_pos_rel_measured", "joint_vel_rel") and list(asset_cfg.joint_names) != list(JOINT_NAMES):
            raise ValueError(f"{agent}/{name}: joints {asset_cfg.joint_names}, not pretraining's JOINT_NAMES order")
        if kind == "body_pose_in_root_xyzw" and list(asset_cfg.body_names) != ["left_wrist_yaw_link",
                                                                              "right_wrist_yaw_link"]:
            raise ValueError(f"{agent}/{name}: bodies {asset_cfg.body_names}, not the two wrist_yaw_links")

    def widths(self) -> dict[str, int]:
        """Each agent's actor vector width (main's ``group_obs_dim``), for 32-wide latents."""
        sizes = {"lowstate_gyro": 3, "lowstate_projected_gravity": 3, "generated_commands": 3, "arm_goal_active": 1,
                 "arm_targets_in_root_xyzw": 14, "joint_pos_rel_measured": len(JOINT_NAMES),
                 "joint_vel_rel": len(JOINT_NAMES), "gait_phase_sin": 2, "gait_phase_cos": 2, "applied_action": 32,
                 "body_pose_in_root_xyzw": 14, "arm_target_error_in_root": 12}
        return {a: sum(sizes[k] for _, k, _ in terms) for a, terms in self.groups.items()}

    def reset(self, n: int, gen: torch.Generator) -> None:
        """Draw ``n`` envs' encoder offsets, gyro bias and IMU bias rotation as main's reset does
        (locomanip_marl_env.py:273-276, lowstate_imu.py:92-107) and restart the gyro low-pass."""
        self.gen, self.n = gen, n
        u = lambda *s: torch.rand(*s, generator=gen, device=gen.device).to(self.device)  # noqa: E731
        self.encoder_offset = (2.0 * u(n, len(JOINT_NAMES)) - 1.0) * self.encoder_offset_range
        self.gyro_bias = (2.0 * u(n, 3) - 1.0) * torch.tensor(self.imu.gyro_bias, device=self.device)
        bias = self.imu.quat_bias_deg
        angles = [math.radians(lo) + (math.radians(hi) - math.radians(lo)) * u(n)
                  for lo, hi in (bias[k] for k in ("roll", "pitch", "yaw"))]
        self.bias_quat = quat_from_euler_xyz(*angles)
        self._y = torch.zeros(n, 3, device=self.device)
        self._dy = torch.zeros(n, 3, device=self.device)
        self._fresh = True

    def zero_randomisation(self) -> None:
        """No offsets or biases (the comparison against a noiseless env)."""
        self.encoder_offset = torch.zeros_like(self.encoder_offset)
        self.gyro_bias = torch.zeros_like(self.gyro_bias)
        self.bias_quat = torch.zeros_like(self.bias_quat)
        self.bias_quat[:, 0] = 1.0

    def _randn(self, *shape) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen, device=self.gen.device).to(self.device)

    def _uniform(self, x: torch.Tensor, noise) -> torch.Tensor:
        u = torch.rand(x.shape, generator=self.gen, device=self.gen.device).to(self.device)
        return x + u * (noise.n_max - noise.n_min) + noise.n_min

    def _noise(self, x: torch.Tensor, noise) -> torch.Tensor:
        if noise is None:
            return x
        if hasattr(noise, "per_speed"):  # joint_noise.SpeedScaledNoiseCfg
            floor = torch.as_tensor(noise.floor, dtype=x.dtype, device=x.device)
            return x + self._randn(*x.shape) * _joint_noise.joint_vel_noise_std(x, floor, noise.per_speed)
        if hasattr(noise, "n_min"):  # AdditiveUniformNoiseCfg
            return self._uniform(x, noise)
        raise ValueError(f"noise {type(noise).__name__} is not modelled")

    def _gyro(self, state: WMState) -> torch.Tensor:
        """One policy step of the torso gyro through the low-pass (imu_filter.GyroLowpass.__call__)."""
        q_torso = state.body_quat[:, TORSO]
        u = quat_apply_inverse(q_torso, state.body_ang_vel[:, TORSO])
        u = u + self.gyro_bias + self.imu.gyro_noise / math.sqrt(self.decimation) * self._randn(*u.shape)
        if self._fresh:
            y, dy = u, torch.zeros_like(u)
        else:
            ad, bd = self._ad, self._bd
            y = ad[0, 0] * self._y + ad[0, 1] * self._dy + bd[0] * u
            dy = ad[1, 0] * self._y + ad[1, 1] * self._dy + bd[1] * u
        self._y, self._dy, self._fresh = y, dy, False
        return y

    def _gravity(self, state: WMState) -> torch.Tensor:
        """lowstate_imu.LowstateImu.quaternion / projected_gravity (lowstate_imu.py:130-141), without the lag."""
        q = state.body_quat[:, TORSO]
        noise = self._randn(q.shape[0], 2) * math.radians(self.imu.quat_noise_deg)
        q = quat_mul(quat_mul(q, self.bias_quat),
                     quat_from_euler_xyz(noise[:, 0], noise[:, 1], torch.zeros_like(noise[:, 0])))
        down = torch.tensor([0.0, 0.0, -1.0], device=q.device, dtype=q.dtype).expand(q.shape[0], 3)
        return quat_apply_inverse(q, down)

    def _phase(self, velocity: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """leg_phase (legs_rewards.py:28-36)."""
        standing = velocity[:, :3].norm(dim=-1) < self.zero_cmd_eps
        phase = torch.where(standing, torch.zeros_like(t), (t % self.gait_period) / self.gait_period)
        offset = torch.where(standing, 0.0, 0.5)
        return torch.stack([phase, (phase + offset) % 1.0], -1)

    def __call__(self, state: WMState, commands: dict[str, torch.Tensor], last_latent: dict[str, torch.Tensor],
                 t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Each agent's actor vector for ``state`` (after the step), ``commands`` (``velocity`` (n, 3), ``arm``
        (n, 2, 7) w-first, optionally ``arm_goal`` (n,)), ``last_latent`` (agent -> (n, 32), the latents just
        applied) and the clock ``t`` (n,) s. Advances the gyro low-pass once."""
        gyro = self._gyro(state)  # one filter, as main's one IMU feeds both actors
        n = state.joint_pos.shape[0]
        wrist_pos, wrist_quat = wrist_poses_in_pelvis(state)
        arm = commands["arm"]
        out = {}
        for agent, terms in self.groups.items():
            corrupt = self.corrupt[agent]
            parts = []
            for _, kind, term in terms:
                if kind == "lowstate_gyro":
                    x = gyro
                elif kind == "lowstate_projected_gravity":
                    x = self._gravity(state)
                elif kind == "generated_commands":
                    x = commands["velocity"]
                elif kind == "arm_goal_active":
                    x = commands.get("arm_goal", torch.ones(n, device=self.device)).float().view(n, 1)
                elif kind == "arm_targets_in_root_xyzw":
                    x = torch.cat([arm[..., :3], _xyzw(arm[..., 3:])], -1).reshape(n, -1)
                elif kind == "joint_pos_rel_measured":
                    x = state.joint_pos - self.default_joint_pos + self.encoder_offset
                elif kind == "joint_vel_rel":
                    x = state.joint_vel
                elif kind in ("gait_phase_sin", "gait_phase_cos"):
                    phase = 2.0 * math.pi * self._phase(commands["velocity"], t)
                    x = phase.sin() if kind == "gait_phase_sin" else phase.cos()
                elif kind == "applied_action":
                    x = last_latent.get(term.params.get("action_name"), last_latent[agent])
                elif kind == "body_pose_in_root_xyzw":
                    x = torch.cat([wrist_pos, _xyzw(_unique(wrist_quat))], -1).reshape(n, -1)
                else:  # arm_target_error_in_root: isaaclab compute_pose_error, axis-angle
                    pos_err = arm[..., :3] - wrist_pos
                    rot_err = axis_angle(quat_mul(arm[..., 3:], quat_conjugate(wrist_quat)))
                    x = torch.cat([pos_err, rot_err], -1).reshape(n, -1)
                if corrupt:
                    x = self._noise(x, getattr(term, "noise", None))
                parts.append(x)
            out[agent] = torch.cat(parts, -1)
        return out
