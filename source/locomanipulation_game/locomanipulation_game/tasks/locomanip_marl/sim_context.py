"""The simulator's per-episode draws as one vector, xi (LocoManipMarlEnv.sim_context): what a world model needs besides
the state to predict the next step, the same for the whole episode.

Everything the env randomizes at a reset and holds until the next one, each scaled to about unit range:
- per policy joint (ALL_JOINTS_NAMES' order) the motor's gain and strength (mdp.randomize_actuator_scales: kp and kd
  x gain x strength, the effort limit x strength), recovered from the actuators against env._actuator_nominal; 1
  before the first draw and for joints no explicit PD actuator drives;
- the static friction of each foot's collision shapes (the mean over its shapes; the physics_material event) and the
  torso's mass less its default (kg / 4; the add_torso_mass event), both as PhysX holds them (env.physx_body_props:
  drawn once per env at startup, so the same for every episode of an env);
- the action delay (physics substeps / 10; LocoManipMarlEnv._delay_steps, 0 without a delay);
- the torso IMU model's draws (lowstate_imu.LowstateImu): the quaternion's bias rotation, its lag (substeps / 10),
  the gyro's bias over its range per axis, the vibration's gain and its frequency / 26 Hz;
- the pre-pose gantry's draws (mdp.prepose_reset): its initial share, release start / 3 s, release duration / 2 s,
  and the hoist's lowering: its start / 2 s, duration / 4 s and drop / 0.3 m (0 where it is not lowered); all 0
  where no rope carries anything (and without the prepose event).

SimContext.refresh runs at the end of the env's _reset_idx, after the reset events and the IMU's draws.
"""

from __future__ import annotations

import torch

# name -> size, in order
SIM_CONTEXT_TERMS = {
    "actuator_gain": 27,
    "actuator_strength": 27,
    "foot_static_friction": 2,
    "torso_mass_delta": 1,
    "action_delay": 1,
    "imu_bias_quat": 4,
    "imu_lag": 1,
    "imu_gyro_bias": 3,
    "imu_vibration_gain": 1,
    "imu_vibration_hz": 1,
    "gantry": 6,
}
SIM_CONTEXT_DIM = sum(SIM_CONTEXT_TERMS.values())

TORSO_MASS_SCALE = 4.0  # kg: the add_torso_mass draw is -2..+4
DELAY_SCALE = 10.0  # physics substeps: ACTION_DELAY_RANGE's top
IMU_LAG_SCALE = 10.0  # physics substeps: LowstateImuCfg.quat_lag_s's top at 500 Hz
VIBRATION_HZ_SCALE = 26.0
# share0, release_start (s), release_duration (s), lower_start (s), lower_duration (s), lower_drop (m)
GANTRY_SCALES = (1.0, 3.0, 2.0, 2.0, 4.0, 0.3)


class SimContext:
    """xi (num_envs, SIM_CONTEXT_DIM) of a LocoManipMarlEnv, refreshed per reset env (refresh)."""

    def __init__(self, env):
        from locomanipulation_game.assets.h1_2 import ALL_JOINTS_NAMES, FOOT_LINK_NAMES

        self.env = env
        self.robot = robot = env.scene["robot"]
        n, dev = env.num_envs, env.device
        if len(ALL_JOINTS_NAMES) != SIM_CONTEXT_TERMS["actuator_gain"]:
            raise ValueError(f"SIM_CONTEXT_TERMS expects 27 policy joints, not {len(ALL_JOINTS_NAMES)}")
        # per actuator, (its columns, those joints' places in ALL_JOINTS_NAMES) for the policy joints it drives
        index = {name: i for i, name in enumerate(ALL_JOINTS_NAMES)}
        self._columns = {}
        for name, actuator in robot.actuators.items():
            pairs = [(c, index[j]) for c, j in enumerate(actuator.joint_names) if j in index]
            if pairs:
                self._columns[name] = tuple(torch.tensor(p, device=dev) for p in zip(*pairs))
        # each foot's collision shapes in PhysX's material buffer (Isaac Lab's randomize_rigid_body_material's
        # bookkeeping: the shapes are stored body after body, in the articulation's link order)
        shapes = [robot._physics_sim_view.create_rigid_body_view(path).max_shapes
                  for path in robot.root_physx_view.link_paths[0]]
        if sum(shapes) != robot.root_physx_view.max_shapes:
            raise ValueError(f"SimContext: {sum(shapes)} link shapes, {robot.root_physx_view.max_shapes} in PhysX")
        self._foot_shapes = []
        for body in robot.find_bodies(FOOT_LINK_NAMES, preserve_order=True)[0]:
            start = sum(shapes[:body])
            if shapes[body] == 0:
                raise ValueError(f"SimContext: foot body {robot.body_names[body]} has no collision shape")
            self._foot_shapes.append(slice(start, start + shapes[body]))
        self._torso = robot.find_bodies("torso_link")[0][0]
        self._gyro_range = torch.tensor(env.cfg.lowstate_imu.gyro_bias, device=dev).clamp(min=1e-9)
        self.xi = torch.zeros(n, SIM_CONTEXT_DIM, device=dev)
        self.refresh(torch.arange(n, device=dev))

    def actuator_scales(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(gain, strength), each (len(ids), 27): stiffness = nominal kp x gain x strength, effort limit = nominal x
        strength (mdp.randomize_actuator_scales). Ones before its first draw and where no actuator was drawn."""
        gain = torch.ones(len(ids), SIM_CONTEXT_TERMS["actuator_gain"], device=self.env.device)
        strength = torch.ones_like(gain)
        nominal = getattr(self.env, "_actuator_nominal", None)
        if nominal is None:
            return gain, strength
        for name, (kp, _, limit) in nominal.items():
            if name not in self._columns:
                continue
            cols, places = self._columns[name]
            actuator = self.robot.actuators[name]
            s = actuator.effort_limit[ids][:, cols] / limit[ids][:, cols].clamp(min=1e-9)
            g = actuator.stiffness[ids][:, cols] / (kp[ids][:, cols] * s).clamp(min=1e-9)
            strength[:, places], gain[:, places] = s, g
        return gain, strength

    def refresh(self, env_ids):
        """Rebuild xi for env_ids (the envs just reset)."""
        env, robot = self.env, self.robot
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        if len(ids) == 0:
            return
        ids_cpu = ids.cpu()
        parts = list(self.actuator_scales(ids))
        masses, materials = env.physx_body_props()
        materials = materials[ids_cpu]  # (m, shapes, 3) static, dynamic, restitution
        parts.append(torch.stack([materials[:, s, 0].mean(dim=1) for s in self._foot_shapes], dim=1).to(env.device))
        mass = masses[ids_cpu, self._torso] - robot.data.default_mass[ids_cpu, self._torso]
        parts.append((mass / TORSO_MASS_SCALE).unsqueeze(1).to(env.device))
        delay = getattr(env, "_delay_steps", None)
        parts.append((delay[ids].float() / DELAY_SCALE if delay is not None and env._target_history
                      else torch.zeros(len(ids), device=env.device)).unsqueeze(1))
        imu = env.lowstate_imu
        parts += [imu.bias_quat[ids], (imu.lag[ids].float() / IMU_LAG_SCALE).unsqueeze(1),
                  imu.gyro_bias[ids] / self._gyro_range, imu.vib_gain[ids].unsqueeze(1),
                  (imu.vib_hz[ids] / VIBRATION_HZ_SCALE).unsqueeze(1)]
        prepose = getattr(env, "_prepose", None)
        if prepose is None:
            parts.append(torch.zeros(len(ids), len(GANTRY_SCALES), device=env.device))
        else:
            share0 = prepose.share0[ids]
            on = (share0 > 0.0).float()
            low = (prepose.lower_drop[ids] > 0.0).float()
            parts.append(torch.stack([share0 / GANTRY_SCALES[0], on * prepose.release_start[ids] / GANTRY_SCALES[1],
                                      on * prepose.release_duration[ids] / GANTRY_SCALES[2],
                                      low * prepose.lower_start[ids] / GANTRY_SCALES[3],
                                      low * prepose.lower_duration[ids] / GANTRY_SCALES[4],
                                      prepose.lower_drop[ids] / GANTRY_SCALES[5]], dim=1))
        self.xi[ids] = torch.cat(parts, dim=1)
