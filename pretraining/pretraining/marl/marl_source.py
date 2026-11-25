"""LocoManip-Marl-Pulse-v0 as stage 2's data source: the command encoders act in main's env through the decoders.

``MarlEnvSource(env)`` wraps the Pulse env (``gym.make("LocoManip-Marl-Pulse-v0", cfg=...)``, built with
``pulse_prior=""``: no latent prior, so the env's decoders take ``z = tanh(u)``, and the policy's latent ``z`` is
handed over as ``u = atanh(clamp(z, +-tanh 5))``, 5 the env's ``clip_actions``). What ``CommandRunner`` reads:

- ``reset()`` -> ``(state, actor_obs, commands)``; ``step(latents)`` -> ``(state, actor_obs, commands, dones)``;
- ``state`` (n, 13 + 2 x 27): the world model's gencoord full state ``[root pos (env frame), root quat wxyz, q, root
  linear and angular velocity (world), dq]`` of the root link and the 27 policy joints (``JOINT_NAMES`` order), read
  from ``scene["robot"].data`` as stage 1's track env's ``full_state`` group reads it (tasks/observations.py:116-142);
- ``actor_obs``: main's actor vectors per agent, as the env computed them (its noise, IMU model and low-pass);
- ``commands``: ``velocity`` (n, 3) the command the policies see (``ModalVelocityCommand.command``, low-passed),
  ``velocity_target`` (n, 3) the drawn one, ``arm`` (n, 2, 7) the pelvis-frame wrist goals (w-first);
- after a step: ``last_action`` (n, 27) both decoders' raw outputs of that step (the world model's action; zero on the
  rows that reset in it, which close their episode, so the world model never trains on them), ``time_outs`` (n,),
  ``clock`` (n,) s (``episode_length_buf x step_dt``, the gait clock main's observation reads);
- ``plant``, ``step_dt``, ``env_cfg`` (the resolved cfg: main's rewards, observations and command laws are read from
  it), ``obs_dims``, ``target_low`` / ``target_high`` (the decoders' clip: the plant's within GOLEM's envelope),
  ``velocity_law`` / ``arm_law`` (``command_law``), ``kernel_stages()``, ``sim_ctx(dim)``.

The world model's sim context (``sim_ctx(sim_context)``): each column of the stage-1 run's layout read from this env,
normalised with the world model's own statistics. Main's draws are read as the stage-1 specs read the fine-tune env's
(``supermimic.sim_context``): ``foot_friction`` the feet shapes' mean static friction (``physics_material``),
``action_delay`` ``env._delay_steps`` (main's own delay, physics substeps as ``TargetDelay``'s), ``actuator_gain`` /
``actuator_strength`` from the explicit actuators against ``env._actuator_nominal`` (``randomize_actuator_scales``),
``torso_mass`` (``add_torso_mass``) and ``encoder_bias`` ``env.encoder_offset`` (main's per-reset offsets, in
``JOINT_NAMES`` order). A quantity the env does not randomise reads its true, nominal value (``_NOMINAL``: main has no
torso-COM event, so ``base_com`` is 0); a column it can neither read nor knows the nominal value of is an error.
"""

from __future__ import annotations


import torch

from pretraining.constants import JOINT_NAMES
from pretraining.marl.command_actor import latent_to_action
from pretraining.marl.command_law import ArmGoalLaw, VelocityLaw
from pretraining.marl.config import AGENTS

_NOMINAL = {"base_com": 0.0, "encoder_bias": 0.0, "action_delay": 0.0, "actuator_gain": 1.0,
            "actuator_strength": 1.0, "torso_mass": 0.0}
"""Each sim-context quantity's raw value in an env that does not randomise it (no event, no delay, no offsets)."""

ARM_COMMAND = "arm_targets"  # locomanip_marl_env_cfg.ARM_COMMAND
VELOCITY_COMMAND = "base_velocity"


class MarlEnvSource:
    """The Pulse env as the stage-2 runner's data source (module docstring)."""

    def __init__(self, env):
        self.env = env
        u = env.unwrapped
        self.u = u
        self.device, self.num_envs, self.step_dt = u.device, u.num_envs, u.step_dt
        self.env_cfg = u.cfg
        if getattr(u.cfg, "pulse_prior", None):
            raise ValueError(f"the Pulse env has a latent prior ({u.cfg.pulse_prior}); stage 2 decodes z = tanh(u): "
                             "build it with pulse_prior=\"\"")
        shared = getattr(u, "_pulse_shared", None)
        if shared is None:
            raise ValueError("not a LocoManip-Marl-Pulse-v0 env: no _pulse_shared decoders")
        self.shared = shared
        self.plant = shared.plant
        self.stage1_run = shared.stage1_run
        self.clip_actions = float(u.cfg.clip_actions)
        self.joint_ids = shared.joint_ids
        self.target_low, self.target_high = shared.decoders.low.clone(), shared.decoders.high.clone()
        self.obs_dims = {a: int(u.cfg.observation_spaces[a]) for a in AGENTS}
        self.robot = u.scene["robot"]
        cmds = u.cfg.commands
        self.velocity_law = VelocityLaw.from_cfg(getattr(cmds, VELOCITY_COMMAND))
        arm = getattr(cmds, ARM_COMMAND)
        # the tables the env's arm command loaded or built (its table_file lives in a log_dir, if any)
        term = u.command_manager.get_term(ARM_COMMAND)
        tables = getattr(term, "_tables", None)
        if not tables:
            # without them the arms encoder would train on goals frozen at each rollout's start
            raise ValueError(f"the env's {ARM_COMMAND} term has no arm-target tables: the imagined arm goals are "
                             "drawn from them")
        self.arm_law = ArmGoalLaw([t.cpu() for t in tables], term.rest_pose_b.cpu(), arm.standing_height,
                                  **ArmGoalLaw.cmd_cfg_kwargs(arm))
        n = self.num_envs
        # main's action delay: its per-term target histories (newest first), in the robot's joint order
        self.has_delay = bool(getattr(u, "_target_history", None))
        self.decimation = int(u.cfg.decimation)
        self.last_action = torch.zeros(n, len(JOINT_NAMES), device=self.device)
        self.time_outs = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.rewards = {a: torch.zeros(n, device=self.device) for a in AGENTS}

    # -- reads ------------------------------------------------------------------------------------------------------

    def state(self) -> torch.Tensor:
        d, ids = self.robot.data, self.joint_ids
        return torch.cat([d.root_link_pos_w - self.u.scene.env_origins, d.root_link_quat_w, d.joint_pos[:, ids],
                          d.root_link_lin_vel_w, d.root_link_ang_vel_w, d.joint_vel[:, ids]], dim=-1)

    def commands(self) -> dict[str, torch.Tensor]:
        vel = self.u.command_manager.get_term(VELOCITY_COMMAND)
        arm = self.u.command_manager.get_term(ARM_COMMAND)
        return {"velocity": vel.command.clone(), "velocity_target": vel.vel_command_b.clone(),
                "arm": arm.command.view(self.num_envs, 2, 7).clone()}

    @property
    def clock(self) -> torch.Tensor:
        return self.u.episode_length_buf.float() * self.step_dt

    def kernel_stages(self) -> dict[str, int]:
        """The velocity command's linear and yaw kernel stages and the arm command's (main's curricula)."""
        vel = self.u.command_manager.get_term(VELOCITY_COMMAND)
        arm = self.u.command_manager.get_term(ARM_COMMAND)
        return {"lin_stage": int(vel.lin_kernel_stage), "ang_stage": int(vel.ang_kernel_stage),
                "arm_stage": int(getattr(arm, "kernel_stage", 0))}

    def prev_targets(self) -> torch.Tensor:
        """(n, 2, 27) the joint targets the next step will apply as one and two steps old, ``JOINT_NAMES`` order,
        read before it from main's histories (``_target_history``; after a reset they hold the pose it was reset to,
        the measured one or the pre-pose's default, which main writes at the reset)."""
        u = self.u
        full = torch.zeros(self.num_envs, 2, self.robot.num_joints, device=self.device)
        for term, history in zip(u._delay_terms, u._target_history):
            rows = history[:, :2] if history.shape[1] >= 2 else history[:, :1].expand(-1, 2, -1)
            full[:, :, term._joint_ids] = rows
        return full[:, :, self.joint_ids]

    def sim_ctx(self, sim_context) -> torch.Tensor:
        """The world model's normalised sim context per env, ``sim_context``'s columns read from the env (module
        docstring); ValueError for a column it cannot supply."""
        raw = torch.cat([self._raw_sim_ctx(s.name, s.dim) for s in sim_context.specs], dim=-1)
        return (raw - sim_context.mean) / sim_context.scale

    def _raw_sim_ctx(self, name: str, dim: int) -> torch.Tensor:
        from pretraining.supermimic import sim_context as sc

        u, n = self.u, self.num_envs
        if name == "encoder_bias":
            offset = getattr(u, "encoder_offset", None)
            raw = None if offset is None else offset[:, self.joint_ids]
        elif name == "base_com":
            if getattr(u.cfg.events, "base_com", None) is not None:
                raise ValueError("sim context base_com: the env draws a torso COM, which stage 2 does not read")
            raw = None
        elif name in sc._CANDIDATES:
            spec_cls = sc._CANDIDATES[name]
            if not spec_cls.available(u):
                raw = None
            else:
                specs = self.__dict__.setdefault("_ctx_specs", {})
                if name not in specs:
                    spec = spec_cls()
                    spec.bind(u)
                    specs[name] = spec
                raw = specs[name].read(u)
        else:
            raise ValueError(f"sim context {name}: the stage-2 source cannot read it from {type(u).__name__}")
        if raw is None:
            if name not in _NOMINAL:
                raise ValueError(f"sim context {name}: {type(u).__name__} does not draw it and it has no nominal value")
            raw = torch.full((n, dim), _NOMINAL[name], device=self.device)
        if raw.shape != (n, dim):
            raise ValueError(f"sim context {name}: the env gives {tuple(raw.shape)}, the world model takes {(n, dim)}")
        return raw.to(self.device, torch.float32)

    # -- the loop ---------------------------------------------------------------------------------------------------

    def reset(self):
        obs, _ = self.env.reset()
        self.last_action.zero_()
        self.time_outs.zero_()
        return self.state(), {a: obs[a] for a in AGENTS}, self.commands()

    def step(self, latents: dict[str, torch.Tensor]):
        """Step every env once with the agents' latents ``z`` (n, 32) each."""
        actions = {a: latent_to_action(latents[a], self.clip_actions) for a in AGENTS}
        obs, rewards, terminated, truncated, _ = self.env.step(actions)
        first = AGENTS[0]  # one body, one episode: every agent's done is the same
        self.time_outs = truncated[first].bool()
        dones = terminated[first].bool() | self.time_outs
        self.last_action = self.shared.decoders.last_action.clone()
        self.rewards = {a: rewards[a] for a in AGENTS}
        return self.state(), {a: obs[a] for a in AGENTS}, self.commands(), dones
