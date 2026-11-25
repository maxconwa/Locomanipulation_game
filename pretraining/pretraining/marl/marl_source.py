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

The world model's sim context: main's randomisation is not stage 1's (friction, COM and encoder-bias events whose
values the world model reads), so every row gets the context's mean (zeros, normalised); the world model fits the
mean over main's randomisation, as h12_train's stage 2 did for its RandomisationCfg.
"""

from __future__ import annotations

import os

import torch

from pretraining.constants import JOINT_NAMES
from pretraining.marl.command_actor import latent_to_action
from pretraining.marl.command_law import ArmGoalLaw, VelocityLaw
from pretraining.marl.config import AGENTS

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
        table = arm.table_file
        if table and os.path.isfile(table):
            self.arm_law = ArmGoalLaw.from_table(table, arm.standing_height, **ArmGoalLaw.cmd_cfg_kwargs(arm))
        else:
            # the imagined rollouts then keep the arm goal the env held at their start (no new arm segments)
            print(f"[MarlEnvSource] no arm-target table at {table!r}: imagined arm goals are not resampled", flush=True)
            self.arm_law = None
        n = self.num_envs
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

    def sim_ctx(self, dim: int) -> torch.Tensor:
        """The world model's normalised sim context per env: its mean (module docstring)."""
        return torch.zeros(self.num_envs, dim, device=self.device)

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
