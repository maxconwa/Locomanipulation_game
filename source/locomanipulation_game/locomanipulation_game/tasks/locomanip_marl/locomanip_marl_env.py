"""DirectMARLEnv that runs Isaac Lab's managers inside, so manager-based mdp terms work unchanged.

DirectMARLEnv owns the step loop, the scene and the event manager; this class builds the other managers and maps
them onto the per-agent dicts:

    actions:      one ActionManager, the agents' terms concatenated in possible_agents order (cfg.agent_action_terms);
                  with cfg.action_delay_range each env's joint targets reach the robot that many physics substeps late
    observations: agent -> its observation group; the env state is the `critic` group; the `world` group, computed
                  with them, is a world model's state (world_state; never an agent's observation or the env state)
    rewards:      agent -> its own RewardManager, floored at 0, plus termination_penalty on terminating steps and
                  golem_estop_penalty on the step a GOLEM e-stop trip starts
    dones:        one TerminationManager, shared (one body, one episode)

After every step it moves the arm command by the pelvis's motion (on the robot, FAST-LIO's odometry does). Each
episode's length is drawn at its reset (cfg.episode_length_range_s). While training, every task is navigation until
the robot walks like sport mode (cfg.warm_start_*, _update_warm_start); then the arm goals start. GolemEstopMonitor
(golem_safety.py) checks GOLEM's e-stops after every physics substep; trips are logged and charged, never end an
episode.

For a world model (SuperMimic, supermimic/) the env also exposes world_state() (the `world` group), sim_context() (the
episode's randomization draws, sim_context.py), goal_state(agent) (goals.py), task_rewards (each agent's reward
without the fall penalty), trip_events (per agent, whether a GOLEM e-stop trip of its started this step: the
golem_estop_penalty in task_rewards) and obs_noise_scale(agent) (each actor observation's noise half-width).
"""

from __future__ import annotations

import os
import torch
from collections.abc import Sequence

from isaaclab.envs import DirectMARLEnv
from isaaclab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms
from isaaclab.utils.noise import UniformNoiseCfg

from . import goals
from .golem_safety import GolemEstopMonitor
from .lowstate_imu import LowstateImu
from .locomanip_marl_env_cfg import ARM_COMMAND, LocoManipMarlEnvCfg
from .mdp.observations import target_history_depth
from .sim_context import SimContext


class LocoManipMarlEnv(DirectMARLEnv):
    cfg: LocoManipMarlEnvCfg

    def __init__(self, cfg: LocoManipMarlEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.sync_derived()
        super().__init__(cfg, render_mode, **kwargs)
        self._set_soft_joint_limits()
        # each episode's own length in policy steps (cfg.episode_length_range_s; mdp.episode_limit_reached), drawn at
        # its reset; None: every episode max_episode_length (episode_length_range_s None)
        self.episode_limit = (torch.full((self.num_envs,), self.max_episode_length, device=self.device)
                              if cfg.episode_length_range_s is not None else None)

        # the torso IMU as /lowstate reports it (mdp.lowstate_gyro, mdp.lowstate_projected_gravity read it; before the
        # observation manager): one sample per physics substep, /lowstate's 500 Hz
        self.lowstate_imu = LowstateImu(self.cfg.lowstate_imu, self.cfg.gyro_lowpass_hz, self.physics_dt, self.num_envs,
                                        self.device)
        # -- the pre-pose resets' gantry (mdp.prepose_reset, the events' prepose term; None without it): before the
        # observation manager, whose world group reads its share
        prepose = getattr(self.cfg.events, "prepose", None)
        self._prepose = self.event_manager.get_term_cfg("prepose").func if prepose is not None else None

        # After super() has started the sim, in ManagerBasedRLEnv.load_managers' order: observations read commands
        # and actions, rewards read terminations.
        self.command_manager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        self.reward_managers = {
            agent: RewardManager(getattr(self.cfg.rewards, agent), self) for agent in self.cfg.possible_agents
        }
        for agent, manager in self.reward_managers.items():
            print(f"[INFO] Reward Manager ({agent}): ", manager)
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)
        self._robot = self.scene["robot"]
        # the warm start: exponential means over the ending episodes of their falls and their minutes (falls per
        # minute is their ratio, from 1: a start far above any threshold that the means forget within ~10k episodes),
        # the sport-mode gate's consecutive passing checks and the last check's metrics
        self._walk_falls = torch.tensor(1.0, device=self.device)
        self._walk_minutes = torch.tensor(1.0, device=self.device)
        self._sport_streak = 0
        self._sport_metrics: dict[str, float] = {}
        self._warm_start_log: dict[str, torch.Tensor] = {}
        self.golem_monitor = GolemEstopMonitor(self._robot, self.cfg.golem_estop, self.num_envs, self.device)
        self.golem_log: dict[str, torch.Tensor] = {}
        # this step's GOLEM e-stop trips (mdp.golem_estop) and the last step's, for the penalty on a trip's first step
        self.golem_tripped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._golem_tripped_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # the same per agent: the envs where one of its own joints tripped (the torso is both's)
        self.golem_tripped_by = {agent: torch.zeros_like(self.golem_tripped) for agent in self.cfg.possible_agents}
        self._golem_tripped_by_prev = {agent: torch.zeros_like(self.golem_tripped) for agent in self.cfg.possible_agents}

        group_dims = self.observation_manager.group_obs_dim
        self.cfg.observation_spaces = {agent: group_dims[agent][0] for agent in self.cfg.possible_agents}
        self.cfg.state_space = group_dims["critic"][0]
        terms = [term for agent in self.cfg.possible_agents for term in self.cfg.agent_action_terms[agent]]
        if terms != list(self.action_manager.active_terms):
            raise ValueError(f"Agents' action terms {terms} are not the ActionManager's {self.action_manager.active_terms}")
        self.cfg.action_spaces = {
            agent: sum(self.action_manager.get_term(term).action_dim for term in self.cfg.agent_action_terms[agent])
            for agent in self.cfg.possible_agents
        }
        self._configure_env_spaces()
        print(
            f"[INFO] Spaces: observations {self.cfg.observation_spaces}, state {self.cfg.state_space},"
            f" actions {self.cfg.action_spaces}"
        )

        # -- the arm command's odometry: the pelvis pose at the last step; envs whose last pose belongs to an
        # earlier episode
        self._arm_command = self.command_manager.get_term(ARM_COMMAND)
        self._arm_command.goals_enabled = not self.cfg.training
        self._prev_root_pos = self._robot.data.root_pos_w.clone()
        self._prev_root_quat = self._robot.data.root_quat_w.clone()
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.state_checkpoint_path:
            print(f"[INFO] Curriculum state: {self.cfg.state_checkpoint_path}")
            self._load_state(torch.load(self.cfg.state_checkpoint_path, map_location=self.device, weights_only=False)["env_state"])

        self._obs_buf: dict[str, torch.Tensor] = {}
        # the groups computed every step: the agents', the env state's and, if the cfg has it, the world model's
        self._obs_groups = list(self.cfg.possible_agents) + ["critic"]
        if "world" in self.observation_manager.active_terms:
            self._obs_groups.append("world")
        # each agent's reward this step without the fall penalty, and whether a GOLEM e-stop trip of its started
        # this step (charged golem_estop_penalty in task_rewards) (_get_rewards)
        self.task_rewards: dict[str, torch.Tensor] = {}
        self.trip_events: dict[str, torch.Tensor] = {}

        # -- action delay: per action term, the last targets, newest first; per env, how many substeps late they apply
        self._delay_terms = [self.action_manager.get_term(name) for name in self.action_manager.active_terms]
        self._target_history: list[torch.Tensor] = []
        self._substep = 0
        if self.cfg.action_delay_range is not None:
            low, high = self.cfg.action_delay_range
            self._delay_steps = torch.randint(low, high + 1, (self.num_envs,), device=self.device)
            depth = target_history_depth(self.cfg.action_delay_range, self.cfg.decimation)
            self._target_history = [
                self._robot.data.joint_pos[:, term._joint_ids].unsqueeze(1).repeat(1, depth + 1, 1).clone()
                for term in self._delay_terms
            ]
            print(f"[INFO] Action delay: {low}-{high} physics substeps per env, drawn at each reset")

        # -- the legs' target rate drives the torso IMU's vibration
        self._legs_term = self.action_manager.get_term("joint_pos")
        self._prev_leg_target = self._legs_term._applied.clone()
        print(f"[INFO] Torso IMU (/lowstate): {self.cfg.lowstate_imu}; gyro low-pass {self.cfg.gyro_lowpass_hz:g} Hz")

        # -- the episode's randomization draws (sim_context.py), last: it reads the delay, the IMU and the gantry
        self._sim_context = SimContext(self)

    """
    DirectMARLEnv hooks.
    """

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
        joint_action = torch.cat([actions[agent] for agent in self.cfg.possible_agents], dim=1)
        if not torch.isfinite(joint_action).all():
            # a NaN joint target hangs the PhysX solver
            raise RuntimeError(f"Non-finite actions at step {self.common_step_counter}")
        self.action_manager.process_action(joint_action.clamp(-self.cfg.clip_actions, self.cfg.clip_actions))
        target = self._legs_term.processed_actions
        self.lowstate_imu.target_rate = (target - self._prev_leg_target).square().mean(dim=1).sqrt()
        self._prev_leg_target = target.clone()
        # the policies observe what they asked for (applied_actions); the robot gets it _delay_steps substeps late
        for term, history in zip(self._delay_terms, self._target_history):
            history[:] = history.roll(1, dims=1)
            history[:, 0] = term.processed_actions
        self._substep = 0

    def _apply_action(self) -> None:
        if self._substep > 0:
            # the IMU sample the previous substep left; the step's last one is taken in _get_observations
            self._sample_imu()
            # the state the previous substep left (mdp.golem_estop checks the last one's; at substep 0 that is the
            # state it has already checked: a second look counted its trip on this step too, and the stale flag then
            # hid the next real trip's start from the penalty)
            self.golem_monitor.update()
        if self._target_history:
            # substep k of this policy step applies the targets of ceil((d - k) / decimation) policy steps ago
            back = (self._delay_steps - self._substep).clamp(min=0)
            back = torch.div(back + self.cfg.decimation - 1, self.cfg.decimation, rounding_mode="floor")
            rows = torch.arange(self.num_envs, device=self.device)
            for term, history in zip(self._delay_terms, self._target_history):
                term._processed_actions = history[rows, back]
        if self._prepose is not None and self._substep == 0:
            # the rope's force, set once per policy step and held over its substeps (set every substep it cost ~6 ms
            # each, ~30 ms per step, about a fifth of the step; the rope's ~1 Hz dynamics don't need 500 Hz)
            self._prepose.apply_support(self.episode_length_buf * self.step_dt)
        self._substep += 1
        self.action_manager.apply_action()

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.termination_manager.compute()
        terminated = self.termination_manager.terminated
        time_outs = self.termination_manager.time_outs
        self._update_warm_start(terminated | time_outs)
        return (
            {agent: terminated for agent in self.cfg.possible_agents},
            {agent: time_outs for agent in self.cfg.possible_agents},
        )

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        terminated = self.termination_manager.terminated.float()
        # a GOLEM e-stop trip costs the agent whose joint tripped golem_estop_penalty once, on its first step, after the
        # floor as a fall's penalty does (from the start: held back to the end of the warm start, a policy tripped on
        # 11% of its steps). The other agent pays nothing for it
        new_trip = self.golem_tripped & ~self._golem_tripped_prev
        self._golem_tripped_prev = self.golem_tripped.clone()
        self.golem_log["Safety/trip_starts"] = new_trip.float().mean()
        trip_cost = {}
        for agent, tripped in self.golem_tripped_by.items():
            new = tripped & ~self._golem_tripped_by_prev[agent]
            self._golem_tripped_by_prev[agent] = tripped.clone()
            self.trip_events[agent] = new
            trip_cost[agent] = self.cfg.golem_estop_penalty * new.float()
            self.golem_log[f"Safety/trip_starts_{agent}"] = new.float().mean()
        rewards = {}
        for agent, manager in self.reward_managers.items():
            reward = manager.compute(dt=self.step_dt)
            if torch.isnan(reward).any():
                bad = torch.isnan(manager._step_reward).any(dim=0).nonzero(as_tuple=True)[0].tolist()
                raise RuntimeError(f"NaN in {agent} reward terms {[manager._term_names[i] for i in bad]}")
            # the reward without the fall's penalty (a world model predicts it, and the fall apart): floored at 0, the
            # trip cost
            self.task_rewards[agent] = torch.clamp(reward, min=0.0) + trip_cost[agent]
            rewards[agent] = self.task_rewards[agent] + self.cfg.termination_penalty * terminated
        return rewards

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # the last substep's IMU sample (after a reset, the refreshed reading the model restarts from)
        self._sample_imu()
        self._update_odometry()
        self.command_manager.compute(dt=self.step_dt)
        self._obs_buf = {g: self.observation_manager.compute_group(g) for g in self._obs_groups}
        self.extras.setdefault("log", {}).update(self._warm_start_log)
        self.extras["log"].update(self.golem_log)
        return {agent: self._obs_buf[agent] for agent in self.cfg.possible_agents}

    def _get_states(self) -> torch.Tensor:
        # computed in _get_observations, which step() and reset() call before state()
        return self._obs_buf["critic"]

    def _reset_idx(self, env_ids: Sequence[int]):
        # ManagerBasedRLEnv._reset_idx's order: curriculum before the scene reset, managers after
        self.curriculum_manager.compute(env_ids=env_ids)
        super()._reset_idx(env_ids)  # scene, reset events, episode_length_buf
        if self.episode_limit is not None:
            ids = torch.as_tensor(env_ids, device=self.device)
            seconds = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.episode_length_range_s)
            steps = torch.ceil(seconds / self.step_dt)
            self.episode_limit[ids] = steps.clamp(max=self.max_episode_length).long()
        self._refresh_sensors_after_reset(env_ids)
        self._fresh[env_ids] = True
        self.lowstate_imu.reset(env_ids)
        self.golem_monitor.reset(env_ids)
        self._golem_tripped_prev[env_ids] = False
        for prev in self._golem_tripped_by_prev.values():
            prev[env_ids] = False

        log = {}
        log.update(self.observation_manager.reset(env_ids))
        log.update(self.action_manager.reset(env_ids))
        self._prev_leg_target[env_ids] = self._legs_term._applied[env_ids]  # the measured pose, the first rate's base
        if self._target_history:
            low, high = self.cfg.action_delay_range
            self._delay_steps[env_ids] = torch.randint(low, high + 1, (len(env_ids),), device=self.device)
            for term, history in zip(self._delay_terms, self._target_history):
                history[env_ids] = self._robot.data.joint_pos[env_ids][:, term._joint_ids].unsqueeze(1)
        for agent, manager in self.reward_managers.items():
            for key, value in manager.reset(env_ids).items():
                log[key.replace("Episode_Reward/", f"Episode_Reward/{agent}/", 1)] = value
        log.update(self.curriculum_manager.reset(env_ids))
        log.update(self.command_manager.reset(env_ids))
        if self._prepose is not None:
            log.update(self._hand_over(torch.as_tensor(env_ids, device=self.device)))
        log.update(self.event_manager.reset(env_ids))
        log.update(self.termination_manager.reset(env_ids))
        # the episode's draws are all in by now (the reset events', the delay's, the IMU's, the gantry's)
        if getattr(self, "_sim_context", None) is not None:
            self._sim_context.refresh(env_ids)
        # skrl logs only one-element tensors; the managers also return floats
        self.extras["log"] = {
            key: value if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=self.device)
            for key, value in log.items()
        }

    def _hand_over(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """At the pre-pose resets (mdp.prepose_reset), what lmg_deploy's reset hands the policies besides what every
        reset already does: a zero velocity command (and low-pass) until its next resample, and the targets before
        the first policy step at the default pose, the pre-pose's hold, not the measured one."""
        ids = env_ids[self._prepose.active[env_ids]]
        if len(ids) > 0:
            velocity = self.command_manager.get_term("base_velocity")
            velocity.vel_command_b[ids] = 0.0
            velocity.vel_command_filtered[ids] = 0.0
            velocity.is_standing_env[ids] = True
            default = self._robot.data.default_joint_pos[ids]
            for term, history in zip(self._delay_terms, self._target_history):
                history[ids] = default[:, term._joint_ids].unsqueeze(1)
        prepose = self._prepose.active[env_ids].float()
        supported = (self._prepose.share0[env_ids] > 0.0).float()
        return {"Prepose/reset_share": prepose.mean(), "Prepose/supported_share": supported.mean()}

    def physx_body_props(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(masses (num_envs, bodies), material properties (num_envs, shapes, 3)) of the robot as PhysX holds them, on
        the CPU. Read once and kept when no reset event redraws them (the mass and friction draws at startup); read anew
        at every call otherwise. A read copies every env's values through the CPU (~10 ms at 4096 envs)."""
        cached = getattr(self, "_physx_props", None)
        if cached is not None:
            return cached
        view = self._robot.root_physx_view
        props = (view.get_masses(), view.get_material_properties())
        redrawn = {"randomize_rigid_body_mass", "randomize_rigid_body_material"}
        reset_terms = self.event_manager._mode_term_cfgs.get("reset", [])
        names = {getattr(c.func, "__name__", type(c.func).__name__) for c in reset_terms}
        if not names & redrawn:
            self._physx_props = props
        return props

    def _sample_imu(self):
        """One /lowstate IMU sample of the current physics state (lowstate_imu.LowstateImu.sample)."""
        data = self.scene["imu"].data
        self.lowstate_imu.sample(data.quat_w, data.ang_vel_b)

    def support_share(self) -> torch.Tensor:
        """Per env, the share of the robot's weight a pre-pose gantry carries now (mdp.prepose_reset; 0 without it)."""
        if self._prepose is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self._prepose.support_share(self.episode_length_buf * self.step_dt)

    """
    World-model hooks (SuperMimic, supermimic/).
    """

    def world_state(self) -> torch.Tensor:
        """(num_envs, S): this step's `world` observation group (MarlObservationsCfg.WorldCfg), the true state in the
        pelvis frame, no noise; computed with the agents' observations."""
        return self._obs_buf["world"]

    def sim_context(self) -> torch.Tensor:
        """(num_envs, C): each env's randomization draws for its episode (sim_context.SIM_CONTEXT_TERMS), constant
        between its resets."""
        return self._sim_context.xi

    def goal_state(self, agent: str) -> torch.Tensor:
        """(num_envs, Gs): agent's goal state now (goals.goal_state)."""
        return goals.goal_state(self, agent)

    def obs_noise_scale(self, agent: str) -> torch.Tensor:
        """(obs_dim,): the half-width of the uniform noise on each dimension of agent's observation (the observation
        manager's term cfgs, times each term's scale; 0 for terms without noise), to add the sensors' noise back to a
        noise-free observation."""
        manager = self.observation_manager
        widths = []
        for term_cfg, dims in zip(manager._group_obs_term_cfgs[agent], manager._group_obs_term_dim[agent]):
            noise = term_cfg.noise
            if noise is None:
                width = torch.zeros(dims, device=self.device)
            elif isinstance(noise, UniformNoiseCfg) and noise.operation == "add":
                low = torch.as_tensor(noise.n_min, dtype=torch.float, device=self.device)
                high = torch.as_tensor(noise.n_max, dtype=torch.float, device=self.device)
                width = (0.5 * (high - low)).expand(dims).clone()
            else:
                raise NotImplementedError(f"obs_noise_scale: {agent}'s noise {type(noise).__name__}")
            if term_cfg.scale is not None:
                width = width * torch.as_tensor(term_cfg.scale, dtype=torch.float, device=self.device).abs()
            widths.append(width.reshape(-1))
        widths = torch.cat(widths)
        if widths.numel() != manager.group_obs_dim[agent][0]:
            raise ValueError(f"obs_noise_scale: {widths.numel()} widths for {agent}'s {manager.group_obs_dim[agent]}")
        return widths

    """
    Arm-command odometry and the warm start.
    """

    def _update_odometry(self):
        """Move each arm goal by the pelvis's motion since the last step, as the robot's SLAM odometry moves it (reset
        envs are skipped: their last pose is the old episode's, and the reset gave them an exact command). While
        training, save the curriculum state every cfg.state_save_every steps."""
        root_pos = self._robot.data.root_pos_w.clone()
        root_quat = self._robot.data.root_quat_w.clone()
        delta_pos, delta_quat = subtract_frame_transforms(self._prev_root_pos, self._prev_root_quat, root_pos, root_quat)
        self._arm_command.apply_pelvis_motion(delta_pos, delta_quat, env_mask=~self._fresh)
        self._prev_root_pos, self._prev_root_quat = root_pos, root_quat
        self._fresh[:] = False
        if (self.cfg.training and self.common_step_counter % self.cfg.state_save_every == 0
                and getattr(self.cfg, "log_dir", None)):
            self.save_state()

    def _update_warm_start(self, ending: torch.Tensor):
        """End the warm start (goals_enabled: the arm goals and the arms' residual ramp in) once the robot walks like
        sport mode: the sport-mode gate passes (_sport_gate)
        and, if cfg.warm_start_max_falls_per_min is set, over the ending episodes (exponential means, decay
        cfg.warm_start_episode_decay per episode) at most that many falls per minute; or at
        cfg.warm_start_deadline_steps. Per minute, not per episode: the share of episodes that
        fall grows with their length. Runs before the resets, while the velocity command still holds the ending
        episodes' sums and episode_length_buf their lengths."""
        cfg = self.cfg
        ids = ending.nonzero(as_tuple=False).flatten()
        if len(ids):
            fell = self.termination_manager.get_term("fell")[ids]
            minutes = self.episode_length_buf[ids].float() * self.step_dt / 60.0
            keep = cfg.warm_start_episode_decay ** len(ids)
            self._walk_falls = keep * self._walk_falls + (1.0 - keep) * fell.float().mean()
            self._walk_minutes = keep * self._walk_minutes + (1.0 - keep) * minutes.mean()
        falls_per_min = self._walk_falls / self._walk_minutes
        sport = self._sport_gate()  # every check, so the metrics are logged before and after the goals start
        max_falls = cfg.warm_start_max_falls_per_min
        walks = (max_falls is None or bool(falls_per_min <= max_falls)) and sport
        deadline = 0 < cfg.warm_start_deadline_steps <= self.common_step_counter
        arm = self._arm_command
        if not arm.goals_enabled and (walks or deadline):
            arm.goals_enabled = True
            arm.goals_enabled_at = self.common_step_counter  # the arm-goal share ramps up from here
            metrics = ", ".join(f"{k} {v:.3f}" for k, v in self._sport_metrics.items() if not k.startswith("n_"))
            print(f"[INFO] Warm start over at step {self.common_step_counter}{' (deadline)' if not walks else ''}:"
                  f" {falls_per_min.item():.3f} falls per minute; sport metrics {metrics or 'none yet'};"
                  " the arm goals start")
        self._warm_start_log["Curriculum/warm_start_falls_per_min"] = falls_per_min
        self._warm_start_log["Curriculum/arm_goals_enabled"] = torch.tensor(float(arm.goals_enabled), device=self.device)

    def _sport_gate(self) -> bool:
        """Every cfg.warm_start_sport_every steps, the velocity command's sport-mode metrics (training's own
        navigation, ModalVelocityCommand.sport_metrics) against the cfg.warm_start_* thresholds; True once
        cfg.warm_start_sport_checks checks in a row have passed. The thresholds can be changed while training: a
        warm_start_gate.yaml in the run's log dir, read at each check, overrides any of the cfg's warm_start_* fields
        it names (the arm goals still start only once)."""
        cfg = self.cfg
        if self.common_step_counter % cfg.warm_start_sport_every:
            return self._sport_streak >= cfg.warm_start_sport_checks
        self._read_gate_overrides()
        m = self.command_manager.get_term("base_velocity").sport_metrics()
        self._sport_metrics = m
        for key, value in m.items():
            self._warm_start_log[f"Metrics/sport/{key}"] = torch.tensor(value, device=self.device)
        speed, turn = cfg.warm_start_speed_ratio_range, cfg.warm_start_turn_ratio_range
        passes = (m["turn_sense"] >= cfg.warm_start_min_turn_sense
                  and (speed is None or speed[0] <= m["speed_ratio"] <= speed[1])
                  and (turn is None or turn[0] <= m["turn_ratio"] <= turn[1])
                  and m["cos"] >= cfg.warm_start_min_sport_cos
                  and m["ang_err"] <= cfg.warm_start_max_ang_err
                  and m["lin_err"] <= cfg.warm_start_max_lin_err)  # nan compares False
        self._sport_streak = self._sport_streak + 1 if passes else 0
        self._warm_start_log["Curriculum/sport_gate_streak"] = torch.tensor(float(self._sport_streak), device=self.device)
        return self._sport_streak >= cfg.warm_start_sport_checks

    def _read_gate_overrides(self):
        log_dir = getattr(self.cfg, "log_dir", None)
        path = os.path.join(log_dir, "warm_start_gate.yaml") if log_dir else None
        if not path or not os.path.isfile(path):
            return
        import yaml

        try:
            with open(path) as f:
                values = yaml.safe_load(f) or {}
        except yaml.YAMLError as error:
            print(f"[WARN] {path}: {error}")
            return
        for key, value in values.items():
            if not key.startswith("warm_start_") or not hasattr(self.cfg, key):
                print(f"[WARN] {path}: no cfg field {key}")
                continue
            value = tuple(value) if isinstance(value, list) else value
            if getattr(self.cfg, key) != value:
                print(f"[INFO] Warm start gate: {key} {getattr(self.cfg, key)} -> {value} ({path})")
                setattr(self.cfg, key, value)

    """
    Curriculum state.
    """

    def save_state(self):
        """Save the curriculum state (the arm and velocity commands' kernel stages, whether the arm goals have
        started) to <log_dir>/curriculum/state_<step>.pt and state_latest.pt: a resumed or evaluated run
        starts from it (cfg.state_checkpoint_path)."""
        env_state = {
            "arm_targets": self._arm_command.state_dict(),
            "base_velocity": self.command_manager.get_term("base_velocity").state_dict(),
            "arm_goals_enabled": self._arm_command.goals_enabled,
        }
        directory = os.path.join(self.cfg.log_dir, "curriculum")
        os.makedirs(directory, exist_ok=True)
        for name in (f"state_{self.common_step_counter}.pt", "state_latest.pt"):
            torch.save({"env_state": env_state}, os.path.join(directory, name))

    def _load_state(self, state: dict):
        """The kernel stages and the end of the warm start saved by save_state."""
        self._arm_command.load_state_dict(state["arm_targets"])
        self.command_manager.get_term("base_velocity").load_state_dict(state["base_velocity"])
        self._arm_command.goals_enabled |= state["arm_goals_enabled"]
        print(
            f"[INFO] Restored curriculum: arm kernel stage {self._arm_command.kernel_stage}, arm goals"
            f" {self._arm_command.goals_enabled}, velocity kernel stages lin"
            f" {state['base_velocity']['lin_kernel_stage']} ang {state['base_velocity']['ang_kernel_stage']}"
        )

    """
    Helpers.
    """

    def _refresh_sensors_after_reset(self, env_ids: Sequence[int]):
        """Make the IMU read the reset state, not the robot's last pose before the reset.

        DirectMARLEnv.step resets envs without updating the kinematics, so the IMU would show the previous
        episode's orientation and a lin_acc of (old velocity) / dt in an episode's first observation. Imu.reset
        also zeroes its previous velocity: the first reading is set to the accelerometer at rest, gravity only.
        """
        self.scene.write_data_to_sim()
        self.sim.forward()
        imu = self.scene["imu"]
        data = imu.data  # recomputes the reset envs from the refreshed view and stores their velocity
        data.lin_acc_b[env_ids] = quat_apply_inverse(data.quat_w[env_ids], imu._gravity_bias_w[env_ids])
        data.ang_acc_b[env_ids] = 0.0

    def _set_soft_joint_limits(self):
        """Soft limits at cfg.soft_joint_pos_limit_factors of the hard range for the joints it names (Articulation's
        formula, which applies the asset's single factor to every joint). Before the managers: the arm target tables
        and joint_pos_limits read them."""
        robot = self.scene["robot"]
        for pattern, factor in self.cfg.soft_joint_pos_limit_factors.items():
            ids, names = robot.find_joints(pattern)
            hard = robot.data.joint_pos_limits[:, ids]
            mean, half_range = hard.mean(dim=-1), 0.5 * (hard[..., 1] - hard[..., 0])
            robot.data.soft_joint_pos_limits[:, ids, 0] = mean - factor * half_range
            robot.data.soft_joint_pos_limits[:, ids, 1] = mean + factor * half_range
            limits = [[round(v, 3) for v in pair] for pair in robot.data.soft_joint_pos_limits[0, ids].tolist()]
            print(f"[INFO] Soft joint limits at {factor} of the hard range: {dict(zip(names, limits))}")
