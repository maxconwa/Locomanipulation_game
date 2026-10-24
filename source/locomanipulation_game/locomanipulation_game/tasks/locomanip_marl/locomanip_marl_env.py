"""DirectMARLEnv that runs Isaac Lab's managers inside, so manager-based mdp terms work unchanged.

DirectMARLEnv owns the step loop, the scene and the event manager; this class builds the other managers and maps
them onto the per-agent dicts:

    actions:      one ActionManager, the agents' terms concatenated in possible_agents order (cfg.agent_action_terms);
                  with cfg.action_delay_range each env's joint targets reach the robot 0-2 policy steps late
                  (with cfg.action_delay_substeps, that many physics substeps late)
    observations: agent -> its observation group; the env state is both agents' groups plus `critic`
    rewards:      agent -> its own RewardManager, floored at 0, plus termination_penalty on terminating steps and
                  golem_estop_penalty on the step a GOLEM e-stop trip starts
    dones:        one TerminationManager, shared (one body, one episode)

observation_scales / state_scales give, per agent, the fixed input scale of every observation dimension
(cfg.observation_scales); scripts/skrl/train.py freezes the agents' skrl scalers at them.

After every step it moves the arm command by the pelvis motion the learned estimator (odometry.py) reports, or with
estimator.odometry "legs" by leg odometry (_leg_odometry), with estimator.vertical_from_feet / tilt_from_gravity its
vertical motion and tilt from the feet and gravity (_from_feet_and_gravity), and fits the estimator to the true motion
(with estimator.train_on_arm_goals_only, the motion during arm goals only). While training, episodes are navigation only
until the robot walks (cfg.warm_start_*: few falls and most episodes tracking their commands; _update_warm_start); then
arm goals start and the legs' base_height term stops (_set_walk_stage). GolemEstopMonitor (golem_safety.py) checks GOLEM's e-stops at every physics substep and logs them; with
cfg.golem_estop_ends_episode they end the episode from the end of the warm start, in every mode (always outside training).
"""

from __future__ import annotations

import math
import os
import torch
from collections.abc import Sequence

from isaaclab.envs import DirectMARLEnv
from isaaclab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    ObservationManager,
    ObservationTermCfg,
    RewardManager,
    TerminationManager,
)
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul

from locomanipulation_game.assets.h1_2 import H1_2_MAGPIE_CFG

from .golem_safety import GolemEstopMonitor
from .imu_filter import GyroLowpass
from .mdp.curriculums import _tracked_episode
from .locomanip_marl_env_cfg import ARM_COMMAND, LocoManipMarlEnvCfg
from .odometry import EstimatorTrainer, align_tilt, motion_to_transform, pelvis_motion, saved_history_frames


WARM_START_DECAY = 0.999  # per policy step: the time-out share weighs episode ends of the last ~20 s of sim time


class LocoManipMarlEnv(DirectMARLEnv):
    cfg: LocoManipMarlEnvCfg

    def __init__(self, cfg: LocoManipMarlEnvCfg, render_mode: str | None = None, **kwargs):
        if not cfg.explicit_pd:
            # cfg.__post_init__ made them explicit before any override could reach cfg.explicit_pd
            cfg.scene.robot = cfg.scene.robot.replace(actuators=H1_2_MAGPIE_CFG.actuators)
        cfg.sync_derived()
        super().__init__(cfg, render_mode, **kwargs)
        self._set_soft_joint_limits()

        # After super() has started the sim, in ManagerBasedRLEnv.load_managers' order: observations read commands
        # and actions, rewards read terminations.
        self.command_manager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        history_frames = self._odometry_history()
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
        # the warm start: decayed counts of episode ends and of the time-outs among them; arm goals start once the
        # share passes cfg.warm_start_timeout_share and stay on. Outside training they are on from the start.
        self._ends = torch.tensor(0.0, device=self.device)
        self._time_outs = torch.tensor(0.0, device=self.device)
        # and the walking gate's exponential means over the ending episodes: the share that fell, that tracked
        self._walk_fell = torch.tensor(1.0, device=self.device)
        self._walk_tracked = torch.tensor(0.0, device=self.device)
        self._walk_cos = torch.tensor(0.0, device=self.device)
        self._warm_start_log: dict[str, torch.Tensor] = {}
        # GOLEM's e-stops (None: no e-stop checks)
        self.golem_monitor = (
            GolemEstopMonitor(self._robot, self.cfg.golem_estop, self.num_envs, self.device) if self.cfg.golem_estop else None
        )
        self.golem_log: dict[str, torch.Tensor] = {}
        # this step's GOLEM e-stop trips (mdp.golem_estop) and the last step's, for the penalty on a trip's first step
        self.golem_tripped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._golem_tripped_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        group_dims = self.observation_manager.group_obs_dim
        self.cfg.observation_spaces = {agent: group_dims[agent][0] for agent in self.cfg.possible_agents}
        self.cfg.state_space = group_dims["critic"][0] + sum(group_dims[agent][0] for agent in self.cfg.possible_agents)
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

        # -- pelvis odometry
        self._arm_command = self.command_manager.get_term(ARM_COMMAND)
        self._arm_command.goals_enabled = not self.cfg.estimator.train
        self._set_walk_stage()
        self.estimator = EstimatorTrainer(
            self.cfg.estimator, group_dims["odometry"][0] + self.action_manager.total_action_dim, self.num_envs,
            self.device, history_frames,
        )
        print(f"[INFO] Pelvis estimator: {self.estimator.model}")
        self._have_prev = False
        self._prev_root_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_root_quat = torch.zeros(self.num_envs, 4, device=self.device)
        # envs whose previous odometry sample belongs to an earlier episode
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # running mean of the estimate's drift at the end of an arm goal; starts closed
        self._goal_drift_ema = torch.tensor(1.0, device=self.device)
        self._estimator_log: dict[str, torch.Tensor] = {}
        self._legs = self.cfg.estimator.odometry == "legs"
        self._vertical = self.cfg.estimator.vertical_from_feet
        self._tilt = self.cfg.estimator.tilt_from_gravity
        if self._legs or self._vertical:
            feet = self.cfg.estimator.foot_body_names
            self._foot_ids = self._robot.find_bodies(feet, preserve_order=True)[0]
            self._prev_feet_in_pelvis = self._feet_in_pelvis()
        if self._legs:
            self._contact_sensor = self.scene.sensors[self.cfg.estimator.contact_sensor_name]
            self._foot_sensor_ids = self._contact_sensor.find_bodies(feet, preserve_order=True)[0]
        if self.cfg.estimator.checkpoint_path:
            print(f"[INFO] Pelvis estimator file: {self.cfg.estimator.checkpoint_path}")
            self._load_env_state(self.estimator.load(self.cfg.estimator.checkpoint_path))

        self._obs_buf: dict[str, torch.Tensor] = {}

        # -- action delay: per action term, the last targets, newest first; per env, how many steps late they apply
        self._delay_terms = [self.action_manager.get_term(name) for name in self.action_manager.active_terms]
        self._target_history: list[torch.Tensor] = []
        self._substep = 0
        if self.cfg.action_delay_range is not None:
            low, high = self.cfg.action_delay_range
            self._delay_steps = torch.randint(low, high + 1, (self.num_envs,), device=self.device)
            # with action_delay_substeps, a delay of d substeps reaches back ceil(d / decimation) policy steps
            depth = -(-high // self.cfg.decimation) if self.cfg.action_delay_substeps else high
            self._target_history = [
                self._robot.data.joint_pos[:, term._joint_ids].unsqueeze(1).repeat(1, depth + 1, 1).clone()
                for term in self._delay_terms
            ]
            unit = "physics substeps" if self.cfg.action_delay_substeps else "policy steps"
            print(f"[INFO] Action delay: {low}-{high} {unit} per env, drawn at each reset")

        # -- the observed joint velocity's lag (cfg.joint_vel_obs_lag_ms): the state after each physics substep, newest
        # last; mdp.joint_vel_rel_lagged reads lagged_joint_vel
        self._dq_ring = None
        if self.cfg.joint_vel_obs_lag_ms is not None:
            low_ms, high_ms = self.cfg.joint_vel_obs_lag_ms
            self._dq_lag_range = (int(round(low_ms / 1000 / self.physics_dt)), int(round(high_ms / 1000 / self.physics_dt)))
            self._dq_lag_ids = torch.as_tensor(
                self._robot.find_joints(self.cfg.joint_vel_obs_lag_joints)[0], device=self.device)
            depth = self._dq_lag_range[1] + 1
            self._dq_ring = self._robot.data.joint_vel.unsqueeze(0).repeat(depth, 1, 1).clone()
            self._dq_slot = 0
            self._dq_lag = torch.randint(self._dq_lag_range[0], self._dq_lag_range[1] + 1, (self.num_envs,), device=self.device)
            print(f"[INFO] Joint velocity observation lag: {low_ms:g}-{high_ms:g} ms ({self._dq_lag_range} substeps) on"
                  f" {self.cfg.joint_vel_obs_lag_joints}")

        # -- the policy's gyro low-pass (mdp.imu_ang_vel_lowpass reads its output): one sample per physics substep,
        # the IMU's rate, as the deploy filters /livox/imu
        self.gyro_lowpass = None
        if self.cfg.gyro_lowpass_hz:
            self.gyro_lowpass = GyroLowpass(self.cfg.gyro_lowpass_hz, self.physics_dt, self.num_envs, self.device)
            print(f"[INFO] Gyro low-pass: {self.cfg.gyro_lowpass_hz:g} Hz Butterworth at {1.0 / self.physics_dt:.0f} Hz")

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
        if self._target_history:
            # the policies observe what they asked for (applied_actions); the robot gets it _delay_steps late
            rows = torch.arange(self.num_envs, device=self.device)
            for term, history in zip(self._delay_terms, self._target_history):
                history[:] = history.roll(1, dims=1)
                history[:, 0] = term.processed_actions
                if not self.cfg.action_delay_substeps:
                    term._processed_actions = history[rows, self._delay_steps]
        self._substep = 0

    def _apply_action(self) -> None:
        if self._dq_ring is not None:
            # the state the previous substep left
            self._dq_ring[self._dq_slot % len(self._dq_ring)] = self._robot.data.joint_vel
            self._dq_slot += 1
        if self.gyro_lowpass is not None and self._substep > 0:
            # the IMU sample the previous substep left; the step's last one is taken in _get_observations
            self.gyro_lowpass(self.scene["imu"].data.ang_vel_b)
        if self.golem_monitor is not None:
            # once per physics substep: the state the previous substep left (mdp.golem_estop checks the last)
            self.golem_monitor.update()
        if self._target_history and self.cfg.action_delay_substeps:
            # substep k of this policy step applies the targets of ceil((d - k) / decimation) policy steps ago
            back = (self._delay_steps - self._substep).clamp(min=0)
            back = torch.div(back + self.cfg.decimation - 1, self.cfg.decimation, rounding_mode="floor")
            rows = torch.arange(self.num_envs, device=self.device)
            for term, history in zip(self._delay_terms, self._target_history):
                term._processed_actions = history[rows, back]
        self._substep += 1
        self.action_manager.apply_action()

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.termination_manager.compute()
        terminated = self.termination_manager.terminated
        time_outs = self.termination_manager.time_outs
        self._update_warm_start(terminated, time_outs)
        return (
            {agent: terminated for agent in self.cfg.possible_agents},
            {agent: time_outs for agent in self.cfg.possible_agents},
        )

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        terminated = self.termination_manager.terminated.float()
        # a GOLEM e-stop trip costs golem_estop_penalty once, on its first step, after the floor as a fall's penalty does
        # (from the end of the warm start, as the arm goals start, so the stepping it learns first isn't buried)
        new_trip = self.golem_tripped & ~self._golem_tripped_prev
        self._golem_tripped_prev = self.golem_tripped.clone()
        trip_cost = self.cfg.golem_estop_penalty * new_trip.float() * float(self._arm_command.goals_enabled)
        self.golem_log["Safety/trip_starts"] = new_trip.float().mean()
        rewards = {}
        for agent, manager in self.reward_managers.items():
            reward = manager.compute(dt=self.step_dt)
            if torch.isnan(reward).any():
                bad = torch.isnan(manager._step_reward).any(dim=0).nonzero(as_tuple=True)[0].tolist()
                raise RuntimeError(f"NaN in {agent} reward terms {[manager._term_names[i] for i in bad]}")
            rewards[agent] = torch.clamp(reward, min=0.0) + self.cfg.termination_penalty * terminated + trip_cost
        return rewards

    def _get_observations(self) -> dict[str, torch.Tensor]:
        if self.gyro_lowpass is not None:
            # the last substep's IMU sample (after a reset, the refreshed reading the filter restarts from)
            self.gyro_lowpass(self.scene["imu"].data.ang_vel_b)
        self._update_odometry()
        self.command_manager.compute(dt=self.step_dt)
        groups = list(self.cfg.possible_agents) + ["critic"]
        self._obs_buf = {g: self.observation_manager.compute_group(g, update_history=True) for g in groups}
        self.extras.setdefault("log", {}).update(self._estimator_log)
        self.extras["log"].update(self._warm_start_log)
        self.extras["log"].update(self.golem_log)
        return {agent: self._obs_buf[agent] for agent in self.cfg.possible_agents}

    def _get_states(self) -> torch.Tensor:
        # computed in _get_observations, which step() and reset() call before state()
        return torch.cat([self._obs_buf[agent] for agent in self.cfg.possible_agents] + [self._obs_buf["critic"]], dim=1)

    def _reset_idx(self, env_ids: Sequence[int]):
        # ManagerBasedRLEnv._reset_idx's order: curriculum before the scene reset, managers after
        self.curriculum_manager.compute(env_ids=env_ids)
        super()._reset_idx(env_ids)  # scene, reset events, episode_length_buf
        self._refresh_sensors_after_reset(env_ids)
        self._fresh[env_ids] = True
        if self.gyro_lowpass is not None:
            self.gyro_lowpass.reset(env_ids)
        if self._dq_ring is not None:
            low, high = self._dq_lag_range
            self._dq_lag[env_ids] = torch.randint(low, high + 1, (len(env_ids),), device=self.device)
            self._dq_ring[:, env_ids] = self._robot.data.joint_vel[env_ids].unsqueeze(0)
        if self.golem_monitor is not None:
            self.golem_monitor.reset(env_ids)
        self._golem_tripped_prev[env_ids] = False

        log = {}
        log.update(self.observation_manager.reset(env_ids))
        log.update(self.action_manager.reset(env_ids))
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
        log.update(self.event_manager.reset(env_ids))
        log.update(self.termination_manager.reset(env_ids))
        # skrl logs only one-element tensors; the managers also return floats
        self.extras["log"] = {
            key: value if isinstance(value, torch.Tensor) else torch.tensor(float(value), device=self.device)
            for key, value in log.items()
        }

    def lagged_joint_vel(self) -> torch.Tensor:
        """Joint velocity (num_envs, joints) with cfg.joint_vel_obs_lag_joints _dq_lag physics substeps late (the
        current state for a lag of 0, and for every joint without cfg.joint_vel_obs_lag_ms)."""
        dq = self.scene["robot"].data.joint_vel
        if getattr(self, "_dq_ring", None) is None:
            return dq
        depth = len(self._dq_ring)
        slot = (self._dq_slot - self._dq_lag) % depth  # lag k: pushed k substeps before now; the ring's newest is 1
        old = self._dq_ring[slot, torch.arange(self.num_envs, device=self.device)]
        out = dq.clone()
        ids = self._dq_lag_ids
        out[:, ids] = torch.where((self._dq_lag > 0).unsqueeze(1), old[:, ids], dq[:, ids])
        return out

    """
    Fixed observation scales.
    """

    def observation_scales(self, group: str) -> torch.Tensor:
        """Per dimension of an observation group, the scale cfg.observation_scales gives its term (1 by default): a
        number, or one per component of the term (repeated over its history)."""
        scales = self.cfg.observation_scales or {}
        manager = self.observation_manager
        parts = []
        for name, dims in zip(manager.active_terms[group], manager.group_obs_term_dim[group]):
            n = math.prod(dims)
            scale = torch.as_tensor(scales.get(name, 1.0), dtype=torch.float, device=self.device).flatten()
            if n % scale.numel():
                raise ValueError(f"observation_scales[{name!r}]: {scale.numel()} values for {n} dimensions")
            parts.append(scale.repeat(n // scale.numel()))
        return torch.cat(parts)

    def state_scales(self) -> torch.Tensor:
        """observation_scales over the env state: both agents' groups, then the critic's (_get_states)."""
        return torch.cat([self.observation_scales(g) for g in list(self.cfg.possible_agents) + ["critic"]])

    """
    Pelvis odometry.
    """

    def _update_odometry(self):
        """Move the arm commands by this step's pelvis motion, fit the estimator, and set the arm command's
        estimate_prob: while training, arm goals follow the estimate only while its drift over whole goals is under
        drift_gate, and never with estimator.true_odometry_while_training.

        Runs after resets: reset envs are skipped (their previous sample is from the old episode) and get an exact
        command from the command manager anyway.
        """
        odometry_obs = self.observation_manager.compute_group("odometry", update_history=True)
        root_pos = self._robot.data.root_pos_w.clone()
        root_quat = self._robot.data.root_quat_w.clone()
        cfg = self.cfg.estimator
        feet_in_pelvis = self._feet_in_pelvis() if self._legs or self._vertical else None

        if self._have_prev:
            valid = ~self._fresh
            inputs = torch.cat([odometry_obs, self.action_manager.action], dim=1)
            true_motion = pelvis_motion(self._prev_root_pos, self._prev_root_quat, root_pos, root_quat, self.step_dt)
            with torch.no_grad():  # the commands don't carry the estimator's graph from step to step
                estimated = motion_to_transform(self.estimator.model(inputs), self.step_dt)
            if self._vertical or self._tilt:
                estimated = self._from_feet_and_gravity(feet_in_pelvis, root_quat, estimated, true_motion, valid)
            if self._legs:
                estimated = self._leg_odometry(feet_in_pelvis, root_quat, estimated, true_motion, valid)
            self._arm_command.apply_pelvis_motion(
                estimated=estimated, true=motion_to_transform(true_motion, self.step_dt), env_mask=valid
            )
            if cfg.train:
                # the command moves by odometry only during arm goals: with train_on_arm_goals_only the estimator
                # fits that motion (standing, crouching, reaching), not the walking it is never used for
                train = valid & self._arm_command.arm_mode if cfg.train_on_arm_goals_only else valid
                self.estimator.add(inputs, true_motion, self._prev_root_quat, train)
                if self.common_step_counter % cfg.train_every == 0:
                    stats = self.estimator.train()
                    self._estimator_log.update({f"Estimator/{k}": v for k, v in stats.items()})
                if self.common_step_counter % cfg.save_every == 0 and self.cfg.log_dir:
                    self._save_estimator()

        self._have_prev = True
        self._prev_root_pos, self._prev_root_quat = root_pos, root_quat
        if self._legs or self._vertical:
            self._prev_feet_in_pelvis = feet_in_pelvis
        self._fresh[:] = False

        # the drift gate: one EMA sample of the estimate's drift (shadow command) per ended arm goal
        arm = self._arm_command
        count = arm.ended_goal_count
        mean_drift = arm.ended_goal_drift_sum / count.clamp(min=1.0)
        alpha = 1.0 - (1.0 - cfg.drift_ema_per_goal) ** count
        self._goal_drift_ema = torch.where(
            count > 0, (1.0 - alpha) * self._goal_drift_ema + alpha * mean_drift, self._goal_drift_ema
        )
        arm.ended_goal_drift_sum.zero_()
        arm.ended_goal_count.zero_()
        gate_open = (self._goal_drift_ema < cfg.drift_gate).float()
        if cfg.true_odometry_while_training:
            gate_open = torch.zeros_like(gate_open)
        arm.estimate_prob = gate_open if cfg.train else torch.tensor(1.0, device=self.device)
        self._estimator_log["Estimator/estimate_prob"] = arm.estimate_prob
        self._estimator_log["Estimator/goal_drift"] = self._goal_drift_ema
        self._estimator_log["Estimator/gate_open"] = gate_open

    @property
    def estops_end_episodes(self) -> bool:
        """With cfg.golem_estop_ends_episode, GOLEM's e-stops end the episode from the end of the warm start (as the
        arm goals start; outside training from the start) in every mode, walking and arm goals alike, so a fresh
        policy's first steps don't end it."""
        return self.cfg.golem_estop_ends_episode and self._arm_command.goals_enabled

    def _update_warm_start(self, terminated: torch.Tensor, time_outs: torch.Tensor):
        """Start the arm goals once the robot walks: over the ending episodes (exponential means, decay
        cfg.warm_start_episode_decay per episode) the share that fell is at most cfg.warm_start_max_fall_share and the
        share that tracked their velocity commands (as the terrain curriculum promotes) at least
        cfg.warm_start_min_tracked_share, and their velocity's mean cos-sim to the commanded one while walking is
        commanded (nav_cos_vel_xy) at least cfg.warm_start_min_cos: the tracked share alone passed a robot that stood
        (UnitreeFall20: its 0.33 m/s error is under the 0.4 limit). With warm_start_max_fall_share None: once the time-out share among the episode
        ends, both counts decayed by WARM_START_DECAY per step, reaches cfg.warm_start_timeout_share; an episode that
        ends in a fall or out of bounds on its last step counts as a fall. Runs before the resets, while the velocity
        command still holds the ending episodes' tracking sums."""
        cfg = self.cfg
        self._ends = WARM_START_DECAY * self._ends + (terminated | time_outs).sum()
        self._time_outs = WARM_START_DECAY * self._time_outs + (time_outs & ~terminated).sum()
        share = self._time_outs / self._ends.clamp(min=1.0)
        ending = (terminated | time_outs).nonzero(as_tuple=False).flatten()
        if len(ending):
            tracked, fell = _tracked_episode(self, ending, "base_velocity", cfg.warm_start_max_tracking_error, "fell",
                                             "time_out")
            keep = cfg.warm_start_episode_decay ** len(ending)
            self._walk_fell = keep * self._walk_fell + (1.0 - keep) * fell.float().mean()
            self._walk_tracked = keep * self._walk_tracked + (1.0 - keep) * tracked.float().mean()
            command = self.command_manager.get_term("base_velocity")
            walked = command._walk_steps[ending] > 0
            if walked.any():
                cos = command._walk_cos[ending][walked].sum() / command._walk_steps[ending][walked].sum()
                keep = cfg.warm_start_episode_decay ** int(walked.sum())
                self._walk_cos = keep * self._walk_cos + (1.0 - keep) * cos
        if cfg.warm_start_max_fall_share is None:
            walks = share.item() >= cfg.warm_start_timeout_share
        else:
            walks = bool((self._walk_fell <= cfg.warm_start_max_fall_share)
                         & (self._walk_tracked >= cfg.warm_start_min_tracked_share)
                         & (self._walk_cos >= cfg.warm_start_min_cos))
        arm = self._arm_command
        if not arm.goals_enabled and walks:
            arm.goals_enabled = True
            arm.goals_enabled_at = self.common_step_counter  # the arm-goal share ramps up from here
            self._set_walk_stage()
            print(f"[INFO] Warm start over at step {self.common_step_counter}: {self._walk_fell.item():.1%} of episodes"
                  f" fall, {self._walk_tracked.item():.0%} track, cos v {self._walk_cos.item():.2f},"
                  f" {share.item():.0%} time out; arm goals start{'' if cfg.base_height_during_navigation else ', base_height stops'}{', GOLEM e-stops end the episode' if self.estops_end_episodes else ''}")
        self._warm_start_log["Curriculum/warm_start_timeout_share"] = share
        self._warm_start_log["Curriculum/warm_start_fell_share"] = self._walk_fell
        self._warm_start_log["Curriculum/warm_start_tracked_share"] = self._walk_tracked
        self._warm_start_log["Curriculum/warm_start_cos_vel_xy"] = self._walk_cos
        self._warm_start_log["Curriculum/arm_goals_enabled"] = torch.tensor(float(arm.goals_enabled), device=self.device)

    def _set_walk_stage(self):
        """The legs' base_height term (pelvis height above the ground) counts only while the robot learns to walk,
        before the arm goals start: after that it would hold the pelvis up against their crouch. Outside training the
        goals are on from the start, so it never counts. With cfg.base_height_during_navigation it stays on (it never
        counts during arm goals: mdp.base_height_l2_navigation)."""
        legs = self.reward_managers["legs"]
        if (self._arm_command.goals_enabled and "base_height" in legs.active_terms
                and not self.cfg.base_height_during_navigation):
            legs.get_term_cfg("base_height").weight = 0.0

    def _feet_in_pelvis(self) -> torch.Tensor:
        """Each foot's ankle-roll origin in the pelvis frame, (N, F, 3): on the robot, forward kinematics."""
        feet = self._robot.data.body_pos_w[:, self._foot_ids] - self._robot.data.root_pos_w.unsqueeze(1)
        n, f = feet.shape[:2]
        quat = self._robot.data.root_quat_w.unsqueeze(1).expand(n, f, 4).reshape(-1, 4)
        return quat_apply_inverse(quat, feet.reshape(-1, 3)).view(n, f, 3)

    def _from_feet_and_gravity(
        self,
        feet_in_pelvis: torch.Tensor | None,
        root_quat: torch.Tensor,
        estimated: tuple[torch.Tensor, torch.Tensor],
        true_motion: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The estimate (delta_pos, delta_quat), previous pelvis frame, with its rotation's roll and pitch made to
        match gravity's (tilt_from_gravity, odometry.align_tilt) and its vertical motion replaced by the feet's
        (vertical_from_feet).

        The pelvis's height above an ankle along gravity is -up . p, with p the ankle in the pelvis frame (forward
        kinematics) and up the world's vertical in the pelvis frame (on the robot, the IMU's gravity). The lowest ankle
        is taken to be on the ground, so the pelvis's height above the ground is the largest of those heights, and the
        step's rise is its change: no contact sensing, and the rises telescope, so over any stretch they add up to the
        change in height above the ground, whichever foot carries the robot. (Tracking one planted foot instead drifts
        down: a pushing-off ankle rises while it is still the lowest, and stepping never gives that back.) Uneven
        ground costs at most the step between the footholds, once.
        """
        z = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        up, up_prev = quat_apply_inverse(root_quat, z), quat_apply_inverse(self._prev_root_quat, z)
        delta_quat = align_tilt(estimated[1], up_prev, up) if self._tilt else estimated[1]
        if not self._vertical:
            return estimated[0], delta_quat
        height = -(feet_in_pelvis * up.unsqueeze(1)).sum(-1)  # (N, F), pelvis above each ankle
        height_prev = -(self._prev_feet_in_pelvis * up_prev.unsqueeze(1)).sum(-1)
        rise = height.max(dim=1).values - height_prev.max(dim=1).values
        delta_pos = estimated[0]
        learned_rise = (delta_pos * up_prev).sum(-1)
        delta_pos = delta_pos + (rise - learned_rise).unsqueeze(-1) * up_prev

        # vertical speed errors during arm goals (m/s): the feet's and the estimator's it replaces
        true_rise = (true_motion[:, :3] * up_prev).sum(-1) * self.step_dt
        arm = (valid & self._arm_command.arm_mode).float()
        n = arm.sum().clamp(min=1.0) * self.step_dt
        self._estimator_log["Estimator/feet_vertical_error"] = ((rise - true_rise).abs() * arm).sum() / n
        self._estimator_log["Estimator/learned_vertical_error"] = ((learned_rise - true_rise).abs() * arm).sum() / n
        self._estimator_log["Estimator/learned_vertical_bias"] = ((learned_rise - true_rise) * arm).sum() / n
        return delta_pos, delta_quat

    def _odometry_history(self) -> int:
        """The odometry group's window. An estimator file to start from sets it to the window it was trained on
        (4 frames before that was saved), so older runs still load."""
        terms = [t for t in vars(self.cfg.observations.odometry).values() if isinstance(t, ObservationTermCfg)]
        path = self.cfg.estimator.checkpoint_path
        if path:
            frames = saved_history_frames(path)
            if frames != terms[0].history_length:
                print(f"[INFO] Odometry window: {frames} frames, as {path} was trained")
            for term in terms:
                term.history_length = frames
        return terms[0].history_length

    def _leg_odometry(
        self,
        feet_in_pelvis: torch.Tensor,
        root_quat: torch.Tensor,
        fallback: tuple[torch.Tensor, torch.Tensor],
        true_motion: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This step's pelvis motion, (delta_pos, delta_quat) in the previous pelvis frame.

        The rotation dR is the change in pelvis orientation (on the robot, the IMU's attitude through the waist
        joint). A planted foot's ankle stays put in the world, so with p(t) its position in the pelvis frame the
        pelvis moved by p(t-1) - dR p(t). A foot counts as planted if its contact force stayed above stance_force at
        every physics substep; with both planted the two are averaged, weighted by that force. Envs with neither
        keep the learned estimate's translation.
        """
        n, f = feet_in_pelvis.shape[:2]
        delta_quat = quat_mul(quat_inv(self._prev_root_quat), root_quat)
        turned = quat_apply(delta_quat.unsqueeze(1).expand(n, f, 4).reshape(-1, 4), feet_in_pelvis.reshape(-1, 3))
        delta_pos = self._prev_feet_in_pelvis - turned.view(n, f, 3)  # (N, F, 3), one estimate per foot

        # lowest contact force of each foot over the step's substeps (history slot 0 is the latest)
        forces = self._contact_sensor.data.net_forces_w_history[:, : self.cfg.decimation, self._foot_sensor_ids]
        load = forces.norm(dim=-1).min(dim=1).values  # (N, F)
        weight = torch.where(load > self.cfg.estimator.stance_force, load, torch.zeros_like(load))
        total = weight.sum(dim=1, keepdim=True)
        stance = total.squeeze(1) > 0
        leg_pos = ((weight / total.clamp(min=1e-6)).unsqueeze(-1) * delta_pos).sum(dim=1)

        # how often it applies during arm goals, and its per-step error there (m/s)
        arm = valid & self._arm_command.arm_mode
        used = arm & stance
        lin_error = torch.norm(leg_pos / self.step_dt - true_motion[:, :3], dim=-1)
        self._estimator_log["Estimator/legs_stance"] = used.sum() / arm.sum().clamp(min=1)
        self._estimator_log["Estimator/legs_lin_vel_error"] = (lin_error * used).sum() / used.sum().clamp(min=1)

        return torch.where(stance.unsqueeze(-1), leg_pos, fallback[0]), delta_quat

    def _save_estimator(self):
        directory = os.path.join(self.cfg.log_dir, "estimator")
        terrain = self.scene.terrain
        env_state = {
            "arm_targets": self._arm_command.state_dict(),
            "base_velocity": self.command_manager.get_term("base_velocity").state_dict(),
            "terrain_levels": terrain.terrain_levels.clone(),
            "terrain_types": terrain.terrain_types.clone(),
            "goal_drift_ema": self._goal_drift_ema.clone(),
            "arm_goals_enabled": self._arm_command.goals_enabled,
        }
        self.estimator.save(os.path.join(directory, f"estimator_{self.common_step_counter}.pt"), env_state)
        self.estimator.save(os.path.join(directory, "estimator_latest.pt"), env_state)

    def _load_env_state(self, state: dict):
        """Curriculum levels, terrain tiles, the drift gate and the end of the warm start saved by _save_estimator."""
        self._arm_command.load_state_dict(state["arm_targets"])
        # runs saved before the velocity kernel curriculum have none: stage 0
        self.command_manager.get_term("base_velocity").load_state_dict(state.get("base_velocity", {}))
        terrain = self.scene.terrain
        levels = state["terrain_levels"].to(self.device)
        if len(levels) == self.num_envs:
            terrain.terrain_levels[:] = levels
            terrain.terrain_types[:] = state["terrain_types"].to(self.device)
        else:
            terrain.terrain_levels[:] = levels[torch.randint(0, len(levels), (self.num_envs,), device=self.device)]
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        self._goal_drift_ema = state["goal_drift_ema"].to(self.device)
        self._arm_command.goals_enabled |= state["arm_goals_enabled"]
        print(
            f"[INFO] Restored curriculum: depth level mean {self._arm_command.level.float().mean().item():.2f},"
            f" drift gate {self._goal_drift_ema.item():.3f}, arm goals {self._arm_command.goals_enabled},"
            f" velocity kernel stages lin {state.get('base_velocity', {}).get('lin_kernel_stage', 0)}"
            f" ang {state.get('base_velocity', {}).get('ang_kernel_stage', 0)}"
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
