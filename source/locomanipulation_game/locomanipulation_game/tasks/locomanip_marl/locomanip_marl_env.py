"""DirectMARLEnv that runs Isaac Lab's managers inside, so manager-based mdp terms work unchanged.

DirectMARLEnv owns the step loop, the scene and the event manager; this class builds the other managers and maps
them onto the per-agent dicts:

    actions:      one ActionManager, the agents' terms concatenated in possible_agents order (cfg.agent_action_terms);
                  with cfg.action_delay_range each env's joint targets reach the robot that many physics substeps late
    observations: agent -> its observation group; the env state is both agents' groups plus `critic`
    rewards:      agent -> its own RewardManager, floored at 0, plus termination_penalty on terminating steps and
                  golem_estop_penalty on the step a GOLEM e-stop trip starts
    dones:        one TerminationManager, shared (one body, one episode)

After every step it moves the arm command by the pelvis's motion (on the robot, FAST-LIO's odometry does). Each
episode's length is drawn at its reset (cfg.episode_length_mean_s). While training, episodes are navigation only until
the robot walks like sport mode (cfg.warm_start_*, _update_warm_start); then the arm goals start. GolemEstopMonitor
(golem_safety.py) checks GOLEM's e-stops at every physics substep; trips are logged and charged, never end an episode.
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

from .golem_safety import GolemEstopMonitor
from .imu_filter import GyroLowpass
from .locomanip_marl_env_cfg import ARM_COMMAND, TERRAIN_MAX_TRACKING_ERROR, LocoManipMarlEnvCfg
from .mdp.curriculums import _tracked_episode


class LocoManipMarlEnv(DirectMARLEnv):
    cfg: LocoManipMarlEnvCfg

    def __init__(self, cfg: LocoManipMarlEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.sync_derived()
        super().__init__(cfg, render_mode, **kwargs)
        self._set_soft_joint_limits()
        # each episode's own length in policy steps (cfg.episode_length_mean_s; mdp.episode_limit_reached), drawn at
        # its reset; None: every episode max_episode_length (episode_length_mean_s 0)
        self.episode_limit = (torch.full((self.num_envs,), self.max_episode_length, device=self.device)
                              if cfg.episode_length_mean_s > 0 else None)

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
        # the warm start: the exponential mean of the share of ending episodes that fell, the sport-mode gate's
        # consecutive passing checks and the last check's metrics
        self._walk_fell = torch.tensor(1.0, device=self.device)
        self._sport_streak = 0
        self._sport_metrics: dict[str, float] = {}
        self._warm_start_log: dict[str, torch.Tensor] = {}
        self.golem_monitor = GolemEstopMonitor(self._robot, self.cfg.golem_estop, self.num_envs, self.device)
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

        # -- action delay: per action term, the last targets, newest first; per env, how many substeps late they apply
        self._delay_terms = [self.action_manager.get_term(name) for name in self.action_manager.active_terms]
        self._target_history: list[torch.Tensor] = []
        self._substep = 0
        if self.cfg.action_delay_range is not None:
            low, high = self.cfg.action_delay_range
            self._delay_steps = torch.randint(low, high + 1, (self.num_envs,), device=self.device)
            depth = -(-high // self.cfg.decimation)  # a delay of d substeps reaches back ceil(d / decimation) steps
            self._target_history = [
                self._robot.data.joint_pos[:, term._joint_ids].unsqueeze(1).repeat(1, depth + 1, 1).clone()
                for term in self._delay_terms
            ]
            print(f"[INFO] Action delay: {low}-{high} physics substeps per env, drawn at each reset")

        # -- the policy's gyro low-pass (mdp.imu_ang_vel_lowpass reads its output): one sample per physics substep,
        # the IMU's rate, as the deploy filters /livox/imu
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
        # the policies observe what they asked for (applied_actions); the robot gets it _delay_steps substeps late
        for term, history in zip(self._delay_terms, self._target_history):
            history[:] = history.roll(1, dims=1)
            history[:, 0] = term.processed_actions
        self._substep = 0

    def _apply_action(self) -> None:
        if self._substep > 0:
            # the IMU sample the previous substep left; the step's last one is taken in _get_observations
            self.gyro_lowpass(self.scene["imu"].data.ang_vel_b)
        # once per physics substep: the state the previous substep left (mdp.golem_estop checks the last)
        self.golem_monitor.update()
        if self._target_history:
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
        self._update_warm_start(terminated | time_outs)
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
        # the last substep's IMU sample (after a reset, the refreshed reading the filter restarts from)
        self.gyro_lowpass(self.scene["imu"].data.ang_vel_b)
        self._update_odometry()
        self.command_manager.compute(dt=self.step_dt)
        groups = list(self.cfg.possible_agents) + ["critic"]
        self._obs_buf = {g: self.observation_manager.compute_group(g) for g in groups}
        self.extras.setdefault("log", {}).update(self._warm_start_log)
        self.extras["log"].update(self.golem_log)
        return {agent: self._obs_buf[agent] for agent in self.cfg.possible_agents}

    def _get_states(self) -> torch.Tensor:
        # computed in _get_observations, which step() and reset() call before state()
        return torch.cat([self._obs_buf[agent] for agent in self.cfg.possible_agents] + [self._obs_buf["critic"]], dim=1)

    def _reset_idx(self, env_ids: Sequence[int]):
        # ManagerBasedRLEnv._reset_idx's order: curriculum before the scene reset, managers after
        self.curriculum_manager.compute(env_ids=env_ids)
        super()._reset_idx(env_ids)  # scene, reset events, episode_length_buf
        if self.episode_limit is not None:
            ids = torch.as_tensor(env_ids, device=self.device)
            seconds = torch.empty(len(ids), device=self.device).exponential_(1.0 / self.cfg.episode_length_mean_s)
            steps = torch.ceil(seconds.clamp(min=self.cfg.episode_length_min_s) / self.step_dt)
            self.episode_limit[ids] = steps.clamp(max=self.max_episode_length).long()
        self._refresh_sensors_after_reset(env_ids)
        self._fresh[env_ids] = True
        self.gyro_lowpass.reset(env_ids)
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
        """Start the arm goals once the robot walks like sport mode: over the ending episodes (an exponential mean,
        decay cfg.warm_start_episode_decay per episode) the share that fell is at most cfg.warm_start_max_fall_share,
        and the sport-mode gate passes (_sport_gate); or at cfg.warm_start_deadline_steps. Runs before the resets,
        while the velocity command still holds the ending episodes' sums."""
        cfg = self.cfg
        ids = ending.nonzero(as_tuple=False).flatten()
        if len(ids):
            _, fell = _tracked_episode(self, ids, "base_velocity", TERRAIN_MAX_TRACKING_ERROR, "fell", "time_out")
            keep = cfg.warm_start_episode_decay ** len(ids)
            self._walk_fell = keep * self._walk_fell + (1.0 - keep) * fell.float().mean()
        sport = self._sport_gate()  # every check, so the metrics are logged before and after the goals start
        walks = bool(self._walk_fell <= cfg.warm_start_max_fall_share) and sport
        deadline = 0 < cfg.warm_start_deadline_steps <= self.common_step_counter
        arm = self._arm_command
        if not arm.goals_enabled and (walks or deadline):
            arm.goals_enabled = True
            arm.goals_enabled_at = self.common_step_counter  # the arm-goal share ramps up from here
            metrics = ", ".join(f"{k} {v:.3f}" for k, v in self._sport_metrics.items() if not k.startswith("n_"))
            print(f"[INFO] Warm start over at step {self.common_step_counter}{' (deadline)' if not walks else ''}:"
                  f" {self._walk_fell.item():.1%} of episodes fall; sport metrics {metrics or 'none yet'};"
                  " arm goals start")
        self._warm_start_log["Curriculum/warm_start_fell_share"] = self._walk_fell
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
        passes = (m["turn_sense"] >= cfg.warm_start_min_turn_sense
                  and cfg.warm_start_speed_ratio_range[0] <= m["speed_ratio"] <= cfg.warm_start_speed_ratio_range[1]
                  and cfg.warm_start_turn_ratio_range[0] <= m["turn_ratio"] <= cfg.warm_start_turn_ratio_range[1]
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
        """Save the curriculum state (the arm and velocity commands' curricula, the terrain tiles, whether the arm
        goals have started) to <log_dir>/curriculum/state_<step>.pt and state_latest.pt: a resumed or evaluated run
        starts from it (cfg.state_checkpoint_path)."""
        terrain = self.scene.terrain
        env_state = {
            "arm_targets": self._arm_command.state_dict(),
            "base_velocity": self.command_manager.get_term("base_velocity").state_dict(),
            "terrain_levels": terrain.terrain_levels.clone(),
            "terrain_types": terrain.terrain_types.clone(),
            "arm_goals_enabled": self._arm_command.goals_enabled,
        }
        directory = os.path.join(self.cfg.log_dir, "curriculum")
        os.makedirs(directory, exist_ok=True)
        for name in (f"state_{self.common_step_counter}.pt", "state_latest.pt"):
            torch.save({"env_state": env_state}, os.path.join(directory, name))

    def _load_state(self, state: dict):
        """Curriculum levels, terrain tiles and the end of the warm start saved by save_state."""
        self._arm_command.load_state_dict(state["arm_targets"])
        self.command_manager.get_term("base_velocity").load_state_dict(state["base_velocity"])
        terrain = self.scene.terrain
        levels = state["terrain_levels"].to(self.device)
        if len(levels) == self.num_envs:
            terrain.terrain_levels[:] = levels
            terrain.terrain_types[:] = state["terrain_types"].to(self.device)
        else:
            terrain.terrain_levels[:] = levels[torch.randint(0, len(levels), (self.num_envs,), device=self.device)]
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        self._arm_command.goals_enabled |= state["arm_goals_enabled"]
        print(
            f"[INFO] Restored curriculum: depth level mean {self._arm_command.level.float().mean().item():.2f},"
            f" arm goals {self._arm_command.goals_enabled}, velocity kernel stages lin"
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
