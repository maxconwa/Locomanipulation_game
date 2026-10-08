"""DirectMARLEnv that runs Isaac Lab's managers inside, so manager-based mdp terms work unchanged.

DirectMARLEnv owns the step loop, the scene and the event manager; this class builds the other managers and maps
them onto the per-agent dicts:

    actions:      one ActionManager, the agents' terms concatenated in possible_agents order
    observations: agent -> its observation group; the env state is both agents' groups plus `critic`
    rewards:      agent -> its own RewardManager, floored at 0, plus termination_penalty on terminating steps
    dones:        one TerminationManager, shared (one body, one episode)

After every step it moves the arm command by the pelvis motion the learned estimator (odometry.py) reports, or with
estimator.odometry "legs" by leg odometry (_leg_odometry), and fits the estimator to the true motion. GolemEstopMonitor
(golem_safety.py) checks GOLEM's e-stops at every physics substep, and joint targets land action_delay_substeps late.
While training, arm goals start, the e-stops end the episode, and RoboCasa's joint physics ramps in only once the
legs walk (_update_walking_gate, _update_robocasa_physics); until then the episodes are walking only.
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
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul

from .golem_safety import GolemEstopMonitor
from .locomanip_marl_env_cfg import ARM_COMMAND, LocoManipMarlEnvCfg
from .odometry import EstimatorTrainer, motion_to_transform, pelvis_motion


WALKING_PROGRESS_EMA = 0.001  # per policy step: a time constant of 20 s of sim time
ROBOCASA_SCALE_STEPS = 20  # the ramp of RoboCasa's joint physics is written in this many steps


class LocoManipMarlEnv(DirectMARLEnv):
    cfg: LocoManipMarlEnvCfg

    def __init__(self, cfg: LocoManipMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._set_soft_joint_limits()

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
        self.golem_monitor = GolemEstopMonitor(self._robot, self.cfg.golem_estop, self.num_envs, self.device)
        self.golem_log: dict[str, torch.Tensor] = {}
        # opens once and stays open; always open outside training
        self.estops_end_episodes = not self.cfg.estimator.train
        self._walking_progress_ema = torch.tensor(0.0, device=self.device)
        self._gate_log: dict[str, torch.Tensor] = {}
        # RoboCasa's joint damping and armature as the startup events drew them, and the step its ramp starts from
        self._robocasa_damping = self._robot.data.joint_viscous_friction_coeff.clone()
        self._robocasa_armature = self._robot.data.joint_armature.clone()
        self._ramp_start: int | None = None if self.cfg.estimator.train else -self.cfg.robocasa_ramp_steps
        self.robocasa_scale = -1.0
        self._update_robocasa_physics()
        # the deploy loop's latency: per env, the substeps a new target waits; until then the robot holds the last
        lo = self.cfg.action_delay_substeps[0]
        self._delay = torch.randint(lo, self._delay_max + 1, (self.num_envs,), device=self.device)
        self._delayed_target = self._robot.data.joint_pos.clone()
        self._substep = 0

        group_dims = self.observation_manager.group_obs_dim
        self.cfg.observation_spaces = {agent: group_dims[agent][0] for agent in self.cfg.possible_agents}
        self.cfg.state_space = group_dims["critic"][0] + sum(group_dims[agent][0] for agent in self.cfg.possible_agents)
        self.cfg.action_spaces = {
            agent: self.action_manager.get_term(term).action_dim for agent, term in self.cfg.agent_action_terms.items()
        }
        self._configure_env_spaces()
        print(
            f"[INFO] Spaces: observations {self.cfg.observation_spaces}, state {self.cfg.state_space},"
            f" actions {self.cfg.action_spaces}"
        )

        # -- pelvis odometry
        self._arm_command = self.command_manager.get_term(ARM_COMMAND)
        self._arm_command.goals_enabled = self.estops_end_episodes
        self.estimator = EstimatorTrainer(
            self.cfg.estimator, group_dims["odometry"][0] + self.action_manager.total_action_dim, self.num_envs, self.device
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
        if self._legs:
            feet = self.cfg.estimator.foot_body_names
            self._contact_sensor = self.scene.sensors[self.cfg.estimator.contact_sensor_name]
            self._foot_ids = self._robot.find_bodies(feet, preserve_order=True)[0]
            self._foot_sensor_ids = self._contact_sensor.find_bodies(feet, preserve_order=True)[0]
            self._prev_feet_in_pelvis = self._feet_in_pelvis()
        if self.cfg.estimator.checkpoint_path:
            print(f"[INFO] Pelvis estimator file: {self.cfg.estimator.checkpoint_path}")
            self._load_env_state(self.estimator.load(self.cfg.estimator.checkpoint_path))

        self._obs_buf: dict[str, torch.Tensor] = {}

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
        self._substep = 0

    def _apply_action(self) -> None:
        # once per physics substep: the state the previous substep left
        self.golem_monitor.update()
        self.action_manager.apply_action()
        # the targets just written are this step's; envs still inside their delay keep the last step's
        new = self._robot.data.joint_pos_target.clone()
        late = self._delay > self._substep
        self._robot.set_joint_position_target(torch.where(late.unsqueeze(1), self._delayed_target, new))
        self._substep += 1
        if self._substep == self.cfg.decimation:
            self._delayed_target = new

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.termination_manager.compute()
        terminated = self.termination_manager.terminated
        time_outs = self.termination_manager.time_outs
        return (
            {agent: terminated for agent in self.cfg.possible_agents},
            {agent: time_outs for agent in self.cfg.possible_agents},
        )

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        terminated = self.termination_manager.terminated.float()
        rewards = {}
        for agent, manager in self.reward_managers.items():
            reward = manager.compute(dt=self.step_dt)
            if torch.isnan(reward).any():
                bad = torch.isnan(manager._step_reward).any(dim=0).nonzero(as_tuple=True)[0].tolist()
                raise RuntimeError(f"NaN in {agent} reward terms {[manager._term_names[i] for i in bad]}")
            rewards[agent] = torch.clamp(reward, min=0.0) + self.cfg.termination_penalty * terminated
        return rewards

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_odometry()
        self._update_walking_gate()
        self.command_manager.compute(dt=self.step_dt)
        groups = list(self.cfg.possible_agents) + ["critic"]
        self._obs_buf = {g: self.observation_manager.compute_group(g, update_history=True) for g in groups}
        self.extras.setdefault("log", {}).update(self._estimator_log)
        self.extras["log"].update(self.golem_log)
        self.extras["log"].update(self._gate_log)
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
        self.golem_monitor.reset(env_ids)
        lo = self.cfg.action_delay_substeps[0]
        ids = torch.as_tensor(env_ids, device=self.device)
        self._delay[ids] = torch.randint(lo, self._delay_max + 1, (len(ids),), device=self.device)
        # a fresh episode starts from a hold at its reset pose, as the deploy node's pre-pose ends
        self._delayed_target[ids] = self._robot.data.joint_pos[ids]

        log = {}
        log.update(self.observation_manager.reset(env_ids))
        log.update(self.action_manager.reset(env_ids))
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
    Pelvis odometry.
    """

    def _update_odometry(self):
        """Move the arm commands by this step's pelvis motion, fit the estimator, and set the arm command's
        estimate_prob: while training, arm goals follow the estimate only while its drift over whole goals is under
        drift_gate.

        Runs after resets: reset envs are skipped (their previous sample is from the old episode) and get an exact
        command from the command manager anyway.
        """
        odometry_obs = self.observation_manager.compute_group("odometry", update_history=True)
        root_pos = self._robot.data.root_pos_w.clone()
        root_quat = self._robot.data.root_quat_w.clone()
        cfg = self.cfg.estimator
        feet_in_pelvis = self._feet_in_pelvis() if self._legs else None

        if self._have_prev:
            valid = ~self._fresh
            inputs = torch.cat([odometry_obs, self.action_manager.action], dim=1)
            true_motion = pelvis_motion(self._prev_root_pos, self._prev_root_quat, root_pos, root_quat, self.step_dt)
            with torch.no_grad():  # the commands don't carry the estimator's graph from step to step
                estimated = motion_to_transform(self.estimator.model(inputs), self.step_dt)
            if self._legs:
                estimated = self._leg_odometry(feet_in_pelvis, root_quat, estimated, true_motion, valid)
            self._arm_command.apply_pelvis_motion(
                estimated=estimated, true=motion_to_transform(true_motion, self.step_dt), env_mask=valid
            )
            if cfg.train:
                self.estimator.add(inputs, true_motion, valid)
                if self.common_step_counter % cfg.train_every == 0:
                    stats = self.estimator.train()
                    self._estimator_log.update({f"Estimator/{k}": v for k, v in stats.items()})
                if self.common_step_counter % cfg.save_every == 0 and self.cfg.log_dir:
                    self._save_estimator()

        self._have_prev = True
        self._prev_root_pos, self._prev_root_quat = root_pos, root_quat
        if self._legs:
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
        arm.estimate_prob = gate_open if cfg.train else torch.tensor(1.0, device=self.device)
        self._estimator_log["Estimator/estimate_prob"] = arm.estimate_prob
        self._estimator_log["Estimator/goal_drift"] = self._goal_drift_ema
        self._estimator_log["Estimator/gate_open"] = gate_open

    def _update_walking_gate(self):
        """Open the walking gate when the EMA of walking progress passes cfg.walking_gate: per step, the mean over
        envs walking on a command of at least walking_gate_min_speed (not just reset) of the pelvis velocity along
        the command over the command's speed. 0 standing still, 1 on the command. Before the command manager
        resamples, so the command is the one the step followed. Opening it starts the arm goals and the e-stops.
        """
        command = self.command_manager.get_command("base_velocity")[:, :2]
        speed = torch.norm(command, dim=1)
        arm = self._arm_command
        nav = ~arm.arm_mode & (speed >= self.cfg.walking_gate_min_speed) & (self.episode_length_buf > 0)
        if nav.any():
            velocity = self._robot.data.root_lin_vel_b[:, :2]
            progress = (torch.sum(velocity * command, dim=1) / speed.square().clamp(min=1e-6))[nav].mean()
            self._walking_progress_ema += WALKING_PROGRESS_EMA * (progress - self._walking_progress_ema)
        if not self.estops_end_episodes and self._walking_progress_ema.item() > self.cfg.walking_gate:
            self.estops_end_episodes = True
            arm.goals_enabled = True
            self._ramp_start = self.common_step_counter
            print(f"[INFO] Walking gate open at step {self.common_step_counter}: arm goals start, GOLEM's e-stops end"
                  f" the episode, RoboCasa's joint physics ramps in over {self.cfg.robocasa_ramp_steps} steps")
        self._update_robocasa_physics()
        self._gate_log["Safety/walking_progress"] = self._walking_progress_ema
        self._gate_log["Safety/estops_end_episodes"] = torch.tensor(float(self.estops_end_episodes), device=self.device)
        self._gate_log["Safety/robocasa_scale"] = torch.tensor(self.robocasa_scale, device=self.device)

    def _update_robocasa_physics(self):
        """Write RoboCasa's joint physics at robocasa_scale of the way from the asset's defaults: 0 until the ramp
        starts, then rising to 1 over cfg.robocasa_ramp_steps in ROBOCASA_SCALE_STEPS steps. It scales the passive
        damping and armature the startup events drew, and the upper end of the action delay new episodes draw.
        """
        scale = 0.0
        if self._ramp_start is not None:
            progress = (self.common_step_counter - self._ramp_start) / max(self.cfg.robocasa_ramp_steps, 1)
            scale = min(1.0, int(progress * ROBOCASA_SCALE_STEPS) / ROBOCASA_SCALE_STEPS)
        if scale == self.robocasa_scale:
            return
        self.robocasa_scale = scale
        data = self._robot.data
        damping = torch.lerp(data.default_joint_viscous_friction_coeff, self._robocasa_damping, scale)
        armature = torch.lerp(data.default_joint_armature, self._robocasa_armature, scale)
        self._robot.write_joint_friction_coefficient_to_sim(
            data.joint_friction_coeff.clone(), joint_viscous_friction_coeff=damping
        )
        self._robot.write_joint_armature_to_sim(armature)
        lo, hi = self.cfg.action_delay_substeps
        self._delay_max = lo + round(scale * (hi - lo))
        print(
            f"[INFO] RoboCasa's joint physics at {scale:.2f} (step {self.common_step_counter}): passive damping"
            f" {damping.max().item():.2f} max, armature {armature.max().item():.3f} max, delay up to"
            f" {self._delay_max} substeps"
        )

    def _feet_in_pelvis(self) -> torch.Tensor:
        """Each foot's ankle-roll origin in the pelvis frame, (N, F, 3): on the robot, forward kinematics."""
        feet = self._robot.data.body_pos_w[:, self._foot_ids] - self._robot.data.root_pos_w.unsqueeze(1)
        n, f = feet.shape[:2]
        quat = self._robot.data.root_quat_w.unsqueeze(1).expand(n, f, 4).reshape(-1, 4)
        return quat_apply_inverse(quat, feet.reshape(-1, 3)).view(n, f, 3)

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
            "terrain_levels": terrain.terrain_levels.clone(),
            "terrain_types": terrain.terrain_types.clone(),
            "goal_drift_ema": self._goal_drift_ema.clone(),
            "walking_progress_ema": self._walking_progress_ema.clone(),
            "estops_end_episodes": self.estops_end_episodes,
            "robocasa_scale": self.robocasa_scale,
        }
        self.estimator.save(os.path.join(directory, f"estimator_{self.common_step_counter}.pt"), env_state)
        self.estimator.save(os.path.join(directory, "estimator_latest.pt"), env_state)

    def _load_env_state(self, state: dict):
        """Curriculum levels, terrain tiles, the drift and walking gates and the RoboCasa ramp saved by
        _save_estimator. Runs from before the walking gate had e-stops end the episode and RoboCasa's joint physics
        on in full from the start."""
        self._arm_command.load_state_dict(state["arm_targets"])
        terrain = self.scene.terrain
        levels = state["terrain_levels"].to(self.device)
        if len(levels) == self.num_envs:
            terrain.terrain_levels[:] = levels
            terrain.terrain_types[:] = state["terrain_types"].to(self.device)
        else:
            terrain.terrain_levels[:] = levels[torch.randint(0, len(levels), (self.num_envs,), device=self.device)]
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        self._goal_drift_ema = state["goal_drift_ema"].to(self.device)
        self._walking_progress_ema = state.get("walking_progress_ema", torch.tensor(1.0)).to(self.device)
        self.estops_end_episodes = self.estops_end_episodes or state.get("estops_end_episodes", True)
        self._arm_command.goals_enabled = self.estops_end_episodes
        if self.estops_end_episodes and self.cfg.estimator.train:
            scale = state.get("robocasa_scale", 1.0)
            self._ramp_start = self.common_step_counter - round(scale * self.cfg.robocasa_ramp_steps)
        self._update_robocasa_physics()
        print(
            f"[INFO] Restored curriculum: arm level mean {self._arm_command.level.float().mean().item():.2f},"
            f" drift gate {self._goal_drift_ema.item():.3f}, e-stops end episodes {self.estops_end_episodes},"
            f" RoboCasa's joint physics at {self.robocasa_scale:.2f}"
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
