"""DirectMARLEnv that runs Isaac Lab's managers inside, so the IBR mdp terms work unchanged.

DirectMARLEnv owns the step loop, the scene and the event manager. The mdp
functions the IBR rounds use read `env.command_manager`, `env.action_manager`
and `env.termination_manager`, and every manager only needs `scene`, `sim`,
`num_envs`, `device` and `max_episode_length_s` from its env, which a
DirectMARLEnv has. So this class builds those managers itself and maps them
onto the per-agent dicts:

    actions:      agent -> its ActionManager term (concatenated, legs first)
    observations: agent -> its ObservationManager group; `critic` -> env state
    rewards:      agent -> its own RewardManager
    dones:        one TerminationManager, shared (one body, one episode)

It also runs the pelvis odometry (odometry.py). After every step it estimates
how the pelvis moved from the `odometry` observation group's history window
(which ends at the step's end) and the action, moves the arm command by that
(ArmTargetsCommand.apply_pelvis_motion), and fits the estimator to the true
motion. The estimator's file also carries the env-side state a resume or
play.py needs: the arm and terrain curriculum levels and the drift gate.

_get_observations does, in order: odometry -> commands (events re-anchor arm
goals, overriding the odometry update) -> the agents' groups. Unlike
ManagerBasedRLEnv, the command manager updates after interval events, not
before.
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
from isaaclab.utils.math import quat_apply_inverse

from .locomanip_marl_env_cfg import ARM_COMMAND, LocoManipMarlEnvCfg
from .odometry import EstimatorTrainer, motion_to_transform, pelvis_motion


class LocoManipMarlEnv(DirectMARLEnv):
    cfg: LocoManipMarlEnvCfg

    def __init__(self, cfg: LocoManipMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._set_soft_joint_limits()

        # Managers resolve physics handles, so they come after super() has
        # started the sim. Same order as ManagerBasedRLEnv.load_managers:
        # observations read commands and actions, rewards read terminations.
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

        # _pre_physics_step concatenates the agents' actions in possible_agents order
        expected_terms = [self.cfg.agent_action_terms[agent] for agent in self.cfg.possible_agents]
        if self.action_manager.active_terms != expected_terms:
            raise ValueError(
                f"Action terms {self.action_manager.active_terms} must be {expected_terms}:"
                " one per agent, in possible_agents order."
            )

        # Spaces come from the managers, not from hand-counted cfg values.
        group_dims = self.observation_manager.group_obs_dim
        self.cfg.observation_spaces = {agent: group_dims[agent][0] for agent in self.cfg.possible_agents}
        self.cfg.state_space = group_dims["critic"][0]
        if self.cfg.state_includes_agent_obs:
            self.cfg.state_space += sum(group_dims[agent][0] for agent in self.cfg.possible_agents)
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
        self._robot = self.scene["robot"]
        odometry_dim = group_dims["odometry"][0]
        self.estimator = EstimatorTrainer(
            self.cfg.estimator, odometry_dim + self.action_manager.total_action_dim, self.num_envs, self.device
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
        if self.cfg.estimator.checkpoint_path:
            model_loaded, env_state = self.estimator.load(self.cfg.estimator.checkpoint_path)
            print(f"[INFO] Pelvis estimator file: {self.cfg.estimator.checkpoint_path} (model loaded: {model_loaded})")
            if env_state is not None:
                self._load_env_state(env_state, restore_gate=model_loaded)

        self._obs_buf: dict[str, torch.Tensor] = {}

    """
    DirectMARLEnv hooks.
    """

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
        joint_action = torch.cat([actions[agent] for agent in self.cfg.possible_agents], dim=1)
        if not torch.isfinite(joint_action).all():
            # a NaN joint target hangs the PhysX solver (flat run A, twice: GPU at 100%, no error)
            bad = {agent: int((~torch.isfinite(actions[agent])).any(dim=1).sum()) for agent in self.cfg.possible_agents}
            raise RuntimeError(f"Non-finite actions at step {self.common_step_counter}, envs per agent: {bad}")
        self.action_manager.process_action(joint_action.clamp(-self.cfg.clip_actions, self.cfg.clip_actions))

    def _apply_action(self) -> None:
        self.action_manager.apply_action()

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
                self._raise_nan(agent, manager)
            floor = self.cfg.reward_clip_min[agent]
            if floor is not None:
                clipped = torch.clamp(reward, min=floor)
                if self.cfg.reward_clip_during_arm_goals[agent]:
                    reward = clipped
                else:
                    reward = torch.where(self._arm_command.arm_mode, reward, clipped)
            rewards[agent] = reward + self.cfg.termination_penalty[agent] * terminated
        return rewards

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_odometry()
        self.command_manager.compute(dt=self.step_dt)
        groups = list(self.cfg.possible_agents) + ["critic"]
        self._obs_buf = {g: self.observation_manager.compute_group(g, update_history=True) for g in groups}
        self.extras.setdefault("log", {}).update(self._estimator_log)
        return {agent: self._obs_buf[agent] for agent in self.cfg.possible_agents}

    def _get_states(self) -> torch.Tensor:
        # computed with the agents' groups in _get_observations, which step()
        # and reset() always call before state()
        if self.cfg.state_includes_agent_obs:
            return torch.cat([self._obs_buf[agent] for agent in self.cfg.possible_agents] + [self._obs_buf["critic"]], dim=1)
        return self._obs_buf["critic"]

    def _reset_idx(self, env_ids: Sequence[int]):
        # Same order as ManagerBasedRLEnv._reset_idx: curriculum before the
        # scene reset (it reads how far the robot got), managers after.
        self.curriculum_manager.compute(env_ids=env_ids)
        super()._reset_idx(env_ids)  # scene, reset events, episode_length_buf
        self._refresh_sensors_after_reset(env_ids)
        self._fresh[env_ids] = True

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
        """Estimate this step's pelvis motion, move the arm commands by it, and fit the estimator.

        Arm goals follow the estimate with probability estimate_prob: 0 during
        warmup, then ramping to 1, but only while the estimate's drift over a
        whole arm goal (Estimator/goal_drift) is under drift_gate. A worse
        estimate would hand the arms targets they can't reach.

        Runs after resets: reset envs are skipped (their previous sample is
        from the old episode) and get a fresh, exact command from the command
        manager anyway.
        """
        odometry_obs = self.observation_manager.compute_group("odometry", update_history=True)
        root_pos = self._robot.data.root_pos_w.clone()
        root_quat = self._robot.data.root_quat_w.clone()
        cfg = self.cfg.estimator

        if self._have_prev:
            valid = ~self._fresh
            inputs = torch.cat([odometry_obs, self.action_manager.action], dim=1)
            true_motion = pelvis_motion(self._prev_root_pos, self._prev_root_quat, root_pos, root_quat, self.step_dt)
            estimated_motion = self.estimator.model(inputs)
            self._arm_command.apply_pelvis_motion(
                estimated=motion_to_transform(estimated_motion, self.step_dt),
                true=motion_to_transform(true_motion, self.step_dt),
                env_mask=valid,
            )

            if cfg.train:
                self.estimator.add(inputs, true_motion, valid)
                if self.common_step_counter % cfg.train_every == 0:
                    stats = self.estimator.train()
                    self._estimator_log.update({f"Estimator/{k}": v for k, v in stats.items()})
                if cfg.save_every > 0 and self.common_step_counter % cfg.save_every == 0 and self.cfg.log_dir:
                    self._save_estimator()

        self._have_prev = True
        self._prev_root_pos, self._prev_root_quat = root_pos, root_quat
        self._fresh[:] = False

        # Quality gate: the estimate's drift over whole arm goals (shadow command),
        # one EMA sample per ended goal. No GPU sync: everything stays a tensor.
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

        # teacher forcing: the share of new arm goals whose command follows the estimate
        if not cfg.use_estimate:
            prob = torch.tensor(0.0, device=self.device)
        elif not cfg.train:
            prob = torch.tensor(1.0, device=self.device)
        else:
            ramp = min(max((self.common_step_counter - cfg.warmup_steps) / max(cfg.ramp_steps, 1), 0.0), 1.0)
            prob = ramp * gate_open
        arm.estimate_prob = prob
        self._estimator_log["Estimator/estimate_prob"] = prob
        self._estimator_log["Estimator/goal_drift"] = self._goal_drift_ema
        self._estimator_log["Estimator/gate_open"] = gate_open

    def _save_estimator(self):
        directory = os.path.join(self.cfg.log_dir, "estimator")
        env_state = self._env_state()
        self.estimator.save(os.path.join(directory, f"estimator_{self.common_step_counter}.pt"), env_state)
        self.estimator.save(os.path.join(directory, "estimator_latest.pt"), env_state)

    def _env_state(self) -> dict:
        terrain = self.scene.terrain
        return {
            "arm_targets": self._arm_command.state_dict(),
            "terrain_levels": terrain.terrain_levels.clone(),
            "terrain_types": terrain.terrain_types.clone(),
            "goal_drift_ema": self._goal_drift_ema.clone(),
        }

    def _load_env_state(self, state: dict, restore_gate: bool):
        """Curriculum levels (and, with a matching estimator, the drift gate) saved by _save_estimator."""
        self._arm_command.load_state_dict(state["arm_targets"])
        terrain = self.scene.terrain
        levels = state["terrain_levels"].to(self.device)
        if len(levels) == self.num_envs:
            terrain.terrain_levels[:] = levels
            terrain.terrain_types[:] = state["terrain_types"].to(self.device)
        else:
            terrain.terrain_levels[:] = levels[torch.randint(0, len(levels), (self.num_envs,), device=self.device)]
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        if restore_gate:
            self._goal_drift_ema = state["goal_drift_ema"].to(self.device)
        print(
            f"[INFO] Restored curriculum: arm level mean {self._arm_command.level.float().mean().item():.2f},"
            f" terrain level mean {terrain.terrain_levels.float().mean().item():.2f}"
        )

    """
    Helpers.
    """

    def _refresh_sensors_after_reset(self, env_ids: Sequence[int]):
        """Make the IMU read the reset state, not the robot's last pose before the reset.

        DirectMARLEnv.step resets envs without updating the kinematics, so the
        IMU's rigid-body view kept the old pose and velocity until the next
        physics step: the first observation of an episode showed the previous
        episode's (often fallen) orientation and a lin_acc of (old velocity) /
        dt, 100-250 m/s^2 against a running std of ~18. The legs answered with
        knee actions near -300 (clipped at 10): run 12, agent_57600, drop level
        10, 20% of episodes after a fall fell again within 1 s, 12% after a
        time-out. Imu.reset also zeroes its previous velocity, so even a fresh
        first reading is (reset velocity) / dt; it is set to the accelerometer
        at rest, gravity only, and the next reading differences from the reset
        velocity.
        """
        self.scene.write_data_to_sim()
        self.sim.forward()
        imu = self.scene["imu"]
        data = imu.data  # recomputes the reset envs from the refreshed view and stores their velocity
        data.lin_acc_b[env_ids] = quat_apply_inverse(data.quat_w[env_ids], imu._gravity_bias_w[env_ids])
        data.ang_acc_b[env_ids] = 0.0

    def _set_soft_joint_limits(self):
        """Soft limits at cfg.soft_joint_pos_limit_factors of the hard range, for the joints it names.

        Same formula as Articulation (mean +- factor * half range), which applies
        the asset's single factor to every joint. Before the managers: the arm
        target table and joint_pos_limits read these.
        """
        robot = self.scene["robot"]
        for pattern, factor in self.cfg.soft_joint_pos_limit_factors.items():
            ids, names = robot.find_joints(pattern)
            hard = robot.data.joint_pos_limits[:, ids]
            mean, half_range = hard.mean(dim=-1), 0.5 * (hard[..., 1] - hard[..., 0])
            robot.data.soft_joint_pos_limits[:, ids, 0] = mean - factor * half_range
            robot.data.soft_joint_pos_limits[:, ids, 1] = mean + factor * half_range
            limits = [[round(v, 3) for v in pair] for pair in robot.data.soft_joint_pos_limits[0, ids].tolist()]
            print(f"[INFO] Soft joint limits at {factor} of the hard range: {dict(zip(names, limits))}")

    def _raise_nan(self, agent: str, manager: RewardManager):
        step_reward = manager._step_reward
        nan_mask = torch.isnan(step_reward)
        env_ids = nan_mask.any(dim=1).nonzero(as_tuple=True)[0]
        bad_terms = [manager._term_names[i] for i in nan_mask.any(dim=0).nonzero(as_tuple=True)[0].tolist()]
        root_pos = self.scene["robot"].data.root_pos_w[env_ids]
        raise RuntimeError(
            f"NaN in {agent} reward terms {bad_terms} at step {self.common_step_counter},"
            f" envs {env_ids.tolist()[:8]}, root_pos_w {root_pos.tolist()[:8]}"
        )
