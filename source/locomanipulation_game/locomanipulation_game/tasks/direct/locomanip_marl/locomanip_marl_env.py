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

Step order differs from ManagerBasedRLEnv in one place: the command manager
updates inside _get_observations, after interval events rather than before.
"""

from __future__ import annotations

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

from .locomanip_marl_env_cfg import LocoManipMarlEnvCfg


class LocoManipMarlEnv(DirectMARLEnv):
    cfg: LocoManipMarlEnvCfg

    def __init__(self, cfg: LocoManipMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

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
        self.cfg.action_spaces = {
            agent: self.action_manager.get_term(term).action_dim for agent, term in self.cfg.agent_action_terms.items()
        }
        self._configure_env_spaces()
        print(
            f"[INFO] Spaces: observations {self.cfg.observation_spaces}, state {self.cfg.state_space},"
            f" actions {self.cfg.action_spaces}"
        )

        self._obs_buf: dict[str, torch.Tensor] = {}

    """
    DirectMARLEnv hooks.
    """

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
        joint_action = torch.cat([actions[agent] for agent in self.cfg.possible_agents], dim=1)
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
                reward = torch.clamp(reward, min=floor)
            rewards[agent] = reward + self.cfg.termination_penalty[agent] * terminated
        return rewards

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self.command_manager.compute(dt=self.step_dt)
        self._obs_buf = self.observation_manager.compute(update_history=True)
        return {agent: self._obs_buf[agent] for agent in self.cfg.possible_agents}

    def _get_states(self) -> torch.Tensor:
        # computed with the agents' groups in _get_observations, which step()
        # and reset() always call before state()
        return self._obs_buf["critic"]

    def _reset_idx(self, env_ids: Sequence[int]):
        # Same order as ManagerBasedRLEnv._reset_idx: curriculum before the
        # scene reset (it reads how far the robot got), managers after.
        self.curriculum_manager.compute(env_ids=env_ids)
        super()._reset_idx(env_ids)  # scene, reset events, episode_length_buf

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
    Helpers.
    """

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
