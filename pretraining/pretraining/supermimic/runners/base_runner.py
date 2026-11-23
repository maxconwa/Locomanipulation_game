# Ported from SuperMimic (HybridRobotics, commit 7e94b07, Apache-2.0), src/supermimic/runners/supermimic_runner.py.
# Changes: BaseRunnerCfg, WorldModel, build_world_model, load_wandb_checkpoint, Ratio and BaseRunner only (not
# SuperMimicRunnerCfg, build_actor, SuperMimicRunner); EnvAdapter added (mjlab's RslRlVecEnvWrapper over Isaac Lab's
# env); typing.Generic for PEP 695 (Python 3.11); default joint positions of the policy joints; wandb optional.
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from pretraining.supermimic.envs.motion_tracking_env import MotionTrackingEnv
from pretraining.supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from pretraining.supermimic.models import Dynamics
from pretraining.supermimic.replay_buffer import FieldSpec, ReplayBuffer
from pretraining.supermimic.representations import StateRepresentation, create_representation
from pretraining.supermimic.sim_context import SimContext
from pretraining.supermimic.utils.helpers import (
    extract_action_scale_offset,
    unpack_obs_dims,
)
from pretraining.supermimic.utils.logger import Logger
from pretraining.supermimic.utils.wandb_artifacts import download_model_dir, optional_wandb


class EnvAdapter:
    """An Isaac Lab ``ManagerBasedRLEnv`` seen as mjlab's ``RslRlVecEnvWrapper``, the env the runner drives.

    Transcribed from mjlab (mjlab_research 6ea3e57, ``src/mjlab/rl/vecenv_wrapper.py``) without rsl_rl's ``VecEnv``
    base, the spaces and the render mode: observations as a ``TensorDict``, ``dones = terminated | truncated`` and,
    on an infinite-horizon env, the time-outs in ``extras["time_outs"]``.
    """

    def __init__(self, env: Any, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions

        self.num_envs = self.unwrapped.num_envs
        self.device = torch.device(self.unwrapped.device)
        self.max_episode_length = self.unwrapped.max_episode_length
        self.num_actions = self.unwrapped.action_manager.total_action_dim

        # Reset at the start since rsl_rl does not call reset.
        self.env.reset()

    @property
    def cfg(self) -> Any:
        return self.unwrapped.cfg

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def _to_tensordict(self, obs_dict: dict[str, Any]) -> TensorDict:
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        obs_dict = self.unwrapped.observation_manager.compute()
        return self._to_tensordict(obs_dict)

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()
        return self._to_tensordict(obs_dict), extras

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self._to_tensordict(obs_dict), rew, dones, extras

    def close(self) -> None:
        return self.env.close()


@dataclass
class BaseRunnerCfg:
    """Shared configuration fields for all runners."""

    wandb_project: str = "supermimic"
    run_name: str | None = None
    wandb_run_path: str | None = None
    headless: bool = True
    buffer_size: int = 1000
    representation: str = "gencoord"  # "gencoord" or "maxcoord"
    actor_obs: str = "rep_state"  # "rep_state" or "proprio"
    log_interval: int = 10
    save_interval: int = 1000
    history_length: int = 1
    termination_softness: float = 0.05
    dynamics_hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    wm: WorldModelLearnerCfg = field(default_factory=WorldModelLearnerCfg)


@dataclass
class WorldModel:
    """The world model plus the env dimensions that sized it."""

    representation: StateRepresentation
    dynamics: Dynamics
    sim_context: SimContext
    num_goals: int
    num_actions: int

    def bundle(self) -> dict:
        """The ``world_model.pt`` contents."""
        return {
            "dynamics": self.dynamics.state_dict(),
            "sim_context": self.sim_context.state_dict(),
        }

    def load_bundle(self, bundle: dict) -> None:
        """Load a ``world_model.pt`` bundle."""
        self.dynamics.load_state_dict(bundle["dynamics"])
        self.sim_context.load_state_dict(bundle["sim_context"])


def build_world_model(cfg: BaseRunnerCfg, env, device: str) -> WorldModel:
    """Construct the representation and dynamics for ``env`` sized by ``cfg``.

    Resets ``env`` to read its observation dimensions.
    """
    if cfg.history_length > 1 and cfg.actor_obs != "proprio":
        raise RuntimeError("history_length > 1 requires actor_obs='proprio' (--runner.actor-obs proprio)")

    obs, _ = env.reset()
    num_full_state, num_goals = unpack_obs_dims(obs)
    num_actions = env.num_actions

    print(f"[INFO] State Dim: {num_full_state}, Goal Dim: {num_goals}")

    sim_context = SimContext.from_env(env.unwrapped)
    sim_ctx_dim = sim_context.output_dim
    print(f"[INFO] Sim Context: {sim_ctx_dim} dim ({[s.name for s in sim_context.specs]})")

    # The policy joints, in the motion command's (JOINT_NAMES) order: the Isaac articulation also has the gripper
    # hinges, and its joint order is not the policy's.
    joint_ids = env.unwrapped.command_manager.get_term("motion").joint_ids
    default_joint_pos = env.unwrapped.scene["robot"].data.default_joint_pos[0][joint_ids].to(device)
    representation = create_representation(
        cfg.representation,
        num_full_state,
        num_actions,
        env.unwrapped.step_dt,
        env.unwrapped.cfg.decimation,
        default_joint_pos,
        actor_obs_mode=cfg.actor_obs,
    ).to(device)

    representation.bind(env.unwrapped, sim_context)

    print(
        f"[INFO] Representation: {cfg.representation} "
        f"(full={representation.full_state_dim}, rep={representation.rep_state_dim}, "
        f"pred={representation.prediction_dim})"
    )

    action_scale, action_offset = extract_action_scale_offset(env)
    dynamics = Dynamics(
        representation,
        num_actions,
        sim_ctx_dim,
        action_scale,
        action_offset,
        hidden_dims=cfg.dynamics_hidden_dims,
    ).to(device)

    return WorldModel(
        representation=representation,
        dynamics=dynamics,
        sim_context=sim_context,
        num_goals=num_goals,
        num_actions=num_actions,
    )


def load_wandb_checkpoint(wandb_run_path: str, wm: WorldModel, device: str) -> None:
    """Warm-start ``wm`` from another W&B run's world-model bundle."""
    print(f"[INFO] Loading World Model from W&B: {wandb_run_path}")
    wandb = optional_wandb()
    if wandb is None:
        raise RuntimeError("wandb_run_path needs wandb (not installed, or WANDB_MODE=disabled)")
    run = wandb.Api().run(wandb_run_path)
    wm.load_bundle(torch.load(download_model_dir(run) / "world_model.pt", map_location=device))


class Ratio:
    def __init__(self, ratio: float) -> None:
        self._ratio = ratio
        self._prev: int | None = None

    def __call__(self, step: int) -> int:
        step = int(step)
        if self._ratio == 0:
            return 0
        if self._ratio < 0:
            return 1
        if self._prev is None:
            self._prev = step
            return 1
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats


CfgT = TypeVar("CfgT", bound=BaseRunnerCfg)


class BaseRunner(Generic[CfgT]):
    """Base class with shared initialization for all runners.

    Handles: env dimension extraction, representation creation, world model
    setup, replay buffer, action manager, and logger. Subclasses implement
    ``_init_policy()`` for their specific policy setup.
    """

    cfg: CfgT

    def __init__(
        self,
        env,
        cfg: CfgT,
        task_id: str,
        registry_name: str | None = None,
        device: str = "cuda:0",
    ):
        self.env = env
        self.cfg = cfg
        self.task_id = task_id
        self.registry_name = registry_name
        self.device = device

        self._init_common()
        self._init_policy()

        self.logger = Logger(
            run_name=cfg.run_name,
            num_envs=env.num_envs,
            max_episode_length=env.max_episode_length,
            device=device,
        )

    def _init_common(self) -> None:
        wm = build_world_model(self.cfg, self.env, self.device)
        self._wm = wm
        self._representation = wm.representation
        self._dynamics = wm.dynamics
        self._sim_context = wm.sim_context
        self._num_goals = wm.num_goals
        self._num_actions = wm.num_actions

        if self.cfg.wandb_run_path is not None:
            load_wandb_checkpoint(self.cfg.wandb_run_path, wm, self.device)

        self.action_manager = self.env.unwrapped.action_manager
        self.motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
        self.motion = self.motion_cmd.motion

        self.replay_buffer = ReplayBuffer(
            schema=self._buffer_schema(),
            buffer_size=self.cfg.buffer_size,
            num_envs=self.env.num_envs,
            device=self.device,
        )
        self.world_model_learner = WorldModelLearner(
            self._dynamics, self._representation, self.cfg.wm, device=self.device
        )

        self._pending_norm_updates: list[tuple] = []
        self._sim_history: torch.Tensor | None = None

    def _seed_sim_history(self) -> None:
        """Seed the real-sim obs-history ring with the noised current sensor."""
        if self.cfg.history_length == 1:
            return
        seed = self._representation.add_proprio_noise(self.env.get_observations()["actor"].to(self.device))
        self._sim_history = seed.unsqueeze(1).repeat(1, self.cfg.history_length, 1)

    def _collection_history(self, sensor: torch.Tensor) -> torch.Tensor:
        """The ``(B, history_length, sensor_dim)`` window for a real-sim actor step.
        history_length > 1 needs ``_seed_sim_history()`` first."""
        if self.cfg.history_length == 1:
            return sensor.unsqueeze(1)
        assert self._sim_history is not None
        return self._sim_history

    def _advance_sim_history(self, next_actor: torch.Tensor, dones: torch.Tensor) -> None:
        """Append the noised next sensor; rows whose episode just reset repeat that
        single realization, matching the warm-start padding."""
        if self.cfg.history_length == 1:
            return
        assert self._sim_history is not None
        next_sensor = self._representation.add_proprio_noise(next_actor)
        shifted = torch.cat([self._sim_history[:, 1:], next_sensor.unsqueeze(1)], dim=1)
        reset = dones.bool().view(-1, 1, 1)
        self._sim_history = torch.where(reset, next_sensor.unsqueeze(1), shifted)

    def _policy_input_dims(self) -> tuple[int, int]:
        """Compute actor and critic input dimensions from the representation."""
        actor_dim = self._representation.actor_obs_dim(self._num_actions, self._num_goals, self.cfg.history_length)
        critic_dim = self._representation.rep_state_dim + self._num_goals + self._sim_context.output_dim
        return actor_dim, critic_dim

    def _build_mlp(
        self,
        name: str,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        **kwargs: object,
    ) -> nn.Module:
        """Build an MLPModel with input obs normalization."""
        obs = TensorDict(
            {name: torch.zeros(1, input_dim, device=self.device)},
            batch_size=[1],
            device=self.device,
        )
        return MLPModel(
            obs=obs,
            obs_groups={name: [name]},
            obs_set=name,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            obs_normalization=True,
            **kwargs,
        ).to(self.device)

    def _create_wm_env(self, batch_size: int) -> MotionTrackingEnv:
        """Create a :class:`MotionTrackingEnv` bound to this runner's world model and buffer."""
        return MotionTrackingEnv(
            dynamics=self._dynamics,
            representation=self._representation,
            env=self.env,
            batch_size=batch_size,
            replay_buffer=self.replay_buffer,
            history_length=self.cfg.history_length,
            termination_softness=self.cfg.termination_softness,
            device=self.device,
        )

    @property
    def _stores_actor_obs(self) -> bool:
        """The buffer carries the real sensor only for its sole reader, the
        history warm-start (``WorldModelEnv.sample_initial_states``)."""
        return self._representation.actor_obs_mode == "proprio" and self.cfg.history_length > 1

    def _buffer_schema(self) -> dict[str, FieldSpec]:
        """Override to customize the replay buffer schema (e.g. extra fields)."""
        schema = {
            "state": FieldSpec(self._representation.full_state_dim),
            "action": FieldSpec(self._num_actions),
            "prev_action": FieldSpec(self._num_actions),
            "motion_time": FieldSpec(dtype=torch.long),
            "terminated": FieldSpec(dtype=torch.bool),
            "truncated": FieldSpec(dtype=torch.bool),
        }
        if self._stores_actor_obs:
            schema["actor_obs"] = FieldSpec(self._representation.proprio_obs_dim(self._num_actions))
        schema["sim_ctx"] = FieldSpec(self._sim_context.output_dim)
        return schema

    def _init_policy(self) -> None:
        """Subclasses implement this to create their policy learner."""
        raise NotImplementedError

    def _step_and_store(
        self,
        actions: torch.Tensor,
        curr_states: torch.Tensor,
        actor_td: TensorDict | None = None,
        extra_fields: dict[str, torch.Tensor] | None = None,
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
        """Step the env and write the transition to the replay buffer.

        When given, ``actor_td`` is queued for the actor-normalizer update.
        """
        motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
        motion_times = motion_cmd.time_steps.clone()

        raw_actions = actions.detach().clone()
        prev_action = self.action_manager.action.clone()

        next_obs, rewards, dones, extras = self.env.step(actions)

        truncated = extras.get(
            "time_outs",
            torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.device),
        )
        terminated = dones.bool() & ~truncated

        fields: dict[str, torch.Tensor] = {
            "state": curr_states,
            "action": raw_actions,
            "prev_action": prev_action,
            "motion_time": motion_times,
            "terminated": terminated,
            "truncated": truncated,
        }
        if extra_fields:
            fields.update(extra_fields)
        self.replay_buffer.insert(TensorDict(fields, batch_size=[self.env.num_envs]))

        if actor_td is not None:
            self._pending_norm_updates.append(
                (
                    curr_states.clone(),
                    raw_actions.clone(),
                    actor_td.detach().clone(),
                )
            )
        return next_obs, rewards, dones, extras

    def _flush_normalization(self) -> None:
        """Apply queued normalizer updates. Call outside ``inference_mode()``."""
        if not self._pending_norm_updates:
            return
        states, actions, actor_inputs = zip(*self._pending_norm_updates, strict=True)
        self.world_model_learner.update_normalization(torch.cat(states), torch.cat(actions))
        self._update_actor_normalization(actor_inputs)
        self._pending_norm_updates.clear()

    def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
        """Subclasses override to update the actor obs normalizer."""
        raise NotImplementedError

    def _create_log_dir(self, *extra_parts: str) -> Path:
        """Create and return a timestamped log directory."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = Path("logs", self.task_id, *extra_parts, timestamp)
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Logging to {log_dir}")
        return log_dir

    def _init_wandb(self, log_dir: Path, name_suffix: str = "") -> None:
        """Initialize the W&B run and link the registry artifacts."""
        wandb = optional_wandb()
        if wandb is None:
            return
        wandb_config = {
            "env_cfg": asdict(self.env.cfg),
            "runner_cfg": asdict(self.cfg),
        }

        default_name = f"{self.task_id}{name_suffix}-{log_dir.name}"
        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.run_name or default_name,
            config=wandb_config,
        )

        if self.registry_name:
            api = wandb.Api()
            for name in self.registry_name.split(","):
                name = name.strip()
                if ":" not in name:
                    name = name + ":latest"
                print(f"[INFO] Linking registry artifact: {name}")
                artifact = api.artifact(name)
                wandb.run.use_artifact(artifact)

    def _cleanup(self) -> None:
        """Close environment and finish W&B run."""
        env_close_fn = getattr(self.env, "close", None)
        if env_close_fn:
            env_close_fn()
        wandb = optional_wandb()
        if wandb is not None:
            wandb.finish()
