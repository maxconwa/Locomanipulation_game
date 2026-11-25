# Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c), src/h12_train/marl/runner.py.
# Changes: imports; stage 1 only (CommandRunner and load_command_runner left out, with TwoAgentRunner._sampling and
# the collision-mask tables of _make_tables); robot constants and sim reads: the estop limits of the policy joints
# (JOINT_NAMES) from pretraining.constants, the encoder bias from env.encoder_bias, the applied joint torque from the
# Isaac articulation's applied_torque (MuJoCo: qfrc_actuator), reward and termination terms as configclass attributes;
# wandb optional; ImitationRunner takes run_dir and motion_files (run() then needs no meta); load_actors,
# IMITATION_GOAL_DIMS and PROPRIO_DIM added.
"""The two-agent SuperMimic runner: sim collection, world model, BPTT actors, stage 1.

Built on ``pretraining.supermimic.runners.base_runner.BaseRunner`` (replay buffer, world-model
learner, logger, history ring, transition storage) with the H1-2 torso-IMU world model
(``representation.build_world_model``). Each iteration steps every sim env once with
both actors on a shared proprio history and their own goals, stores the transition and,
at ``replay_ratio``, fits the world model on replayed windows and runs one
``TwoAgentLearner`` update, as ``SuperMimicRunner.run`` does.

``ImitationRunner`` (stage 1) tracks the clips. (h12_train's stage 2, ``CommandRunner``, is not ported.)

Runs are resumable (``load_checkpoint``): the checkpoint holds the world model and sim
context, actors and critics with their normalisers, every optimiser's state, the stage-1
reset sampler's statistics, the iteration and the elapsed wall time. The replay buffer
is not saved: a resumed run collects ``resume_warmup`` iterations before it updates again.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from pretraining.constants import JOINT_NAMES
from pretraining.plants import get_plant
from pretraining.marl.actors import build_imitation_actor
from pretraining.marl.config import ACTION_DIMS, AGENTS, EE_DIM, JOINT_SLICES, ImitationRewardCfg
from pretraining.marl.learner import TwoAgentLearner
from pretraining.marl.representation import build_world_model
from pretraining.marl.rewards import JointLimits, estop_ratios, max_step_rewards
from pretraining.marl.runs import (
    ProgressCsv,
    atomic_save,
    atomic_write_text,
    config_from_dict,
    default_runs_dir,
    keep_last,
    read_checkpoint_info,
    to_repo_path,
)
from pretraining.marl.wm_env import ClipTables, ImitationWorldModelEnv, arms_goal, ee_window, legs_goal
from pretraining.supermimic.goal_window import NUM_GOAL_FRAMES
from pretraining.supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg
from pretraining.supermimic.learners.wm_learner import WorldModelLearner
from pretraining.supermimic.models import build_mlp
from pretraining.supermimic.replay_buffer import FieldSpec, ReplayBuffer
from pretraining.supermimic.sim_context import widen_input_weight
from pretraining.supermimic.runners.base_runner import BaseRunner, BaseRunnerCfg, Ratio
from pretraining.supermimic.runners.reset_sampling import ReturnSampler, ReturnSamplerCfg
from pretraining.supermimic.utils.wandb_artifacts import optional_wandb

# Shares of golem_safety's estop limits whose crossing the sim logs (``Safety/*``).
ESTOP_LOG_LEVELS = (0.9, 1.0)
# rad: the sim also logs how often a joint is this close to its estop position limit.
ESTOP_POSITION_BAND = 0.05
# |applied torque| at or above this share of the effort limit counts as saturated.
SATURATED = 0.999

# The policy contract: one proprio frame is [gravity 3, gyro 3, q - default 27, dq 27, last action 27], and each
# agent's stage-1 goal per frame of SuperMimic's window is its joints' [pos, vel] plus the pelvis orientation
# relative to the robot (legs, rot6d) or both tool frames (arms, position + rot6d each).
PROPRIO_DIM = 3 + 3 + 3 * len(JOINT_NAMES)
IMITATION_GOAL_DIMS = {
    "legs": NUM_GOAL_FRAMES * (2 * ACTION_DIMS["legs"] + 6),
    "arms": NUM_GOAL_FRAMES * (2 * ACTION_DIMS["arms"] + 2 * EE_DIM),
}


@dataclass
class TwoAgentRunnerCfg(BaseRunnerCfg):
    """Runner settings; the defaults not set here are ``BaseRunnerCfg``'s (SuperMimic's)."""

    actor_obs: str = "proprio"
    max_iterations: int = 1_000_000  # SuperMimicRunnerCfg's
    replay_ratio: int = 16
    latent_dim: int = 32
    actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    command_hidden_dims: tuple[int, ...] = (512, 256)
    command_proprio: bool = False
    """Stage 2 (not ported): the command encoders also read the proprio history (h12_train's ``actors.CommandActor``)."""
    decoder_history_length: int = 0
    """Stage 2 (not ported): the stage-1 actors' history length when the command encoders read a
    longer one (``history_length`` is then the command encoders' and the buffers');
    0: the same as ``history_length``."""
    command_prior: str = ""
    """Stage 2 (not ported): a latent prior file of ``scripts/fit_prior.py`` (``{agent: state dict}``);
    the command encoders then output residuals on it. Empty: no prior."""
    command_prior_hidden_dims: tuple[int, ...] = (512, 256)
    policy: ActorCriticPolicyLearnerCfg = field(default_factory=ActorCriticPolicyLearnerCfg)
    sampling_mode: Literal["uniform", "adaptive", "start", "return"] = "return"
    """Reset-frame distribution. 'uniform' and stage 1's default 'return' start from a
    frame drawn per frame (or per 0.2 s bin) of the concatenated motion, as mjlab and
    SuperMimic do, so a clip is drawn in proportion to its length; 'start' always starts
    at frame 0."""
    return_sampler: ReturnSamplerCfg = field(default_factory=ReturnSamplerCfg)
    runs_dir: str = field(default_factory=default_runs_dir)
    max_seconds: float = 0.0
    """Stop once the run's wall time (s, summed over resumed segments) passes this,
    saving a checkpoint; 0: run max_iterations."""
    seed: int | None = None
    """Seed of torch, numpy, random and the sim (mjlab's ``seed_rng``); the scripts draw
    one when unset and record it here. A resumed segment seeds with seed + its first
    iteration."""
    checkpoint_interval_s: float = 1800.0
    """Wall time (s) between checkpoints, besides every ``save_interval`` iterations."""
    keep_checkpoints: int = 3
    """Periodic checkpoints kept in ``checkpoints/``; ``checkpoint.pt`` is the newest."""
    resume_warmup: int = 50
    """Iterations a resumed run collects into its empty replay buffer before updating."""


def eta_seconds(
    it: int, start_iteration: int, max_iterations: int, segment_s: float, wall_s: float, max_seconds: float
) -> float:
    """Seconds left after iteration ``it``: this segment's mean time per iteration (counted from ``start_iteration``,
    so right after a resume) times the iterations left to ``max_iterations``; with a wall-time budget
    (``max_seconds`` > 0, the run's summed ``wall_s``) the smaller of that and the budget left."""
    done = it + 1 - start_iteration
    eta = segment_s / done * max(max_iterations - it - 1, 0) if done > 0 else 0.0
    if max_seconds > 0:
        eta = min(eta, max(max_seconds - wall_s, 0.0))
    return eta


def safety_flags(
    limits: JointLimits,
    q: torch.Tensor,
    dq: torch.Tensor,
    q_next: torch.Tensor,
    dq_next: torch.Tensor,
    target: torch.Tensor,
    torque: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return, per env (bool), which of golem_safety's estop limits a step crossed, by joint group (``Safety/*``).

    ``q, dq``: the pre-step joint state; ``q_next, dq_next``: the post-step one; ``target``: the actuator's position
    target (the clipped processed action less the encoder bias); ``torque``: the applied joint torque at the step's
    last substep. Velocity: the post-step ``|dq|``; torque: the PD demand of the target on the pre-step state (the
    first substep's, as the world model's penalty); force: the applied torque at its effort limit.
    """
    force = torque.abs() / limits.effort
    vel, tau = estop_ratios(limits, q, dq, dq_next, target)
    # Room (rad) to the nearer estop position limit; negative past it.
    room = torch.minimum(q_next - limits.q_low, limits.q_high - q_next)
    out = {}
    for a in AGENTS:
        sl = JOINT_SLICES[a]
        out[f"{a}_force_saturated"] = force[:, sl].max(-1).values >= SATURATED
        least = room[:, sl].min(-1).values
        out[f"{a}_q_over_estop"] = least < 0.0
        out[f"{a}_q_within_{ESTOP_POSITION_BAND}"] = least < ESTOP_POSITION_BAND
        for level in ESTOP_LOG_LEVELS:
            out[f"{a}_dq_over_{level}"] = vel[:, sl].max(-1).values > level
            out[f"{a}_tau_demand_over_{level}"] = tau[:, sl].max(-1).values > level
    return out


class TwoAgentRunner(BaseRunner[TwoAgentRunnerCfg]):
    """Sim collection and the training loop."""

    stage: str
    wm_env: ImitationWorldModelEnv
    learner: TwoAgentLearner
    actors: dict[str, nn.Module]
    critics: dict[str, nn.Module]
    run_dir: Path | None = None
    """The run directory; None: ``<cfg.runs_dir>/<cfg.run_name or stage>``."""

    def _init_common(self) -> None:
        """Do ``BaseRunner._init_common``'s set-up with the torso-IMU world model."""
        wm = build_world_model(self.cfg, self.env, self.device, getattr(self, "sim_context_layout", None),
                               fine_tune=getattr(self, "sim_context_fine_tune", False))
        self._wm = wm
        self._representation = wm.representation
        self._dynamics = wm.dynamics
        self._sim_context = wm.sim_context
        self._num_goals = wm.num_goals
        self._num_actions = wm.num_actions
        self.action_manager = self.env.unwrapped.action_manager
        # a delayed env (tasks.delay.TargetDelay): the world model takes each step's effective target, from the
        # stored targets of the two steps before it (Dynamics.set_delay)
        term = self.action_manager.get_term("joint_pos")
        self._delay_term = term if hasattr(term, "previous_targets") else None
        if self._delay_term is not None:
            self._dynamics.set_delay(self.env.unwrapped.cfg.decimation, self._sim_context)
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
        # The policy joints, in the full state's (JOINT_NAMES) order.
        # The PD of the env's plant (TrackMainEnvCfg.plant; the pretrain plant's for envs that name none).
        plant = get_plant(getattr(self.env.unwrapped.cfg, "plant", "pretrain"))
        self.joint_limits = JointLimits.for_joints(JOINT_NAMES, self.device, plant)
        self._safety: dict[str, torch.Tensor] = {}
        self._safety_steps = torch.zeros((), device=self.device)
        self.start_iteration = 0
        self._wall_offset = 0.0

    def _buffer_schema(self) -> dict:
        """The base schema, plus the two steps' previous targets for a delayed env (``Dynamics.delay``)."""
        schema = super()._buffer_schema()
        if self._dynamics.delay is not None:
            schema["prev_target"] = FieldSpec(2 * self._num_actions)
        return schema

    def _make_tables(self) -> None:
        """Tabulate the clips' tool-frame poses."""
        self.tables = ClipTables(self.motion)
        self.obs_dim = self._representation.proprio_obs_dim(self._num_actions)

    def _build_critics(self, critic_dim: int) -> None:
        """Build one critic per agent on the privileged input."""
        self.critics = {
            a: build_mlp(critic_dim, self.cfg.critic_hidden_dims, output_dim=1).to(self.device) for a in AGENTS
        }

    def _finish_policy(self, wm_env) -> None:
        """Compile the WM env's step and reset (as SuperMimic does) and build the learner."""
        if torch.cuda.is_available():
            wm_env.step = torch.compile(wm_env.step)
            wm_env.reset = torch.compile(wm_env.reset)
        self.wm_env = wm_env
        self.learner = TwoAgentLearner(self.actors, self.critics, wm_env, self.cfg.policy, device=self.device)
        self._sampler: ReturnSampler | None = None
        if self.cfg.sampling_mode == "return":
            self._sampler = ReturnSampler(
                cfg=self.cfg.return_sampler,
                bin_count=self.motion_cmd.bin_count,
                frame_count=self.motion.time_step_total,
                num_envs=self.env.num_envs,
                max_step_reward=self.env.unwrapped.step_dt
                # Isaac Lab cfgs are configclasses, not dicts: terms are attributes, disabled ones None.
                * sum(t.weight for t in vars(self.env.cfg.rewards).values() if t is not None and t.weight > 0.0),
                device=self.device,
            )

    def _goals(self, motion_times: torch.Tensor, aux_state: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        """Return each agent's goal keys for the sim envs."""
        raise NotImplementedError

    def act(
        self, history: torch.Tensor, agent_goals: dict[str, dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, TensorDict]:
        """Return both agents' concatenated action and their inputs."""
        n = history.shape[0]
        tds = {a: TensorDict({"history": history, **agent_goals[a]}, batch_size=[n], device=self.device) for a in AGENTS}
        actions = torch.cat([self.actors[a](tds[a]) for a in AGENTS], dim=-1)
        return actions, TensorDict(tds, batch_size=[n], device=self.device)

    def step(self) -> int:
        """Step every sim env once with both actors and store the transition."""
        with torch.inference_mode():
            obs = self.env.get_observations()
            curr_states = obs["full_state"].to(self.device)
            sim_actor_obs = obs["actor"].to(self.device)
            motion_times = self.motion_cmd.time_steps.to(self.device).clone()
            rep_state = self._representation.represent(curr_states)
            aux_state = self._representation.extract_aux_state(curr_states)
            sim_ctx = self._sim_context.read(self.env.unwrapped)
            sensor = self._representation.compute_actor_sensor(
                rep_state, sim_ctx, aux_state=aux_state, sim_actor_obs=sim_actor_obs, noisy=True
            )
            actions, actor_td = self.act(self._collection_history(sensor), self._goals(motion_times, aux_state))
            extra = {"sim_ctx": sim_ctx}
            if self._dynamics.delay is not None:
                extra["prev_target"] = self._delay_term.previous_targets().flatten(1)
            if self._stores_actor_obs:
                extra["actor_obs"] = sim_actor_obs
            next_obs, rewards, dones, extras = self._step_and_store(
                actions, curr_states, actor_td=actor_td, extra_fields=extra
            )
            self._advance_sim_history(next_obs["actor"].to(self.device), dones)
            terminated = self.env.unwrapped.termination_manager.terminated
            next_states = next_obs["full_state"].to(self.device)
            self._record_safety(curr_states, next_states, dones)
            self.logger.process_env_step(rewards, terminated, self._log_extras(extras))
            self._after_step(next_states, dones, motion_times, rewards, terminated)
        self._flush_normalization()
        return self.env.num_envs

    def _log_extras(self, extras: dict) -> dict:
        """Return the env's episode extras to log."""
        return extras

    def _record_safety(self, curr_states: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor) -> None:
        """Accumulate, per joint group, how often the step crossed golem_safety's estop limits.

        Velocity: the post-step ``|dq|``; torque: the PD demand of the clipped target on the
        pre-step state (the first substep's, as the world model's penalty); force: the
        actuator's applied joint torque at the step's last substep at its effort limit. Rows
        that reset in the step are not counted.
        """
        from pretraining.tasks.events import encoder_bias

        n = self._num_actions
        q, dq = curr_states[:, 7 : 7 + n], curr_states[:, 13 + n :]
        q_next, dq_next = next_states[:, 7 : 7 + n], next_states[:, 13 + n :]
        robot = self.env.unwrapped.scene["robot"]
        # The actuator's target: the clipped processed action less the encoder bias.
        target = self.action_manager.get_term("joint_pos")._processed_actions - encoder_bias(self.env.unwrapped)
        torque = robot.data.applied_torque[:, self.motion_cmd.joint_ids]
        alive = (~dones.bool()).float()
        for k, v in safety_flags(self.joint_limits, q, dq, q_next, dq_next, target, torque).items():
            self._safety[k] = self._safety.get(k, 0.0) + (v.float() * alive).sum()
        self._safety_steps = self._safety_steps + alive.sum()

    def _after_step(self, next_states, dones, motion_times, rewards, terminated) -> None:
        """Update the reset sampler from the step."""
        if self._sampler is None:
            return
        manager = self.env.unwrapped.termination_manager
        truncated = manager.time_outs.to(self.device, torch.bool)
        clip_end = manager.get_term("motion_end").to(self.device, torch.bool)
        weights = self._sampler.record(
            motion_times,
            rewards,
            terminated.to(self.device, torch.bool) & ~truncated,
            clip_end,
            truncated & ~clip_end,
        )
        if weights is not None:
            self.motion_cmd.set_sampling_distribution(weights)

    def _on_sim_reset(self) -> None:
        """Prepare stage-specific sim state after the env's first reset."""

    def _update_actor_normalization(self, actor_inputs: tuple[TensorDict, ...]) -> None:
        """Update each actor's normalisers from the collected inputs."""
        for a in AGENTS:
            self.actors[a].train()
            for td in actor_inputs:
                self.actors[a].update_normalization(td[a])

    def _sim_metrics(self) -> dict[str, float]:
        """Return the sim's estop crossings since the last log, and reset them."""
        steps = float(self._safety_steps)
        out = {f"Safety/{k}": float(v) / max(steps, 1.0) for k, v in self._safety.items()}
        self._safety.clear()
        self._safety_steps = torch.zeros((), device=self.device)
        return out

    # Checkpoints and resume.

    def _optimizers(self) -> dict[str, list[torch.optim.Optimizer]]:
        """Return every optimiser of the run by name."""
        out = {"world_model": self.world_model_learner.optimizers}
        for a in AGENTS:
            out[f"actor_{a}"] = self.learner.actor_optimizers[a]
            out[f"critic_{a}"] = self.learner.critic_optimizers[a]
        return out

    def training_state(self, iteration: int, wall_s: float) -> dict:
        """Return everything a resumed run needs except the replay buffer."""
        state = {
            "iteration": iteration,
            "wall_s": wall_s,
            "world_model": self._wm.bundle(),
            "actors": {a: m.state_dict() for a, m in self.actors.items()},
            "critics": {a: m.state_dict() for a, m in self.critics.items()},
            "optimizers": {k: [o.state_dict() for o in opts] for k, opts in self._optimizers().items()},
        }
        if self._sampler is not None:
            state["return_sampler"] = {k: getattr(self._sampler, k) for k in ("value", "count", "effort")}
        return state

    def load_checkpoint(self, path: Path) -> None:
        """Load a ``training_state`` into this runner, before its first update.

        The optimisers' learning rates stay device tensors (the captured updates read them).
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        widen_sim_context(state, self._sim_context)
        self._load_world_model(state["world_model"])
        for a in AGENTS:
            self.actors[a].load_state_dict(state["actors"][a])
            self.critics[a].load_state_dict(state["critics"][a])
        for name, opts in self._optimizers().items():
            for opt, sd in zip(opts, state["optimizers"][name], strict=True):
                opt.load_state_dict(sd)
                _pad_optimizer_state(opt)
                for group in opt.param_groups:
                    if torch.is_tensor(group["lr"]):
                        group["lr"] = group["lr"].to(self.device)
        if self._sampler is not None and "return_sampler" in state:
            for k, v in state["return_sampler"].items():
                if tuple(v.shape) != tuple(getattr(self._sampler, k).shape):
                    raise ValueError(
                        f"{path}: the reset sampler's {k} has shape {tuple(v.shape)}, this run's clips give "
                        f"{tuple(getattr(self._sampler, k).shape)}: the checkpoint was trained on other clips"
                    )
                setattr(self._sampler, k, v.to(self.device))
        self.start_iteration = int(state["iteration"]) + 1
        self._wall_offset = float(state["wall_s"])

    def _load_world_model(self, bundle: dict) -> None:
        """Load a world-model bundle; refuse one whose action scale or offset differs from the env's."""
        for key in ("action_scale", "action_offset"):
            saved = bundle["dynamics"][key].to(self.device)
            if not torch.allclose(saved, getattr(self._dynamics, key)):
                raise ValueError(
                    f"the saved world model's {key} differs from this env's: the run was trained with "
                    "another action space (h12_train/README.md)"
                )
        self._wm.load_bundle(bundle)

    def save(self, log_dir: Path, iteration: int, wall_s: float, periodic: bool = False) -> None:
        """Write the deploy files and ``checkpoint.pt`` atomically; ``periodic`` also keeps a numbered copy."""
        atomic_save(self._wm.bundle(), log_dir / "world_model.pt")
        atomic_save({a: m.state_dict() for a, m in self.actors.items()}, log_dir / "actors.pt")
        atomic_save({a: m.state_dict() for a, m in self.critics.items()}, log_dir / "critics.pt")
        state = self.training_state(iteration, wall_s)
        atomic_save(state, log_dir / "checkpoint.pt")
        if periodic:
            (log_dir / "checkpoints").mkdir(exist_ok=True)
            atomic_save(state, log_dir / "checkpoints" / f"it_{iteration:08d}.pt")
            keep_last(log_dir / "checkpoints", "it_*.pt", self.cfg.keep_checkpoints)
        atomic_write_text(
            log_dir / "checkpoint.json",
            json.dumps({"iteration": iteration, "wall_s": wall_s}) + "\n",
        )

    # The training loop.

    def _run_meta(self) -> dict:
        """Return the ``run.json`` entries of a new run besides the stage, the cfg and the wandb id."""
        return {}

    def run(self, meta: dict | None = None, resume: bool = False) -> Path:
        """Train until ``max_iterations`` or ``max_seconds``; return the run directory.

        A new run refuses an existing run directory; ``resume`` continues one (after
        ``load_checkpoint``). ``meta`` (motion files, reward cfg, ...; default ``_run_meta()``) is
        written to ``run.json`` with the cfg on a new run; its paths are repository-relative
        (``runs.to_repo_path``). Every log interval appends a row to ``progress.csv``: timings,
        losses, sim episode statistics, estop crossings and the peak GPU memory.
        """
        if meta is None:
            meta = self._run_meta()
        log_dir = self.run_dir or Path(self.cfg.runs_dir) / (self.cfg.run_name or self.stage)
        if resume:
            run_json = json.loads((log_dir / "run.json").read_text())
            run_json.setdefault("resumed", []).append({"iteration": self.start_iteration, "wall_s": self._wall_offset,
                                                      "max_seconds": self.cfg.max_seconds})
            progress = ProgressCsv(log_dir / "progress.csv", keep_through_iteration=self.start_iteration - 1)
        else:
            if log_dir.exists():
                raise FileExistsError(f"{log_dir} exists; choose another run name or resume it")
            log_dir.mkdir(parents=True)
            run_json = {
                "stage": self.stage,
                "runner": asdict(self.cfg),
                **meta,
                "wandb_id": uuid.uuid4().hex[:8],
            }
            progress = ProgressCsv(log_dir / "progress.csv")
        atomic_write_text(log_dir / "run.json", json.dumps(run_json, indent=2, default=str))
        wandb = optional_wandb()
        if wandb is not None:
            wandb.init(
                project=self.cfg.wandb_project,
                name=self.cfg.run_name,
                id=run_json.setdefault("wandb_id", uuid.uuid4().hex[:8]),
                resume="allow",
                dir=str(log_dir),
                config={k: v for k, v in run_json.items() if k != "resumed"},
                mode=os.environ.get("WANDB_MODE", "offline"),
            )
        self.env.reset()
        self._seed_sim_history()
        self._on_sim_reset()
        wm = self.cfg.wm
        should_train = Ratio(self.env.num_envs / wm.num_batches / wm.batch_size / wm.horizon * self.cfg.replay_ratio)
        first_update = self.start_iteration + (self.cfg.resume_warmup if resume else 0)
        run_start = time.time() - self._wall_offset
        segment_start = time.time()
        last_checkpoint = time.time()
        t0 = time.time()
        wm_time = policy_time = 0.0
        collected = updates = 0
        loss_dict: dict[str, torch.Tensor] = {}
        it = self.start_iteration
        for it in range(self.start_iteration, self.cfg.max_iterations):
            collected += self.step()
            for _ in range(should_train(it)):
                if self.replay_buffer.num_transitions < wm.horizon or it < first_update:
                    continue
                data = (self.replay_buffer.sample(wm.horizon, wm.batch_size) for _ in range(wm.num_batches))
                t = time.time()
                loss_dict.update(self.world_model_learner.update(data))
                wm_time += time.time() - t
                t = time.time()
                loss_dict.update(self.learner.update())
                policy_time += time.time() - t
                updates += 1
            wall = time.time() - run_start
            out_of_time = self.cfg.max_seconds > 0 and wall > self.cfg.max_seconds
            if it % self.cfg.log_interval == 0 or out_of_time:
                elapsed = time.time() - t0
                extras = self.logger._episode_extras()
                losses = {k: float(v) for k, v in loss_dict.items()}
                row = {
                    "iteration": it,
                    "wall_s": wall,
                    "interval_s": elapsed,
                    "s_per_iteration": elapsed / max(self.cfg.log_interval if it > self.start_iteration else 1, 1),
                    "env_steps_per_s": collected / max(elapsed, 1e-6),
                    "collect_s": elapsed - wm_time - policy_time,
                    "wm_s": wm_time,
                    "policy_s": policy_time,
                    "updates": updates,
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
                    "mean_episode_length": sum(self.logger.lenbuffer) / max(len(self.logger.lenbuffer), 1),
                    **losses,
                    **self._sim_metrics(),
                    **{k: v for k, v in extras.items() if k.startswith(("Episode_Termination", "Metrics/motion/error"))},
                }
                self.logger.log(
                    it=it // self.cfg.log_interval,
                    total_it=self.cfg.max_iterations // self.cfg.log_interval,
                    collect_time=row["collect_s"],
                    wm_learn_time=wm_time,
                    policy_learn_time=policy_time,
                    loss_dict=loss_dict,
                    num_collected_steps=collected,
                    elapsed_s=wall,
                    eta_s=eta_seconds(it, self.start_iteration, self.cfg.max_iterations, time.time() - segment_start,
                                      wall, self.cfg.max_seconds),
                )
                progress.append(row)
                print(
                    " ".join(
                        f"{k} {row[k]:.4g}" if isinstance(row.get(k), float) else f"{k} {row.get(k, '-')}"
                        for k in ("iteration", "wall_s", "s_per_iteration", "peak_gpu_gb", "Loss/state",
                                  "Metrics/motion/error_body_pos")
                    ),
                    flush=True,
                )
                t0 = time.time()
                wm_time = policy_time = 0.0
                collected = updates = 0
            by_iteration = it > self.start_iteration and it % self.cfg.save_interval == 0
            by_clock = time.time() - last_checkpoint > self.cfg.checkpoint_interval_s
            if by_iteration or by_clock or out_of_time:
                self.save(log_dir, it, wall, periodic=not out_of_time)
                last_checkpoint = time.time()
            if out_of_time:
                break
        else:
            self.save(log_dir, it, time.time() - run_start)
        self._cleanup()
        return log_dir


class ImitationRunner(TwoAgentRunner):
    """Stage 1: both agents track the clips; one world model for the whole body."""

    stage = "imitation"

    def __init__(
        self,
        env,
        cfg: TwoAgentRunnerCfg,
        rewards: ImitationRewardCfg | None = None,
        run_dir: Path | None = None,
        motion_files: list[str] | None = None,
        task_id: str = "imitation",
        registry_name: str | None = None,
        device: str = "cuda:0",
        sim_context_layout: list[dict] | None = None,
        sim_context_fine_tune: bool = False,
    ):
        """Hold the reward cfg, the run directory and the motion files (recorded in ``run.json``), then build
        everything. ``env`` is the env wrapped in ``EnvAdapter``; ``sim_context_layout`` is the sim-context layout of
        the run a fine-tune or resume starts from, ``sim_context_fine_tune`` whether it is a fine-tune's
        (``SimContext.from_env``)."""
        self.sim_context_layout = sim_context_layout
        self.sim_context_fine_tune = sim_context_fine_tune
        self.reward_cfg = rewards or ImitationRewardCfg()
        self.run_dir = None if run_dir is None else Path(run_dir)
        self.motion_files = list(motion_files or [])
        super().__init__(env, cfg, task_id, registry_name, device)

    def _run_meta(self) -> dict:
        """Return the motion files (repository-relative), the env count and the reward cfg, as train_imitation's."""
        return {
            "motion_files": [to_repo_path(f) for f in self.motion_files],
            "num_envs": self.env.num_envs,
            "rewards": asdict(self.reward_cfg),
        }

    def _goals(self, motion_times, aux_state):
        """Return both agents' stage-1 goals from the clip."""
        return {
            "legs": {"goal_motion": legs_goal(self.motion, motion_times, aux_state)},
            "arms": {"goal_motion": arms_goal(self.tables, motion_times)},
        }

    def _init_policy(self) -> None:
        """Build both actors, both critics and the stage-1 world-model env."""
        self._make_tables()
        dims = imitation_goal_dims(self)
        self.actors = {
            a: build_imitation_actor(
                self.obs_dim,
                ACTION_DIMS[a],
                dims[a],
                self.cfg.latent_dim,
                self.cfg.actor_hidden_dims,
                self.cfg.history_length,
            ).to(self.device)
            for a in AGENTS
        }
        ee = ee_window(self.tables, torch.zeros(1, dtype=torch.long, device=self.device)).shape[-1]
        self._build_critics(self._representation.rep_state_dim + self._num_goals + ee + self._sim_context.output_dim)
        wm_env = ImitationWorldModelEnv(
            dynamics=self._dynamics,
            representation=self._representation,
            env=self.env,
            batch_size=self.cfg.policy.batch_size,
            replay_buffer=self.replay_buffer,
            tables=self.tables,
            history_length=self.cfg.history_length,
            termination_softness=self.cfg.termination_softness,
            max_step_reward=max_step_rewards(self.reward_cfg, self.env.unwrapped.step_dt),
            joint_limits=self.joint_limits,
            rewards=self.reward_cfg,
            device=self.device,
        )
        self._finish_policy(wm_env)


def widen_sim_context(state: dict, sim_context) -> None:
    """Fit a ``training_state`` saved with another sim context (a fine-tune's source: fewer columns, or narrower
    normalisation) to ``sim_context``, in place: the world model's and each critic's first layer by
    ``widen_input_weight`` (the sim context is the last block of both inputs), and the saved normalisation replaced by
    ``sim_context``'s. The loaded function is the source's for the same raw values. A no-op for the same context."""
    saved = state["world_model"]["sim_context"]
    mean, scale = sim_context.mean, sim_context.scale
    if saved["mean"].shape == mean.shape and torch.equal(saved["scale"].to(scale), scale) \
            and torch.equal(saved["mean"].to(mean), mean):
        return
    args = (saved["mean"], saved["scale"], mean, scale)
    dyn = state["world_model"]["dynamics"]
    dyn["mlp.0.weight"] = widen_input_weight(dyn["mlp.0.weight"], *args)
    for critic in state["critics"].values():
        critic["0.weight"] = widen_input_weight(critic["0.weight"], *args)
    state["world_model"]["sim_context"] = {"mean": mean.clone(), "scale": scale.clone()}


def _pad_optimizer_state(opt: torch.optim.Optimizer) -> None:
    """Zero-pad, at the end of each dimension, any per-parameter state (moments, momentum) smaller than its now
    wider parameter (``widen_sim_context``'s first layers): the appended columns start without history."""
    for p, st in opt.state.items():
        for k, v in st.items():
            if torch.is_tensor(v) and v.ndim == p.ndim and v.shape != p.shape:
                pad = v.new_zeros(p.shape)
                pad[tuple(slice(0, n) for n in v.shape)] = v
                st[k] = pad


def imitation_goal_dims(runner: TwoAgentRunner) -> dict[str, int]:
    """Return each agent's stage-1 goal width, measured on frame 0."""
    t = torch.zeros(1, dtype=torch.long, device=runner.device)
    aux = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]], device=runner.device)
    return {
        "legs": legs_goal(runner.motion, t, aux).shape[-1],
        "arms": arms_goal(runner.tables, t).shape[-1],
    }


def resume_info(run_dir: Path) -> tuple[dict, int]:
    """Return a run's ``run.json`` and the first iteration a resume of it runs."""
    run_dir = Path(run_dir)
    return json.loads((run_dir / "run.json").read_text()), read_checkpoint_info(run_dir)["iteration"] + 1


def load_actors(run_dir: Path, device) -> dict[str, nn.Module]:
    """Return a stage-1 run's actors (``actors.pt``, from this package or h12_train) on ``device``, in eval mode.

    The actors are built to the policy contract (``PROPRIO_DIM`` proprio frame, ``IMITATION_GOAL_DIMS`` goals) with
    the latent width, hidden sizes and history length of the run's ``run.json`` (the defaults without one). A state
    dict whose keys or shapes differ is refused, naming the saved and the expected shape.
    """
    run_dir = Path(run_dir)
    run_json = run_dir / "run.json"
    cfg = (
        config_from_dict(TwoAgentRunnerCfg, json.loads(run_json.read_text())["runner"])
        if run_json.exists()
        else TwoAgentRunnerCfg()
    )
    path = run_dir / "actors.pt"
    state = torch.load(path, map_location=device, weights_only=False)
    actors = {}
    for a in AGENTS:
        actor = build_imitation_actor(
            PROPRIO_DIM, ACTION_DIMS[a], IMITATION_GOAL_DIMS[a], cfg.latent_dim, cfg.actor_hidden_dims,
            cfg.history_length,
        ).to(device)
        expected, saved = actor.state_dict(), state[a]
        for k, v in expected.items():
            if k in saved and tuple(saved[k].shape) != tuple(v.shape):
                raise ValueError(
                    f"{path}: {a} {k} has shape {tuple(saved[k].shape)}, the policy contract with {run_json.name}'s "
                    f"settings expects {tuple(v.shape)}"
                )
        if expected.keys() != saved.keys():
            raise ValueError(
                f"{path}: {a} has keys {sorted(saved.keys() - expected.keys())} this build does not and lacks "
                f"{sorted(expected.keys() - saved.keys())}"
            )
        actor.load_state_dict(saved)
        actors[a] = actor.eval()
    return actors
