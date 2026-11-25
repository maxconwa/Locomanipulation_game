# Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c), src/h12_train/marl/runner.py
# (CommandRunner, lines 639-869) with its world-model env (marl/wm_env.py CommandWorldModelEnv) and goals
# (marl/goals.py CommandProcess). Changes: the data source is main's LocoManip-Marl-Pulse-v0 (marl_source.py), not
# the stage-1 track env, so the runner is not a TwoAgentRunner: it builds the world model from the stage-1 run's
# world_model.pt and the plant (no env), keeps its own replay buffer schema, and runs its own BPTT update; the command
# encoders are recurrent (command_actor.py) and read main's actor observation (main_obs.py); the reward is main's
# tracking terms plus stage 1's e-stop penalties (main_rewards.py), the command law main's (command_law.py); clip
# segments only in imagination (the sim runs main's own commands); plain Adam on the encoders and critics (no Muon);
# the policy update is not CUDA-graph captured (the observation model draws its noise from a generator).
"""Stage 2 through the world model (arm C): command encoders into the frozen stage-1 decoders, trained by gradients.

``CommandRunner(source, stage1_run, cfg)``:

- refuses a stage-1 run whose plant (``run.json["plant"]``) is not the source's, naming both, before loading anything
  (a run decoded on another plant fell in 23% of zero-policy episodes), and a world model whose action scale or
  offset is not that plant's;
- loads and freezes the stage-1 decoders and encoders with their normalisers (``runner.load_actors``); loads the
  stage-1 world model (dynamics and sim-context normalisation, ``world_model.pt``) and keeps training it, as stage 1
  did, on the data collected here (main's plant, randomisation, delay and sensing), frozen inside the encoders'
  update;
- collects ``rollout`` sim steps per iteration with both encoders acting (``steps_per_env`` counts them), storing
  per env and step the full state, the decoders' raw action and the one before, each encoder's LSTM state, the
  commands, the gait clock and the last latents;
- per iteration, ``updates`` times: fits the world model on replayed ``wm.horizon``-step windows (stage 1's
  ``WorldModelLearner``), then imagines ``policy.horizon``-step rollouts from ``policy.batch_size`` buffered states
  and steps each encoder on its own return: ``main_rewards`` (main's velocity and wrist tracking on the predicted
  state, plus the soft e-stop penalties), discounted, survival-masked by a soft fall probability (main's
  ``pelvis_contact_or_tilt``: roll 0.8, pitch 1.0 rad, and a pelvis height for the contact), and bootstrapped by
  the agent's critic; minus the latent-BC term ``prior_weight * mean((z - z_stage1)^2)`` on clip-segment rows,
  ``z_stage1`` the frozen stage-1 encoder's latent of the clip's stage-1 goal at the row's clip frame;
- the imagined commands follow ``CommandProcess`` (main's laws, independent velocity and arm segments of mean
  ``mean_hold_s``); every rollout starts a segment of each command, a clip segment with ``clip_command_prob`` (its
  command the clip's own at a random frame, advanced a frame per step), else the command the env held there.

What an imagined step does (the sim's order): main's actor observation from the predicted state (``MainActorObs``,
the encoders' applied latents detached: the action's own feedback through ``atanh`` would scale gradients by up to
1/(1 - z^2)); each encoder's ``z``; both decoders on one proprio frame (the torso IMU the world model synthesises, the
joints, and both decoders' raw outputs of the step before), into the 27 raw actions; the dynamics' next state; the
rewards from the state pair and the gait-cycle mean velocity (the cycle window starts empty at the rollout start).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from tensordict import TensorDict

from pretraining.constants import JOINT_NAMES, TRACKED_BODIES
from pretraining.kinematics import URDF, ForwardKinematics, UrdfTree
from pretraining.marl.command_actor import RecurrentCommandActor, latent_bc, latent_to_action, stage1_latent
from pretraining.marl.command_law import CommandProcess
from pretraining.marl.config import AGENTS, EstopPenaltyCfg
from pretraining.marl.main_obs import MainActorObs
from pretraining.marl.main_rewards import (
    PELVIS,
    WRISTS,
    MainRewardCfg,
    WMState,
    cycle_init,
    cycle_push,
    cycle_window,
    main_rewards,
    wrist_errors,
)
from pretraining.marl.representation import ClampedDynamics, TorsoImuGenCoordRepresentation
from pretraining.marl.runner import load_actors
from pretraining.marl.runs import (
    ProgressCsv,
    atomic_save,
    atomic_write_text,
    from_repo_path,
    keep_last,
    to_repo_path,
)
from pretraining.marl.wm_env import ClipTables, arms_goal, legs_goal
from pretraining.math import quat_apply_inverse, quat_conjugate, quat_mul
from pretraining.plants import Plant, get_plant
from pretraining.supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg, gae_returns
from pretraining.supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from pretraining.supermimic.models import build_mlp
from pretraining.supermimic.replay_buffer import FieldSpec, ReplayBuffer
from pretraining.supermimic.runners.base_runner import WorldModel
from pretraining.supermimic.sim_context import SimContext
from pretraining.supermimic.utils.gencoord_helpers import decompose_rep_state

COMMAND_DIMS = {"legs": 3, "arms": 14}
"""Each encoder's command: the legs' velocity command (vx, vy, wz), the arms' two pelvis-frame wrist goals (xyz, xyzw
each, as main's ``arm_targets_in_root_xyzw``)."""

# The stage-1 track env's sim-context specs in their column order (SimContext.from_env) and widths; a stage-1 run's
# world_model.pt holds the normalisation of the ones its env had.
_SIM_CONTEXT_SPECS = (("foot_friction", 1), ("base_com", 3), ("encoder_bias", len(JOINT_NAMES)))


@dataclass
class CommandRunnerCfg:
    """Stage-2 settings (``configs/commands.yaml``)."""

    rollout: int = 24
    """Sim steps per env per iteration."""
    replay_ratio: float = 8.0
    """World-model samples (num_batches x batch_size x horizon) per collected sample: the updates per iteration are
    ``max(1, round(rollout x num_envs x replay_ratio / (wm.num_batches x wm.batch_size x wm.horizon)))``."""
    buffer_size: int = 100
    """Steps per env the replay buffer holds."""
    wm: WorldModelLearnerCfg = field(default_factory=lambda: WorldModelLearnerCfg(batch_size=1024, horizon=16))
    policy: ActorCriticPolicyLearnerCfg = field(default_factory=lambda: ActorCriticPolicyLearnerCfg(
        batch_size=128, horizon=64, gamma=0.995, lam=0.975))
    lstm: int = 64
    command_hidden_dims: tuple[int, ...] = (512, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    clip_command_prob: float = 0.25
    prior_weight: float = 0.1
    mean_hold_s: float = 6.0
    max_clips: int = 2000
    """Clips (of the stage-1 run's, drawn with ``seed``) held for clip segments; 0: all. All of a 13,578-clip set at
    100 Hz with its tables would take about 4 GB of the GPU, for a quarter of the segments' regulariser."""
    termination_softness: float = 0.05
    fall_height: float = 0.5
    """m: a predicted pelvis below this counts as a fall (main's pelvis-contact term, which the world model cannot
    see)."""
    max_roll: float = 0.8  # rad, main's MarlTerminationsCfg.fell
    max_pitch: float = 1.0
    estop: EstopPenaltyCfg = field(default_factory=EstopPenaltyCfg)
    max_iterations: int = 1_000_000
    max_steps: int = 40032
    """Sim steps per env to stop at (counted over resumed segments); 0: no cap."""
    max_seconds: float = 0.0
    """Wall time (s, over resumed segments) to stop at; 0: no cap."""
    log_interval: int = 1
    save_interval: int = 100  # iterations between periodic checkpoints
    checkpoint_interval_s: float = 1800.0
    keep_checkpoints: int = 5
    seed: int = 0


def _joint_noise():
    """Main's joint_noise module (loaded by path in main_obs: the task package's __init__ imports Isaac Lab)."""
    from pretraining.marl import main_obs

    return main_obs._joint_noise


class _Spec:
    """A sim-context spec stand-in: a name and a width (the values come from the source, not an env)."""

    def __init__(self, name: str, dim: int):
        self.name, self.dim = name, dim


def _sim_context(bundle: dict, device) -> SimContext:
    """The stage-1 world model's sim-context normalisation, its specs inferred from its width."""
    width = bundle["sim_context"]["mean"].numel()
    for mask in range(1 << len(_SIM_CONTEXT_SPECS)):
        specs = [s for i, s in enumerate(_SIM_CONTEXT_SPECS) if mask >> i & 1]
        if sum(d for _, d in specs) == width:
            ctx = SimContext([_Spec(n, d) for n, d in specs], [(0.0, 0.0)] * width)
            ctx.load_state_dict(bundle["sim_context"])
            return ctx.to(device)
    raise ValueError(f"the world model's sim context is {width} wide; no combination of {_SIM_CONTEXT_SPECS} is")


def build_world_model(bundle: dict, plant: Plant, hidden_dims, target_low, target_high, device) -> WorldModel:
    """Stage 1's torso-IMU gencoord world model for ``plant``, from a ``world_model.pt`` bundle, without an env
    (``representation.build_world_model`` reads the same from the track env). Refuses a bundle whose action scale or
    offset is not the plant's (the world model of another plant's run)."""
    n = len(JOINT_NAMES)
    default = torch.tensor([plant.default_joint_pos[j] for j in JOINT_NAMES], dtype=torch.float32, device=device)
    scale = torch.tensor([plant.action_scale[j] for j in JOINT_NAMES], dtype=torch.float32, device=device)
    saved_scale = bundle["dynamics"]["action_scale"].to(device).reshape(-1)
    saved_offset = bundle["dynamics"]["action_offset"].to(device).reshape(-1)
    if not (torch.allclose(saved_scale, scale, atol=1e-5) and torch.allclose(saved_offset, default, atol=1e-4)):
        raise ValueError(f"the world model's action scale or offset is not the {plant.name} plant's: it was trained "
                         "on another plant")
    ctx = _sim_context(bundle, device)
    rep = TorsoImuGenCoordRepresentation(n, default)
    rep.actor_obs_mode = "proprio"
    rep.register_buffer("_proprio_noise_bounds", rep.proprio_noise_bounds(n, device), persistent=False)
    rep.dt = plant.step_dt
    rep.substep_coeff = (plant.decimation + 1) / (2 * plant.decimation)
    rep = rep.to(device)
    rep.fk = ForwardKinematics(UrdfTree.from_file(URDF, JOINT_NAMES), TRACKED_BODIES, str(device))
    columns = ctx.find("encoder_bias")
    rep._encoder_bias = None if columns is None else (columns, ctx)
    dynamics = ClampedDynamics(rep, n, ctx.output_dim, bundle["dynamics"]["action_scale"].to(device),
                               bundle["dynamics"]["action_offset"].to(device), hidden_dims=tuple(hidden_dims),
                               target_low=target_low.to(device), target_high=target_high.to(device)).to(device)
    dynamics.load_state_dict(bundle["dynamics"])
    return WorldModel(representation=rep, dynamics=dynamics, sim_context=ctx, num_goals=0, num_actions=n)


def check_stage1_plant(stage1_run, source_plant: Plant) -> tuple[dict, Plant]:
    """The stage-1 run's ``run.json`` and plant; ValueError naming the run, its plant and the env's if they differ."""
    run_json = Path(stage1_run) / "run.json"
    meta = json.loads(run_json.read_text()) if run_json.exists() else {}
    plant = get_plant(meta.get("plant", "pretrain"))
    if plant.name != source_plant.name:
        raise ValueError(f"stage-1 run {stage1_run} was trained on the {plant.name} plant; the env is on the "
                         f"{source_plant.name} plant: its decoders and world model would run on another robot")
    return meta, plant


def command_vectors(commands: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Each encoder's command input (``COMMAND_DIMS``): the velocity command; both wrist goals, xyz + xyzw."""
    arm = commands["arm"]
    xyzw = torch.cat([arm[..., 4:], arm[..., 3:4]], dim=-1)
    return {"legs": commands["velocity"], "arms": torch.cat([arm[..., :3], xyzw], dim=-1).flatten(1)}


class ClipCommands:
    """The clips as a command source for clip segments: per frame the pelvis-frame velocity command, the wrist goals
    (``*_wrist_yaw_link`` in the pelvis frame, main's goal frame, no grasp offset) and both stage-1 goals."""

    _CHUNK = 4096

    def __init__(self, motion, velocity_ranges: torch.Tensor | None = None):
        self.motion = motion
        self.tables = ClipTables(motion)
        vel, arm = [], []
        for start in range(0, motion.time_step_total, self._CHUNK):
            f = torch.arange(start, min(start + self._CHUNK, motion.time_step_total), device=motion.joint_pos.device)
            pos, quat, _, _ = motion.body_state(f)
            q_root = motion.root_quat_w[f]
            v = quat_apply_inverse(q_root, motion.root_lin_vel_w[f])
            w = quat_apply_inverse(q_root, motion.root_ang_vel_w[f])
            vel.append(torch.cat([v[:, :2], w[:, 2:3]], dim=-1))
            qp = quat[:, PELVIS : PELVIS + 1].expand(-1, 2, 4)
            p = quat_apply_inverse(qp, pos[:, list(WRISTS)] - pos[:, PELVIS : PELVIS + 1])
            q = quat_mul(quat_conjugate(qp), quat[:, list(WRISTS)])
            arm.append(torch.cat([p, torch.where(q[..., :1] < 0, -q, q)], dim=-1))
        self.velocity = torch.cat(vel)
        if velocity_ranges is not None:  # h12_train clamps the clip's velocity commands to the law's ranges
            r = velocity_ranges.to(self.velocity)
            self.velocity = torch.maximum(torch.minimum(self.velocity, r[:, 1]), r[:, 0])
        self.arm = torch.cat(arm)
        self.frame_end = torch.zeros(motion.time_step_total, dtype=torch.long, device=self.velocity.device)
        for s, e in zip(motion.motion_starts.tolist(), motion.motion_ends.tolist()):
            self.frame_end[s:e] = e - 1

    def draw(self, n: int, gen: torch.Generator) -> torch.Tensor:
        """``n`` frames uniform over every clip frame."""
        u = torch.rand(n, generator=gen, device=gen.device).to(self.velocity.device)
        return (u * self.motion.time_step_total).long().clamp(max=self.motion.time_step_total - 1)

    def advance(self, frames: torch.Tensor) -> torch.Tensor:
        """One frame on, held at its clip's last frame."""
        return torch.minimum(frames + 1, self.frame_end[frames])


class CommandRunner:
    """Stage 2 through the world model (module docstring)."""

    stage = "commands"

    def __init__(self, source, stage1_run, cfg: CommandRunnerCfg, run_dir: Path | None = None):
        self.source, self.cfg = source, cfg
        self.stage1_run = Path(stage1_run)
        self.run_dir = None if run_dir is None else Path(run_dir)
        # first: a run of another plant is refused before anything is loaded
        self.stage1_meta, self.plant = check_stage1_plant(self.stage1_run, source.plant)
        if abs(self.plant.step_dt - source.step_dt) > 1e-9:
            raise ValueError(f"stage-1 run {self.stage1_run}: the {self.plant.name} plant's policy step is "
                             f"{self.plant.step_dt} s, the env's {source.step_dt} s")
        self.device = torch.device(source.device)
        self.num_envs, self.dt = source.num_envs, source.step_dt
        dev = self.device
        self.gen = torch.Generator(device=dev).manual_seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # -- the frozen stage-1 decoders and encoders
        self.stage1 = load_actors(self.stage1_run, dev)
        for actor in self.stage1.values():
            actor.eval().requires_grad_(False)
        latent_dim = self.stage1[AGENTS[0]].latent_dim

        # -- the world model, trained on as stage 1 did
        runner_meta = self.stage1_meta.get("runner", {})
        hidden = runner_meta.get("dynamics_hidden_dims", (2048, 2048, 2048))
        bundle = torch.load(self.stage1_run / "world_model.pt", map_location=dev, weights_only=False)
        self.wm = build_world_model(bundle, self.plant, hidden, source.target_low, source.target_high, dev)
        self.rep, self.dynamics, self.sim_context = self.wm.representation, self.wm.dynamics, self.wm.sim_context
        self.wm_learner = WorldModelLearner(self.dynamics, self.rep, cfg.wm, device=str(dev))
        if dev.type != "cuda":  # torch.compile is for the GPU (and fails on some CPU toolchains)
            self.wm_learner._compiled_rollout = self.wm_learner._rollout
            self.wm_learner._compiled_compute_losses = self.wm_learner._compute_losses
            self.wm_learner._compiled_represent = self.rep.represent

        # -- the encoders, critics and their optimisers
        self.encoders = {
            a: RecurrentCommandActor(source.obs_dims[a], COMMAND_DIMS[a], latent_dim, cfg.lstm,
                                     tuple(cfg.command_hidden_dims)).to(dev) for a in AGENTS}
        ctx_dim = self.sim_context.output_dim
        self.critic_dim = self.rep.rep_state_dim + 3 + 3 + 14 + ctx_dim + 2
        self.critics = {a: build_mlp(self.critic_dim, tuple(cfg.critic_hidden_dims), output_dim=1).to(dev)
                        for a in AGENTS}
        p = cfg.policy
        self.encoder_opt = {a: torch.optim.Adam(self.encoders[a].parameters(), lr=p.lr, betas=(0.9, p.beta2))
                            for a in AGENTS}
        self.critic_opt = {a: torch.optim.Adam(self.critics[a].parameters(), lr=p.lr) for a in AGENTS}

        # -- main's task: observation model, reward terms, command law
        self.obs_model = MainActorObs(source.env_cfg, dev)
        widths = self.obs_model.widths()
        if widths != {a: source.obs_dims[a] for a in widths}:
            raise ValueError(f"main's observation model gives {widths}, the env {source.obs_dims}")
        jn = _joint_noise()
        self._vel_floor = torch.tensor(jn.per_joint(list(JOINT_NAMES), jn.JOINT_VEL_NOISE_FLOOR,
                                                    jn.JOINT_VEL_NOISE_FLOOR_DEFAULT), device=dev)
        self.gait_window = cycle_window(source.env_cfg.gait_period, self.dt)
        self._refresh_rewards()
        self.process = CommandProcess(self.dt, cfg.mean_hold_s, cfg.clip_command_prob, source.velocity_law,
                                      source.arm_law)
        self.clips = self._load_clips() if cfg.clip_command_prob > 0.0 else None

        # -- the replay buffer
        n = len(JOINT_NAMES)
        h = 2 * cfg.lstm
        schema = {
            "state": FieldSpec(self.rep.full_state_dim), "action": FieldSpec(n), "prev_action": FieldSpec(n),
            "sim_ctx": FieldSpec(ctx_dim), "hc_legs": FieldSpec(h), "hc_arms": FieldSpec(h),
            "latent_legs": FieldSpec(latent_dim), "latent_arms": FieldSpec(latent_dim),
            "velocity": FieldSpec(3), "velocity_target": FieldSpec(3), "arm": FieldSpec(14), "clock": FieldSpec(),
            "terminated": FieldSpec(dtype=torch.bool), "truncated": FieldSpec(dtype=torch.bool),
        }
        self.buffer = ReplayBuffer(schema, cfg.buffer_size, self.num_envs, device=str(dev))
        per_update = cfg.wm.num_batches * cfg.wm.batch_size * cfg.wm.horizon
        self.updates_per_iteration = max(1, round(cfg.rollout * self.num_envs * cfg.replay_ratio / per_update))

        self.iteration = 0
        self.steps_per_env = 0
        self._wall_offset = 0.0
        self._sim: dict | None = None
        self._stats: dict[str, float] = {}

    # -- set-up -----------------------------------------------------------------------------------------------------

    def _refresh_rewards(self) -> None:
        """Main's reward terms at the env's current kernel stages (its curricula move them)."""
        stages = self.source.kernel_stages() if hasattr(self.source, "kernel_stages") else {}
        self.reward_cfg = MainRewardCfg.from_env_cfg(self.source.env_cfg, estop=self.cfg.estop, plant=self.plant,
                                                     **stages)

    def _load_clips(self) -> ClipCommands:
        from pretraining.motions import MotionLoader

        files = [from_repo_path(f) for f in self.stage1_meta.get("motion_files", [])]
        if 0 < self.cfg.max_clips < len(files):
            pick = torch.randperm(len(files), generator=torch.Generator().manual_seed(self.cfg.seed))
            files = [files[i] for i in sorted(pick[: self.cfg.max_clips].tolist())]
        missing = [f for f in files if not f.exists()]
        if not files or missing:
            raise FileNotFoundError(
                f"clip segments (clip_command_prob {self.cfg.clip_command_prob}) need the stage-1 run's clips: "
                f"{len(missing)} of {len(files)} motion files in {self.stage1_run}/run.json are missing")
        motion = MotionLoader(files, self.rep.fk, str(self.device), fps=round(1.0 / self.dt))
        law = self.source.velocity_law
        return ClipCommands(motion, None if law is None else law.ranges)

    # -- collection -------------------------------------------------------------------------------------------------

    def _start_sim(self) -> None:
        state, obs, commands = self.source.reset()
        n, dev = self.num_envs, self.device
        self._sim = {
            "state": state, "obs": obs, "commands": commands,
            "hc": {a: self.encoders[a].initial_state(n, dev) for a in AGENTS},
            "latent": {a: torch.zeros(n, self.encoders[a].latent_dim, device=dev) for a in AGENTS},
            "prev_action": torch.zeros(n, len(JOINT_NAMES), device=dev),
            "clock": self.source.clock.clone(),
        }

    @torch.no_grad()
    def collect(self) -> dict[str, float]:
        """Step every env ``rollout`` times with both encoders acting; store each transition."""
        if self._sim is None:
            self._start_sim()
        s = self._sim
        ctx = self.source.sim_ctx(self.sim_context.output_dim)
        sums = {"vel_xy_error": 0.0, "wrist_pos_error": 0.0, "done": 0.0, "reward_legs": 0.0, "reward_arms": 0.0}
        for _ in range(self.cfg.rollout):
            cmd = command_vectors(s["commands"])
            z, hc_next = {}, {}
            for a in AGENTS:
                self.encoders[a].update_normalization(s["obs"][a], cmd[a])
                z[a], hc_next[a] = self.encoders[a](s["obs"][a], cmd[a], s["hc"][a])
            state, obs, commands, dones = self.source.step(z)
            truncated = self.source.time_outs.bool() & dones
            self.buffer.insert(TensorDict({
                "state": s["state"], "action": self.source.last_action, "prev_action": s["prev_action"],
                "sim_ctx": ctx, "hc_legs": s["hc"]["legs"].flatten(1), "hc_arms": s["hc"]["arms"].flatten(1),
                "latent_legs": s["latent"]["legs"], "latent_arms": s["latent"]["arms"],
                "velocity": s["commands"]["velocity"], "velocity_target": s["commands"]["velocity_target"],
                "arm": s["commands"]["arm"].flatten(1), "clock": s["clock"],
                "terminated": dones & ~truncated, "truncated": truncated,
            }, batch_size=[self.num_envs]))
            self.wm_learner.update_normalization(s["state"], self.source.last_action)
            sums["vel_xy_error"] += self._tracking(state, s["commands"], sums)
            sums["done"] += float(dones.float().mean())
            rewards = getattr(self.source, "rewards", None) or {}
            for a in AGENTS:
                if a in rewards:
                    sums[f"reward_{a}"] += float(rewards[a].float().mean())
            keep = (~dones).float().view(-1, 1)
            s["state"], s["obs"], s["commands"] = state, obs, commands
            s["hc"] = {a: hc_next[a] * keep.unsqueeze(-1) for a in AGENTS}
            s["latent"] = {a: latent_to_action(z[a]) * keep for a in AGENTS}
            s["prev_action"] = self.source.last_action * keep
            s["clock"] = self.source.clock.clone()
            self.steps_per_env += 1
        return {f"Sim/{k}": v / self.cfg.rollout for k, v in sums.items()}

    def _tracking(self, state: torch.Tensor, commands: dict, sums: dict) -> float:
        """The step's sim tracking errors (logged): pelvis-frame (vx, vy) to the velocity command it was taken under,
        wrist position to its goal."""
        n = len(JOINT_NAMES)
        quat = state[:, 3:7]
        v_b = quat_apply_inverse(quat, state[:, 7 + n : 10 + n])
        w_b = quat_apply_inverse(quat, state[:, 10 + n : 13 + n])
        ws = WMState.from_root(self.rep.fk, state[:, :3], quat, v_b, w_b, state[:, 7 : 7 + n], state[:, 13 + n :])
        pos_err, _ = wrist_errors(ws, commands["arm"])
        sums["wrist_pos_error"] += float(pos_err.mean())
        return float((commands["velocity"][:, :2] - v_b[:, :2]).norm(dim=-1).mean())

    # -- the world model's update -----------------------------------------------------------------------------------

    def update_world_model(self) -> dict[str, torch.Tensor]:
        wm = self.cfg.wm
        data = (self.buffer.sample(wm.horizon, wm.batch_size) for _ in range(wm.num_batches))
        return self.wm_learner.update(data)

    # -- the encoders' update ---------------------------------------------------------------------------------------

    def _soft_fall(self, rep_state: torch.Tensor) -> torch.Tensor:
        """Main's fall (pelvis contact or tilt) as a probability on the predicted state."""
        z, g_b, _, _, _, _ = decompose_rep_state(rep_state)
        roll = torch.atan2(-g_b[:, 1], -g_b[:, 2]).abs()
        pitch = torch.asin(g_b[:, 0].clamp(-1.0, 1.0)).abs()
        s = self.cfg.termination_softness
        alive = ((1 - torch.sigmoid((roll - self.cfg.max_roll) / s))
                 * (1 - torch.sigmoid((pitch - self.cfg.max_pitch) / s))
                 * (1 - torch.sigmoid((self.cfg.fall_height - z[:, 0]) / s)))
        return 1.0 - alive

    def decoder_sensors(self, sensors: torch.Tensor) -> torch.Tensor:
        """The decoders' measured joints as the env gives them (pulse_actions.py:96-102): compute_wm_obs adds no
        noise (its _proprio_noise_bounds are only read by add_proprio_noise), so add the rollout's encoder offsets
        (MainActorObs.reset's draw from main's ranges, the same the actor observation reads), main's uniform position
        noise and its speed-scaled velocity noise. Columns: gravity 3, gyro 3, q - default 27, dq 27."""
        jn = _joint_noise()
        n = len(JOINT_NAMES)
        q, dq = sensors[:, 6 : 6 + n], sensors[:, 6 + n : 6 + 2 * n]
        g = self.gen
        u = torch.rand(q.shape, generator=g, device=g.device).to(q)
        q = q + self.obs_model.encoder_offset + (2.0 * u - 1.0) * jn.JOINT_POS_NOISE
        noise = torch.randn(dq.shape, generator=g, device=g.device).to(dq)
        dq = dq + noise * jn.joint_vel_noise_std(dq.detach(), self._vel_floor.to(dq))
        return torch.cat([sensors[:, :6], q, dq], dim=-1)

    def _decode(self, rep: torch.Tensor, aux: torch.Tensor, sim_ctx: torch.Tensor, prev_action: torch.Tensor,
                z: dict[str, torch.Tensor]) -> torch.Tensor:
        """Both frozen decoders on one proprio frame, into the 27 raw actions (pulse.PulseDecoders.step's order)."""
        sensors = self.decoder_sensors(self.rep.compute_wm_obs(rep, aux, self.rep.joint_state(rep), sim_ctx))
        frame = torch.cat([sensors, prev_action], dim=-1).unsqueeze(1)
        out = []
        for a in AGENTS:
            actor = self.stage1[a]
            hist = actor.obs_normalizer(frame).flatten(1)
            out.append(actor.decoder(torch.cat([z[a], hist], dim=-1)))
        return torch.cat(out, dim=-1)

    def _critic_obs(self, rep, commands, sim_ctx, clip_v, clip_a) -> torch.Tensor:
        return torch.cat([rep, commands["velocity"], commands["velocity_target"], commands["arm"].flatten(1), sim_ctx,
                          clip_v.float().unsqueeze(-1), clip_a.float().unsqueeze(-1)], dim=-1)

    def _clip_goals(self, frames_v, frames_a, aux) -> dict[str, torch.Tensor]:
        """The stage-1 goals of the clip frames: the legs' at the velocity segment's frame (relative to the robot's
        pose), the arms' at the arm segment's."""
        return {"legs": legs_goal(self.clips.motion, frames_v, aux), "arms": arms_goal(self.clips.tables, frames_a)}

    def _start_clip_segments(self, cmds: dict, frames: dict, new_v: torch.Tensor, new_a: torch.Tensor) -> dict:
        """Clip rows that start a segment draw a frame; every clip row's command is its clip's at its frame."""
        if self.clips is None:
            return cmds
        b = new_v.shape[0]
        frames["v"] = torch.where(new_v, self.clips.draw(b, self.gen), frames["v"])
        frames["a"] = torch.where(new_a, self.clips.draw(b, self.gen), frames["a"])
        return CommandProcess.refresh(cmds, self.clips.velocity[frames["v"]], self.clips.arm[frames["a"]])

    def update_encoders(self) -> dict[str, torch.Tensor]:
        """One imagined rollout from buffered states; each encoder stepped on its own return, each critic fitted."""
        cfg, p, dev = self.cfg, self.cfg.policy, self.device
        b, h = p.batch_size, cfg.lstm
        start = self.buffer.sample(1, b)[:, 0]
        for m in (self.dynamics, self.rep):
            m.eval()
            m.requires_grad_(False)
        for e in self.encoders.values():
            e.train()
        state0 = start["state"]
        rep, aux = self.rep.represent(state0), self.rep.extract_aux_state(state0)
        sim_ctx = start["sim_ctx"]
        hc = {a: start[f"hc_{a}"].view(b, 2, h) for a in AGENTS}
        last_u = {a: start[f"latent_{a}"] for a in AGENTS}
        prev_action = start["prev_action"]
        clock = start["clock"]
        cmds = self.process.start({"velocity": start["velocity"], "velocity_target": start["velocity_target"],
                                   "arm": start["arm"].view(b, 2, 7)})
        # every rollout starts a segment of each command: a clip one with clip_command_prob
        if self.clips is not None:
            clip_v = torch.rand(b, generator=self.gen, device=dev) < cfg.clip_command_prob
            clip_a = torch.rand(b, generator=self.gen, device=dev) < cfg.clip_command_prob
            cmds["clip_velocity"], cmds["clip_arm"] = clip_v, clip_a
            frames = {"v": torch.zeros(b, dtype=torch.long, device=dev),
                      "a": torch.zeros(b, dtype=torch.long, device=dev)}
            cmds = self._start_clip_segments(cmds, frames, clip_v, clip_a)
            # the seen velocity jumps to the clip's at a clip segment's start: no low-pass from another command
            cmds["velocity"] = torch.where(clip_v[:, None], cmds["velocity_target"], cmds["velocity"])
        else:
            frames = None
        self.obs_model.reset(b, self.gen)
        cycle = cycle_init(b, self.gait_window, dev)
        state = WMState.from_wm(self.rep, aux, rep)
        ret = {a: torch.zeros(b, device=dev) for a in AGENTS}
        aux_loss = {a: torch.zeros(b, device=dev) for a in AGENTS}
        survival = torch.ones(b, device=dev)
        steps: list[dict] = []
        logs: dict[str, torch.Tensor] = {}
        for t in range(p.horizon):
            commands = {"velocity": cmds["velocity"], "arm": cmds["arm"]}
            obs = self.obs_model(state, commands, {a: last_u[a].detach() for a in AGENTS}, clock)
            cvec = command_vectors(commands)
            z = {}
            for a in AGENTS:
                z[a], hc[a] = self.encoders[a](obs[a], cvec[a], hc[a])
            if frames is not None and cfg.prior_weight > 0.0:
                goals = self._clip_goals(frames["v"], frames["a"], aux.detach())
                for a, mask in (("legs", cmds["clip_velocity"]), ("arms", cmds["clip_arm"])):
                    aux_loss[a] = aux_loss[a] + latent_bc(z[a], stage1_latent(self.stage1[a], goals[a]), mask,
                                                          cfg.prior_weight)
            action = self._decode(rep, aux, sim_ctx, prev_action, z)
            next_aux, next_rep = self.dynamics(rep, action, aux, sim_ctx)
            next_state = WMState.from_wm(self.rep, next_aux, next_rep)
            cycle, cycle_mean = cycle_push(cycle, next_state)
            target = self.dynamics.proc(action) - self.rep.encoder_bias(sim_ctx)
            r = main_rewards(state, next_state, commands, self.reward_cfg, cycle_mean=cycle_mean, target=target)
            fall = self._soft_fall(next_rep)
            critic_obs = self._critic_obs(rep, cmds, sim_ctx, cmds["clip_velocity"], cmds["clip_arm"])
            steps.append({"critic": critic_obs.detach(), "done": fall.detach(),
                          **{f"reward_{a}": r[a].detach() for a in AGENTS}})
            for a in AGENTS:
                ret[a] = ret[a] + p.gamma**t * survival * r[a]
            if p.survival_masking:
                survival = survival * (1.0 - fall)
            if t == 0:
                for k, v in r.items():
                    if "/" in k:
                        logs[f"Reward/{k}"] = v.mean().detach()
            rep, aux, state, prev_action = next_rep, next_aux, next_state, action
            last_u = {a: latent_to_action(z[a]) for a in AGENTS}
            clock = clock + self.dt
            cmds = self.process.step(cmds, self.gen)
            if frames is not None:
                frames["v"] = self.clips.advance(frames["v"])
                frames["a"] = self.clips.advance(frames["a"])
                cmds = self._start_clip_segments(cmds, frames, cmds["new_velocity"] & cmds["clip_velocity"],
                                                 cmds["new_arm"] & cmds["clip_arm"])
        last_obs = self._critic_obs(rep, cmds, sim_ctx, cmds["clip_velocity"], cmds["clip_arm"])
        losses, params = {}, {}
        for a in AGENTS:
            critic = self.critics[a]
            critic.requires_grad_(False)
            value = critic(last_obs).squeeze(-1)
            ret[a] = ret[a] + p.gamma**p.horizon * survival * value
            losses[a] = (-ret[a] + aux_loss[a]).mean()
            params[a] = [q for q in self.encoders[a].parameters() if q.requires_grad]
            logs[f"Policy/return_{a}"] = ret[a].mean().detach()
            logs[f"Policy/latent_bc_{a}"] = aux_loss[a].mean().detach()
        grads = {a: torch.autograd.grad(losses[a], params[a], retain_graph=i < len(AGENTS) - 1)
                 for i, a in enumerate(AGENTS)}
        for a in AGENTS:
            for q, g in zip(params[a], grads[a], strict=True):
                q.grad = g
            logs[f"Policy/grad_norm_{a}"] = torch.nn.utils.clip_grad_norm_(params[a], max_norm=1.0)
            if p.grad_clip_value > 0:
                torch.nn.utils.clip_grad_value_(params[a], p.grad_clip_value)
            self.encoder_opt[a].step()
            self.encoder_opt[a].zero_grad(set_to_none=True)
        logs["Policy/survival"] = survival.mean().detach()
        logs.update(self._fit_critics(steps, last_obs.detach()))
        return logs

    def _fit_critics(self, steps: list[dict], last_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Each critic to the GAE returns of its agent's reward (TwoAgentLearner._fit_critics)."""
        p = self.cfg.policy
        obs = torch.stack([s["critic"] for s in steps])
        done = torch.stack([s["done"] for s in steps]).unsqueeze(-1)
        logs = {}
        for a in AGENTS:
            critic = self.critics[a]
            rewards = torch.stack([s[f"reward_{a}"] for s in steps]).unsqueeze(-1)
            with torch.no_grad():
                returns = gae_returns(critic(obs), critic(last_obs), rewards, done, p.gamma, p.lam)
            x, y = obs.flatten(0, 1), returns.flatten(0, 1)
            critic.requires_grad_(True)
            perm = torch.randperm(len(y) // 4 * 4, device=y.device)
            losses = []
            for _ in range(5):
                for idx in perm.view(4, -1):
                    loss = (y[idx] - critic(x[idx])).pow(2).mean()
                    self.critic_opt[a].zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                    self.critic_opt[a].step()
                    losses.append(loss.detach())
            logs[f"Policy/value_loss_{a}"] = torch.stack(losses).mean()
        return logs

    # -- checkpoints ------------------------------------------------------------------------------------------------

    def commands_bundle(self) -> dict:
        """``commands.pt``: both encoders and what decodes them (the frozen stage-1 run, its plant)."""
        return {
            "encoders": {a: m.state_dict() for a, m in self.encoders.items()},
            "obs_dims": {a: m.obs_dim for a, m in self.encoders.items()},
            "command_dims": dict(COMMAND_DIMS),
            "latent_dim": self.encoders[AGENTS[0]].latent_dim,
            "lstm": self.cfg.lstm,
            "command_hidden_dims": list(self.cfg.command_hidden_dims),
            "stage1": to_repo_path(self.stage1_run),
            "stage1_actors": to_repo_path(self.stage1_run / "actors.pt"),
            "plant": self.plant.name,
            "steps_per_env": self.steps_per_env,
        }

    def training_state(self, wall_s: float) -> dict:
        return {
            "iteration": self.iteration, "steps_per_env": self.steps_per_env, "wall_s": wall_s,
            "world_model": self.wm.bundle(),
            "encoders": {a: m.state_dict() for a, m in self.encoders.items()},
            "critics": {a: m.state_dict() for a, m in self.critics.items()},
            "encoder_opt": {a: o.state_dict() for a, o in self.encoder_opt.items()},
            "critic_opt": {a: o.state_dict() for a, o in self.critic_opt.items()},
            "world_model_opt": [o.state_dict() for o in self.wm_learner.optimizers],
        }

    def load_checkpoint(self, path: Path) -> None:
        """Restore a ``training_state`` (the replay buffer is not saved: a resumed run refills it first)."""
        st = torch.load(path, map_location=self.device, weights_only=False)
        self.wm.load_bundle(st["world_model"])
        for a in AGENTS:
            self.encoders[a].load_state_dict(st["encoders"][a])
            self.critics[a].load_state_dict(st["critics"][a])
            self.encoder_opt[a].load_state_dict(st["encoder_opt"][a])
            self.critic_opt[a].load_state_dict(st["critic_opt"][a])
        for opt, sd in zip(self.wm_learner.optimizers, st["world_model_opt"], strict=True):
            opt.load_state_dict(sd)
            for group in opt.param_groups:
                if torch.is_tensor(group["lr"]):
                    group["lr"] = group["lr"].to(self.device)
        self.iteration = int(st["iteration"]) + 1
        self.steps_per_env = int(st["steps_per_env"])
        self._wall_offset = float(st["wall_s"])

    def save(self, log_dir: Path, wall_s: float, periodic: bool = False) -> None:
        atomic_save(self.commands_bundle(), log_dir / "commands.pt")
        atomic_save(self.wm.bundle(), log_dir / "world_model.pt")
        state = self.training_state(wall_s)
        atomic_save(state, log_dir / "checkpoint.pt")
        if periodic:
            (log_dir / "checkpoints").mkdir(exist_ok=True)
            atomic_save(state, log_dir / "checkpoints" / f"it_{self.iteration:08d}.pt")
            keep_last(log_dir / "checkpoints", "it_*.pt", self.cfg.keep_checkpoints)
        atomic_write_text(log_dir / "checkpoint.json", json.dumps(
            {"iteration": self.iteration, "steps_per_env": self.steps_per_env, "wall_s": wall_s}) + "\n")

    # -- the loop ---------------------------------------------------------------------------------------------------

    def _done(self, wall: float, completed: int) -> str | None:
        """Why the run stops after ``completed`` iterations (None: it goes on)."""
        cfg = self.cfg
        if cfg.max_steps > 0 and self.steps_per_env >= cfg.max_steps:
            return "max_steps"
        if cfg.max_seconds > 0 and wall >= cfg.max_seconds:
            return "max_seconds"
        if completed >= cfg.max_iterations:
            return "max_iterations"
        return None

    def run(self, meta: dict | None = None, resume: bool = False) -> Path:
        """Train until ``max_steps``, ``max_seconds`` or ``max_iterations``; checkpoints every ``save_interval``
        iterations, every ``checkpoint_interval_s`` and at the end. A new run writes ``run.json`` (``meta`` plus the
        cfg) and refuses an existing directory; ``resume`` continues one after ``load_checkpoint``. Every
        ``log_interval`` iterations a line and a ``progress.csv`` row, with the peak GPU memory since the last."""
        log_dir = self.run_dir
        if resume:
            run_json = json.loads((log_dir / "run.json").read_text())
            run_json.setdefault("resumed", []).append({"iteration": self.iteration, "wall_s": self._wall_offset,
                                                      "steps_per_env": self.steps_per_env})
            progress = ProgressCsv(log_dir / "progress.csv", keep_through_iteration=self.iteration - 1)
        else:
            if log_dir.exists():
                raise FileExistsError(f"{log_dir} exists; choose another run name or resume it")
            log_dir.mkdir(parents=True)
            run_json = {"stage": self.stage, **(meta or {}), "cfg": asdict(self.cfg),
                        "updates_per_iteration": self.updates_per_iteration}
            progress = ProgressCsv(log_dir / "progress.csv")
        atomic_write_text(log_dir / "run.json", json.dumps(run_json, indent=2, default=str))
        on_cuda = self.device.type == "cuda"
        print(f"[commands] {self.num_envs} envs x {self.cfg.rollout} steps per iteration, "
              f"{self.updates_per_iteration} updates per iteration; left out of main's reward: "
              f"{', '.join(self.reward_cfg.left_out) or 'none'}", flush=True)
        run_start = time.time() - self._wall_offset
        last_checkpoint = time.time()
        stop = self._done(time.time() - run_start, self.iteration)
        while stop is None:
            t0 = time.time()
            self._refresh_rewards()
            row: dict = dict(self.collect())
            t_collect = time.time() - t0
            wm_s = policy_s = 0.0
            losses: dict[str, torch.Tensor] = {}
            if self.buffer.num_transitions >= self.cfg.wm.horizon:
                for _ in range(self.updates_per_iteration):
                    t = time.time()
                    losses.update(self.update_world_model())
                    wm_s += time.time() - t
                    t = time.time()
                    losses.update(self.update_encoders())
                    policy_s += time.time() - t
            wall = time.time() - run_start
            stop = self._done(wall, self.iteration + 1)
            if self.iteration % self.cfg.log_interval == 0 or stop is not None:
                peak = torch.cuda.max_memory_allocated(self.device) / 1e9 if on_cuda else 0.0
                if on_cuda:
                    torch.cuda.reset_peak_memory_stats(self.device)
                row = {"iteration": self.iteration, "steps_per_env": self.steps_per_env, "wall_s": wall,
                       "collect_s": t_collect, "wm_s": wm_s, "policy_s": policy_s, "peak_gpu_gb": peak, **row,
                       **{k: float(v) for k, v in losses.items()}}
                progress.append(row)
                print(" ".join(f"{k} {row[k]:.4g}" if isinstance(row.get(k), float) else f"{k} {row.get(k, '-')}"
                               for k in ("iteration", "steps_per_env", "wall_s", "collect_s", "wm_s", "policy_s",
                                         "peak_gpu_gb", "Loss/state", "Policy/return_legs", "Policy/return_arms",
                                         "Sim/vel_xy_error", "Sim/wrist_pos_error", "Sim/done")), flush=True)
            by_iteration = self.iteration > 0 and self.iteration % self.cfg.save_interval == 0
            by_clock = time.time() - last_checkpoint > self.cfg.checkpoint_interval_s
            if stop is not None or by_iteration or by_clock:
                self.save(log_dir, wall, periodic=stop is None)
                last_checkpoint = time.time()
            if stop is None:
                self.iteration += 1
        print(f"[commands] stopped at iteration {self.iteration} ({stop}), {self.steps_per_env} steps per env",
              flush=True)
        return log_dir


def load_commands(path, device) -> dict[str, RecurrentCommandActor]:
    """The encoders of a ``commands.pt`` on ``device``, in eval mode."""
    b = torch.load(Path(path), map_location=device, weights_only=False)
    out = {}
    for a in AGENTS:
        m = RecurrentCommandActor(b["obs_dims"][a], b["command_dims"][a], b["latent_dim"], b["lstm"],
                                  tuple(b["command_hidden_dims"])).to(device)
        m.load_state_dict(b["encoders"][a])
        out[a] = m.eval()
    return out
