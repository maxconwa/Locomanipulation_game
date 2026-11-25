"""Latent policies in the Isaac evals: a policy that acts in the 32-wide latent of the frozen stage-1 decoders, which
LocoManip-Marl-Pulse-v0's action terms decode into joint targets (mdp/pulse_actions.py), so the scorecards and the
video drive it as they drive a joint-space policy: obs in, actions out, the env does the rest.

Two kinds, both run with both agents (the legs and the arms of one run), as `--latent_run <dir>`:

- B, an skrl MAPPO run trained in LocoManip-Marl-Pulse-v0 (params/task.txt): its MarlRunner as the joint-space
  evals load it; `<dir>` is the run (its newest checkpoint) or one checkpoints/agent_<N>.pt. The env is the run's,
  its overrides replayed, its prior as it trained.
- C, a `pretraining.scripts.train_commands` run (commands.pt and run.json in `<dir>`): each agent's recurrent command
  encoder (RecurrentCommandActor) maps the agent's observation and its command to z = tanh(...); the env gets
  u = atanh(clamp(z, +-tanh 5)) (clip_actions 5), in LocoManip-Marl-Pulse-v0 with the prior off (pulse_prior null)
  and the run's own stage 1 as the decoders, as it trained (pretraining.marl.marl_source). Its stage 1 is run.json's
  "stage1", else commands.pt's.

Everything else (events, randomisation, the program, the scoring) is the scorecard's own, so A (a joint-space run of
LocoManip-Marl-Torso-v0 with --arms_checkpoint/--legs_checkpoint same), B and C are scored on the same conditions.

What C's loading takes from pretraining's stage 2 is in the small functions marked STAGE-2 CONTRACT below.
Imports only the standard library and torch at module level (the Isaac parts after the app is up).
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
PULSE = "LocoManip-Marl-Pulse-v0"
AGENTS = ("legs", "arms")
COMMANDS_FILE = "commands.pt"
LATENT_CLIP = 5.0  # MarlPulseEnvCfg's clip_actions: u is clamped there, so z beyond tanh(5) cannot be reached


@dataclass
class LatentRun:
    kind: str  # "skrl" (B) or "commands" (C)
    run_dir: Path
    checkpoint: Path  # B: the skrl checkpoint; C: commands.pt
    task: str
    step: int | None
    stage1: Path | None = None  # C: the run's stage-1 run (B: the env's pulse_stage1_run, as the run trained)


def resolve_latent_run(path: str | Path) -> LatentRun:
    """What `--latent_run <path>` names (the module docstring); SystemExit naming what is missing otherwise."""
    p = Path(path).resolve()
    if p.is_dir() and (p / COMMANDS_FILE).is_file():
        meta = json.loads((p / "run.json").read_text()) if (p / "run.json").is_file() else {}
        stage1 = meta.get("stage1") or _bundle_key(p / COMMANDS_FILE, "stage1")
        if not stage1:
            raise SystemExit(f"[evals] {p}/run.json does not name its stage-1 run (\"stage1\")")
        stage1 = Path(stage1) if Path(stage1).is_absolute() else (REPO / stage1)
        task = meta.get("task") or PULSE
        if task != PULSE:
            raise SystemExit(f"[evals] {p} trained in {task}; command encoders are evaluated in {PULSE}")
        return LatentRun("commands", p, p / COMMANDS_FILE, PULSE, commands_step(p / COMMANDS_FILE, meta),
                         stage1.resolve())
    if p.is_file():
        run_dir, ckpt = p.parents[1], p
    elif p.is_dir():
        ckpts = sorted(((int(m.group(1)), c) for c in (p / "checkpoints").glob("agent_*.pt")
                        if (m := re.fullmatch(r"agent_(\d+)\.pt", c.name))), key=lambda x: x[0])
        if not ckpts:
            raise SystemExit(f"[evals] {p}: neither {COMMANDS_FILE} (a train_commands run) nor"
                             " checkpoints/agent_<N>.pt")
        run_dir, ckpt = p, ckpts[-1][1]
    else:
        raise SystemExit(f"[evals] --latent_run {path}: no such run or checkpoint")
    task_file = run_dir / "params" / "task.txt"
    task = task_file.read_text().strip() if task_file.is_file() else None
    if task != PULSE:
        raise SystemExit(f"[evals] {run_dir} trained in {task or 'an unrecorded task'}, not {PULSE}: a joint-space run"
                         " is scored with --checkpoint (and --arms_checkpoint/--legs_checkpoint same)")
    m = re.search(r"_(\d+)\.pt$", ckpt.name)
    return LatentRun("skrl", run_dir, ckpt, PULSE, int(m.group(1)) if m else None)


def _bundle_key(path: Path, key: str):
    try:
        bundle = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    return bundle.get(key) if isinstance(bundle, dict) else None


def hydra_overrides(run: LatentRun) -> list[str]:
    """C's env: the Pulse task on the run's own stage 1, no prior (B's come from its run, replay_run)."""
    if run.kind != "commands":
        return []
    return [f"env.pulse_stage1_run={run.stage1}", "env.pulse_prior=null"]


def latent_from_z(z: torch.Tensor) -> torch.Tensor:
    """The env's action for a command encoder's z: u = atanh(z), z clamped to +-tanh(clip_actions) so u stays finite
    and inside the env's clamp; with the prior off the env decodes tanh(u) = z (command_actor.latent_to_action, which
    trained it; restated here so the CPU tests need no pretraining import)."""
    bound = math.tanh(LATENT_CLIP)
    return torch.atanh(z.clamp(-bound, bound))


# ---- STAGE-2 CONTRACT: what C's loading takes from pretraining's stage 2 (marl/command_runner.py, command_actor.py) --

def commands_step(path: Path, meta: dict) -> int | None:
    """The env steps a commands.pt was saved at (the scorecard's <N>): its steps_per_env times run.json's env count
    ("num_envs"), else steps_per_env alone, else None."""
    try:
        bundle = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001  (an unreadable file fails later, loading the encoders, with its own error)
        return None
    if not isinstance(bundle, dict) or bundle.get("steps_per_env") is None:
        return None
    n = meta.get("num_envs") or meta.get("env_count")
    return int(bundle["steps_per_env"]) * (int(n) if n else 1)


def load_command_actors(path: Path, obs_dims: dict[str, int], device, loader=None) -> dict[str, torch.nn.Module]:
    """Each agent's RecurrentCommandActor of a commands.pt (command_runner.load_commands), frozen; stops when an
    encoder reads another observation width than the env gives."""
    if loader is None:
        from pretraining.marl.command_runner import load_commands as loader
    actors = loader(path, device)
    for agent in AGENTS:
        if actors[agent].obs_dim != obs_dims[agent]:
            raise SystemExit(f"[evals] {path}: the {agent} command encoder reads {actors[agent].obs_dim} observation"
                             f" values, this env gives {obs_dims[agent]}")
    return {a: actors[a].requires_grad_(False) for a in AGENTS}


def command_inputs(base) -> dict[str, torch.Tensor]:
    """Each agent's command as its encoder trained on it (command_runner.command_vectors on MarlEnvSource.commands):
    the legs the velocity command as the policies see it (n, 3), the arms both pelvis-frame wrist goals as the
    policies believe them, xyz + xyzw (n, 14)."""
    from pretraining.marl.command_runner import command_vectors

    vel = base.command_manager.get_term("base_velocity")
    arm = base.command_manager.get_term("arm_targets")
    return command_vectors({"velocity": vel.command, "arm": arm.command.view(base.num_envs, 2, 7)})


def initial_hc(actor, n: int, device) -> torch.Tensor:
    """The encoder's zero state for n envs, (n, 2, lstm): h then c."""
    return actor.initial_state(n, device)

# -------------------------------------------------------------------------------------------------------------------


def reset_ids(done) -> torch.Tensor:
    """Env indices from a bool mask ((n,) or (n, 1)) or an index tensor/list."""
    t = torch.as_tensor(done)
    if t.dtype == torch.bool:
        return t.view(-1).nonzero(as_tuple=False)[:, 0]
    return t.view(-1).long()


class LatentPolicy:
    """A latent policy's two agents (module docstring). from_run, then bind(env, agent_cfg) once the env is up;
    act(obs) -> {agent: action}; reset(done) zeroes the reset envs' recurrent state."""

    def __init__(self, run: LatentRun, settings: LatentRun | None = None):
        self.run, self.settings = run, settings
        if run.kind == "commands" and settings is None:
            raise SystemExit("[evals] a train_commands run records no env overrides, IMU model or curriculum state:"
                             " give --settings_run <a LocoManip-Marl-Pulse-v0 skrl run> so C runs on B's conditions")
        if settings is not None and settings.kind != "skrl":
            raise SystemExit(f"[evals] --settings_run {settings.run_dir}: not an skrl run of {PULSE}")
        self.base = self.runner = self.actors = self.hc = None
        self.stochastic = False

    @classmethod
    def from_run(cls, run_dir: str | Path, settings_run: str | Path | None = None) -> "LatentPolicy":
        """settings_run: for C (required there), the Pulse skrl run whose env overrides, IMU model, engage blend and
        curriculum state C is scored with; ignored for B (its own run's)."""
        run = resolve_latent_run(run_dir)
        settings = resolve_latent_run(settings_run) if (settings_run and run.kind == "commands") else None
        return cls(run, settings)

    @property
    def cfg_run(self) -> tuple[str, str]:
        """(checkpoint, run dir) evaluation_cfgs reads the IMU model, engage blend and curriculum state from: B's own,
        C's settings run's."""
        src = self.run if self.kind == "skrl" else self.settings
        return str(src.checkpoint), str(src.run_dir)

    @property
    def settings_run(self) -> str | None:
        return None if self.settings is None else str(self.settings.run_dir)

    @property
    def kind(self) -> str:
        return self.run.kind

    @property
    def trainer(self) -> str:
        return "skrl" if self.kind == "skrl" else "commands"

    def replay(self, extra: list[str]) -> tuple[str, str, str, list[str]]:
        """As run_files.replay_run (call after the app is up): (checkpoint, run dir, task, the run's overrides) and
        sys.argv set for hydra_task_config. B replays its run's overrides; C the Pulse env on its stage 1, no prior."""
        if self.kind == "skrl":
            from locomanipulation_game.tasks.locomanip_marl.run_files import replay_run
            return replay_run(str(self.run.checkpoint), PULSE, extra)
        from locomanipulation_game.tasks.locomanip_marl.run_files import replay_run
        # the settings run's overrides, then C's own decoders (its stage 1, no prior), then the caller's
        _, _, _, replayed = replay_run(str(self.settings.checkpoint), PULSE, hydra_overrides(self.run) + extra)
        return str(self.run.checkpoint), str(self.run.run_dir), PULSE, replayed + hydra_overrides(self.run)

    def bind(self, env, agent_cfg: dict, stochastic: bool = False) -> None:
        """Load the agents into the built env (SkrlVecEnvWrapper)."""
        self.base, self.stochastic = env.unwrapped, stochastic
        base = self.base
        if self.kind == "skrl":
            from locomanipulation_game.tasks.locomanip_marl.recurrent import MarlRunner

            from evals.isaac import env as E
            self.runner = MarlRunner(env, agent_cfg)
            for agent in AGENTS:
                E.check_agent_fits(base, agent, self.run.run_dir, self.run.checkpoint)
                E.load_agent(self.runner, self.run.checkpoint, agent)
            self.runner.agent.enable_training_mode(False, apply_to_models=True)
            return
        if base.cfg.pulse_prior:
            raise SystemExit(f"[evals] a command encoder runs with the prior off; the env has {base.cfg.pulse_prior}")
        obs_dims = {a: base.observation_manager.group_obs_dim[a][0] for a in AGENTS}
        self.actors = load_command_actors(self.run.checkpoint, obs_dims, base.device)
        self.hc = {a: initial_hc(self.actors[a], base.num_envs, base.device) for a in AGENTS}

    def act(self, obs: dict) -> dict[str, torch.Tensor]:
        if self.kind == "skrl":
            states = self.base.state()
            out = self.runner.agent.act(obs, {a: states for a in self.base.possible_agents}, timestep=0, timesteps=0)
            if self.stochastic:
                return {a: out[0][a] for a in self.base.possible_agents}
            return {a: out[-1][a].get("mean_actions", out[0][a]) for a in self.base.possible_agents}
        commands = command_inputs(self.base)
        actions = {}
        for a in AGENTS:
            z, self.hc[a] = self.actors[a](obs[a], commands[a], self.hc[a])
            actions[a] = latent_from_z(z)
        return actions

    def reset(self, done) -> None:
        if self.kind == "skrl":
            self.runner.agent.reset_rnn(torch.as_tensor(done, dtype=torch.bool))
            return
        ids = reset_ids(done)
        if ids.numel():
            for hc in self.hc.values():
                hc[ids] = 0.0

    def policy_std(self, agent: str) -> torch.Tensor | None:
        """The skrl policy's action std (B); None for a deterministic encoder (C)."""
        if self.kind != "skrl":
            return None
        return self.runner.agent.policies[agent].log_std_parameter.detach().exp()
