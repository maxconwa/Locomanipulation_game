"""Latent policies in MuJoCo: the frozen stage-1 decoders in the loop, as LocoManip-Marl-Pulse-v0 decodes in Isaac
(mdp/pulse_actions.py, pretraining/marl/pulse.py).

The decoders come from pretraining/scripts/export_decoders.py (TorchScript, plant.json: no pretraining import here).
Each policy step, before acting:

    frame  = [gravity 3, gyro 3, q - default 27, dq 27, last action 27]   (87, Unitree motor order = sim.ALL_NAMES)
             gravity and gyro: the torso IMU as the policies read it (the export's lowstate model, its bias) but the
             gyro before the low-pass, as the Isaac terms read imu.gyro_raw; q with the encoders' position noise, dq
             with their speed noise (training's; noise=False for none); the last action both decoders' raw outputs
    u      = the agent's encoder: an skrl export's actor on its observation (B), or a command encoder on its
             observation and command, u = atanh(clamp(z, +-tanh 5)) (C)
    z      = tanh(clamp(u, +-clip_actions) + prior(frame))                (prior: B as its run trained; C none)
    target = clip(default + action_scale x decoder(z, frame), the plant's target limits, the export's target_limits
             and the URDF's)                                                (no IK, no residual, no torque bound)

The agents' observations are MarlMujoco's, their "actions" term the applied latent u. LatentMarlMujoco is MarlMujoco
with that action path; everything else (physics, sensors, commands, arm goals, odometry) is the parent's.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .sim import ALL_NAMES, MarlMujoco

AGENTS = ("legs", "arms")
LATENT_CLIP = 5.0
DEFAULT_POSE_TOL = 1e-4


class CommandEncoder:
    """A command encoder exported by export_decoders --commands: (obs, command) -> u, its LSTM state carried."""

    def __init__(self, path: str | Path, lstm: int = 64):
        self.net = torch.jit.load(str(path), map_location="cpu").eval()
        self.lstm, self.hc = lstm, None

    def __call__(self, obs: np.ndarray, command: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32)[None]
        c = torch.as_tensor(command, dtype=torch.float32)[None]
        if self.hc is None:
            self.hc = torch.zeros(1, 2, self.lstm, dtype=torch.float32)
        with torch.no_grad():
            z, self.hc = self.net(x, c, self.hc)
        bound = math.tanh(LATENT_CLIP)
        return torch.atanh(z[0].clamp(-bound, bound)).numpy().astype(float)

    def reset_state(self):
        self.hc = None

    def state_norm(self) -> float:
        return 0.0 if self.hc is None else float(self.hc[:, 1].abs().mean())


def command_vector(sim, agent: str) -> np.ndarray:
    """The agent's command as its encoder trained on it (command_runner.command_vectors): the velocity command as
    the policies see it; both pelvis-frame wrist goals, xyz + xyzw."""
    if agent == "legs":
        return sim.observed_cmd()
    b = np.asarray(sim.believed, float)
    return np.concatenate([np.r_[g[:3], g[4:7], g[3]] for g in b])


class LatentMujocoPolicy:
    """The decoders of an export_decoders folder and each agent's encoder ({agent: callable}; an skrl export's
    `policy.Recurrent` called on obs[None], or a CommandEncoder). prior: True / False, None: plant.json's."""

    def __init__(self, export_dir: str | Path, encoder: dict, prior: bool | None = None):
        export_dir = Path(export_dir)
        self.info = json.loads((export_dir / "plant.json").read_text())
        if list(self.info["joint_names"]) != list(ALL_NAMES):
            raise ValueError(f"{export_dir}: the decoders' joint order is not the sim's")
        load = lambda name: torch.jit.load(str(export_dir / name), map_location="cpu").eval()  # noqa: E731
        self.decoders = {"legs": load("legs_decoder.pt"), "arms": load("upper_decoder.pt")}
        use_prior = bool(self.info.get("prior")) if prior is None else prior
        if use_prior and not (export_dir / "prior.pt").is_file():
            raise FileNotFoundError(f"{export_dir}: no prior.pt (export_decoders without a prior)")
        self.prior = load("prior.pt") if use_prior else None
        self.encoder = encoder
        self.commands = {a: isinstance(e, CommandEncoder) for a, e in encoder.items()}
        self.default = np.array(self.info["default_joint_pos"], float)
        self.scale = np.array(self.info["action_scale"], float)
        self.low, self.high = (np.array([lim[i] for lim in self.info["target_limits"]], float) for i in (0, 1))
        self.slices = {a: slice(*self.info["agent_joints"][a]) for a in AGENTS}
        self.latent_dim = int(self.info["latent_dim"])
        self.last_action = np.zeros(len(ALL_NAMES))

    def reset(self):
        self.last_action[:] = 0.0
        for e in self.encoder.values():
            e.reset_state()

    def frame(self, sim, noise: bool = True) -> np.ndarray:
        """The 87-wide proprio frame (module docstring) from the sim's state."""
        _, _, grav = sim.imu()
        gyro = sim._raw_gyro()
        q, dq = sim.measured_q(), sim.measured_dq()
        if noise:
            n, r = sim.NOISE, sim.rng
            grav = grav + r.uniform(-n["grav"], n["grav"], 3)
            gyro = gyro + r.uniform(-n["ang_vel"], n["ang_vel"], 3)
            q = q + r.uniform(-n["jpos"], n["jpos"], q.shape)
            dq = sim.noisy_dq(n["jvel"])
        return np.concatenate([grav, gyro, q - self.default, dq, self.last_action])

    def act(self, sim, obs: dict) -> dict[str, np.ndarray]:
        """Each agent's latent u from its encoder."""
        out = {}
        with torch.no_grad():
            for a in AGENTS:
                e = self.encoder[a]
                if self.commands[a]:
                    out[a] = e(obs[a], command_vector(sim, a))
                else:
                    out[a] = e(torch.as_tensor(obs[a], dtype=torch.float32)[None])[0].numpy().astype(float)
        return out

    def decode(self, u: dict, frame: np.ndarray) -> np.ndarray:
        """The 27 joint targets for both latents on one frame; moves the last action on after both."""
        f = torch.as_tensor(frame, dtype=torch.float32)[None]
        with torch.no_grad():
            prior = self.prior(f) if self.prior is not None else None
            raw = np.zeros(len(ALL_NAMES))
            for i, a in enumerate(AGENTS):
                ua = torch.as_tensor(u[a], dtype=torch.float32)[None]
                if prior is not None:
                    ua = ua + prior[i]
                raw[self.slices[a]] = self.decoders[a](torch.tanh(ua), f)[0].numpy()
        self.last_action = raw
        return np.clip(self.default + self.scale * raw, self.low, self.high)


class LatentMarlMujoco(MarlMujoco):
    """MarlMujoco driven by a LatentMujocoPolicy (module docstring)."""

    def __init__(self, policy: LatentMujocoPolicy, settings: dict, physics: str = "robocasa", noise: bool = True,
                 **kwargs):
        super().__init__(dict(policy.encoder), settings, physics=physics, **kwargs)
        self.policy, self.frame_noise = policy, noise
        off = np.abs(self.default - policy.default)
        if (off > DEFAULT_POSE_TOL).any():
            bad = ", ".join(f"{ALL_NAMES[i]} ({self.default[i] - policy.default[i]:+.3f})" for i in np.nonzero(
                off > DEFAULT_POSE_TOL)[0])
            raise SystemExit(f"[evals] the decoders' plant ({policy.info['plant']}) and the export's default pose "
                             f"differ on {bad}: the stage-1 run was trained on another robot")
        # the env's clip: the plant's target limits within the URDF's and the run's (GOLEM's envelope)
        self.lower, self.upper = np.maximum(self.lower, policy.low), np.minimum(self.upper, policy.high)
        self.u = {a: np.zeros(policy.latent_dim) for a in AGENTS}

    def reset(self, rng: np.random.Generator):
        super().reset(rng)
        self.policy.reset()
        self.u = {a: np.zeros(self.policy.latent_dim) for a in AGENTS}

    def applied_actions(self) -> tuple[np.ndarray, np.ndarray]:
        return self.u["legs"], self.u["arms"]

    def step(self, obs: dict):
        frame = self.policy.frame(self, self.frame_noise)
        u = self.policy.act(self, obs)
        self.u = {a: np.clip(u[a], -self.clip_actions, self.clip_actions) for a in AGENTS}
        self.applied[:] = np.clip(self.policy.decode(self.u, frame), self.lower, self.upper)
        self.policy_target = self.applied[self.leg_idx].copy()
        self.physics_step()
        self.steps += 1
        self._update_odometry()
        self._filter_command()
        return self.observations()


def encoders(kind: str, export_dir: str | Path) -> dict:
    """Each agent's encoder: "skrl" an export_marl folder's actors (legs.pt, arms.pt), "commands" the
    export_decoders folder's command encoders (legs_commands.pt, arms_commands.pt)."""
    from .policy import actor
    if kind == "skrl":
        return {a: actor(export_dir, a) for a in AGENTS}
    info = json.loads((Path(export_dir) / "plant.json").read_text()).get("commands")
    if not info:
        raise SystemExit(f"[evals] {export_dir}: no command encoders (export_decoders --commands <run>)")
    return {a: CommandEncoder(Path(export_dir) / f"{a}_commands.pt", info["lstm"]) for a in AGENTS}


# ---- --latent_run in the MuJoCo evals -----------------------------------------------------------------------------

ISAAC_PYTHON = "/home/max/miniconda3/envs/env_isaaclab/bin/python"


def _run_env(run_dir: Path) -> dict:
    from .common import _yaml_load_any
    return _yaml_load_any((Path(run_dir) / "params" / "env.yaml").read_text())


def _sha256(path) -> str | None:
    import hashlib
    path = Path(path) if path else None
    return hashlib.sha256(path.read_bytes()).hexdigest() if path is not None and path.is_file() else None


def stale(decoders: Path, stage1: Path, prior: str, checkpoint: str) -> str | None:
    """Why the decoders in `decoders` are not the ones asked for (None: they are): no plant.json, another stage-1
    run, prior or checkpoint, or the stage-1 actors or the prior changed since (their sha256)."""
    path = Path(decoders) / "plant.json"
    if not path.is_file():
        return "no export yet"
    info = json.loads(path.read_text())
    want_prior = None if prior == "none" else str(prior)
    checks = {"stage1": (info.get("stage1"), str(Path(stage1).resolve())),
              "prior": (info.get("prior"), want_prior),
              "checkpoint": (info.get("checkpoint"), str(checkpoint)),
              "actors_sha256": (info.get("actors_sha256"), _sha256(Path(stage1) / "actors.pt")),
              "prior_sha256": (info.get("prior_sha256"), _sha256(want_prior))}
    diff = [f"{k}: {have!r} -> {want!r}" for k, (have, want) in checks.items() if have != want]
    return "; ".join(diff) or None


def prepare(latent_run: str, settings_run: str | None = None) -> dict:
    """What a MuJoCo eval needs for `--latent_run`: {kind, run_dir, step, checkpoint, settings, encoders (folder),
    decoders (folder)}. The deploy settings (gains, IMU model, rates, target clip, command filter) are an export_marl
    export's: B's own checkpoint's (its actors are B's encoders); C has no env.yaml, so a Pulse skrl run's
    (--settings_run, same env cfg). The decoders: export_decoders on the stage-1 run (B: its env's pulse_stage1_run and
    pulse_prior; C: its own, no prior, its encoders with them), in the isaac python ($LMG_ISAAC_PYTHON, which has
    pretraining), under logs/evals/<run>/mujoco/exports/."""
    import os
    import subprocess
    import sys

    from evals.common import outputs, runs
    from evals.isaac.latent import resolve_latent_run

    from .common import REPO
    from .policy import combined_settings, export_for

    r = resolve_latent_run(latent_run)
    root = outputs.eval_dir(runs.name(r.run_dir), "mujoco", "exports")
    if r.kind == "skrl":
        export = export_for(r.checkpoint, root)
        env = _run_env(r.run_dir)
        stage1 = Path(env["pulse_stage1_run"])
        stage1 = stage1 if stage1.is_absolute() else REPO / stage1
        prior = env.get("pulse_prior")
        prior = "none" if not prior else str(prior if Path(prior).is_absolute() else stage1 / prior)
        decoders, commands = root / "decoders", None
    else:
        if not settings_run:
            raise SystemExit("[evals] a train_commands run records no env.yaml: give --settings_run <a"
                             " LocoManip-Marl-Pulse-v0 skrl run> for the deploy settings (gains, IMU, rates, clip)")
        s = resolve_latent_run(settings_run)
        if s.kind != "skrl":
            raise SystemExit(f"[evals] --settings_run {settings_run}: not an skrl run of LocoManip-Marl-Pulse-v0")
        export = export_for(s.checkpoint, outputs.eval_dir(runs.name(s.run_dir), "mujoco", "exports"))
        stage1, prior = r.stage1, "none"
        decoders, commands = root / "decoders_commands", r.run_dir
    why = "the command encoders are re-exported each time" if commands is not None else \
        stale(decoders, stage1, prior, str(r.checkpoint))
    if why:
        print(f"[evals] exporting the decoders ({why})", flush=True)
        python = os.environ.get("LMG_ISAAC_PYTHON", ISAAC_PYTHON)
        cmd = [python, "-m", "pretraining.scripts.export_decoders", str(stage1), "--out", str(decoders),
               "--prior", prior, "--checkpoint", str(r.checkpoint)] + \
            (["--commands", str(commands)] if commands is not None else [])
        print(f"[evals] exporting the decoders: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=str(REPO), env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                       stdout=sys.stdout, stderr=sys.stderr)
    return {"kind": r.kind, "run_dir": r.run_dir, "step": r.step, "checkpoint": str(r.checkpoint),
            "settings": combined_settings(export, export), "settings_export": str(export),
            "encoders": str(export if r.kind == "skrl" else decoders), "decoders": str(decoders)}


def make_latent_sim(latent: tuple, settings: dict, plant: str, **kwargs) -> LatentMarlMujoco:
    """latent: (kind, encoders folder, decoders folder), as the scorecards' jobs carry it."""
    kind, enc_dir, dec_dir = latent
    return LatentMarlMujoco(LatentMujocoPolicy(dec_dir, encoders(kind, enc_dir)), settings, physics=plant, **kwargs)
