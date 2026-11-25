"""Export a stage-1 run's frozen decoders (and its latent prior, and optionally a train_commands run's command
encoders) to TorchScript, for latent policies outside Isaac (evals/mujoco/latent.py: the MuJoCo scorecards).

    python -m pretraining.scripts.export_decoders <stage-1 run> --out <dir under data/ or logs/>
        [--prior <file>|none] [--commands <train_commands run>]

Writes into --out:
  legs_decoder.pt, upper_decoder.pt   forward(z (n, 32), frame (n, 87)) -> the decoder's raw output (n, 12) / (n, 15):
                                      the actor's observation normaliser on the one-frame history, then its decoder
                                      on [z, normalised frame] (pulse.PulseDecoders.step without the tanh, the prior
                                      and the target: the caller's)
  prior.pt                            forward(frame (n, 87)) -> {legs, arms} pre-tanh prior means as a (2, n, 32)
                                      tensor, legs first (the run's prior.pt, or --prior; none: not written)
  legs_commands.pt, arms_commands.pt  with --commands: forward(obs, command, hc (n, 2, 64)) -> (z, hc), the
                                      RecurrentCommandActor as trained
  plant.json                          the stage-1 run's plant: joint order (Unitree motor order), default pose, action
                                      scales, target limits, policy step; the stage-1 run, the prior and the commands
                                      run these came from; the commands' observation and command widths

The decoders and encoders are AMASS-derived: --out must be under the repository's data/ or logs/ (untracked).
Imports torch and pretraining only; no simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
UNTRACKED = ("data", "logs")


class Decoder(nn.Module):
    """One agent's frozen decoder on a one-frame history: frame -> normalised -> decoder([z, hist])."""

    def __init__(self, actor: nn.Module):
        super().__init__()
        self.norm = actor.obs_normalizer
        self.decoder = actor.decoder

    def forward(self, z: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        hist = self.norm(frame.unsqueeze(1)).flatten(1)
        return self.decoder(torch.cat([z, hist], dim=-1))


class Prior(nn.Module):
    """Both agents' latent priors, each on its own actor's normalised frame: (2, n, latent), legs first."""

    def __init__(self, actors: dict, priors: dict):
        super().__init__()
        self.legs_norm, self.arms_norm = actors["legs"].obs_normalizer, actors["arms"].obs_normalizer
        self.legs, self.arms = priors["legs"], priors["arms"]

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        x = frame.unsqueeze(1)
        return torch.stack([self.legs(self.legs_norm(x).flatten(1)), self.arms(self.arms_norm(x).flatten(1))])


def to_torchscript(module: nn.Module, example: tuple) -> torch.jit.ScriptModule:
    """Scripted when the module scripts (rsl_rl's normaliser may not), else traced on `example` (the modules here
    have no data-dependent control flow in eval mode)."""
    module = module.eval()
    try:
        return torch.jit.script(module)
    except Exception:  # noqa: BLE001
        with torch.no_grad():
            return torch.jit.trace(module, example, check_trace=False)


def check_out(out: Path) -> Path:
    out = out.resolve()
    if not any(out.is_relative_to(REPO / d) for d in UNTRACKED):
        raise SystemExit(f"--out {out}: the decoders are AMASS-derived; write them under {REPO}/data or {REPO}/logs")
    return out


def export(stage1: Path, out: Path, prior: Path | str | None = "auto", commands: Path | None = None) -> dict:
    """Write the files (module docstring) and return plant.json's content. prior: "auto" the run's prior.pt if it
    has one, None no prior, else that file."""
    from pretraining.constants import JOINT_NAMES
    from pretraining.marl.actors import load_latent_prior
    from pretraining.marl.config import AGENTS
    from pretraining.marl.pulse import check_prior_plant
    from pretraining.marl.runner import PROPRIO_DIM, load_actors
    from pretraining.plants import get_plant

    stage1, out = Path(stage1).resolve(), check_out(Path(out))
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((stage1 / "run.json").read_text()) if (stage1 / "run.json").is_file() else {}
    plant = get_plant(meta.get("plant", "pretrain"))
    actors = load_actors(stage1, "cpu")
    for agent in AGENTS:
        if actors[agent].history_length != 1:
            raise SystemExit(f"{stage1}: the {agent} actor reads {actors[agent].history_length} frames; only 1")
    latent = actors["legs"].latent_dim
    frame, z = torch.randn(4, PROPRIO_DIM), torch.randn(4, latent)
    for agent, name in (("legs", "legs_decoder.pt"), ("arms", "upper_decoder.pt")):
        to_torchscript(Decoder(actors[agent]), (z, frame)).save(str(out / name))
    if prior == "auto":
        prior = stage1 / "prior.pt" if (stage1 / "prior.pt").is_file() else None
    if prior is not None:
        check_prior_plant(prior, plant)
        to_torchscript(Prior(actors, load_latent_prior(prior, "cpu")), (frame,)).save(str(out / "prior.pt"))
    info = {
        "plant": plant.name, "step_dt": plant.step_dt, "latent_dim": latent, "proprio_dim": PROPRIO_DIM,
        "joint_names": list(JOINT_NAMES),
        "default_joint_pos": [plant.default_joint_pos[j] for j in JOINT_NAMES],
        "action_scale": [plant.action_scale[j] for j in JOINT_NAMES],
        "target_limits": [list(plant.target_limits[j]) for j in JOINT_NAMES],
        "agent_joints": {"legs": [0, 12], "arms": [12, 27]},
        "stage1": str(stage1), "prior": str(prior) if prior is not None else None,
    }
    if commands is not None:
        from pretraining.marl.command_runner import load_commands

        bundle = torch.load(Path(commands) / "commands.pt", map_location="cpu", weights_only=False)
        if Path(bundle.get("stage1", "")).name != stage1.name:
            print(f"[export_decoders] WARNING: {commands} trained on stage 1 {bundle.get('stage1')}, not {stage1}")
        encoders = load_commands(Path(commands) / "commands.pt", "cpu")
        for agent, enc in encoders.items():
            ex = (torch.randn(4, enc.obs_dim), torch.randn(4, enc.command_dim), enc.initial_state(4, "cpu"))
            to_torchscript(enc.requires_grad_(False), ex).save(str(out / f"{agent}_commands.pt"))
        info["commands"] = {"run": str(Path(commands).resolve()), "obs_dims": dict(bundle["obs_dims"]),
                            "command_dims": dict(bundle["command_dims"]), "lstm": int(bundle["lstm"])}
    (out / "plant.json").write_text(json.dumps(info, indent=1))
    print(f"[export_decoders] {stage1} ({plant.name} plant) -> {out}: decoders"
          f"{', prior' if prior is not None else ''}{', command encoders' if commands is not None else ''}")
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage1")
    p.add_argument("--out", required=True)
    p.add_argument("--prior", default=None, help="A fit_prior.py file (default: the run's prior.pt); none: no prior.")
    p.add_argument("--commands", default=None, help="A train_commands run whose encoders to export too.")
    a = p.parse_args(argv)
    prior = "auto" if a.prior is None else None if a.prior == "none" else Path(a.prior)
    export(Path(a.stage1), Path(a.out), prior, Path(a.commands) if a.commands else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
