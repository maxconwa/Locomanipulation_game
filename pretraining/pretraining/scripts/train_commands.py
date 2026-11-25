# Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c),
# src/h12_train/scripts/train_commands.py. Changes: argparse and a yaml config (configs/commands.yaml) as train.py, not
# tyro; the env is main's LocoManip-Marl-Pulse-v0 (no prior) on the stage-1 run's plant; run.json records the stage-1
# run, plant, task, settings, seed and env count; the step and wall-time caps; leaves with os._exit.
"""Stage 2 through the world model (arm C): train a command encoder per agent into a stage-1 run's frozen decoders.

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m pretraining.scripts.train_commands \\
      --stage1 <stage-1 run dir> --run-name <name> [--num-envs N] [--max-steps 40032] [--max-seconds S] [--seed N]
  python -m pretraining.scripts.train_commands --resume <run dir> [--max-steps N] [--max-seconds S]

Data comes from ``LocoManip-Marl-Pulse-v0`` built on the stage-1 run (``pulse_stage1_run``) without a latent prior,
so the env decodes the encoders' latents with the same decoders the world model's imagined rollouts use
(``marl/command_runner.py``). The stage-1 run must be on main's plant (the env's): another is refused.

A new run goes to ``<runs dir>/<run name>`` (``$PRETRAINING_RUNS_DIR``, default ``data/pretraining/runs``); an existing
directory is refused. ``run.json`` records the stage-1 run, the plant, the task, every setting, the seed and the env
count used. ``commands.pt`` (both encoders and the stage-1 run they decode through), ``world_model.pt`` and
``checkpoint.pt`` are written every ``save_interval`` iterations, every ``checkpoint_interval_s`` and at the end.

The run stops at ``--max-steps`` sim steps per env (40,032 = 1,668 iterations of 24 steps, at 100 Hz about 400 s of
robot time per env), at ``--max-seconds`` of this invocation's wall time, or at ``--max-iterations``, whichever comes
first; each is checked after an iteration. ``--resume`` continues from ``checkpoint.pt`` with run.json's settings
(only the caps and ``--device`` are taken from the command line); the replay buffer is refilled first.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "pretraining" / "configs" / "commands.yaml"
TASK = "LocoManip-Marl-Pulse-v0"
RESUME_OPTIONS = ("resume", "max_steps", "max_seconds", "max_iterations", "device")
# command-line overrides of the config: option -> (section, key)
OVERRIDES = {
    "rollout": (None, "rollout"), "buffer_size": (None, "buffer_size"), "replay_ratio": (None, "replay_ratio"),
    "wm_batch_size": ("wm", "batch_size"), "wm_horizon": ("wm", "horizon"),
    "policy_batch_size": ("policy", "batch_size"), "policy_horizon": ("policy", "horizon"),
    "max_clips": (None, "max_clips"), "max_steps": (None, "max_steps"), "max_iterations": (None, "max_iterations"),
}


class UsageError(Exception):
    """A bad command line or config: reported without a traceback, before the app starts."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage1", type=Path, default=None, help="the stage-1 run directory (main's plant)")
    p.add_argument("--run-name", default=None, help="default: the config file's name")
    p.add_argument("--config", type=Path, default=None, help=f"default {DEFAULT_CONFIG.relative_to(REPO)}")
    p.add_argument("--resume", type=Path, default=None, help="a stage-2 run directory to continue")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None, help="default: drawn and recorded in run.json")
    p.add_argument("--max-seconds", type=float, default=None, help="wall time of this invocation")
    p.add_argument("--max-steps", type=int, default=None, help="sim steps per env to stop at (40032 in the config)")
    p.add_argument("--max-iterations", type=int, default=None)
    for opt, kind in (("rollout", int), ("buffer_size", int), ("replay_ratio", float), ("wm_batch_size", int),
                      ("wm_horizon", int), ("policy_batch_size", int), ("policy_horizon", int), ("max_clips", int)):
        p.add_argument(f"--{opt.replace('_', '-')}", type=kind, default=None)
    return p


def _checkpoint(run_dir: Path) -> dict:
    from pretraining.marl.runs import read_checkpoint_info

    if not (run_dir / "checkpoint.pt").exists() or not (run_dir / "checkpoint.json").exists():
        raise UsageError(f"{run_dir} has no checkpoint.pt / checkpoint.json")
    return read_checkpoint_info(run_dir)


def _apply_overrides(cfg, args) -> None:
    for opt, (section, key) in OVERRIDES.items():
        value = getattr(args, opt, None)
        if value is not None:
            setattr(getattr(cfg, section) if section else cfg, key, value)


def plan(args: argparse.Namespace) -> dict:
    """Check the command line and config; return what ``train`` needs. No app, no GPU."""
    from pretraining.marl.command_runner import CommandRunnerCfg
    from pretraining.marl.runs import config_from_dict, default_runs_dir, from_repo_path

    if args.resume is not None:
        extra = [f"--{k.replace('_', '-')}" for k in vars(build_parser().parse_args([]))
                 if k not in RESUME_OPTIONS and getattr(args, k) is not None]
        if extra:
            raise UsageError(f"{', '.join(extra)} cannot be given with --resume: the run's run.json has its settings")
        run_dir = args.resume.resolve()
        if not (run_dir / "run.json").exists():
            raise UsageError(f"{run_dir} has no run.json")
        info = _checkpoint(run_dir)
        meta = json.loads((run_dir / "run.json").read_text())
        cfg = config_from_dict(CommandRunnerCfg, meta["cfg"])
        _apply_overrides(cfg, args)
        return {"mode": "resume", "run_dir": run_dir, "cfg": cfg, "stage1": from_repo_path(meta["stage1"]),
                "plant": meta["plant"], "task": meta.get("task", TASK), "num_envs": int(meta["num_envs"]),
                "max_seconds": args.max_seconds, "checkpoint": info}
    if args.stage1 is None:
        raise UsageError("--stage1 is required for a new run")
    stage1 = args.stage1.resolve()
    if not (stage1 / "run.json").exists() or not (stage1 / "actors.pt").exists() \
            or not (stage1 / "world_model.pt").exists():
        raise UsageError(f"{stage1} is not a stage-1 run (run.json, actors.pt, world_model.pt)")
    plant = json.loads((stage1 / "run.json").read_text()).get("plant", "pretrain")
    if plant != "main":
        raise UsageError(f"{stage1} was trained on the {plant} plant; {TASK} is on main's")
    config_path = (args.config or DEFAULT_CONFIG).resolve()
    if not config_path.is_file():
        raise UsageError(f"no config file {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    body = {k: v for k, v in config.items() if k != "num_envs"}
    try:
        cfg = config_from_dict(CommandRunnerCfg, body, strict=True)
    except (ValueError, TypeError) as e:
        raise UsageError(f"{config_path}: {e}") from e
    _apply_overrides(cfg, args)
    run_name = args.run_name or config_path.stem
    run_dir = (Path(default_runs_dir()) / run_name).resolve()
    if run_dir.exists():
        raise UsageError(f"{run_dir} exists; choose another --run-name, or continue it with --resume {run_dir}")
    cfg.seed = args.seed if args.seed is not None else random.randrange(2**31)
    return {"mode": "new", "run_dir": run_dir, "cfg": cfg, "stage1": stage1, "plant": plant, "task": TASK,
            "num_envs": args.num_envs or int(config.get("num_envs", 4096)), "max_seconds": args.max_seconds,
            "config": config}


def train(p: dict, device: str) -> Path:
    """Build the Pulse env, the source and the runner, and train. Needs the app."""
    import gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry

    import locomanipulation_game.tasks  # noqa: F401  (registers the gym ids)
    from pretraining.marl.command_runner import CommandRunner
    from pretraining.marl.marl_source import MarlEnvSource
    from pretraining.marl.runs import to_repo_path

    cfg = p["cfg"]
    start = p["checkpoint"]["iteration"] + 1 if p["mode"] == "resume" else 0
    env_cfg = load_cfg_from_registry(p["task"], "env_cfg_entry_point")
    env_cfg.scene.num_envs = p["num_envs"]
    env_cfg.sim.device = device
    env_cfg.seed = cfg.seed + start  # a resumed segment seeds with seed + its first iteration
    env_cfg.pulse_stage1_run, env_cfg.pulse_prior = str(p["stage1"]), ""
    env = gym.make(p["task"], cfg=env_cfg)
    source = MarlEnvSource(env)
    runner = CommandRunner(source, p["stage1"], cfg, run_dir=p["run_dir"])
    if p["mode"] == "resume":
        runner.load_checkpoint(p["run_dir"] / "checkpoint.pt")
    # this invocation's wall time: the runner counts the run's, over its segments
    runner.cfg.max_seconds = p["max_seconds"] + runner._wall_offset if p["max_seconds"] else 0.0
    meta = None
    if p["mode"] == "new":
        meta = {"stage1": to_repo_path(p["stage1"]), "plant": p["plant"], "task": p["task"], "seed": cfg.seed,
                "num_envs": source.num_envs, "config": p["config"]}
    print(f"[train_commands] run directory: {p['run_dir']}", flush=True)
    return runner.run(meta, resume=p["mode"] == "resume")


def main() -> None:
    # before torch touches the GPU: the policy update's activations fragment the allocator
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        p = plan(args)
    except UsageError as e:
        print(f"[train_commands] error: {e}", file=sys.stderr)
        sys.exit(2)
    args.headless = True
    AppLauncher(args)
    try:
        log_dir = train(p, args.device or "cuda:0")
        print(f"[train_commands] done: {log_dir}", flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    # the app's close has hung after a run: everything is written, leave without it
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
