"""Stage 1: train the legs and arms actors and the world model to track motion clips in the Isaac env.

  python -m pretraining.scripts.train --config pretraining/configs/base.yaml --run-name amass_s1 \\
      [--motion-dir data/motions/h12_amass] [--num-envs N] [--max-iterations N] [--max-seconds S] [--seed N]
  python -m pretraining.scripts.train --config pretraining/configs/golem_aware.yaml --run-name amass_s1_golem_aware \\
      --fine-tune-from <run dir>
  python -m pretraining.scripts.train --resume <run dir> [--max-iterations N] [--max-seconds S]

A config file has ``num_envs``, ``runner:`` (``TwoAgentRunnerCfg``'s fields) and ``rewards:``
(``ImitationRewardCfg``'s); an unknown key is an error. A new run goes to ``<runs dir>/<run name>``
(``$PRETRAINING_RUNS_DIR``, default ``data/pretraining/runs``) and an existing directory is refused. ``run.json``
records every setting, the config file's content, the env count, the motion files and the seed.

``--resume`` continues a run from its ``checkpoint.pt`` with its ``run.json``'s settings; only ``--max-iterations``
(the iteration to stop at) and ``--max-seconds`` are taken from the command line. ``--fine-tune-from`` starts a new
run directory from another run's checkpoint (world model, actors, critics, optimisers, normalisers, reset sampler)
with this config's rewards and settings, as h12_train made amass_s1_golem_aware: the source's iteration count, wall
time and ``progress.csv`` carry on, the config's ``max_iterations`` is the number of iterations to add, and
``run.json`` records ``fine_tuned_from`` and the start in ``resumed``.

``--max-seconds``: wall time of this invocation's training loop; when reached the run saves a checkpoint and exits 0.
The clock is checked after each iteration finishes, so a run can overrun it by up to one iteration (an update window
included). The first update's compile and CUDA-graph capture (about 8 minutes at full size) counts against it, and is
paid again on every resume (after the ``resume_warmup`` iterations of collection).

A fine-tune must use the source run's clips (the same resolved files in the same order): the reset sampler's saved
per-bin statistics belong to them, so other clips are refused.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import shutil
import sys
import traceback
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "pretraining" / "configs" / "base.yaml"
DEFAULT_MOTION_DIR = REPO / "data" / "motions" / "h12_amass"
TASK = "LocoManip-Pretrain-Track-v0"
CONFIG_KEYS = {"num_envs", "runner", "rewards"}
RESUME_OPTIONS = ("max_iterations", "max_seconds", "resume", "device")


class UsageError(Exception):
    """A bad command line or config: reported without a traceback, before the app starts."""


def build_parser() -> argparse.ArgumentParser:
    """The script's own arguments (``main`` adds the app launcher's, ``--device`` among them)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, default=None, help=f"default {DEFAULT_CONFIG.relative_to(REPO)}")
    p.add_argument("--motion-dir", type=Path, default=None, help=f"default {DEFAULT_MOTION_DIR.relative_to(REPO)}")
    p.add_argument("--run-name", default=None, help="default: the config file's name")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=None,
                   help="new run and --resume: the iteration to stop at (exclusive); --fine-tune-from: the number of "
                        "iterations to add to the source's count. Default: the config's (or run.json's) max_iterations")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--seed", type=int, default=None, help="default: drawn and recorded in run.json")
    p.add_argument("--fine-tune-from", type=Path, default=None, help="a stage-1 run directory to start from")
    p.add_argument("--resume", type=Path, default=None, help="a run directory to continue")
    return p


def _checkpoint(run_dir: Path) -> int:
    """Return the iteration of ``run_dir``'s checkpoint; refuse a directory without one."""
    from pretraining.marl.runs import read_checkpoint_info

    if not (run_dir / "checkpoint.pt").exists() or not (run_dir / "checkpoint.json").exists():
        raise UsageError(f"{run_dir} has no checkpoint.pt / checkpoint.json")
    return int(read_checkpoint_info(run_dir)["iteration"])


def plan(args: argparse.Namespace) -> dict:
    """Check the command line and config and return what ``train`` needs; no app, no GPU."""
    from pretraining.marl.config import ImitationRewardCfg
    from pretraining.marl.runner import TwoAgentRunnerCfg
    from pretraining.marl.runs import config_from_dict, from_repo_path

    if args.resume is not None:
        extra = [f"--{k.replace('_', '-')}" for k in vars(build_parser().parse_args([]))
                 if k not in RESUME_OPTIONS and getattr(args, k) is not None]
        if extra:
            raise UsageError(f"{', '.join(extra)} cannot be given with --resume: the run's run.json has its settings")
        run_dir = args.resume.resolve()
        if not (run_dir / "run.json").exists():
            raise UsageError(f"{run_dir} has no run.json")
        start = _checkpoint(run_dir) + 1
        meta = json.loads((run_dir / "run.json").read_text())
        cfg = config_from_dict(TwoAgentRunnerCfg, meta["runner"])
        cfg.runs_dir, cfg.run_name = str(run_dir.parent), run_dir.name
        if args.max_iterations is not None:
            cfg.max_iterations = args.max_iterations
        return {
            "mode": "resume", "run_dir": run_dir, "cfg": cfg, "start": start,
            "rewards": config_from_dict(ImitationRewardCfg, meta["rewards"]),
            "num_envs": int(meta["num_envs"]), "files": [str(from_repo_path(f)) for f in meta["motion_files"]],
        }

    config_path = (args.config or DEFAULT_CONFIG).resolve()
    if not config_path.is_file():
        raise UsageError(f"no config file {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    unknown = sorted(set(config) - CONFIG_KEYS)
    if unknown:
        raise UsageError(f"{config_path}: unknown key(s) {', '.join(unknown)} (expected {sorted(CONFIG_KEYS)})")
    try:
        cfg = config_from_dict(TwoAgentRunnerCfg, config.get("runner") or {}, strict=True, _where="runner.")
        rewards = config_from_dict(ImitationRewardCfg, config.get("rewards") or {}, strict=True, _where="rewards.")
    except (ValueError, TypeError) as e:
        raise UsageError(f"{config_path}: {e}") from e
    cfg.run_name = args.run_name or config_path.stem
    run_dir = (Path(cfg.runs_dir) / cfg.run_name).resolve()
    if run_dir.exists():
        raise UsageError(f"{run_dir} exists; choose another --run-name, or continue it with --resume {run_dir}")
    if args.max_iterations is not None:
        cfg.max_iterations = args.max_iterations
    cfg.seed = args.seed if args.seed is not None else (cfg.seed if cfg.seed is not None else random.randrange(2**31))
    motion_dir = (args.motion_dir or DEFAULT_MOTION_DIR).resolve()
    files = sorted(str(f) for f in motion_dir.glob("*.npz"))
    if not files:
        raise UsageError(f"no .npz clips in {motion_dir}")
    out = {
        "mode": "new", "run_dir": run_dir, "cfg": cfg, "rewards": rewards, "start": 0, "config": config,
        "num_envs": args.num_envs or int(config.get("num_envs", 4096)), "files": files,
    }
    if args.fine_tune_from is not None:
        source = args.fine_tune_from.resolve()
        out["start"] = _checkpoint(source) + 1
        if not (source / "run.json").exists():
            raise UsageError(f"{source} has no run.json")
        theirs = [str(from_repo_path(f).resolve()) for f in json.loads((source / "run.json").read_text())["motion_files"]]
        ours = [str(Path(f).resolve()) for f in files]
        if ours != theirs:
            first = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), min(len(ours), len(theirs)))
            raise UsageError(
                f"--motion-dir {motion_dir} gives {len(ours)} clips, {source} was trained on {len(theirs)}; they first "
                f"differ at position {first}. A fine-tune needs the source run's clips (its reset sampler's state)"
            )
        cfg.max_iterations += out["start"]  # the config's count is the fine-tune's own; the run's carries on
        out.update(mode="fine_tune", source=source)
    return out


def train(p: dict, max_seconds: float | None, device: str) -> Path:
    """Build the env and runner, train, and return the run directory. Needs the app."""
    import gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry

    import pretraining.tasks  # noqa: F401  (registers the gym ids)
    from pretraining.marl.runner import ImitationRunner
    from pretraining.marl.runs import atomic_write_text, to_repo_path
    from pretraining.supermimic.runners.base_runner import EnvAdapter

    os.environ.setdefault("WANDB_MODE", "offline")
    cfg, run_dir = p["cfg"], p["run_dir"]
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = p["num_envs"]
    env_cfg.sim.device = device
    env_cfg.seed = cfg.seed + p["start"]  # a resumed segment seeds with seed + its first iteration
    env_cfg.commands.motion.motion_file = p["files"]
    env_cfg.commands.motion.sampling_mode = "external" if cfg.sampling_mode == "return" else cfg.sampling_mode
    env = EnvAdapter(gym.make(TASK, cfg=env_cfg).unwrapped)
    runner = ImitationRunner(env, cfg, rewards=p["rewards"], run_dir=run_dir, motion_files=p["files"], device=device)
    if p["mode"] == "resume":
        runner.load_checkpoint(run_dir / "checkpoint.pt")
    elif p["mode"] == "fine_tune":
        source = p["source"]
        runner.load_checkpoint(source / "checkpoint.pt")
        run_dir.mkdir(parents=True)
        if (source / "progress.csv").exists():
            shutil.copy2(source / "progress.csv", run_dir / "progress.csv")  # the curves carry on
        run_json = {
            "stage": runner.stage, "runner": dataclasses.asdict(cfg), **runner._run_meta(), "config": p["config"],
            "fine_tuned_from": f"{to_repo_path(source)} at iteration {p['start'] - 1}",
        }
        atomic_write_text(run_dir / "run.json", json.dumps(run_json, indent=2, default=str))
    # this invocation's wall time: the runner counts the run's, summed over its segments
    runner.cfg.max_seconds = max_seconds + runner._wall_offset if max_seconds else 0.0
    print(f"[train] run directory: {run_dir}", flush=True)
    meta = {**runner._run_meta(), "config": p["config"]} if p["mode"] == "new" else None
    return runner.run(meta, resume=p["mode"] != "new")


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        p = plan(args)
    except UsageError as e:
        print(f"[train] error: {e}", file=sys.stderr)
        sys.exit(2)
    args.headless = True
    app = AppLauncher(args).app
    try:
        log_dir = train(p, args.max_seconds, args.device or "cuda:0")
        print(f"[train] done: {log_dir}")
    except BaseException:
        # A failed run (a CUDA device-side assert in particular) can leave the app unable to shut down: the
        # process then spins with the GPU idle. Report and exit without closing it.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()  # closing the app ends the process without flushing
    sys.stderr.flush()
    app.close()


if __name__ == "__main__":
    main()
