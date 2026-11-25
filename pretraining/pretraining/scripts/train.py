"""Stage 1: train the legs and arms actors and the world model to track motion clips in the Isaac env.

  python -m pretraining.scripts.train --config pretraining/configs/base.yaml --run-name amass_s1 \\
      [--motion-dir data/motions/h12_amass] [--num-envs N] [--max-iterations N] [--max-seconds S] [--seed N]
  python -m pretraining.scripts.train --config pretraining/configs/golem_aware.yaml --run-name amass_s1_golem_aware \\
      --fine-tune-from <run dir>
  python -m pretraining.scripts.train --resume <run dir> [--max-iterations N] [--max-seconds S]
  python -m pretraining.scripts.train --config pretraining/configs/main_plant.yaml --plant main --run-name <name>
  python -m pretraining.scripts.train --config pretraining/configs/main_plant_finetune.yaml --plant main --fine-tune \\
      --run-name <name> --fine-tune-from <main-plant run dir>

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

``--plant`` picks the plant a new run trains on (``plants``): ``pretrain`` (the default; ``TASK``) or ``main``
(``TASK_MAIN``, or ``TASK_MAIN_FINE_TUNE``, main's domain randomization, with ``--fine-tune``). ``run.json`` records
``plant`` and ``task``; ``--resume`` reads both from it (a run.json without them is a pretrain-plant run), and
``--fine-tune-from`` takes the source run's plant and refuses another.

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
TASK_MAIN = "LocoManip-Pretrain-Track-Main-v0"
TASK_MAIN_FINE_TUNE = "LocoManip-Pretrain-Track-Main-FineTune-v0"
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
    p.add_argument("--resume", type=Path, default=None, help="a run directory to continue. Without --max-seconds it runs on to "
                        "--max-iterations or run.json's max_iterations, whatever wall-time budget the run had")
    p.add_argument("--plant", choices=("pretrain", "main"), default=None,
                   help="a new run's plant (default pretrain; a fine-tune: its source's)")
    p.add_argument("--fine-tune", action="store_true", default=None,
                   help="with --plant main: the env with main's domain randomization (TASK_MAIN_FINE_TUNE)")
    return p


def _plant_of(meta: dict) -> tuple[str, str]:
    """(plant, task) a run.json records; one written before they were recorded is a pretrain-plant run."""
    return meta.get("plant", "pretrain"), meta.get("task", TASK)


def run_json_entries(p: dict) -> dict:
    """The entries ``train`` adds to a new or fine-tuned run's run.json."""
    return {"plant": p["plant"], "task": p["task"]}


def source_sim_context_layout(run_dir: Path) -> list[dict]:
    """The sim-context layout of ``run_dir``'s world model: run.json's ``sim_context``, or for a run from before it
    was recorded, inferred from its saved normalisation (``world_model.pt``, else ``checkpoint.pt``)."""
    import torch

    from pretraining.constants import JOINT_NAMES
    from pretraining.supermimic.sim_context import layout_from_bundle

    meta = json.loads((run_dir / "run.json").read_text())
    if "sim_context" in meta:
        return meta["sim_context"]
    if (run_dir / "world_model.pt").exists():
        bundle = torch.load(run_dir / "world_model.pt", map_location="cpu", weights_only=False)
    else:
        bundle = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)["world_model"]
    return layout_from_bundle(bundle["sim_context"], len(JOINT_NAMES))


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
        plant, task = _plant_of(meta)
        cfg.runs_dir, cfg.run_name = str(run_dir.parent), run_dir.name
        if args.max_iterations is not None:
            cfg.max_iterations = args.max_iterations
        return {
            "mode": "resume", "run_dir": run_dir, "cfg": cfg, "start": start, "plant": plant, "task": task,
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
    plant = args.plant
    out = {
        "mode": "new", "run_dir": run_dir, "cfg": cfg, "rewards": rewards, "start": 0, "config": config,
        "num_envs": args.num_envs or int(config.get("num_envs", 4096)), "files": files,
    }
    if args.fine_tune_from is not None:
        source = args.fine_tune_from.resolve()
        out["start"] = _checkpoint(source) + 1
        if not (source / "run.json").exists():
            raise UsageError(f"{source} has no run.json")
        source_meta = json.loads((source / "run.json").read_text())
        source_plant = _plant_of(source_meta)[0]
        if plant is not None and plant != source_plant:
            raise UsageError(f"--plant {plant}, but {source} was trained on the {source_plant} plant")
        plant = source_plant
        theirs = [str(from_repo_path(f).resolve()) for f in source_meta["motion_files"]]
        ours = [str(Path(f).resolve()) for f in files]
        if ours != theirs:
            first = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), min(len(ours), len(theirs)))
            raise UsageError(
                f"--motion-dir {motion_dir} gives {len(ours)} clips, {source} was trained on {len(theirs)}; they first "
                f"differ at position {first}. A fine-tune needs the source run's clips (its reset sampler's state)"
            )
        cfg.max_iterations += out["start"]  # the config's count is the fine-tune's own; the run's carries on
        out.update(mode="fine_tune", source=source)
    plant = plant or "pretrain"
    if args.fine_tune and plant != "main":
        raise UsageError("--fine-tune picks main's fine-tune env: it needs the main plant (--plant main)")
    out.update(plant=plant, task=(TASK_MAIN_FINE_TUNE if args.fine_tune else TASK_MAIN) if plant == "main" else TASK)
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
    env_cfg = load_cfg_from_registry(p["task"], "env_cfg_entry_point")
    env_cfg.scene.num_envs = p["num_envs"]
    env_cfg.sim.device = device
    env_cfg.seed = cfg.seed + p["start"]  # a resumed segment seeds with seed + its first iteration
    env_cfg.commands.motion.motion_file = p["files"]
    env_cfg.commands.motion.sampling_mode = "external" if cfg.sampling_mode == "return" else cfg.sampling_mode
    env_plant = getattr(env_cfg, "plant", "pretrain")
    if env_plant != p["plant"]:
        raise RuntimeError(f"{p['task']} is on the {env_plant} plant, the run's is {p['plant']}")
    env = EnvAdapter(gym.make(p["task"], cfg=env_cfg).unwrapped)
    # a fine-tune or resume keeps its source's sim-context columns (the world model's and critics' input)
    source = {"resume": run_dir, "fine_tune": p.get("source")}.get(p["mode"])
    layout = source_sim_context_layout(source) if source is not None else None
    # (a fine-tune's: then this env's new columns appended, its first layers widened at load: SimContext.from_env)
    runner = ImitationRunner(env, cfg, rewards=p["rewards"], run_dir=run_dir, motion_files=p["files"], device=device,
                             sim_context_layout=layout, sim_context_fine_tune=p["mode"] == "fine_tune")
    if p["mode"] == "resume":
        runner.load_checkpoint(run_dir / "checkpoint.pt")
    elif p["mode"] == "fine_tune":
        runner.load_checkpoint(p["source"] / "checkpoint.pt")
    # this invocation's wall time: the runner counts the run's, summed over its segments; set before run.json is
    # written, so that it records the budget used
    runner.cfg.max_seconds = max_seconds + runner._wall_offset if max_seconds else 0.0
    if p["mode"] == "fine_tune":
        run_dir.mkdir(parents=True)
        if (source / "progress.csv").exists():
            shutil.copy2(source / "progress.csv", run_dir / "progress.csv")  # the curves carry on
        run_json = {
            "stage": runner.stage, "runner": dataclasses.asdict(runner.cfg), **runner._run_meta(), "config": p["config"],
            "fine_tuned_from": f"{to_repo_path(source)} at iteration {p['start'] - 1}", **run_json_entries(p),
            "sim_context": runner._sim_context.layout(),
        }
        atomic_write_text(run_dir / "run.json", json.dumps(run_json, indent=2, default=str))
    print(f"[train] run directory: {run_dir}", flush=True)
    meta = {**runner._run_meta(), "config": p["config"], **run_json_entries(p),
            "sim_context": runner._sim_context.layout()} if p["mode"] == "new" else None
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
