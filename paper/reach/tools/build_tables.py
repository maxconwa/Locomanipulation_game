"""Build the task's standing-reachable wrist-pose tables once, with a fixed seed, for the fixed goal grid.

    python paper/reach/tools/build_tables.py [--out logs/reach/goals] [--num_envs 1024] [--seed 0]

Instantiates LocoManip-Marl-Direct-v0 with log_dir=--out: ArmTargetsCommand builds its tables there exactly as at the
start of a training run (random arm joint angles within the soft limits, the rest of the robot standing at its
default joint angles, collision-free by the contact sensor, at least 0.1 m ahead of the pelvis, balanced over 10 cm
cells) and saves <out>/arm_target_tables_v3.pt. Writes <out>/tables_meta.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import os as _os
import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from reachlib.common import DEFAULT_ASSETS as _ASSETS, GOALS_DIR as _GOALS_DIR  # noqa: E402
_os.environ.setdefault("CL_ASSETS_DIR", str(_ASSETS))   # the task reads it when imported
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", default=str(_GOALS_DIR))
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--task", default="LocoManip-Marl-Direct-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reachlib.common import DEFAULT_CODE, git_info, sha256  # noqa: E402

out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.log_dir = str(out)
cfg.estimator.train = False
env = gym.make(args.task, cfg=cfg)
arm = env.unwrapped.command_manager.get_term("arm_targets")
table_file = out / arm.cfg.table_file
meta = {
    "table_file": table_file.name, "sha256": sha256(table_file), "seed": args.seed, "num_envs": args.num_envs,
    "rows": [len(t) for t in arm._tables], "standing_min_z": arm._standing_min_z.tolist(),
    "rest_pose_b": arm.rest_pose_b.tolist(), "settings": arm._table_settings(), "task": args.task,
    "code": {"path": str(DEFAULT_CODE), **git_info(DEFAULT_CODE)},
}
(out / "tables_meta.json").write_text(json.dumps(meta, indent=1))
print(f"[build_tables] {table_file}: {meta['rows']} rows, standing min z {meta['standing_min_z']}", flush=True)
sys.stdout.flush()
os._exit(0)
