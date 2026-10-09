"""Evaluate every run named in conditions.yaml that has no current results, then rebuild every paper output.

    python paper/reach/tools/run_all.py [--device cuda:0] [--seeds 0,1,2] [--no-mujoco] [--dry-run]
    make -C paper/reach all            # the same with the defaults

For each condition whose run directory holds its checkpoint (conditions.yaml: run, optional checkpoint, default the
newest checkpoints/agent_<N>.pt):
    Isaac   full and legs_blind on --seeds, arms_ik on the first seed, and velocity tracking (eval_walk.py)
    MuJoCo  full with RoboCasa's joint dynamics and with the training plant's, legs_blind with RoboCasa's
A one-agent whole-body checkpoint (LocoManip-WholeBody-Direct-v0) skips legs_blind, which needs a separate leg actor.
A result is current when results/<label>/<sim>/meta.json names the same checkpoint file hash; anything else is
(re)run. Then analyze.py, the robot renders, figures.py and build_page.py. Isaac jobs go through the workstation's
resguard memory guard when it exists.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REACH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REACH))
from reachlib import common as C  # noqa: E402

GUARD = Path.home() / ".claude" / "bin" / "resguard.sh"


def checkpoint_of(cond: dict) -> Path | None:
    run = cond.get("run")
    if not run or "REPLACE" in run or not Path(run).is_dir():
        return None
    if cond.get("checkpoint"):
        p = Path(run) / "checkpoints" / cond["checkpoint"]
        return p if p.is_file() else None
    return C.newest_checkpoint(Path(run))


def current(label: str, sim: str, ckpt: Path) -> bool:
    meta = C.RESULTS_DIR / label / sim / "meta.json"
    if not meta.is_file() or not (meta.parent / "trials.csv").is_file():
        return False
    try:
        return json.loads(meta.read_text()).get("checkpoint_sha256") == C.sha256(ckpt)
    except (json.JSONDecodeError, OSError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0", help="Isaac physics device; cpu when another job holds the GPU")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--workers", type=int, default=8, help="MuJoCo worker processes")
    p.add_argument("--no-mujoco", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--py", default=sys.executable, help="python with Isaac Lab (and MuJoCo, torch)")
    a = p.parse_args()
    first = a.seeds.split(",")[0]
    guard = [str(GUARD), "run", "--mem", "14G", "--cpu", "800", "--"] if GUARD.is_file() else []
    guard_mj = [str(GUARD), "run", "--mem", "10G", "--cpu", str(100 * (a.workers + 1)), "--"] if GUARD.is_file() else []
    from reachlib.policy import checkpoint_agents
    jobs = []
    for cond in C.load_conditions():
        ckpt = checkpoint_of(cond)
        if ckpt is None:
            print(f"[run_all] {cond['label']}: no checkpoint on disk ({cond.get('run') or 'run not set'}), skipped")
            continue
        label = cond["label"]
        if cond.get("evaluate") is False:
            print(f"[run_all] {label}: evaluate: false, existing results only")
            continue
        whole = checkpoint_agents(str(ckpt)) == ["whole"]
        isaac = [("isaac", ["--seeds", a.seeds]), ("isaac_legs_blind", ["--seeds", a.seeds, "--variant", "legs_blind"]),
                 ("isaac_arms_ik", ["--seeds", first, "--variant", "arms_ik"])]
        if whole:
            isaac = [j for j in isaac if "legs_blind" not in j[0]]
        for sim, extra in isaac:
            if not current(label, sim, ckpt):
                jobs.append(guard + [a.py, str(REACH / "tools" / "eval_isaac.py"), "--checkpoint", str(ckpt),
                                     "--label", label, "--device", a.device, *extra])
        walk = C.RESULTS_DIR / label / "isaac_walk" / "meta.json"
        if not (walk.is_file() and json.loads(walk.read_text()).get("checkpoint_sha256") == C.sha256(ckpt)):
            jobs.append(guard + [a.py, str(REACH / "tools" / "eval_walk.py"), "--checkpoint", str(ckpt), "--label",
                                 label, "--device", a.device])
        if not a.no_mujoco:
            mj = [("mujoco", []), ("mujoco_isaacphys", ["--physics", "isaac"])] + \
                ([] if whole else [("mujoco_legs_blind", ["--variant", "legs_blind"])])
            for sim, extra in mj:
                if not current(label, sim, ckpt):
                    jobs.append(guard_mj + [a.py, str(REACH / "tools" / "eval_mujoco.py"), "--checkpoint", str(ckpt),
                                            "--label", label, "--seeds", first, "--workers", str(a.workers), *extra])
    tail = [[a.py, str(REACH / "tools" / "analyze.py")],
            [a.py, str(REACH / "tools" / "training_curves.py")],
            [a.py, str(REACH / "tools" / "render_robot.py"), "--pose", "standing", "--out",
             str(C.FIG_DIR / "render" / "render_standing.png")],
            [a.py, str(REACH / "tools" / "render_robot.py"), "--auto", "--out",
             str(C.FIG_DIR / "render" / "render_crouch.png")],
            [a.py, str(REACH / "tools" / "figures.py")],
            [a.py, str(REACH / "tools" / "build_page.py")]]
    print(f"[run_all] {len(jobs)} evaluations to run")
    for cmd in jobs + tail:
        print("[run_all] $", " ".join(cmd), flush=True)
        if a.dry_run:
            continue
        r = subprocess.run(cmd, cwd=REACH)
        if r.returncode != 0 and cmd not in tail[3:4]:      # the crouch render may have nothing to show yet
            print(f"[run_all] exit {r.returncode}: {' '.join(cmd)}", flush=True)


if __name__ == "__main__":
    main()
