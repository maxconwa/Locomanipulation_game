"""Learning curves: every saved checkpoint of a run on the unextended goals, with its curriculum levels.

    python paper/reach/tools/eval_curve.py --run <run dir> --label l1 [--every 1] [--device cuda:0] [--seeds 0]
    make -C paper/reach curve RUN=<run dir> LABEL=l1

For each checkpoints/agent_<N>.pt (every --every-th, the newest always): eval_isaac.py on the 1056 goals without
extension, one seed, as results/<label>@<N>/isaac/ (skipped when already evaluated for the same file). Then
results/curves/<label>.csv, one row per checkpoint: environment transitions (N policy steps x the run's num_envs),
strict success over all goals and over those below / above the standing table, the median closest approach and
1 s hold error, and the curriculum levels saved beside the checkpoint (estimator_<N>.pt): mean k_d, its quartiles
and the share of environments at the deepest level. figures.py draws learning_curve from these files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REACH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REACH))
from reachlib import common as C  # noqa: E402

GUARD = Path.home() / ".claude" / "bin" / "resguard.sh"


def levels(run: Path, step: int) -> dict:
    import torch
    path = run / "estimator" / f"estimator_{step}.pt"
    if not path.is_file():
        return {}
    state = torch.load(path, map_location="cpu", weights_only=False).get("env_state", {})
    lv = state.get("arm_targets", {}).get("level")
    if lv is None:
        return {}
    lv = lv.numpy().astype(float)
    return {"kd_mean": float(lv.mean()), "kd_q25": float(np.percentile(lv, 25)), "kd_q75": float(np.percentile(lv, 75)),
            "kd_top_share": float((lv >= lv.max()).mean()) if lv.size else float("nan"),
            "arm_goals_enabled": bool(state.get("arm_goals_enabled", False))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", default="0")
    p.add_argument("--py", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    run = Path(a.run).resolve()
    ckpts = sorted(run.glob("checkpoints/agent_*.pt"), key=lambda q: int(re.search(r"_(\d+)\.pt$", q.name).group(1)))
    chosen = [c for i, c in enumerate(ckpts) if i % a.every == 0 or c == ckpts[-1]]
    cfg = C._yaml_load_any((run / "params" / "env.yaml").read_text()) if (run / "params" / "env.yaml").is_file() else {}
    num_envs = int(cfg.get("scene", {}).get("num_envs", 4096))
    guard = [str(GUARD), "run", "--mem", "14G", "--cpu", "800", "--"] if GUARD.is_file() else []
    for ck in chosen:
        step = int(re.search(r"_(\d+)\.pt$", ck.name).group(1))
        out = C.RESULTS_DIR / f"{a.label}@{step}" / "isaac"
        meta = out / "meta.json"
        if meta.is_file() and json.loads(meta.read_text()).get("checkpoint_sha256") == C.sha256(ck):
            continue
        cmd = guard + [a.py, str(REACH / "tools" / "eval_isaac.py"), "--checkpoint", str(ck), "--label",
                       f"{a.label}@{step}", "--seeds", a.seeds, "--device", a.device, "--ext0", "--trace_bases", "0"]
        print("[eval_curve] $", " ".join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, cwd=REACH)
    floor = C.STANDING_PELVIS_HEIGHT + min(json.loads((C.GOALS_DIR / "eval_goals_v1.json").read_text())["standing_min_z"])
    rows = []
    for ck in chosen:
        step = int(re.search(r"_(\d+)\.pt$", ck.name).group(1))
        tf = C.RESULTS_DIR / f"{a.label}@{step}" / "isaac" / "trials.csv"
        if not tf.is_file():
            continue
        tr = list(csv.DictReader(open(tf)))
        f = lambda k, sel=tr: np.array([float(r[k]) for r in sel])  # noqa: E731
        below = [r for r in tr if float(r["target_min_height"]) < floor]
        above = [r for r in tr if float(r["target_min_height"]) >= floor]
        rows.append({"label": a.label, "step": step, "transitions": step * num_envs, "n": len(tr),
                     "lambda_share": C.read_reward_share(run), "success": f("success").mean(),
                     "success_below": f("success", below).mean() if below else float("nan"),
                     "success_above": f("success", above).mean() if above else float("nan"),
                     "closest_med": float(np.median(f("closest"))),
                     "hold_err_pos_med": float(np.median(f("hold_err_pos"))) if "hold_err_pos" in tr[0] else float("nan"),
                     "pelvis_drop_below_med": float(np.nanmedian(f("pelvis_drop_last1s", below))) if below else float("nan"),
                     "fall_rate": f("fell").mean(), **levels(run, step)})
    if rows:
        C.write_csv(C.RESULTS_DIR / "curves" / f"{a.label}.csv", rows)
    print(f"[eval_curve] {a.label}: {len(rows)} checkpoints -> {C.RESULTS_DIR / 'curves' / (a.label + '.csv')}")


if __name__ == "__main__":
    main()
