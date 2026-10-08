"""The fixed goal grid every controller is scored on.

    python paper/reach/tools/make_goals.py [--table goals/arm_target_tables_v3.pt] [--bases 96] \
        [--depths 0,0.1,0.2,0.3,0.4,0.5,0.6] [--exts 0,0.1,0.2] [--seed 2026] [--name v1]

Base goals are pairs of wrist poses drawn uniformly (independently per arm, as in training) from the task's
standing-reachable tables, in the standing frame: the pelvis's x, y and yaw, standing_height above the ground. Each
base pair is then

    lowered by a depth d (both wrists, as the task's z_offset), and
    pushed out horizontally by an extension e: each wrist target moves e metres further from its shoulder in the
    ground plane (straight ahead when it is within 5 cm of the shoulder's vertical),

for every (d, e) on the grid, orientation unchanged. Every controller sees the same goals in the same order, so a
difference between controllers at a (base, d, e) cell is paired.

Writes goals/eval_goals_<name>.csv (one row per goal: id, base, d, e, both poses as x y z qw qx qy qz) and
goals/eval_goals_<name>.json (settings, table hash, the base pairs' table rows).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reachlib.common import GOALS_DIR, SHOULDER_B, STANDING_PELVIS_HEIGHT, sha256, write_csv  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--table", default=str(GOALS_DIR / "arm_target_tables_v3.pt"))
p.add_argument("--bases", type=int, default=96)
p.add_argument("--depths", default="0,0.1,0.2,0.3,0.4,0.5,0.6")
p.add_argument("--exts", default="0,0.1,0.2")
p.add_argument("--seed", type=int, default=2026)
p.add_argument("--name", default="v1")
a = p.parse_args()

data = torch.load(a.table, map_location="cpu", weights_only=False)
tables = [t.double().numpy() for t in data["tables"]]          # per arm (rows, 7): x y z qw qx qy qz, pelvis frame
rng = np.random.default_rng(a.seed)
idx = [rng.integers(0, len(t), a.bases) for t in tables]
depths = [float(x) for x in a.depths.split(",")]
exts = [float(x) for x in a.exts.split(",")]


def extend(pose: np.ndarray, shoulder, e: float) -> np.ndarray:
    out = pose.copy()
    h = pose[:2] - np.asarray(shoulder[:2])
    n = np.linalg.norm(h)
    u = h / n if n > 0.05 else np.array([1.0, 0.0])
    out[:2] = pose[:2] + e * u
    return out


rows = []
for b in range(a.bases):
    base = {"left": tables[0][idx[0][b]], "right": tables[1][idx[1][b]]}
    for d in depths:
        for e in exts:
            row = {"goal_id": f"b{b:03d}_d{round(100 * d):02d}_e{round(100 * e):02d}", "base_id": b, "depth": d, "ext": e}
            for arm, pre in (("left", "l"), ("right", "r")):
                pose = extend(base[arm], SHOULDER_B[arm], e)
                pose[2] -= d
                for k, v in zip(("x", "y", "z", "qw", "qx", "qy", "qz"), pose):
                    row[f"{pre}_{k}"] = float(v)
            row["min_height"] = STANDING_PELVIS_HEIGHT + min(row["l_z"], row["r_z"])
            rows.append(row)

name = f"eval_goals_{a.name}"
write_csv(GOALS_DIR / f"{name}.csv", rows)
meta = {
    "name": a.name, "bases": a.bases, "depths": depths, "exts": exts, "seed": a.seed, "goals": len(rows),
    "table": str(Path(a.table).resolve()), "table_sha256": sha256(Path(a.table)),
    "table_rows": [len(t) for t in tables], "standing_min_z": [float(t[:, 2].min()) for t in tables],
    "base_rows": {"left": idx[0].tolist(), "right": idx[1].tolist()},
    "frame": "standing frame: pelvis x, y, yaw at the goal's start; origin standing_height above the ground",
    "standing_height": STANDING_PELVIS_HEIGHT, "shoulders_b": SHOULDER_B,
}
(GOALS_DIR / f"{name}.json").write_text(json.dumps(meta, indent=1))
heights = np.array([r["min_height"] for r in rows])
print(f"[make_goals] {len(rows)} goals ({a.bases} bases x {len(depths)} depths x {len(exts)} extensions) -> "
      f"{GOALS_DIR / name}.csv; lower target height {heights.min():.2f}-{heights.max():.2f} m above the ground")
