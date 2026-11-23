"""The legs comparison plots (mujoco): this run's scorecards against the newest prior runs of the same task, by env
steps and by wall-clock hours, sport mode dashed, and the runs' training curves (evals/common/plots.py).

    python -m evals.mujoco.legs.get_plot --run <run name or dir> [--prior 4] [--with <run> ...]

Reads the cached scorecards in logs/evals/<run>/mujoco/legs/ (get_scorecard writes them; nothing is scored here) and
writes plot_steps.png, plot_hours.png, plot_training_steps.png and plot_training_hours.png beside them. Needs
matplotlib, numpy and tensorboard; no simulator.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # the repo root
from evals.common import legs_scorecard as L  # noqa: E402
from evals.common import plots  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--prior", type=int, default=4, help="Newest prior runs of the same task (up to 7).")
    parser.add_argument("--with", dest="extra", action="append", default=[], help="Also compare this run.")
    a = parser.parse_args()
    out = plots.make(a.run, "mujoco", "legs", L.PLOT_KEYS, L.TRAIN_PANELS, (lambda res, ref: L.rows(res)), "sport mode", min(a.prior, 7), a.extra)
    sys.exit(0 if out else 1)
