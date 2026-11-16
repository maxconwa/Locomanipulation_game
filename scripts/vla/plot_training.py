"""Plot a VLA training run's log.jsonl: policy success, agreement with the teacher, the action error, block localization.

    python scripts/vla/plot_training.py logs/vla/<run>/log.jsonl [--out <png>]

Four panels over the env steps, one measure each (no shared second axis). Success is the policy-driven episodes'
that ended in each `--window` log rows, the teacher-driven ones' beside it.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, GREEN, INK, MUTED, SURFACE, GRID = "#2a78d6", "#eb6834", "#2f9e6e", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log")
    parser.add_argument("--out", default=None)
    parser.add_argument("--window", type=int, default=5, help="log rows per success point")
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.log)]
    steps = np.array([r["step"] for r in rows])

    def pooled(who):
        """Success over each window of rows, weighted by the episodes that ended in it."""
        xs, ys = [], []
        for i in range(0, len(rows), args.window):
            chunk = rows[i:i + args.window]
            n = sum(r[f"{who}_episodes"] for r in chunk)
            if n >= 5:
                wins = sum((r[f"{who}_success"] or 0.0) * r[f"{who}_episodes"] for r in chunk)
                xs.append(chunk[-1]["step"])
                ys.append(100.0 * wins / n)
        return xs, ys

    def series(key):
        keep = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
        return [k[0] for k in keep], [k[1] for k in keep]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), facecolor=SURFACE, sharex=True)
    panels = [
        ("Episodes stacked (%)", [("policy-driven", pooled("student"), BLUE), ("teacher-driven", pooled("teacher"), ORANGE)]),
        ("Cosine of the policy's and the teacher's position delta", [("replay", series("cosine"), BLUE), ("new scenes", series("fresh_cosine"), ORANGE)]),
        ("Position-delta error (action units)", [("policy", series("l1_pos"), BLUE), ("predicting no motion", series("zero_pos"), ORANGE)]),
        ("Instructed blocks' position error from the keypoints (cm)", [("replay", series("blocks_cm"), BLUE), ("new scenes, instructed", series("fresh_pick_cm"), ORANGE),
                                                                       ("new scenes, target", series("fresh_target_cm"), GREEN)]),
    ]
    for ax, (title, lines) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        for label, (x, y), color in lines:
            if len(x):
                ax.plot(x, y, color=color, linewidth=2, label=label or None)
                if label:  # direct label at the line's end
                    ax.annotate(label, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", color=INK,
                                fontsize=9, va="center")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_ylim(bottom=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        if len(lines) > 1:
            ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="best")
        ax.margins(x=0.18 if len(lines) > 1 else 0.03)
    for ax in axes[1]:
        ax.set_xlabel(f"training step (x {rows[-1]['env_steps'] // rows[-1]['step']} envs)", color=MUTED, fontsize=9)
    fig.suptitle(os.path.basename(os.path.dirname(os.path.abspath(args.log))), x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.log)), "training.png")
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
