"""Sample efficiency of the paper's runs: training-time curves of the three lambda policies (paper00, paper05, paper1)
and the one-agent whole-body baseline (paperwb) against environment transitions (policy steps x 4096 envs).

    python logs/figures/sample_efficiency/sample_efficiency.py

Same metrics, smoothing (50 updates) and style as paper/reach/tools/figures.py's training_curves; lambda 0 blue,
lambda 1 orange, lambda 0.5 between them, the whole-body agent black. Writes sample_efficiency.{pdf,png}, the plotted
series as sample_efficiency.csv, and milestones.csv: the transitions each run needs to reach a few training targets.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

HERE = Path(__file__).resolve().parent
LOGS = HERE.parents[1] / "skrl" / "locomanip_marl"
NUM_ENVS = 4096
ROT_TOL = 0.35  # rad, the reach test's orientation tolerance


# lambda 0 blue (Fig. 1's legs blue), lambda 1 a deep orange, lambda 0.5 their OKLab midpoint (blue and orange are
# near-opposite, so it is a muted neutral), the whole-body agent the paper's ink. Validated: CVD dE 11.2, normal
# 16.0, every line >= 3:1 on white.
RUNS = [
    ("paperwb", "Whole-body agent", "#1a2127"),
    ("paper1", "λ = 1", "#DB6E00"),
    ("paper05", "λ = 0.5", "#8b7979"),
    ("paper00", "λ = 0", "#0072B2"),
]
TAGS = {
    "depth_level": "Curriculum/arm_target_levels/depth_level",
    "reached": "Metrics/arm_targets/goals_reached",
    "missed": "Metrics/arm_targets/goals_missed",
    "crouch_reached": "Metrics/arm_targets/crouch_goals_reached",
    "crouch_missed": "Metrics/arm_targets/crouch_goals_missed",
    "goal_position_error": "Metrics/arm_targets/goal_position_error",
    "goal_orientation_error": "Metrics/arm_targets/goal_orientation_error",
    "best_error": "Metrics/arm_targets/goal_best_error",
    "pelvis_drop": "Metrics/arm_targets/pelvis_drop",
    "target_drop": "Metrics/arm_targets/target_drop",
    "walk_error": "Metrics/base_velocity/nav_error_vel_xy",
    "falls": "Episode_Termination/fell",
}
PANELS = [  # (main, dashed second, y label)
    ("reach_rate", "crouch_reach_rate", "Goals reached in training (%)"),
    ("depth_level", None, "Curriculum depth level $k_d$"),
    ("best_error_cm", None, "Closest approach per goal (cm)"),
    ("goal_orientation_error_rad", None, "Wrist orientation error (rad)"),
    ("pelvis_drop_cm", "target_drop_cm", "Drop during arm goals (cm)"),
    ("walk_error_m_s", None, "Walking velocity error (m/s)"),
]

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#55595e", "axes.labelcolor": "#1a2127",
    "xtick.color": "#55595e", "ytick.color": "#55595e", "grid.color": "#e6e8ea", "grid.linewidth": 0.5,
})
MUTED = "#77848d"


def load(run: str) -> dict[str, np.ndarray]:
    series = {k: {} for k in TAGS}
    for f in sorted((LOGS / run).glob("events.out.tfevents.*")):
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        have = set(ea.Tags()["scalars"])
        for key, tag in TAGS.items():
            if tag in have:
                for ev in ea.Scalars(tag):
                    series[key][ev.step] = ev.value
    steps = np.array(sorted(set().union(*[set(v) for v in series.values()])))
    g = {k: np.array([series[k].get(s, np.nan) for s in steps]) for k in TAGS}
    with np.errstate(invalid="ignore", divide="ignore"):
        ended, cended = g["reached"] + g["missed"], g["crouch_reached"] + g["crouch_missed"]
        return {
            "step": steps, "transitions": steps * NUM_ENVS, "depth_level": g["depth_level"],
            "reach_rate": np.where(ended > 0, g["reached"] / ended, np.nan),
            "crouch_reach_rate": np.where(cended > 0, g["crouch_reached"] / cended, np.nan),
            "best_error_cm": 100 * g["best_error"], "goal_orientation_error_rad": g["goal_orientation_error"],
            "pelvis_drop_cm": 100 * g["pelvis_drop"], "target_drop_cm": 100 * g["target_drop"],
            "walk_error_m_s": g["walk_error"], "falls": g["falls"],
        }


def smooth(y, w=50):
    y = np.asarray(y, dtype=float)
    k = np.ones(w) / w
    ok = np.isfinite(y)
    num = np.convolve(np.where(ok, y, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.2, num / den, np.nan)


def first_crossing(x, y, target, below=False):
    hit = np.nonzero((y <= target) if below else (y >= target))[0]
    return float(x[hit[0]]) if len(hit) else float("nan")


data = {run: load(run) for run, _, _ in RUNS}
fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.6), sharex=True)
for run, name, color in RUNS:
    d = data[run]
    x = d["transitions"] / 1e6
    for ax, (main, second, label) in zip(axes.flat, PANELS):
        scale = 100 if "rate" in main else 1
        ax.plot(x, scale * smooth(d[main]), color=color, lw=1.5, label=name, solid_capstyle="round")
        if second:
            ax.plot(x, scale * smooth(d[second]), color=color, lw=1.0, ls=(0, (3, 1.5)))
        ax.set_ylabel(label)
        ax.grid(True, axis="y")
axes[0, 2].axhline(12.0, color="#9aa1a7", lw=0.8, ls=(0, (2, 2)))
axes[1, 0].axhline(ROT_TOL, color="#9aa1a7", lw=0.8, ls=(0, (2, 2)))
for ax in axes[1]:
    ax.set_xlabel("Environment transitions (millions)")
for letter, ax in zip("abcdef", axes.flat):
    ax.text(-0.02, 1.03, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
# one legend row above the panels, lambda in order then the baseline
handles, labels = axes[0, 0].get_legend_handles_labels()
order = [labels.index(n) for n in ("λ = 0", "λ = 0.5", "λ = 1", "Whole-body agent")]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=2.0)
fig.text(0.5, -0.01, "Dashed: (a) goals below standing reach, (e) the goals' lowering. Dotted: (c) 12 cm, the closest "
         "approach the depth level rises at; (d) 0.35 rad, the reach test's orientation tolerance. "
         "Smoothed over 50 updates.", ha="center", va="top", fontsize=6.2, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(HERE / "sample_efficiency.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(HERE / "sample_efficiency.png", bbox_inches="tight", pad_inches=0.02, dpi=300)

keys = ["step", "transitions"] + [k for k in data[RUNS[0][0]] if k not in ("step", "transitions")]
with open(HERE / "sample_efficiency.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run"] + keys)
    for run, _, _ in RUNS:
        d = data[run]
        for i in range(len(d["step"])):
            w.writerow([run] + [d[k][i] for k in keys])

# milestones on the smoothed curves, in millions of transitions
milestones = [("depth level 5", "depth_level", 5.0, False), ("depth level 9", "depth_level", 9.0, False),
              ("closest approach <= 12 cm", "best_error_cm", 12.0, True), ("goals reached >= 2%", "reach_rate", 0.02, False),
              ("drop in arm goals >= 15 cm", "pelvis_drop_cm", 15.0, False)]
rows = []
for run, name, _ in RUNS:
    d = data[run]
    x = d["transitions"] / 1e6
    row = {"run": run, "name": name}
    for label, key, target, below in milestones:
        y = smooth(d[key])
        # after the warm start: the first crossing once arm goals have run (reach metrics are nan before)
        valid = np.isfinite(smooth(d["reach_rate"]))
        start = np.argmax(valid) if valid.any() else 0
        row[label] = first_crossing(x[start:], y[start:], target, below)
    last = slice(-100, None)
    row["final goals reached (%)"] = 100 * np.nanmean(d["reach_rate"][last])
    row["final depth level"] = np.nanmean(d["depth_level"][last])
    rows.append(row)
with open(HERE / "milestones.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for row in rows:
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()})
print("[sample_efficiency]", HERE / "sample_efficiency.pdf")
