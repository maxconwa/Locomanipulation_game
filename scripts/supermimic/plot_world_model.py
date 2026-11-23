"""Plot a SuperMimic run's world-model and SHAC training curves from its TensorBoard events.

    python scripts/supermimic/plot_world_model.py <run dir> <out png> [--smooth 15]

One panel per quantity, env policy steps (thousands) on x so it lines up with the PPO runs' progress plots
(evals/isaac/legs/get_plot.py draws the walking metrics; this draws what only a SuperMimic run has):
the world model's prediction errors against "no better than the mean", its loss terms, the reward and termination
fits, imagined against simulated returns and survival, the critic and actor, and the time per iteration. Every
other wm/* tag the run logs (e.g. per-term errors) gets a panel of its own after those. Needs matplotlib, numpy and
tensorboard; no simulator.
"""

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# series colors in fixed order (validated: scripts of the dataviz palette check, light surface), inks and grid
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#8a4fd3")
REF, INK, MUTED, GRID, SPINE, SURFACE = "#0b0b0b", "#0b0b0b", "#52514e", "#e4e3df", "#c3c2b7", "#fcfcfb"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("run")
parser.add_argument("out")
parser.add_argument("--smooth", type=int, default=15, help="moving-average window, in logged points")
args = parser.parse_args()

ea = EventAccumulator(args.run, size_guidance={"scalars": 0})
ea.Reload()
tags = set(ea.Tags()["scalars"])
agents = sorted({t.split("/")[1] for t in tags if t.startswith("shac/") and t.count("/") == 2})


def curve(tag: str):
    """(steps in thousands, smoothed values) of a tag's finite points, or None."""
    if tag not in tags:
        return None
    events = ea.Scalars(tag)
    x, y = np.array([e.step for e in events], float), np.array([e.value for e in events], float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) == 0:
        return None
    if len(y) > args.smooth > 1:
        y, x = np.convolve(y, np.ones(args.smooth) / args.smooth, mode="valid"), x[args.smooth - 1:]
    return x / 1e3, y


def ratio(tag: str, scale_tag: str):
    """tag's curve divided by scale_tag's latest value (None without either)."""
    c, s = curve(tag), curve(scale_tag)
    return None if c is None or s is None or s[1][-1] == 0 else (c[0], c[1] / s[1][-1])


def per_agent(pattern: str, label: str):
    return [(f"{label} ({a})" if len(agents) > 1 else label, curve(pattern.format(a))) for a in agents]


k_tags = sorted((t for t in tags if t.startswith("wm/state_rmse_") and t != "wm/state_rmse_1step"),
                key=lambda t: int("".join(c for c in t if c.isdigit()) or 0))
k_tag = k_tags[-1] if k_tags else None
k_label = f"{k_tag.split('_')[-1].replace('step', '')}-step" if k_tag else "k-step"

# (title, better, [(label, curve)], log y, reference (value, label))
panels = [
    ("State prediction error (normalized RMSE)", "lower",
     [("1-step", curve("wm/state_rmse_1step")), (k_label, curve(k_tag) if k_tag else None)], False,
     (1.0, "predicting the mean")),
    ("World-model loss terms", "lower",
     [("state", curve("wm/state_loss")), ("observation", curve("wm/obs_loss")), ("reward", curve("wm/reward_loss")),
      ("termination", curve("wm/termination_loss"))], True, None),
    ("World-model total loss", "lower", [("loss", curve("wm/loss"))], True, None),
    ("Reward error / reward std", "lower", [("RMSE / std", ratio("wm/reward_rmse", "smc/reward_scale"))], False,
     (1.0, "predicting a constant")),
    ("Termination: share of steps that end in a fall", "predicted near actual",
     [("actual", curve("wm/termination_positive_rate")), ("predicted", curve("wm/termination_predicted_rate"))],
     False, None),
    ("Termination accuracy", "higher", [("accuracy", curve("wm/termination_accuracy"))], False, None),
    ("Return over the imagination horizon", "imagined near simulated",
     [s for a in agents for s in ((f"simulated ({a})" if len(agents) > 1 else "simulated",
                                   curve(f"smc/{a}/return_H_sim")),
                                  (f"imagined ({a})" if len(agents) > 1 else "imagined",
                                   curve(f"smc/{a}/return_H_imagined")))], False, None),
    ("Share still standing after the horizon", "imagined near simulated",
     [("simulated", curve("smc/alive_H_sim")), ("imagined", curve("smc/alive_H_imagined"))], False, None),
    ("Critic loss", "lower", per_agent("shac/{}/critic_loss", "critic"), True, None),
    ("Actor gradient norm before clipping", "steady", per_agent("shac/{}/actor_grad_norm", "actor"), True, None),
    ("Simulated reward per step", "higher", per_agent("smc/{}/sim_step_reward", "reward"), False, None),
    ("Mean |action|", "steady", [("|action|", curve("shac/action_abs"))], False, None),
    ("Seconds per iteration", "lower",
     [("collect (Isaac)", curve("perf/collect_s_per_iter")), ("world model", curve("perf/model_s_per_iter")),
      ("policy", curve("perf/policy_s_per_iter"))], False, None),
]
shown = {"wm/state_rmse_1step", k_tag, "wm/state_loss", "wm/obs_loss", "wm/reward_loss", "wm/termination_loss",
         "wm/loss", "wm/reward_rmse", "wm/termination_positive_rate", "wm/termination_predicted_rate",
         "wm/termination_accuracy", "wm/grad_norm"}
# every other world-model tag the run logs, grouped by its name up to the last "/" (per-term errors and the like)
extra: dict[str, list[str]] = {}
for tag in sorted(t for t in tags if t.startswith("wm/") and t not in shown):
    extra.setdefault(tag.rsplit("/", 1)[0] if tag.count("/") > 1 else tag, []).append(tag)
for group, members in extra.items():
    for i in range(0, len(members), len(SERIES)):  # at most one color per series: more members, more panels
        part = members[i:i + len(SERIES)]
        panels.append((group, "see tag", [(t.rsplit("/", 1)[-1] if len(members) > 1 else t, curve(t)) for t in part],
                       False, None))
panels = [p for p in panels if any(c is not None for _, c in p[2])]

cols = 4
rows = math.ceil(len(panels) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.3 * rows), facecolor=SURFACE, squeeze=False)
x_max = max((c[0][-1] for p in panels for _, c in p[2] if c is not None), default=1.0)
for ax, (title, better, series, log, ref) in zip(axes.flat, panels):
    ax.set_facecolor(SURFACE)
    drawn = [(label, c) for label, c in series if c is not None]
    ends = []
    for i, (label, (x, y)) in enumerate(drawn):
        if log:
            y = np.where(y > 0, y, np.nan)
        ax.plot(x, y, color=SERIES[i % len(SERIES)], lw=1.4)
        ends.append((label, x[-1], y[-1]))
    if log:
        ax.set_yscale("log")
    # direct labels at the lines' ends, in ink (identity never by color alone), nudged apart where two ends meet
    fig.canvas.draw()
    pixels = [ax.transData.transform((x, y))[1] if np.isfinite(y) else 0.0 for _, x, y in ends]
    shift = [0.0] * len(ends)
    order = sorted(range(len(ends)), key=lambda i: pixels[i])
    for lower, upper in zip(order, order[1:]):
        gap = (pixels[upper] + shift[upper]) - (pixels[lower] + shift[lower])
        if gap < 11.0:
            shift[upper] += 11.0 - gap
    for (label, x, y), up in zip(ends, shift):
        if np.isfinite(y):
            ax.annotate(f"{label} {y:.3g}", (x, y), textcoords="offset points", xytext=(4, up * 72.0 / fig.dpi),
                        fontsize=7.5, color=INK, va="center", annotation_clip=False)
    if ref is not None:
        ax.axhline(ref[0], color=REF, lw=1.1, ls=(0, (5, 3)))
        ax.text(0.01, ref[0], f"{ref[1]}: {ref[0]:g}", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7.5,
                color=INK)
    ax.set_title(f"{title}\n({better})", fontsize=9, loc="left", color=INK)
    ax.set_xlim(0, x_max * 1.28)  # room for the end labels
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
for ax in axes.flat[len(panels):]:
    ax.axis("off")
for ax in axes.flat[:len(panels)]:
    ax.set_xlabel("env policy steps (thousands)", fontsize=8, color=MUTED)
name = args.run.rstrip("/").split("/")[-1]
fig.suptitle(f"{name}: world model and SHAC training", fontsize=12, x=0.01, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(args.out, dpi=110, facecolor=SURFACE)
print(f"wrote {args.out}: {len(panels)} panels")
