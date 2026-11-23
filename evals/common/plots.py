"""The comparison plots: a run's scorecards against prior runs', checkpoint by checkpoint, by env steps and by
wall-clock hours, with the baseline dashed; and the runs' training curves the same two ways.

Series: the run being looked at takes slot 1 (blue) always; prior runs (the newest of the same task, runs.prior_runs)
take the next slots in order, newest first; runs differing only in their seed are one series, the seeds' mean with
their range as a band. Each series also has its own marker and a direct label at its last point, so no series is told
apart by colour alone. A scorecard taken under other conditions than the run's newest (conditions.stamp's hash; e.g.
the history imported from logs/candidates) is drawn hollow and thin, joined only to its own kind.
Reads cached scorecards only: it never scores anything.

Writes <dir>/plot_steps.png, plot_hours.png, plot_training_steps.png, plot_training_hours.png.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import outputs, runs, tb
from .scorecard import num

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, MUTED, GRID, SPINE, SURFACE, REF, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#c3c2b7", "#fcfcfb", "#0b0b0b", "#1baf7a"
DASH = (0, (5, 3))


def reference(run_dirs: list[Path], backend: str, body: str) -> tuple[dict | None, Path | None]:
    """The first of the runs' IK-only scorecards (scorecard_ik_only.json: the arms' reference), or (None, None)."""
    for r in run_dirs:
        f = outputs.eval_dir(runs.name(r), backend, body) / "scorecard_ik_only.json"
        if f.is_file():
            return json.load(open(f)), f
    return None, None


def scorecards(run_dir: Path, backend: str, body: str, row_fn, ref: dict | None = None) -> list[dict]:
    """The run's plain scorecards (scorecard_<N>.json: no sampled, coupled or off-table variants), by step, each with
    "values": {row key: value} (rows recomputed by row_fn for a json written before rows existed)."""
    d = outputs.eval_dir(runs.name(run_dir), backend, body)
    out = []
    for f in glob.glob(str(d / "scorecard_*.json")):
        m = re.fullmatch(r"scorecard_(\d+)\.json", os.path.basename(f))
        if not m:
            continue
        res = json.load(open(f))
        try:  # the current code's rows (labels, references, targets); the stored ones if it can't read the json
            rows = [vars(r) for r in row_fn(res, ref)]
        except (KeyError, TypeError, ValueError):
            rows = res.get("rows", [])
        res["values"] = {r["key"]: num(r["value"]) for r in rows}
        res["meta"] = {r["key"]: r for r in rows}
        res["step"] = int(m.group(1))
        out.append(res)
    return sorted(out, key=lambda r: r["step"])


def has_scorecards(backend: str, body: str):
    return lambda run_dir: bool(glob.glob(str(outputs.eval_dir(runs.name(run_dir), backend, body) / "scorecard_[0-9]*.json")))


def _style(ax, title, xlabel):
    import matplotlib.pyplot as plt  # noqa: F401
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)


def _groups(run_dir: Path, prior: list[Path]) -> list[tuple[str, list[Path]]]:
    """[(label, member runs)]: the run's seed group first, then the prior runs' groups, newest first."""
    order, members = [], defaultdict(list)
    for r in [run_dir] + prior:
        g = runs.seed_group(r)
        if g not in members:
            order.append(g)
        members[g].append(r)
    return [(g if len(members[g]) == 1 else f"{g} ({len(members[g])} seeds)", members[g]) for g in order]


def _series(cards_by_run: dict, members: list[Path], key: str, hours: bool, chash: str | None):
    """[(x, y, lo, hi, matching conditions)] points of a seed group for one row key, joined across seeds on the
    steps every member has."""
    per_run = []
    for r in members:
        clock = tb.hours_at(r, [c["step"] for c in cards_by_run[r]]) if hours else None
        pts = {}
        for c in cards_by_run[r]:
            v = c["values"].get(key, float("nan"))
            if not np.isfinite(v) or (hours and c["step"] not in clock):
                continue
            x = clock[c["step"]] if hours else c["step"] / 1e3
            pts[c["step"]] = (x, v, chash is None or c.get("conditions_hash") == chash)
        per_run.append(pts)
    steps = sorted(set.intersection(*[set(p) for p in per_run])) if per_run else []
    out = []
    for s in steps:
        xs, vs, ok = zip(*[p[s] for p in per_run])
        out.append((float(np.mean(xs)), float(np.mean(vs)), float(min(vs)), float(max(vs)), all(ok)))
    return out


def eval_figure(run_dir: Path, prior: list[Path], backend: str, body: str, keys: list[str], row_fn, hours: bool,
                title: str, baseline_label: str, out: str) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref, ref_path = reference([run_dir] + prior, backend, body)
    if ref_path is not None:
        print(f"[plots] reference: {ref_path}")
    cards_by_run = {r: scorecards(r, backend, body, row_fn, ref) for r in [run_dir] + prior}
    if not cards_by_run[run_dir]:
        print(f"[plots] no scorecards for {runs.short(run_dir)} in {outputs.eval_dir(runs.name(run_dir), backend, body)}")
        return None
    newest = cards_by_run[run_dir][-1]
    chash, meta = newest.get("conditions_hash"), newest["meta"]
    keys = [k for k in keys if k in meta]
    groups = _groups(run_dir, prior)
    if hours:
        xs = [h for r, cs in cards_by_run.items() for h in tb.hours_at(r, [c["step"] for c in cs]).values()]
    else:
        xs = [c["step"] / 1e3 for cs in cards_by_run.values() for c in cs]
    xmax = max(xs) if xs else 1.0
    cols = 4
    rows_n = int(np.ceil(len(keys) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.2 * rows_n + 0.9), facecolor=SURFACE, squeeze=False)
    for ax, key in zip(axes.flat, keys):
        m = meta[key]
        better = {"ge": "higher is better", "le": "lower is better"}.get(m.get("sense"), "")
        for i, (label, members) in enumerate(groups):
            color, marker = SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)]
            pts = _series(cards_by_run, members, key, hours, chash)
            for matching in (False, True):
                p = [q for q in pts if q[4] == matching]
                if not p:
                    continue
                x, y, lo, hi, _ = map(np.array, zip(*p))
                if len(members) > 1:
                    ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
                ax.plot(x, y, color=color, lw=2 if matching else 1, ls="-" if matching else ":", marker=marker,
                        ms=6 if matching else 5, mfc=color if matching else SURFACE, mec=color, mew=1.2)
            if pts and i == 0:  # the run's own last value; the others are told apart by marker and legend
                ax.annotate(f"{pts[-1][1]:.3g}", (pts[-1][0], pts[-1][1]), textcoords="offset points", xytext=(6, 0),
                            fontsize=8, color=INK, va="center", fontweight="bold")
        ax.set_xlim(0, xmax * 1.08)
        ref = m.get("reference")
        if isinstance(ref, (int, float)) and np.isfinite(ref):
            ax.axhline(ref, color=REF, lw=1.1, ls=DASH)
            ax.text(0.01, ref, f"{baseline_label} {ref:.3g}", transform=ax.get_yaxis_transform(), va="bottom",
                    fontsize=7.5, color=INK)
        t = m.get("target")
        if isinstance(t, (int, float)) and m.get("sense") in ("ge", "le"):
            lo_hi = (t, ax.get_ylim()[1]) if m["sense"] == "ge" else (min(0.0, ax.get_ylim()[0]), t)
            ax.axhspan(*lo_hi, color=BAND, alpha=0.08, lw=0)
        _style(ax, f"{m['label']}\n({better})" if better else m["label"],
               "wall-clock hours since the run started" if hours else "env policy steps (thousands)")
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=SERIES[i % len(SERIES)], marker=MARKERS[i % len(MARKERS)], lw=2, ms=6)
               for i in range(len(groups))]
    labels = [label + (" (this run)" if i == 0 else "") for i, (label, _) in enumerate(groups)]
    handles += [plt.Line2D([], [], color=REF, lw=1.1, ls=DASH),
                plt.Line2D([], [], color=MUTED, lw=1, ls=":", marker="o", mfc=SURFACE, mec=MUTED)]
    labels += [baseline_label, "scored under other conditions"]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False, fontsize=8.5,
               labelcolor=INK)
    fig.suptitle(title, fontsize=11.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.06 + 0.02 * (len(labels) > 4), 1, 0.96))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def training_figure(run_dir: Path, prior: list[Path], panels: list[tuple[str, str]], hours: bool, title: str,
                    out: str) -> str | None:
    """panels: [(TensorBoard tag, title)]: each run's own curve (seed groups: one line per seed, same colour)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = _groups(run_dir, prior)
    panels = [(t, n) for t, n in panels if any(t in tb.tags(r) for _, ms in groups for r in ms)]
    if not panels:
        return None
    cols = 4
    rows_n = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.0 * rows_n + 0.8), facecolor=SURFACE, squeeze=False)
    for ax, (tag, name) in zip(axes.flat, panels):
        for i, (label, members) in enumerate(groups):
            for r in members:
                c = tb.curve(r, tag, hours=hours)
                if c is not None:
                    x = c[0] if hours else c[0] / 1e3
                    ax.plot(x, c[1], color=SERIES[i % len(SERIES)], lw=1.4 if i == 0 else 1.0)
                    if i == 0:
                        ax.annotate(f"{c[1][-1]:.3g}", (x[-1], c[1][-1]), textcoords="offset points", xytext=(4, 0),
                                    fontsize=7.5, color=INK, va="center", fontweight="bold")
        _style(ax, name, "wall-clock hours since the run started" if hours else "env policy steps (thousands)")
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=SERIES[i % len(SERIES)], lw=2) for i in range(len(groups))]
    fig.legend(handles, [g + (" (this run)" if i == 0 else "") for i, (g, _) in enumerate(groups)], loc="lower center",
               ncol=min(len(groups), 4), frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle(title, fontsize=11.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def make(run: str, backend: str, body: str, keys: list[str], train_panels: list[tuple[str, str]], row_fn,
         baseline_label: str, prior_n: int = 4, extra_prior: list[str] = ()) -> list[str]:
    """All four figures for a run; returns their paths."""
    run_dir = runs.resolve(run)
    prior = runs.prior_runs(run_dir, prior_n, has_scorecards(backend, body))
    prior += [runs.resolve(p) for p in extra_prior if runs.resolve(p) not in prior]
    d = outputs.eval_dir(runs.name(run_dir), backend, body)
    task = runs.task(run_dir) or "?"
    name = f"{runs.short(run_dir)} ({task}, {backend} {body} scorecard)"
    written = []
    for hours in (False, True):
        tag = "hours" if hours else "steps"
        written.append(eval_figure(run_dir, prior, backend, body, keys, row_fn, hours,
                                   f"{name}: against {len(prior)} prior run(s) and {baseline_label}", baseline_label,
                                   str(d / f"plot_{tag}.png")))
        written.append(training_figure(run_dir, prior, train_panels, hours,
                                       f"{runs.short(run_dir)}: training curves (sampled actions, the run's own"
                                       f" randomization)", str(d / f"plot_training_{tag}.png")))
    for w in written:
        if w:
            print(f"[plots] wrote {w}")
    return [w for w in written if w]
