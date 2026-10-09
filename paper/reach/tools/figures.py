"""Paper figures from results/summary (run analyze.py first). Every figure draws the conditions that exist.

    python paper/reach/tools/figures.py

Which conditions appear, their names and order come from conditions.yaml; GOLEM's arm IK is always the reference.
The hero condition (conditions.yaml `hero`, else the highest lambda evaluated) also contributes its blind-legs and
IK-arms variants and its MuJoCo runs. Writes figures/<name>.pdf, .png and the plotted numbers as
figures/data/<name>.csv:

    fig_hero               double column: reached-target workspace over the H1-2 | success | pelvis height,
                           both against the lower wrist target's height (the paper's main figure)
    fig_hero_depth         the same against the lowering d of the standing-table goals
    fig_hero_col           single column: workspace | success
    fig_filmstrip          double column: the robot every second of the hero's deepest reached goal
    success_vs_height, success_vs_depth, pelvis_vs_height, pelvis_drop_vs_depth     single-column panels
    success_vs_tolerance   success against the position tolerance of the 1 s hold, above and below the table floor
    error_vs_depth         wrist error of reached goals and closest approach of all goals
    balance_vs_depth       CoM and DCM margins, tilt, wrist jitter, foot slip, falls
    sim2sim                the hero in Isaac and in MuJoCo (RoboCasa and training joint dynamics)
    workspace_maps         per-wrist success over the sagittal plane for every condition

Colours: lambda on one orange ramp (validated ordinal, light 0 -> dark 1); blind legs blue; GOLEM's IK dark grey
dashed; IK arms dotted and MuJoCo dash-dot in their condition's colour.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REACH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REACH))
sys.path.insert(0, str(REACH / "tools"))
from reachlib import common as C  # noqa: E402

SUMMARY = C.RESULTS_DIR / "summary"
FIG = C.FIG_DIR
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.8,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#55595e", "axes.labelcolor": "#1a2127",
    "xtick.color": "#55595e", "ytick.color": "#55595e", "grid.color": "#e6e8ea", "grid.linewidth": 0.5,
    "savefig.dpi": 300, "figure.dpi": 150, "legend.frameon": False,
})
INK, BLUE, REF, GREY_BAND, MUTED = "#1a2127", "#4A7FC4", "#4b4f54", "#eef0f2", "#77848d"
COL_W, DBL_W = 3.45, 7.16


def lam_color(lam: float) -> str:
    """The validated ordinal orange ramp in OKLCH: lambda 0 -> #f2a359 ... lambda 1 -> #9d4900."""
    lo, hi = (0.78, 0.13, 62.0), (0.50, 0.15, 62.0)
    t = min(max(lam, 0.0), 1.0)
    L, Cc, h = (lo[i] + t * (hi[i] - lo[i]) for i in range(3))
    a, b = Cc * math.cos(math.radians(h)), Cc * math.sin(math.radians(h))
    l_, m_, s_ = (L + 0.3963377774 * a + 0.2158037573 * b, L - 0.1055613458 * a - 0.0638541728 * b,
                  L - 0.0894841775 * a - 1.2914855480 * b)
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    rgb = (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
           -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
           -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)
    enc = [12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in (max(0, min(1, x)) for x in rgb)]
    return "#%02x%02x%02x" % tuple(round(255 * c) for c in enc)


# ---------------------------------------------------------------- data and selection

def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            if k not in ("condition", "name", "group"):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return rows


class Data:
    def __init__(self):
        self.summary = json.loads((SUMMARY / "summary.json").read_text())
        self.infos = self.summary["conditions"]
        self.goals = self.summary["goals"]
        self.floor = C.STANDING_PELVIS_HEIGHT + min(self.goals["standing_min_z"])
        self.by_depth, self.by_height, self.tol = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in read_csv(SUMMARY / "by_depth.csv"):
            self.by_depth[r["condition"]].append(r)
        for r in read_csv(SUMMARY / "by_height.csv"):
            self.by_height[r["condition"]].append(r)
        for r in read_csv(SUMMARY / "tolerance.csv"):
            self.tol[(r["condition"], r["group"])].append(r)
        self.registry = {c["label"]: {**c, "order": i} for i, c in enumerate(C.load_conditions())}
        listed = [lab for lab in self.registry if f"{lab}/isaac" in self.infos]
        if not self.registry:                     # no registry: every evaluated policy
            listed = sorted({k.split("/")[0] for k, v in self.infos.items() if v["kind"] == "policy"})
        self.main = [f"{lab}/isaac" for lab in listed]
        heroes = [lab for lab in listed if self.registry.get(lab, {}).get("hero")]
        if heroes:
            self.hero = heroes[0]
        elif listed:
            self.hero = max(listed, key=lambda lab: self.infos[f"{lab}/isaac"]["lambda"] or 0)
        else:
            self.hero = None
        h = self.hero
        self.blind = f"{h}/isaac_legs_blind" if h and f"{h}/isaac_legs_blind" in self.infos else None
        self.armsik = f"{h}/isaac_arms_ik" if h and f"{h}/isaac_arms_ik" in self.infos else None
        self.golem = "golem_ik/kinematic" if "golem_ik/kinematic" in self.infos else None
        self.mujoco = [k for k in (f"{h}/mujoco", f"{h}/mujoco_isaacphys", f"{h}/mujoco_legs_blind")
                       if h and k in self.infos]
        self.lines = self.main + [k for k in (self.blind, self.armsik, self.golem) if k]
        # the hero figure: the runs that are not marked hero_figure: false (e.g. an intermediate checkpoint)
        self.hero_lines = [k for k in self.main if self.registry.get(k.split("/")[0], {}).get("hero_figure", True)] + \
            [k for k in (self.blind, self.golem) if k]

    def name(self, key: str) -> str:
        return self.infos[key]["name"]

    def style(self, key: str) -> dict:
        info = self.infos[key]
        if info["kind"] == "golem_ik":
            return {"color": REF, "ls": (0, (4, 2)), "z": 2}
        label = key.split("/")[0]
        reg = self.registry.get(label, {})
        col = reg.get("color") or lam_color(info["lambda"] if info["lambda"] is not None else 1.0)
        ls = {"dotted": (0, (1, 1.2)), "dashed": (0, (4, 2))}.get(reg.get("dash"), "-")
        if "legs_blind" in info["variant"]:
            col = BLUE
        if "arms_ik" in info["variant"]:
            ls = (0, (1, 1.2))
        if info["base_sim"] == "mujoco":
            ls = (0, (5, 1.5, 1, 1.5)) if "isaacphys" not in info["variant"] else (0, (2.5, 1.5))
        return {"color": col, "ls": ls, "z": 3 if label == self.hero else 2}


def save(fig, name: str, rows: list[dict] | None = None):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    if rows:
        C.write_csv(FIG / "data" / f"{name}.csv", rows)
    print(f"[figures] {FIG / name}.pdf")


# ---------------------------------------------------------------- line panels

def curve(D: Data, ax, keys, table, xkey, ykey, lo=None, hi=None, scale=1.0, min_n=10, horizontal=False,
          band=True):
    out = []
    for k in keys:
        rows = sorted([r for r in table.get(k, []) if r["n"] >= min_n and not math.isnan(r.get(ykey, math.nan))],
                      key=lambda r: r[xkey])
        if not rows:
            continue
        x = np.array([r[xkey] for r in rows])
        y = scale * np.array([r[ykey] for r in rows])
        st = D.style(k)
        ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.6, label=D.name(k), zorder=st["z"],
                solid_capstyle="round", dash_capstyle="round")
        if band and lo and hi:
            ylo, yhi = scale * np.array([r[lo] for r in rows]), scale * np.array([r[hi] for r in rows])
            ax.fill_between(x, ylo, yhi, color=st["color"], alpha=0.13, lw=0, zorder=st["z"] - 1)
        out += [{"condition": k, xkey: r[xkey], "n": r["n"], ykey: r[ykey],
                 **({lo: r[lo], hi: r[hi]} if lo and hi else {})} for r in rows]
    return out


def shade_below_floor(D: Data, ax, axis="x", label=True):
    if axis == "x":
        ax.axvspan(0.0, D.floor, color=GREY_BAND, zorder=0, lw=0)
        if label:
            ax.text(D.floor / 2, 0.5, "Below the\nstanding table", transform=ax.get_xaxis_transform(),
                    ha="center", va="center", fontsize=6.3, color=MUTED)
    else:
        ax.axhspan(0.0, D.floor, color=GREY_BAND, zorder=0, lw=0)


def success_panel(D, ax, axis="height", keys=None, metric="success"):
    """Success against the lower wrist target's height (or the goal lowering), 95% Wilson band. metric "success" is
    the strict test (5 cm, 0.35 rad, 1 s), "success_10cm" the same test at 10 cm and 0.6 rad."""
    table, xkey = (D.by_height, "bin_mid") if axis == "height" else (D.by_depth, "depth")
    rows = curve(D, ax, keys or D.lines, table, xkey, metric, f"{metric}_lo", f"{metric}_hi", scale=100)
    ax.set_ylabel({"success": "Strict success (%)", "success_10cm": "Success within 10 cm (%)"}[metric])
    ax.set_ylim(-2, 102)
    ax.grid(True, axis="y")
    if axis == "height":
        shade_below_floor(D, ax)
        ax.set_xlim(0.0, 1.8)
        ax.set_xlabel("Lower wrist target height (m)")
    else:
        ax.set_xlabel("Goal lowering below the standing table (m)")
    return rows


def pelvis_panel(D, ax, axis="height", keys=None):
    table, xkey = (D.by_height, "bin_mid") if axis == "height" else (D.by_depth, "depth")
    rows = curve(D, ax, keys or D.lines, table, xkey, "pelvis_h_last1s_med", "pelvis_h_last1s_q25",
                 "pelvis_h_last1s_q75")
    ax.set_ylabel("Pelvis height (m)")
    ax.axhline(C.STANDING_PELVIS_HEIGHT, color="#9aa1a7", lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.grid(True, axis="y")
    lo = min([r["pelvis_h_last1s_q25"] for r in rows] + [0.9]) if rows else 0.6
    ax.set_ylim(max(0.3, lo - 0.05), 1.06)
    if axis == "height":
        shade_below_floor(D, ax, label=False)
        ax.set_xlim(0.0, 1.8)
        ax.set_xlabel("Lower wrist target height (m)")
    else:
        ax.set_xlabel("Goal lowering below the standing table (m)")
    return rows


# ---------------------------------------------------------------- workspace and renders

CRITERIA = {"strict": "held within 5 cm and 0.35 rad for 1 s", "final10": "within 10 cm at the goal's end"}


def wrist_points(key: str, criterion: str = "strict"):
    """(x forward, height, reached) per wrist target; reached = that wrist's strict 1 s hold ("strict") or its
    final position error under 10 cm ("final10")."""
    label, sim = key.split("/", 1)
    pts = []
    for r in csv.DictReader(open(C.RESULTS_DIR / label / sim / "trials.csv")):
        for pre in ("l", "r"):
            if criterion == "strict":
                ok = float(r[f"success_{pre}"]) > 0.5
            else:
                e = float(r[f"err_final_{pre}"]) if r[f"err_final_{pre}"] not in ("", "nan") else float("inf")
                ok = e < 0.10
            pts.append((float(r[f"target_{pre}_x"]), float(r[f"target_{pre}_z"]) + C.STANDING_PELVIS_HEIGHT, ok))
    return np.array(pts, dtype=float)


def success_field(pts, xs, zs, sigma=0.05, min_weight=2.0):
    X, Z = np.meshgrid(xs, zs)
    num, den = np.zeros_like(X), np.zeros_like(X)
    for x, z, s in pts:
        w = np.exp(-((X - x) ** 2 + (Z - z) ** 2) / (2 * sigma ** 2))
        num += w * s
        den += w
    with np.errstate(invalid="ignore"):
        f = num / den
    f[den < min_weight] = np.nan
    return f


def deepest_reached(D: Data):
    """(goal row, traces.npz, frame, kind) for the pose drawn over the workspace: the lowest goal the hero reached
    among its traced seed-0 trials, at the step its 1 s hold completed ("reached"); if none was reached, the traced
    trial with the deepest pelvis drop that stayed upright, at its closest approach ("closest"). None without traces."""
    if not D.hero:
        return None
    tr = C.RESULTS_DIR / D.hero / "isaac" / "traces.npz"
    if not tr.is_file():
        return None
    z = np.load(tr)
    names = set(z.files)
    rows = [r for r in csv.DictReader(open(C.RESULTS_DIR / D.hero / "isaac" / "trials.csv"))
            if r["seed"] == "0" and f"trace_{r['goal_id']}_state" in names]
    reached = [r for r in rows if r["success"] in ("1", "True")]
    if reached:
        r = min(reached, key=lambda r: float(r["target_min_height"]))
        return r, tr, max(0, int(round(float(r["t_success"]) / C.POLICY_DT)) - 1), "reached"
    upright = [r for r in rows if r["fell"] in ("0", "False")]
    if not upright:
        return None
    r = max(upright, key=lambda r: float(r["pelvis_drop_last1s"]))
    err = z[f"trace_{r['goal_id']}_err_mean"]
    return r, tr, int(np.nanargmin(err)), "closest"


def robot_layers(D: Data):
    """[(image, extent, alpha)]: the standing robot (faint when a crouch exists) and the hero's deepest reach."""
    from render_robot import load_model, render, render_trace_frames, standing_pose
    import mujoco
    layers = []
    m = load_model()
    d = mujoco.MjData(m)
    standing_pose(m, d)
    img, mp = render(m, d, width=600, height=800)
    pick = deepest_reached(D)
    layers.append((img, mp, 0.30 if pick else 1.0))
    target = None
    if pick:
        row, tr, t, kind = pick
        (img2, mp2), = render_trace_frames(tr, row["goal_id"], [t], width=600, height=800)
        layers.append((img2, mp2, 1.0))
        target = {**row, "pose_kind": kind, "pose_frame": t}
        C.write_csv(FIG / "data" / "hero_pose.csv", [target])
    return layers, target


def extent_of(mp):
    return (mp["x_left"], mp["x_left"] + mp["width"] / mp["px_per_m"], mp["z_top"] - mp["height"] / mp["px_per_m"],
            mp["z_top"])


def workspace_panel(D: Data, ax, layers, target, criterion: str = "strict"):
    xs, zs = np.linspace(-0.3, 1.1, 141), np.linspace(0.0, 2.0, 201)
    ax.set_aspect("equal")
    shade_below_floor(D, ax, axis="y")
    for img, mp, alpha in layers:
        ax.imshow(img, extent=extent_of(mp), alpha=alpha, zorder=1, interpolation="lanczos")
    standing = D.blind or D.golem
    crouch = f"{D.hero}/isaac" if D.hero else None
    drawn = []
    for key, col in ((standing, BLUE), (crouch, lam_color(D.infos[crouch]["lambda"] or 1.0) if crouch else None)):
        if not key:
            continue
        f = np.nan_to_num(success_field(wrist_points(key, criterion), xs, zs), nan=0.0)
        ax.contourf(xs, zs, f, levels=[0.5, 1.01], colors=[col], alpha=0.28, zorder=2)
        ax.contour(xs, zs, f, levels=[0.5], colors=[col], linewidths=1.1, zorder=3)
        drawn.append((key, col))
    if target is not None:
        for pre in ("l", "r"):
            ax.plot(float(target[f"target_{pre}_x"]), float(target[f"target_{pre}_z"]) + C.STANDING_PELVIS_HEIGHT,
                    marker="x", ms=5, mew=1.4, color=INK, zorder=4)
    ax.axhline(0.0, color="#9aa1a7", lw=0.8, zorder=0)
    ax.set_xlim(-0.3, 1.1)
    ax.set_ylim(0.0, 2.0)
    ax.set_xlabel("Forward of the pelvis (m)")
    ax.set_ylabel("Height above the ground (m)")
    return drawn


def hero_legend(D, fig, ax_lines, drawn, y=0.995, ncol=4, criterion="final10"):
    handles, labels = ax_lines.get_legend_handles_labels()
    for key, col in drawn:
        handles.append(Patch(facecolor=col, alpha=0.3, edgecolor=col))
        labels.append(f"Wrist {CRITERIA[criterion]}: {D.name(key)}")
    handles.append(Line2D([], [], marker="x", ls="", color=INK, ms=5, mew=1.4))
    labels.append("Wrist targets of the pose shown")
    fig.legend(handles, labels, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, y), handlelength=2.0,
               columnspacing=1.1)


def fig_hero(D: Data, layers, target, axis="height", criterion="final10"):
    fig = plt.figure(figsize=(DBL_W, 2.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.2, 1.2], wspace=0.36, left=0.06, right=0.995, bottom=0.14,
                          top=0.80)
    ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))
    drawn = workspace_panel(D, ax0, layers, target, criterion)
    metric = "success" if criterion == "strict" else "success_10cm"
    rows = success_panel(D, ax1, axis, keys=D.hero_lines, metric=metric) + pelvis_panel(D, ax2, axis, keys=D.hero_lines)
    for ax, letter in zip((ax0, ax1, ax2), "abc"):
        ax.text(-0.02, 1.03, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom",
                ha="right")
    hero_legend(D, fig, ax1, drawn, criterion=criterion)
    name = "fig_hero" if axis == "height" else "fig_hero_depth"
    save(fig, name + ("" if criterion == "final10" else f"_{criterion}"), rows)


def fig_hero_col(D: Data, layers, target):
    fig = plt.figure(figsize=(COL_W, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.85, 1.15], wspace=0.42, left=0.13, right=0.99, bottom=0.16, top=0.74)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    drawn = workspace_panel(D, ax0, layers, target, "final10")
    ax0.set_xlabel("Forward (m)")
    ax0.set_ylabel("Height (m)")
    rows = success_panel(D, ax1, "height", keys=D.hero_lines, metric="success_10cm")
    ax1.set_xlabel("Lower target height (m)")
    hero_legend(D, fig, ax1, drawn, y=1.0, ncol=2, criterion="final10")
    save(fig, "fig_hero_col", rows)


def fig_filmstrip(D: Data):
    pick = deepest_reached(D)
    if not pick:
        return
    from render_robot import render_trace_frames
    row, tr, t_pose, kind = pick
    t_done = (t_pose + 1) * C.POLICY_DT
    times = [0.0, 1.0, 2.0, round(min(t_done, 3.98), 2), 3.98]
    times = sorted(set(round(t, 2) for t in times))
    frames = render_trace_frames(tr, row["goal_id"], [int(round(t / C.POLICY_DT)) for t in times], width=480,
                                 height=640)
    fig, axes = plt.subplots(1, len(frames), figsize=(DBL_W, 2.25), sharey=True)
    for ax, (img, mp), t in zip(np.atleast_1d(axes), frames, times):
        ax.imshow(img, extent=extent_of(mp), interpolation="lanczos", zorder=1)
        shade_below_floor(D, ax, axis="y")
        for pre in ("l", "r"):
            ax.plot(float(row[f"target_{pre}_x"]), float(row[f"target_{pre}_z"]) + C.STANDING_PELVIS_HEIGHT,
                    marker="x", ms=5, mew=1.4, color=lam_color(D.infos[f"{D.hero}/isaac"]["lambda"] or 1.0), zorder=4)
        ax.axhline(0.0, color="#9aa1a7", lw=0.8)
        ax.set_xlim(-0.35, 0.95)
        ax.set_ylim(0.0, 1.9)
        ax.set_aspect("equal")
        tag = ("  (held 1 s)" if kind == "reached" else "  (closest)") if abs(t - t_done) < 1e-6 else ""
        ax.set_title(f"t = {t:.1f} s" + tag, fontsize=7.2, pad=2)
        ax.set_xticks([0.0, 0.5])
    np.atleast_1d(axes)[0].set_ylabel("Height (m)")
    fig.supxlabel("Forward of the pelvis (m)", fontsize=7.5, y=0.01)
    fig.suptitle(f"{D.name(D.hero + '/isaac')}, goal {row['goal_id']}: lower wrist target "
                 f"{float(row['target_min_height']):.2f} m above the ground", fontsize=7.2, y=0.995)
    fig.tight_layout(pad=0.3, w_pad=0.4, rect=(0, 0.05, 1, 0.93))
    save(fig, "fig_filmstrip", [{"goal_id": row["goal_id"], "t": t} for t in times])


def single(D: Data, name, panel, axis):
    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    rows = panel(D, ax, axis)
    ax.legend(loc="best")
    save(fig, name, rows)


def pelvis_drop_vs_depth(D: Data):
    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    rows = curve(D, ax, [k for k in D.lines if D.infos[k]["kind"] != "golem_ik"], D.by_depth, "depth",
                 "pelvis_drop_last1s_med", "pelvis_drop_last1s_q25", "pelvis_drop_last1s_q75", scale=100)
    ax.plot([0, 1.0], [0, 100], color="#9aa1a7", lw=0.8, ls=(0, (2, 2)))
    ax.text(0.62, 66, "drop = lowering", fontsize=6.3, color=MUTED, rotation=33)
    ax.set_xlabel("Goal lowering below the standing table (m)")
    ax.set_ylabel("Pelvis drop, last 1 s (cm)")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")
    save(fig, "pelvis_drop_vs_depth", rows)


def success_vs_tolerance(D: Data):
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W * 0.66, 2.15), sharey=True)
    rows = []
    for ax, group, title in zip(axes, ("above", "below"), ("Targets above the standing table",
                                                            "Targets below the standing table")):
        for k in D.lines:
            tr = sorted(D.tol.get((k, group), []), key=lambda r: r["tau"])
            if not tr:
                continue
            st = D.style(k)
            ax.plot([100 * r["tau"] for r in tr], [100 * r["success_pos"] for r in tr], color=st["color"],
                    ls=st["ls"], lw=1.6, label=D.name(k), zorder=st["z"])
            rows += [{"condition": k, "group": group, "tau": r["tau"], "success_pos": r["success_pos"]} for r in tr]
        ax.axvline(100 * C.POS_TOL, color="#9aa1a7", lw=0.8, ls=(0, (2, 2)))
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.set_xlabel("Position tolerance of the 1 s hold (cm)")
        ax.set_xlim(1, 20)
        ax.set_ylim(-2, 102)
        ax.grid(True, axis="y")
    axes[0].set_ylabel("Success, position only (%)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "success_vs_tolerance", rows)


def error_vs_depth(D: Data):
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W * 0.66, 2.1), sharex=True)
    rows = curve(D, axes[0], D.lines, D.by_depth, "depth", "err_last1s_success_med", scale=100, band=False)
    rows += curve(D, axes[1], D.lines, D.by_depth, "depth", "closest_med", "closest_q25", "closest_q75", scale=100)
    axes[0].set_ylabel("Wrist error, reached goals (cm)")
    axes[1].set_ylabel("Closest approach, all goals (cm)")
    axes[1].axhline(100 * C.POS_TOL, color="#9aa1a7", lw=0.8, ls=(0, (2, 2)))
    for ax in axes:
        ax.set_xlabel("Goal lowering (m)")
        ax.grid(True, axis="y")
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "error_vs_depth", rows)


def balance_vs_depth(D: Data):
    # margins over the goal's last second (the hold): their minimum over the whole goal is set by the steps the legs
    # take, during which the static margin to the one supporting foot is negative by construction
    panels = [("com_margin_last1s_med", "CoM margin, last 1 s (cm)", 100),
              ("dcm_margin_last1s_med", "DCM margin, last 1 s (cm)", 100),
              ("tilt_max_deg_med", "Max base tilt (deg)", 1), ("ee_jitter_l_med", "Wrist jitter, last 1 s (mm)", 1),
              ("steps_mean", "Foot lift-offs per goal, mean", 1), ("fall_rate", "Falls (%)", 100)]
    keys = [k for k in D.lines if D.infos[k]["kind"] != "golem_ik"]
    fig, axes = plt.subplots(2, 3, figsize=(DBL_W, 3.3), sharex=True)
    rows = []
    for ax, (field, label, scale) in zip(axes.flat, panels):
        rows += curve(D, ax, keys, D.by_depth, "depth", field, scale=scale, band=False)
        ax.set_ylabel(label)
        ax.grid(True, axis="y")
        lo, hi = ax.get_ylim()                   # from zero: the margins' zero is the support polygon's edge
        ax.set_ylim(min(0.0, lo), hi + 0.08 * (hi - min(0.0, lo)))
    for ax in axes[1]:
        ax.set_xlabel("Goal lowering (m)")
    if keys:
        axes[0, 0].legend(loc="best")
    fig.tight_layout()
    save(fig, "balance_vs_depth", rows)


def sim2sim(D: Data):
    if not D.mujoco:
        return
    keys = [f"{D.hero}/isaac"] + [k for k in (f"{D.hero}/isaac_legs_blind",) if k in D.infos] + D.mujoco
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.35))
    rows = curve(D, axes[0], keys, D.by_height, "bin_mid", "success", "success_lo", "success_hi", scale=100)
    rows += curve(D, axes[1], keys, D.by_height, "bin_mid", "success_10cm", "success_10cm_lo", "success_10cm_hi",
                  scale=100)
    rows += curve(D, axes[2], keys, D.by_height, "bin_mid", "pelvis_h_last1s_med", band=False)
    axes[0].set_ylabel("Strict success (%)")
    axes[1].set_ylabel("Success within 10 cm (%)")
    for ax in axes[:2]:
        ax.set_ylim(-2, 102)
    axes[2].set_ylabel("Pelvis height (m)")
    axes[2].axhline(C.STANDING_PELVIS_HEIGHT, color="#9aa1a7", lw=0.8, ls=(0, (2, 2)), zorder=1)
    for ax in axes:
        shade_below_floor(D, ax, label=False)
        ax.set_xlim(0, 1.8)
        ax.set_xlabel("Lower wrist target height (m)")
        ax.grid(True, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.86 if len(labels) > 3 else 0.9))
    save(fig, "sim2sim", rows)


def learning_curve(D: Data):
    """Strict success (all goals, and below the standing table) and the mean curriculum level against environment
    transitions, one line per run with curve data (results/curves/<label>.csv from eval_curve.py)."""
    files = [f for f in sorted((C.RESULTS_DIR / "curves").glob("*.csv")) if not f.stem.endswith("_train")]
    curves = {f.stem: read_csv(f) for f in files}
    curves = {k: sorted(v, key=lambda r: r["transitions"]) for k, v in curves.items() if len(v) >= 2}
    if not curves:
        return
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W * 0.66, 2.15))
    rows = []
    order = [lab for lab in D.registry if lab in curves] + [lab for lab in curves if lab not in D.registry]
    for lab in order:
        cv = curves[lab]
        lam = cv[0].get("lambda_share")
        col = lam_color(lam if isinstance(lam, float) and not math.isnan(lam) else 1.0)
        x = np.array([r["transitions"] for r in cv]) / 1e6
        name = D.registry.get(lab, {}).get("name", lab)
        axes[0].plot(x, [100 * r["success"] for r in cv], color=col, lw=1.6, label=f"{name}, all goals")
        axes[0].plot(x, [100 * r["success_below"] for r in cv], color=col, lw=1.2, ls=(0, (3, 1.5)),
                     label=f"{name}, below the table")
        if "kd_mean" in cv[0]:
            axes[1].plot(x, [r["kd_mean"] for r in cv], color=col, lw=1.6, label=name)
            axes[1].fill_between(x, [r["kd_q25"] for r in cv], [r["kd_q75"] for r in cv], color=col, alpha=0.13, lw=0)
        rows += [{"label": lab, **r} for r in cv]
    axes[0].set_ylabel("Strict success (%)")
    axes[0].set_ylim(-2, 102)
    axes[1].set_ylabel("Curriculum depth level $k_d$")
    axes[1].set_ylim(-0.3, 10.3)
    for ax in axes:
        ax.set_xlabel("Environment transitions (millions)")
        ax.grid(True, axis="y")
    axes[0].legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    save(fig, "learning_curve", rows)


def training_curves(D: Data):
    """Training-time curves of every listed run (training_curves.py), smoothed over 50 updates."""
    runs = {}
    for lab in D.registry:
        path = C.RESULTS_DIR / "curves" / f"{lab}_train.csv"
        if path.is_file():
            runs[lab] = read_csv(path)
    if not runs:
        return
    panels = [("depth_level", None, "Curriculum depth level $k_d$"),
              ("reach_rate", "crouch_reach_rate", "Goals reached in training (%)"),
              ("goal_position_error_cm", "best_error_cm", "Wrist position error (cm)"),
              ("goal_orientation_error_rad", None, "Wrist orientation error (rad)"),
              ("pelvis_drop_cm", "target_drop_cm", "Drop during arm goals (cm)"),
              ("walk_error_m_s", None, "Walking velocity error (m/s)")]
    fig, axes = plt.subplots(2, 3, figsize=(DBL_W, 3.5), sharex=True)
    rows = []

    def smooth(y, w=50):
        y = np.asarray(y, dtype=float)
        k = np.ones(w) / w
        ok = np.isfinite(y)
        num = np.convolve(np.where(ok, y, 0.0), k, mode="same")
        den = np.convolve(ok.astype(float), k, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    for lab, cv in runs.items():
        key = f"{lab}/isaac"
        lam = D.infos[key]["lambda"] if key in D.infos else C.read_reward_share(Path(D.registry[lab]["run"]))
        col = lam_color(lam if lam is not None else 1.0)
        x = np.array([r["transitions"] for r in cv]) / 1e6
        name = D.registry[lab].get("name", lab)
        for ax, (main_key, second, label) in zip(axes.flat, panels):
            scale = 100 if "rate" in main_key else 1
            ax.plot(x, scale * smooth([r[main_key] for r in cv]), color=col, lw=1.4, label=name)
            if second:
                ax.plot(x, scale * smooth([r[second] for r in cv]), color=col, lw=1.1, ls=(0, (3, 1.5)))
            ax.set_ylabel(label)
            ax.grid(True, axis="y")
        rows += [{"label": lab, **r} for r in cv]
    axes[0, 1].text(0.03, 0.95, "dashed: goals below the table", transform=axes[0, 1].transAxes, ha="left", va="top",
                    fontsize=6.2, color=MUTED)
    axes[0, 2].text(0.98, 0.95, "dashed: closest approach", transform=axes[0, 2].transAxes, ha="right", va="top",
                    fontsize=6.2, color=MUTED)
    axes[1, 0].axhline(C.ROT_TOL, color="#9aa1a7", lw=0.8, ls=(0, (2, 2)))
    axes[1, 1].text(0.98, 0.05, "dashed: goal lowering", transform=axes[1, 1].transAxes, ha="right", va="bottom",
                    fontsize=6.2, color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("Environment transitions (millions)")
    axes[0, 0].legend(loc="lower right")
    fig.tight_layout()
    save(fig, "training_curves", rows[::10])


def walking(D: Data):
    """Velocity tracking error per command for every listed run (eval_walk.py), and the reach-walk trade-off."""
    walks = {}
    for lab in D.registry:
        path = C.RESULTS_DIR / lab / "isaac_walk" / "walk.csv"
        if path.is_file():
            walks[lab] = read_csv(path)
    if not walks:
        return
    cmds = [r["command"] for r in next(iter(walks.values()))]
    fig, ax = plt.subplots(figsize=(DBL_W * 0.62, 2.1))
    width = 0.8 / len(walks)
    rows = []
    for i, (lab, w) in enumerate(walks.items()):
        key = f"{lab}/isaac"
        col = D.style(key)["color"] if key in D.infos else lam_color(w[0]["lambda_share"] or 1.0)
        x = np.arange(len(cmds)) + (i - (len(walks) - 1) / 2) * width
        ax.bar(x, [100 * r["err_xy"] for r in w], width=width * 0.92, color=col, label=D.registry[lab].get("name", lab))
        rows += [{"label": lab, **r} for r in w]
    ax.set_xticks(np.arange(len(cmds)), [c.replace(" + ", "+\n") for c in cmds], fontsize=6.3)
    ax.set_ylabel("Velocity error (cm/s)")
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "walking", rows)
    # trade-off: reach success below the standing table against walking error, one point per run
    pts = []
    for lab, w in walks.items():
        key = f"{lab}/isaac"
        below = (D.infos.get(key) or {}).get("below_floor")
        if not below:
            continue
        moving = [r for r in w if r["command"] != "stand"]
        pts.append((lab, 100 * float(np.mean([r["err_xy"] for r in moving])), 100 * below["success"]))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for lab, ex, sy in pts:
        st = D.style(f"{lab}/isaac")
        ax.plot(ex, sy, "o", ms=7, color=st["color"], mec="white", mew=1.2, zorder=3)
        ax.annotate(D.registry[lab].get("name", lab), (ex, sy), textcoords="offset points", xytext=(6, 4), fontsize=6.5)
    ax.set_xlabel("Walking velocity error, moving commands (cm/s)")
    ax.set_ylabel("Strict success below the table (%)")
    ax.grid(True)
    save(fig, "tradeoff", [{"label": lab, "walk_err_cm_s": ex, "success_below_pct": sy} for lab, ex, sy in pts])


def workspace_maps(D: Data):
    keys = D.main + [k for k in (D.blind, D.golem) if k] + D.mujoco[:1]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(min(DBL_W, 1.75 * len(keys) + 0.6), 2.6), squeeze=False)
    xs, zs = np.linspace(-0.3, 1.1, 141), np.linspace(0.0, 2.0, 201)
    im = None
    for ax, k in zip(axes[0], keys):
        f = success_field(wrist_points(k), xs, zs)
        im = ax.imshow(100 * f, origin="lower", extent=(xs[0], xs[-1], zs[0], zs[-1]), cmap="Oranges", vmin=0,
                       vmax=100, aspect="equal", interpolation="bilinear")
        ax.set_title(D.name(k), fontsize=6.8, loc="left")
        ax.set_xlabel("Forward (m)")
        ax.axhline(D.floor, color="#55595e", lw=0.6, ls=(0, (2, 2)))
    axes[0][0].set_ylabel("Height (m)")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="Per-wrist success (%)")
    save(fig, "workspace_maps")


def main():
    D = Data()
    try:
        layers, target = robot_layers(D)
    except Exception as e:  # rendering needs MuJoCo with EGL; the plots do not
        print(f"[figures] robot render skipped: {e}")
        layers, target = [], None
    fig_hero(D, layers, target, "height")
    fig_hero(D, layers, target, "height", criterion="strict")
    fig_hero(D, layers, target, "depth")
    fig_hero_col(D, layers, target)
    fig_filmstrip(D)
    single(D, "success_vs_height", success_panel, "height")
    single(D, "success_vs_depth", success_panel, "depth")
    single(D, "pelvis_vs_height", pelvis_panel, "height")
    pelvis_drop_vs_depth(D)
    success_vs_tolerance(D)
    error_vs_depth(D)
    balance_vs_depth(D)
    sim2sim(D)
    learning_curve(D)
    training_curves(D)
    walking(D)
    workspace_maps(D)


if __name__ == "__main__":
    main()
