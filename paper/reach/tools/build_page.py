"""Build the result page paper/20261008-reach_cooperation.html from results/summary and figures/.

    python paper/reach/tools/build_page.py

Every number on the page comes from results/summary (analyze.py) and every figure from figures/ (figures.py,
downsampled and embedded), so rerunning `make paper` after a new evaluation refreshes the page. The page opens with
the summary, the hero figure, the key numbers and a gallery of every figure; definitions, the evaluation design and
the review of the brief are folded below. Equations are typeset by tools/texsvg.py (cache figures/texsvg_cache.json).
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
import sys
from datetime import date
from pathlib import Path

REACH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REACH))
sys.path.insert(0, str(REACH / "tools"))
from reachlib import common as C  # noqa: E402

import texsvg  # noqa: E402

PAGE = C.PAPER / "20261008-reach_cooperation.html"
SUMMARY = C.RESULTS_DIR / "summary"
ARTIFACT_URL = "https://claude.ai/artifact/LHVKbdmtDGtoQpmfghMZBn"


def esc(s) -> str:
    return html.escape(str(s))


def isnan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x, nd=0) -> str:
    return "–" if isnan(x) else f"{100 * x:.{nd}f}%"


def num(x, scale=1.0, nd=1, unit="") -> str:
    return "–" if isnan(x) else f"{scale * x:.{nd}f}{unit}"


def img(name: str, alt: str, max_px: int = 1700) -> str:
    path = C.FIG_DIR / f"{name}.png"
    if not path.is_file():
        return f'<div class="pending">{esc(name)} not drawn yet: it needs the conditions it compares.</div>'
    from PIL import Image
    im = Image.open(path)
    if im.width > max_px:
        im = im.resize((max_px, round(im.height * max_px / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="PNG", optimize=True)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" alt="{esc(alt)}">'


def table(head, rows, text_cols=(0,), caption=None):
    """Columns not in text_cols are numeric: right-aligned, tabular figures, no wrapping."""
    th = "".join(f"<th>{esc(h)}</th>" if i in text_cols else f'<th class="num">{esc(h)}</th>' for i, h in enumerate(head))
    body = []
    for r in rows:
        if isinstance(r, str):
            body.append(f'<tr class="grp"><td colspan="{len(head)}">{esc(r)}</td></tr>')
            continue
        body.append("<tr>" + "".join(f"<td>{c}</td>" if i in text_cols else f'<td class="num">{c}</td>'
                                     for i, c in enumerate(r)) + "</tr>")
    cap = f'<p class="tcap">{caption}</p>' if caption else ""
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>{cap}'


EQ = {
    "coupling": (r"\begin{bmatrix} r_U \\ r_L \end{bmatrix} = \begin{bmatrix} 1 & \lambda \\ \lambda & 1 \end{bmatrix}"
                 r"\begin{bmatrix} r_U^{\mathrm{task}} \\ r_L^{\mathrm{task}} \end{bmatrix} + "
                 r"\begin{bmatrix} r_U^{\mathrm{shape}} \\ r_L^{\mathrm{shape}} \end{bmatrix}", True),
    "success": (r"\begin{gathered} S = 1 \iff \exists\, t \le 3\,\mathrm{s}:\ \forall \tau \in [t, t+1\,\mathrm{s}],\ "
                r"\forall k \in \{l, r\}, \\ \lVert p_k(\tau) - p_k^\ast \rVert < 5\,\mathrm{cm}"
                r"\ \text{and}\ \theta_k(\tau) < 0.35\,\mathrm{rad} \end{gathered}", True),
    "dcm": (r"\xi = c_{xy} + \frac{\dot c_{xy}}{\omega}, \qquad \omega = \sqrt{g / z_c}", True),
    "ik": (r"\Delta q_k = \operatorname{clip}\!\big(J_k^\top (J_k J_k^\top + \mu^2 I)^{-1} e_k,\ \pm 0.1\,\mathrm{rad}\big),"
           r"\qquad q^{\mathrm{tgt}}_U = q_U + \Delta q + 0.2\,\bar a_U", True),
    "lam": (r"\lambda", False), "xi": (r"\xi", False), "dl": (r"d", False), "ee": (r"e", False),
}


def eq(svgs, key, number):
    return f'<div class="eq">{svgs[key]}<span class="eqno">({number})</span></div>'


GALLERY = [
    ("fig_hero", "Hero, double column: the reached-target workspace over the H1-2 with and without the legs' "
                 "cooperation, success and pelvis height against the lower wrist target's height."),
    ("fig_hero_strict", "The hero with the strict workspace: each wrist held within 5 cm and 0.35 rad for 1 s in at "
                        "least half of the trials (empty while per-wrist success stays below half everywhere)."),
    ("fig_hero_depth", "The hero with goal lowering d on the right panels: the same base goals at every d."),
    ("fig_hero_col", "Single-column hero: workspace and success."),
    ("fig_filmstrip", "The robot every second of the deepest goal the hero policy reached, wrist targets marked."),
    ("success_vs_tolerance", "Success against the position tolerance of the 1 s hold, for targets above and below "
                             "the standing table: how much a stricter test changes the ranking."),
    ("success_vs_height", "Strict success against the lower target's height, single column."),
    ("success_vs_depth", "Strict success against goal lowering, single column."),
    ("pelvis_vs_height", "Pelvis height over the goal's last second against target height."),
    ("pelvis_drop_vs_depth", "Pelvis drop against goal lowering; the dashed line is a drop equal to the lowering."),
    ("error_vs_depth", "Wrist error of reached goals, and closest approach of all goals."),
    ("balance_vs_depth", "Balance against goal lowering: CoM and DCM margins over the goal's last second, maximum "
                         "base tilt, wrist jitter, mean foot lift-offs per goal, falls."),
    ("sim2sim", "The hero policy in Isaac and in MuJoCo with RoboCasa's and with the training plant's joint dynamics: "
                "strict success, success within 10 cm, pelvis height."),
    ("tradeoff", "Reach against walking: success within 10 cm below and above the standing table against the "
                 "velocity-tracking error on the eight moving commands, one point per run."),
    ("walking", "Velocity-tracking error per command, one bar per run."),
    ("training_curves", "Training curves from each run's TensorBoard log: curriculum level, goals reached (all and "
                        "below the table), wrist errors, pelvis drop against goal lowering, walking error."),
    ("learning_curve", "Strict success on the fixed grid and curriculum depth level against environment transitions, "
                       "every saved checkpoint of each run (make curve)."),
    ("workspace_maps", "Per-wrist success over the sagittal plane, one map per condition."),
]


def s_walk() -> list[dict]:
    import csv
    path = SUMMARY / "walk.csv"
    return list(csv.DictReader(open(path))) if path.is_file() else []


def s_by_height(key: str) -> list[dict]:
    import csv
    out = []
    for r in csv.DictReader(open(SUMMARY / "by_height.csv")):
        if r["condition"] == key:
            out.append({k: (float(v) if k not in ("condition", "name") and v not in ("", None) else v)
                        for k, v in r.items()})
    return out


def main():
    s = json.loads((SUMMARY / "summary.json").read_text())
    conds, goals = s["conditions"], s["goals"]
    registry = {c["label"]: c for c in C.load_conditions()}
    floor_m = C.STANDING_PELVIS_HEIGHT + min(goals["standing_min_z"])
    svgs = texsvg.render([EQ[k] for k in EQ], cache_path=str(C.FIG_DIR / "texsvg_cache.json"))
    svgs = dict(zip(EQ, svgs))

    # -- the hero: as figures.py chooses it
    listed = [lab for lab in registry if f"{lab}/isaac" in conds]
    heroes = [lab for lab in listed if registry[lab].get("hero")]
    hero = heroes[0] if heroes else (max(listed, key=lambda lab: conds[f"{lab}/isaac"]["lambda"] or 0)
                                     if listed else None)

    def side(key, which, field="success"):
        b = (conds.get(key) or {}).get(f"{which}_floor")
        return None if not b else b.get(field)

    hk, bk, ik = (f"{hero}/isaac", f"{hero}/isaac_legs_blind", "golem_ik/kinematic") if hero else (None,) * 3
    if hero:
        mj = conds.get(f"{hero}/mujoco")
        prose = registry.get(hero, {}).get("prose") or f"the {conds[hk]['name']} policy"
        cap = esc(prose[:1].upper() + prose[1:])
        t10 = "success_10cm"
        lede = (f"{cap} brings both wrists within 10 cm and 0.6 rad of their targets for 1 s in"
                f" {pct(side(hk, 'below', t10))} of the goals whose lower wrist target lies below the standing table"
                f" ({floor_m:.2f} m) and in {pct(side(hk, 'above', t10))} of those above it.")
        if ik in conds:
            lede += (f" GOLEM's arm IK on a standing robot does so for {pct(side(ik, 'below', t10))} and"
                     f" {pct(side(ik, 'above', t10))}; the lowest 0.1 m band in which half of the unextended goals"
                     f" succeed is {num(conds[hk].get('floor10_m'), 1, 1, ' m')} for the policy and"
                     f" {num(conds[ik].get('floor10_m'), 1, 1, ' m')} for the IK.")
        if bk in conds:
            lede += (f" With its legs blind to the goal the same networks reach {pct(side(bk, 'below', t10))} and"
                     f" {pct(side(bk, 'above', t10))}.")
        lede += (f" Under the strict test (5 cm and 0.35 rad) the policy reaches {pct(side(hk, 'below'))} and"
                 f" {pct(side(hk, 'above'))}.")
        drop = side(hk, "below", "pelvis_drop_last1s_med")
        if not isnan(drop):
            lede += (f" Its pelvis drops a median {num(drop, 100, 0)}&nbsp;cm for the goals below the table and"
                     f" {num(side(hk, 'above', 'pelvis_drop_last1s_med'), 100, 0)}&nbsp;cm for those above.")
        # the other runs in the hero figure, one sentence each
        for lab, c in registry.items():
            k = f"{lab}/isaac"
            if lab == hero or k not in conds or c.get("hero_figure") is False:
                continue
            name = esc(c.get("prose") or f"the {conds[k]['name']} policy")
            lede += (f" {name[:1].upper() + name[1:]} reaches {pct(side(k, 'below', t10))} and"
                     f" {pct(side(k, 'above', t10))} within 10 cm, with its pelvis"
                     f" {num(side(k, 'below', 'pelvis_drop_last1s_med'), 100, 0)} and"
                     f" {num(side(k, 'above', 'pelvis_drop_last1s_med'), 100, 0)}&nbsp;cm low.")
        walk_rows = {r["label"]: r for r in s_walk()}
        if walk_rows:
            lede += " Velocity-tracking error on the eight moving walking commands: " + ", ".join(
                f"{esc(registry.get(lab, {}).get('name', lab))}: {num(float(r['err_xy_moving']), 100, 0)}&nbsp;cm/s"
                for lab, r in walk_rows.items() if lab in registry) + "."
        if mj:
            lede += (f" In MuJoCo with RoboCasa's joint dynamics {esc(prose)} reaches {pct(mj.get('success_10cm'))} of all goals"
                     f" within 10 cm, against {pct(conds[hk].get('success_10cm'))} in Isaac.")
    else:
        lede = "No trained policy has been evaluated yet."

    # -- key numbers: one group per run, one row per evaluation
    variants = [("isaac", "Isaac"), ("isaac_legs_blind", "Isaac, legs blind"), ("isaac_arms_ik", "Isaac, arms IK only"),
                ("isaac_trueodom", "Isaac, true odometry"), ("mujoco", "MuJoCo, RoboCasa joints"),
                ("mujoco_trueodom", "MuJoCo, RoboCasa joints, true odometry"),
                ("mujoco_isaacphys", "MuJoCo, training joints"),
                ("mujoco_isaacphys_trueodom", "MuJoCo, training joints, true odometry"),
                ("mujoco_legs_blind", "MuJoCo, legs blind")]

    def key_row(k, label):
        v = conds[k]
        return [esc(label), f"{v['n']:,}", pct(v["success"]), pct(side(k, "below")), pct(side(k, "above")),
                pct(v.get("success_10cm")),
                num(v.get("floor10_m"), 1, 1, " m") if not isnan(v.get("floor10_m", float("nan"))) else "–",
                num(side(k, "below", "pelvis_drop_last1s_med"), 100, 0, " cm"),
                num(v.get("hold_err_pos_med"), 100, 1, " cm"), pct(v["fall_rate"], 1)]

    key_rows = []
    for lab, c in registry.items():
        present = [(f"{lab}/{sim}", text) for sim, text in variants if f"{lab}/{sim}" in conds]
        key_rows.append(c.get("name", lab) + ("" if present else f" (pending: {c.get('status', 'not evaluated')})"))
        key_rows += [key_row(k, text) for k, text in present]
    if "golem_ik/kinematic" in conds:
        key_rows.append("GOLEM arm IK, standing robot")
        key_rows.append(key_row("golem_ik/kinematic", "Kinematic, deployed settings"))

    gallery = "".join(
        f'<figure class="card">{img(name, cap)}<figcaption><code>figures/{name}.pdf</code> {cap}</figcaption></figure>'
        for name, cap in GALLERY)

    parity = C.RESULTS_DIR / "_parity" / "report.json"
    if parity.is_file():
        rep = json.loads(parity.read_text())
        obs_err = max(v for k, v in rep.items() if not k.startswith("target") and v == v)
        parity_text = (f"From the same state, the MuJoCo port reproduces every observation term to {obs_err:.1e} and the"
                       f" joint targets to {rep['target.arm_pos']:.1e}&nbsp;rad (<code>tools/parity_check.py</code>). The"
                       " arms' IK step matches only with the wrist Jacobian at the link's centre of mass, where PhysX"
                       " computes it.")
    else:
        parity_text = ""

    # -- open items: each names the measurement that settles it, with the numbers that raise it
    items = []
    pending = [c.get("name", lab) for lab, c in registry.items() if f"{lab}/isaac" not in conds]
    if pending:
        items.append(f"{esc(', '.join(pending))}: set <code>run:</code> in <code>conditions.yaml</code> and run "
                     "<code>make -C paper/reach all</code>. Comparing their pelvis drop and 10 cm success below the "
                     "table with the hero's tells whether the crouch depends on the reward share.")
    if hero:
        h = conds[hk]
        it = (f"Position limits the strict test: {pct(h.get('pos_ok'))} of the hero's trials hold both wrists within "
              f"5 cm for 1 s and {pct(h.get('rot_ok'))} within 0.35 rad")
        if ik in conds:
            it += f" (GOLEM IK: {pct(conds[ik].get('pos_ok'))} and {pct(conds[ik].get('rot_ok'))})"
        ext0 = [(k, t) for k, t in ((f"{hero}/isaac_arms_ik", "arms IK only"), (f"{hero}/isaac_trueodom", "true odometry"))
                if k in conds]
        if ext0:
            it += ". On the unextended goals, within 10 cm: " + ", ".join(
                f"{t} {pct(conds[k].get('success_10cm'))}" for k, t in ext0) + \
                f", against {pct((h.get('ext0') or {}).get('success_10cm'))} for the full policy."
        else:
            it += (". <code>eval_isaac.py --variant arms_ik --ext0</code> and <code>--odometry true --ext0</code> "
                   "separate the learned residual and the pelvis estimator's drift.")
        items.append(it)
        hb = sorted([r for r in s_by_height(hk) if r["n"] >= 10], key=lambda r: r["bin_lo"])
        if hb:
            top = max(hb, key=lambda r: r.get("success_10cm", 0.0))
            high = [r for r in hb if r["bin_lo"] > top["bin_lo"] and r.get("success_10cm", 1.0) < 0.5 * top["success_10cm"]]
            if high:
                r0 = high[0]
                items.append(f"High goals: the hero's 10 cm success falls from {pct(top['success_10cm'])} at "
                             f"{top['bin_lo']:.1f}&ndash;{top['bin_lo'] + 0.1:.1f} m to {pct(r0['success_10cm'])} at "
                             f"{r0['bin_lo']:.1f}&ndash;{r0['bin_lo'] + 0.1:.1f} m, with the pelvis "
                             f"{num(C.STANDING_PELVIS_HEIGHT - r0['pelvis_h_last1s_med'], 100, 0, ' cm')} below "
                             "standing height. Next: the training curriculum's goal heights at depth level 10 against "
                             "the grid's (<code>commands.py</code> lowers goals by up to 1.0 m).")
        mjk = [(k, t) for k, t in ((f"{hero}/mujoco", "RoboCasa's joints"), (f"{hero}/mujoco_isaacphys",
                                                                           "the training plant's joints"))
               if k in conds]
        if mjk:
            items.append("MuJoCo transfer: within 10 cm the hero reaches " + ", ".join(
                f"{pct(conds[k].get('success_10cm'))} with {t}" for k, t in mjk) +
                f" against {pct(h.get('success_10cm'))} in Isaac. The policy loop matches Isaac to 2e-5 from the same "
                "state, so the gap lies in the plant; next: <code>tools/parity_check.py</code> extended from one step "
                "to a second of stance from the same state, to find the first quantity that diverges.")
    for r in s_walk():
        if r["label"] in registry and float(r["err_xy_moving"]) > 0.15:
            items.append(f"{esc(registry[r['label']].get('name', r['label']))} does not follow velocity commands "
                         f"(mean error {num(float(r['err_xy_moving']), 100, 0)}&nbsp;cm/s on the moving commands, no "
                         "falls; the same with the training friction range). Next: its legs' shared arm reward terms "
                         "during navigation in TensorBoard (<code>Episode_Reward/legs/arms_*</code>), to see whether "
                         "they pay the legs for standing still.")
    items.append("Learning curves on the fixed grid: <code>make -C paper/reach curve RUN=&lt;run&gt; LABEL=&lt;label&gt;</code>"
                 " (every saved checkpoint, unextended goals).")
    open_items = "\n".join(f"<li>{x}</li>" for x in items)

    css = (REACH / "tools" / "page.css").read_text()
    today = date.today().isoformat()
    body = f"""<title>H1-2 cooperative reach</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
{css}
{texsvg.CSS}
</style>
<div class="wrap">
<header><div class="col">
  <div class="eyebrow">{today} · LocoManip-Marl-Direct-v0, marl-direct · Isaac Sim 5.1 and MuJoCo 3.15</div>
  <h1>Reach below standing height with reward-coupled leg and arm policies on the Unitree H1-2</h1>
  <p class="lede">{lede}</p>
  <p class="byline">Local file <code>paper/20261008-reach_cooperation.html</code> in Locomanipulation_game · artifact
  {esc(ARTIFACT_URL)} · pipeline <code>paper/reach/</code> · {goals['goals']:,} fixed goals, one trial per goal and seed</p>
</div></header>

<section id="hero"><div class="col">
<figure>{img("fig_hero", "Workspace, success and pelvis height of the hero policy")}
<figcaption>(a) Wrist targets that a wrist ended within 10 cm of in at least half of the trials (5 cm kernel), side
view over the H1-2: the hero policy (orange) and GOLEM's arm IK on a standing robot (blue); the robot is drawn at the
end of the deepest goal the policy reached under the strict test, the standing robot faint behind it, its two wrist
targets marked &times;. (b) Success within 10 cm and 0.6 rad for 1 s against the lower wrist target's height, 0.1 m
bins, 95% Wilson band; the strict version is <code>fig_hero_strict</code> in the gallery. (c) Median pelvis height
over the goal's last second, interquartile band; the dotted line is the nominal standing height. Oranges: &lambda;
runs; dark grey: the hero's networks with blind legs; dashed blue: GOLEM's arm IK. Grey band: below the standing
table's lowest pose.</figcaption></figure>
</div>
<div class="col">
{table(["Evaluation", "Trials", "Success", "Below table", "Above table", "At 10 cm", "10 cm floor", "Pelvis drop, low goals",
        "1 s hold error", "Falls"], key_rows, text_cols=(0,),
       caption="Success: strict test, both wrists within 5 cm and 0.35 rad for 1 s; below / above table: the goals whose "
               "lower target is below / above the standing table's lowest pose; at 10 cm: the same test at 10 cm and "
               "0.6 rad. 10 cm floor: lowest 0.1 m band of lower target height in which half of the unextended "
               "goals succeed at 10 cm. 1 s hold error: median over trials of the best "
               "1 s window's worst wrist position error.")}
</div></section>

<section id="gallery"><div class="col">
<h2>Figures</h2>
<p>Every figure is in <code>paper/reach/figures/</code> as PDF (fonts embedded) and PNG, with its plotted numbers in
<code>figures/data/</code>.</p>
</div>
<div class="gallery">{gallery}</div>
</section>

<section id="add"><div class="col">
<h2>Adding a run</h2>
<p>Set the run directory (<code>run:</code>) of each &lambda; in <code>paper/reach/conditions.yaml</code>, and of
<code>whole</code> for the one-agent whole-body baseline (<code>LocoManip-WholeBody-Direct-v0</code>, recognised from
its checkpoint), then from the Locomanipulation_game checkout:</p>
<pre>conda activate env_isaaclab51        # Isaac Sim 5.1, Isaac Lab main 2.3.2, skrl 2.1
make -C paper/reach all              # evaluates every listed run without current results, then figures, numbers, page
                                     # (DEVICE=cpu when another job holds the GPU)</pre>
<p>Each Isaac variant takes about 5 minutes on the GPU or 20 on 12 CPU cores, each MuJoCo variant 3 minutes on 8
cores. The paper text quotes numbers as <code>\\reach{{l1/isaac}}{{success_below}}</code> from
<code>paper/reach/tex/results_macros.tex</code>; a number not computed yet prints ??.</p>
</div></section>

<section id="details"><div class="col">
<details><summary>Terms and metrics</summary>
<dl class="terms">
<dt>{svgs['lam']}, reward share</dt><dd>The share of the other agent's task reward each agent receives
(<code>REWARD_SHARE</code>), one scalar in both directions, read from each run's <code>params/env.yaml</code>:</dd>
</dl>
{eq(svgs, 'coupling', 1)}
<dl class="terms">
<dt>Goal</dt><dd>A pair of wrist poses placed in the standing frame (the pelvis's x, y and yaw when the goal
starts, origin 1.0024&nbsp;m above the ground) and fixed in the world for 4&nbsp;s.</dd>
<dt>Goal lowering {svgs['dl']}, extension {svgs['ee']} (m)</dt><dd>Base goals come from the task's
standing-reachable table; each is lowered by {svgs['dl']} and pushed {svgs['ee']} outward from each shoulder in the
ground plane. The table's lowest pose is {floor_m:.2f}&nbsp;m above the ground.</dd>
<dt>Strict success</dt><dd>Both wrists inside 5&nbsp;cm and 0.35&nbsp;rad of their true world targets for 1&nbsp;s
without a break (the task's reach test):</dd>
</dl>
{eq(svgs, 'success', 2)}
<dl class="terms">
<dt>1 s hold error (cm)</dt><dd>Over every 1&nbsp;s window of the goal, the worst position error of either wrist;
the best window's value. A trial succeeds at position tolerance &tau; alone when this is below &tau;.</dd>
<dt>Pelvis drop (cm)</dt><dd>Pelvis height when the goal starts minus its mean over the goal's last second.</dd>
<dt>CoM / DCM margin (cm)</dt><dd>Signed distance from the centre of mass's ground projection (or {svgs['xi']}) to
the edge of the support polygon of the feet carrying more than 20&nbsp;N, positive inside:</dd>
</dl>
{eq(svgs, 'dcm', 3)}
<dl class="terms">
<dt>Fall</dt><dd>Base tilt beyond 1&nbsp;rad, the task's termination.</dd>
<dt>Foot slip, steps, wrist jitter</dt><dd>Horizontal speed of a foot carrying more than 50&nbsp;N; foot lift-offs
during the goal; RMS distance of a wrist from its mean position over the last second.</dd>
</dl>
</details>

<details><summary>Evaluation design</summary>
<p>{goals['goals']:,} goals: {goals['bases']} base pose pairs &times; lowerings 0 to {max(goals['depths']):.1f}&nbsp;m in
0.1&nbsp;m steps &times; extensions {", ".join(f"{e:.1f}" for e in goals['exts'])}&nbsp;m. A trial resets the robot
with the training distribution, stands for 2&nbsp;s under a zero velocity command, then holds one goal for 4&nbsp;s.
Actors act through their mean actions with observation noise on; the learned pelvis estimator moves the goal they
observe; the depth curriculum is frozen; Isaac's training randomization is pinned at its midpoint and pushes are off.
Variants of one checkpoint: <b>legs blind</b> (the legs observe navigation inputs, without the goal), <b>arms IK
only</b> (the arms' residual held at zero), <b>true odometry</b>. GOLEM's arm IK (<code>h12_ros2_controller</code>,
<code>IKSolver.solve_ik_reduced</code>) solves each goal on a standing robot with its deployed settings. The arm action
of the policies is</p>
{eq(svgs, 'ik', 4)}
<p>{parity_text}</p>
</details>

<details><summary>Review of the experiment brief</summary>
<ul>
<li>Success test, curriculum thresholds and the single symmetric &lambda; match the code.</li>
<li>The brief's depth grid (0 to 0.4 m) is out of date: the code lowers goals up to 1.0&nbsp;m
(z<sub>max</sub>&nbsp;=&nbsp;0.1&nbsp;k<sub>d</sub>), and the run at 52.8k steps had 77% of its environments at the
deepest level. The grid spans 0 to 1.0&nbsp;m and results are reported against absolute target height.</li>
<li>The blind-legs variant measures the reach the legs add from one checkpoint; &lambda;&nbsp;=&nbsp;0 is still
needed to tie the crouch to reward sharing.</li>
<li>Learning efficiency can be read from saved files: each <code>estimator_&lt;N&gt;.pt</code> stores every
environment's curriculum level.</li>
<li>A model-based whole-body baseline would be a stronger fourth line than a monolithic policy.</li>
</ul>
</details>

<details><summary>Open items</summary>
<ul>
{open_items}
</ul>
</details>
</div></section>

<footer><div class="col">
<p>Rebuild: <code>make -C paper/reach paper</code>. Trials: <code>paper/reach/results/&lt;label&gt;/&lt;sim&gt;/trials.csv</code>;
goals <code>paper/reach/goals/eval_goals_v1.csv</code>.</p>
</div></footer>
</div>
"""
    PAGE.write_text(body)
    print(f"[build_page] {PAGE} ({PAGE.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
