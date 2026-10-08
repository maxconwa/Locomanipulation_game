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
    ("fig_hero_final10", "The hero with a looser workspace: wrist targets within 10 cm at the goal's end in at least "
                         "half of the trials."),
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
    ("balance_vs_depth", "Balance: CoM and DCM margins, base tilt, wrist jitter, foot slip, falls."),
    ("sim2sim", "The hero policy in Isaac and in MuJoCo with RoboCasa's and with the training plant's joint dynamics."),
    ("tradeoff", "Reach against walking: strict success below the standing table against the velocity-tracking "
                 "error of the same checkpoint, one point per run (the two agents' tasks together)."),
    ("walking", "Velocity-tracking error per command, one bar per run."),
    ("learning_curve", "Strict success and curriculum depth level against environment transitions, every saved "
                       "checkpoint of each run."),
    ("workspace_maps", "Per-wrist success over the sagittal plane, one map per condition."),
]


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
        lede = (f"{esc(conds[hk]['name'])} reaches {pct(side(hk, 'below'))} of the goals whose lower wrist target"
                f" lies below the standing table ({floor_m:.2f} m) and {pct(side(hk, 'above'))} of those above it.")
        if bk in conds:
            lede += (f" With its legs blind to the goal the same networks reach {pct(side(bk, 'below'))} and"
                     f" {pct(side(bk, 'above'))}.")
        if ik in conds:
            lede += f" GOLEM's arm IK on a standing robot reaches {pct(side(ik, 'below'))} and {pct(side(ik, 'above'))}."
        drop = side(hk, "below", "pelvis_drop_last1s_med")
        if not isnan(drop):
            lede += f" For the low goals the pelvis drops a median {num(drop, 100, 0)}&nbsp;cm"
            dropb = side(bk, "below", "pelvis_drop_last1s_med") if bk in conds else None
            lede += f" ({num(dropb, 100, 0)}&nbsp;cm with blind legs)." if not isnan(dropb) else "."
        if mj:
            lede += (f" In MuJoCo with RoboCasa's joint dynamics it reaches {pct(mj['success'])} of all goals,"
                     f" against {pct(conds[hk]['success'])} in Isaac.")
    else:
        lede = "No trained policy has been evaluated yet."

    # -- key numbers
    order = [f"{lab}/{sim}" for lab in registry for sim in ("isaac", "isaac_legs_blind", "isaac_arms_ik", "isaac_trueodom",
                                                            "mujoco", "mujoco_isaacphys", "mujoco_legs_blind")
             if f"{lab}/{sim}" in conds] + (["golem_ik/kinematic"] if "golem_ik/kinematic" in conds else [])
    key_rows = []
    for k in order:
        v = conds[k]
        key_rows.append([esc(v["name"]), esc(v["base_sim"]), f"{v['n']:,}", pct(v["success"]),
                         pct(side(k, "below")), pct(side(k, "above")),
                         num(v.get("operational_floor_m"), 1, 2, " m") if not isnan(v.get("operational_floor_m")) else "–",
                         num(side(k, "below", "pelvis_drop_last1s_med"), 100, 0, " cm"),
                         num(v.get("hold_err_pos_med"), 100, 1, " cm"), pct(v["fall_rate"], 1)])
    for lab, c in registry.items():
        if f"{lab}/isaac" not in conds:
            key_rows.append([esc(c.get("name", lab)), "–", "–", "–", "–", "–", "–", "–", "–",
                             f'<span class="pend">{esc(c.get("status", "not evaluated"))}</span>'])

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
<figcaption>(a) Wrist targets reached in at least half of the trials (per-wrist success, 5 cm kernel), side view
over the H1-2, with blind legs (blue) and with the legs cooperating (orange); the robot is drawn at the end of the
deepest goal it reached, the standing robot faint behind it, its targets marked &times;. (b) Strict success against
the lower wrist target's height, 0.1 m bins, 95% Wilson band. (c) Median pelvis height over the goal's last second,
interquartile band. Grey: below the standing table's lowest pose. GOLEM's arm IK is dashed.</figcaption></figure>
</div>
<div class="col">
{table(["Condition", "Sim", "Trials", "Success", "Below table", "Above table", "Floor", "Pelvis drop, low goals",
        "1 s hold error", "Falls"], key_rows, text_cols=(0, 1),
       caption="Strict success: both wrists within 5 cm and 0.35 rad for 1 s. Below / above table: goals whose lower "
               "target is below / above the standing table's lowest pose. Floor: lowest height from which every "
               "0.1 m band above reaches 80% (unextended goals). 1 s hold error: median over trials of the best "
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
<p>Set the run directory of each &lambda; in <code>paper/reach/conditions.yaml</code>, then from the
Locomanipulation_game checkout:</p>
<pre>conda activate env_isaaclab51        # Isaac Sim 5.1, Isaac Lab main 2.3.2, skrl 2.1
make -C paper/reach all              # evaluates every listed run without results, then figures, numbers, page
                                     # (DEVICE=cpu when another job holds the GPU)</pre>
<p>A run takes about 5 minutes on the GPU (Isaac, three variants) and 20 on 8 CPU cores (MuJoCo). The paper text
quotes numbers as <code>\\reach{{l1/isaac}}{{success_below}}</code> from <code>paper/reach/tex/results_macros.tex</code>;
a number not computed yet prints ??.</p>
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
<li>&lambda;&nbsp;=&nbsp;1, 0.5 and 0 at 91.2k steps: set their runs in <code>conditions.yaml</code> and run
<code>make all</code>.</li>
<li>The MuJoCo gap: compare <code>sim2sim</code> with RoboCasa's and the training plant's joint dynamics; the
policy loop matches Isaac to 2e-5 from the same state, so the gap lies in the dynamics.</li>
<li>Learning curves: evaluate every 4800-step checkpoint of each run with <code>eval_isaac.py --seeds 0</code>.</li>
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
