"""Score a GOLEM deploy run (evals/deploy/golem/run.sh's output): did the stack come up and engage, did the robot
stay up, did anything e-stop or fault, did only golem_safety write /lowcmd, and did it follow THE program's moves.

    python -m evals.deploy.golem.score --out <run.sh's output dir> [--gate evals/deploy/gate.yaml]

From the sim's own recording (sim.npz, 50 Hz: the pelvis's pose, the whole qpos; model.mjb for the joints' places)
the truth, not the robot's estimate: per program segment (events.json's marks, sim time), the pelvis velocity in its
yaw frame (central differences over 0.1 s) and yaw rate, resampled to 100 Hz; scored per move (its first second left
out) with the sport-mode tracking code (evals/common/legs_metrics.py): direction cos and turn sense, the gate's
criteria, plus linear and angular error and the feet's lift from the leg angles. Falls: the pelvis tilted past
fall_tilt_rad or under fall_height_m after the band's release. E-stops: golem_safety's estop_command_written or
estop diagnostic lines in launch.log. Faults: a "fault" node state. Arm goals: the node's own wrist distances
(~/wrist_error) over each goal's last 0.5 s. Writes <out>/score.json and score.md; exit 0 when every check passes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))  # the repo root
from evals.common import legs_metrics as lm  # noqa: E402
from evals.common import sportmode as S  # noqa: E402

GATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gate.yaml")
LEG_NAMES = ["left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint",
             "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_yaw_joint", "right_hip_pitch_joint",
             "right_hip_roll_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]
SKIP_S = 1.0


def yaw_of(q):
    w, x, y, z = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rp_of(q):
    w, x, y, z = q.T
    return np.stack([np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                     np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))], 1)


def leg_qpos_addrs(model_path):
    """The 12 leg joints' places in the recorded qpos: from joint_qpos.json (run.sh asks the sim's own mujoco), else
    from model.mjb (needs this mujoco to read that version's file)."""
    names_file = os.path.join(os.path.dirname(model_path), "joint_qpos.json")
    if os.path.isfile(names_file):
        places = json.load(open(names_file))
        out = []
        for n in LEG_NAMES:
            hits = [v for k, v in places.items() if k.endswith(n)]
            if not hits:
                return None
            out.append(hits[0])
        return np.array(out)
    import mujoco
    m = mujoco.MjModel.from_binary_path(model_path)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(m.njnt)]
    out = []
    for n in LEG_NAMES:
        hits = [j for j, name in enumerate(names) if name.endswith(n)]
        if not hits:
            return None
        out.append(int(m.jnt_qposadr[hits[0]]))
    return np.array(out)


def score(out_dir: str, gate: dict) -> dict:
    g = gate["golem"]
    ev = json.load(open(os.path.join(out_dir, "events.json")))
    events, info = ev["events"], ev.get("info", {})
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    released = next((e for e in events if e["name"] == "band_released"), None)
    check("engaged: the band released by the policy node", released is not None,
          f"at sim t {released['sim']:.1f} s" if released else "never (see launch.log)")
    states = [e.get("state", "") for e in events if e["name"] == "node_state"]
    faults = [s for s in states if "fault" in s]
    check("no controller fault", not faults, ", ".join(faults) or f"states: {' -> '.join(states)}")
    estops = []
    if os.path.isfile(os.path.join(out_dir, "launch.log")):
        for line in open(os.path.join(out_dir, "launch.log"), errors="replace"):
            if '"estop_command_written"' in line or "estop diagnostic" in line:
                estops.append(line.strip()[:200])
    check("no e-stop", not estops, f"{len(estops)} line(s); first: {estops[0]}" if estops else "none in launch.log")
    writers = info.get("lowcmd_writers") or []
    most = max((len(w[2]) for w in writers), default=0)
    check("one /lowcmd writer", 0 < most <= g["max_lowcmd_writers"],
          f"most at once: {most} ({sorted({n for w in writers for n in w[2]})})")

    sim = np.load(os.path.join(out_dir, "sim.npz"))
    t, pelvis, quat = sim["t"], sim["pelvis"], sim["quat"]
    upz = sim["upz"] if "upz" in sim else np.cos(np.linalg.norm(rp_of(quat), axis=1))
    after = t >= (released["sim"] if released else np.inf)
    fell_mask = after & ((upz < math.cos(g["fall_tilt_rad"])) | (pelvis[:, 2] < g["fall_height_m"]))
    fell_t = float(t[fell_mask][0]) if fell_mask.any() else None
    check("no fall after the release", fell_t is None, f"fell at sim t {fell_t:.1f} s" if fell_t else
          f"pelvis z min {pelvis[after, 2].min():.2f} m" if after.any() else "no samples after a release")

    # the program, at 100 Hz from its first segment to its end (or the fall)
    segs = [e for e in events if e["name"].startswith("segment:")]
    end_ev = next((e for e in events if e["name"] == "program_end"), None)
    result = {"checks": checks, "moves": []}
    if segs:
        t0, t1 = segs[0]["sim"], (end_ev["sim"] if end_ev else float(t[-1]))
        if fell_t is not None:
            t1 = min(t1, fell_t)
        grid = np.arange(t0, t1, S.DT)
        px = np.stack([np.interp(grid, t, pelvis[:, i]) for i in range(3)], 1)
        yaw = np.interp(grid, t, np.unwrap(yaw_of(quat)))
        h = int(round(0.05 / S.DT))
        vw = (np.roll(px, -h, 0) - np.roll(px, h, 0)) / (2 * h * S.DT)
        wz = (np.roll(yaw, -h) - np.roll(yaw, h)) / (2 * h * S.DT)
        c, s = np.cos(yaw), np.sin(yaw)
        vel = np.stack([c * vw[:, 0] + s * vw[:, 1], -s * vw[:, 0] + c * vw[:, 1], wz], 1)[h:-h]
        grid = grid[h:-h]
        cmd = np.zeros((len(grid), 3))
        seg_id = np.full(len(grid), -1)
        for i, e in enumerate(segs):
            m = grid >= e["sim"]
            cmd[m] = (e.get("vx", 0.0), e.get("vy", 0.0), e.get("wz", 0.0))
            seg_id[m] = i
        addrs = leg_qpos_addrs(os.path.join(out_dir, "model.mjb")) if "qpos" in sim else None
        q = (np.stack([np.interp(grid, t, sim["qpos"][:, a]) for a in addrs], 1) if addrs is not None else None)
        rp = np.stack([np.interp(grid, t, rp_of(quat)[:, i]) for i in range(2)], 1)
        step_mask = S.stepping(q) if q is not None else np.ones(len(grid), bool)
        moves = {}
        for i, e in enumerate(segs):
            if e.get("kind") != "move":
                continue
            in_seg = seg_id == i
            start = grid[in_seg][0] if in_seg.any() else None
            moves[e["name"][len("segment:"):]] = in_seg & (grid >= (start or 0) + SKIP_S)
        stride = int(round(0.70 / S.DT))
        res = lm.score([cmd], [vel], [np.ones(len(grid), bool)], stride, subsets={k: [v] for k, v in moves.items()})
        result["overall"] = res["tracking_vs_lagged"]
        for name, m in moves.items():
            seg = next(e for e in segs if e["name"] == "segment:" + name)
            tr = res.get("subsets", {}).get(name, {})
            linear, turn = math.hypot(seg.get("vx", 0), seg.get("vy", 0)) > 0.1, abs(seg.get("wz", 0)) > 0.1
            cos, sense = tr.get("cos_v_mean", float("nan")), tr.get("turn_sign_agree", float("nan"))
            ok = (not linear or (np.isfinite(cos) and cos >= g["min_cos_v"])) and \
                 (not turn or (np.isfinite(sense) and sense >= g["min_turn_sense"]))
            result["moves"].append({"move": name, "cmd": [seg.get("vx", 0), seg.get("vy", 0), seg.get("wz", 0)],
                                    "seconds": float(m.sum() * S.DT), "cos_v": cos, "turn_sense": sense,
                                    "lin_err": tr.get("lin_err_mean"), "ang_err": tr.get("ang_err_mean"),
                                    "stepping_share": float(step_mask[m].mean()) if m.any() else float("nan"),
                                    "pass": bool(ok and m.any())})
        # what arrived on /cmd_vel against what the driver meant (another publisher, e.g. Nav2's velocity_smoother,
        # would make the scores the driver's artifact)
        dn = os.path.join(out_dir, "drive.npz")
        if os.path.isfile(dn):
            got = np.load(dn)["cmds"]
            got = got[(got[:, 1] >= t0) & (got[:, 1] <= t1)]
            want = np.array([cmd[max(np.searchsorted(grid, x) - 1, 0)] for x in got[:, 1]]) if len(got) else got
            # a message within 0.1 s of a segment change may be either side of it
            near = np.array([min(abs(x - e["sim"]) for e in segs) < 0.1 for x in got[:, 1]]) if len(got) else []
            off = (np.abs(got[:, 2:5] - want).max(1) > 1e-3) & ~near if len(got) else np.array([])
            check("/cmd_vel carried only the program", len(got) > 0 and off.mean() <= 0.01,
                  f"{int(off.sum())} of {len(got)} messages off the program; publishers"
                  f" {info.get('cmd_vel_publishers')}")
        missing = [mv for mv in moves if not moves[mv].any()]
        check("every move followed (cos v, turn sense)", all(mv["pass"] for mv in result["moves"]) and not missing,
              ", ".join(f"{mv['move']} cos {mv['cos_v']:.2f} sense {mv['turn_sense']:.2f}" for mv in result["moves"])
              + (f"; never reached: {missing}" if missing else ""))
        if q is not None:
            result["feet"] = lm.foot_stats([q], [rp], [step_mask & (np.abs(cmd).sum(1) > 0)])
    else:
        check("every move followed (cos v, turn sense)", False, "the program never started")
    goals = [e for e in events if e["name"] == "arm_goal"]
    if goals:
        w = np.load(os.path.join(out_dir, "lmg_evals.npz"))["wrist"]
        reached = []
        for i, e in enumerate(goals):
            t_end = goals[i + 1]["sim"] if i + 1 < len(goals) else e["sim"] + 5.0
            rows = w[(w[:, 0] > t_end - 0.5) & (w[:, 0] <= t_end)] if len(w) else np.zeros((0, 1))
            # the node's ~/wrist_error: each wrist's distance to its goal (m)
            err = float(np.nanmax(rows[:, 1:3])) if len(rows) else float("nan")
            reached.append(bool(np.isfinite(err) and err <= g["arm_reach_m"]))
        share = float(np.mean(reached))
        check("arm goals reached", share >= g["min_arm_reached"], f"{sum(reached)} of {len(reached)}")
    result["passed"] = all(c["pass"] for c in checks)
    return result


def markdown(result: dict, out_dir: str) -> str:
    lines = [f"# GOLEM deploy run: {'PASS' if result['passed'] else 'FAIL'}", "", f"Run: `{out_dir}`", ""]
    meta = os.path.join(out_dir, "lmg_evals.yaml")
    if os.path.isfile(meta):
        lines += ["```", open(meta).read().strip(), "```", ""]
    lines += ["| Check | Pass | Detail |", "| --- | --- | --- |"]
    lines += [f"| {c['check']} | {'yes' if c['pass'] else '**no**'} | {c['detail']} |" for c in result["checks"]]
    if result["moves"]:
        lines += ["", "| Move | cmd (vx, vy, wz) | s | cos v | turn sense | lin err | ang err | stepping | pass |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        f = lambda v: "—" if v is None or not np.isfinite(v) else f"{v:.3f}"  # noqa: E731
        for m in result["moves"]:
            lines.append(f"| {m['move']} | {', '.join(f'{c:+.2f}' for c in m['cmd'])} | {m['seconds']:.1f} |"
                         f" {f(m['cos_v'])} | {f(m['turn_sense'])} | {f(m['lin_err'])} | {f(m['ang_err'])} |"
                         f" {f(m['stepping_share'])} | {'yes' if m['pass'] else '**no**'} |")
    if result.get("feet"):
        ft = result["feet"]
        lines += ["", f"Feet while commanded and stepping: clearance {ft['clearance_median_cm']:.1f} cm (sport mode"
                      f" 5.95), {ft['steps_per_s']:.2f} steps/s (2.78), knee swing {ft['knee_swing_median']:.2f} rad (0.40)."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", default=GATE)
    a = ap.parse_args()
    result = score(a.out, yaml.safe_load(open(a.gate)))
    with open(os.path.join(a.out, "score.json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    md = markdown(result, a.out)
    with open(os.path.join(a.out, "score.md"), "w") as f:
        f.write(md)
    print(md)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
