"""A video of a policy in MuJoCo running THE command program (evals/common/program.py).

    python -m evals.mujoco.get_video --legs <ckpt or export dir> [--arms <ckpt or export dir>]
    python -m evals.mujoco.get_video --arms <ckpt or export dir>                  (the pelvis pinned)
        [--plant isaac|robocasa] [--seed 0] [--every 2] [--out <mp4>]

Legs active: hung on the hoist with the feet off the ground, lowered onto its feet, the rope released (training's
rope model: a spring-damper at a hook on the torso carrying the program's share of the weight), then the moves,
each followed by a stop. Without --arms every arm joint is held at 0 rad. Arms active: a goal every GOAL_EVERY_S from
the arms' own goal table, drawn as a thick RGB frame (x red, y green, z blue) at the goal and a thin one at the wrist
it moves. Arms alone: the pelvis pinned, goals only. On the ground: velocity arrows and turn-rate arcs, commanded green,
actual orange. Two tiles: a third-person tracking view and a side view.

A checkpoint is exported once into logs/evals/<run>/mujoco/exports. Writes logs/evals/<run>/mujoco/video_<N>[...].mp4
(the legs' run when given, else the arms') and a smaller copy (_small.mp4, 960 px, if ffmpeg is there). Needs
mujoco with an offscreen GL (MUJOCO_GL=egl is set if unset), imageio[ffmpeg], opencv.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # the repo root

import numpy as np  # noqa: E402

from evals.common import outputs, program, runs  # noqa: E402

AXIS_RGBA = [np.array(c, np.float32) for c in ([0.95, 0.1, 0.1, 1.0], [0.1, 0.85, 0.1, 1.0], [0.15, 0.3, 1.0, 1.0])]
CMD_RGBA = np.array([0.1, 0.9, 0.2, 0.9], np.float32)
ACT_RGBA = np.array([1.0, 0.55, 0.0, 0.9], np.float32)
ROPE_RGBA, HOOK_RGBA = np.array([1.0, 0.5, 0.05, 1.0], np.float32), np.array([0.9, 0.1, 0.1, 1.0], np.float32)
M_PER_MPS, RAD_PER_RADPS, ARROW_Z = 2.0, 2.0, 0.03
# training's rope (MarlEventCfg.prepose): spring-damper at a hook 0.55 m above the torso link
HOOK, ROPE_TOP = np.array([0.0, 0.0, 0.55]), 2.6
ROPE_KZ, ROPE_CZ, ROPE_KXY, ROPE_CXY = 3000.0, 400.0, 200.0, 100.0
TILE_W, TILE_H = 960, 540


def resolve(spec: str) -> tuple[Path, Path | None]:
    """(export dir, checkpoint or None) for a checkpoint path or an export folder."""
    from evals.mujoco.policy import export_for
    p = Path(spec).resolve()
    if p.is_dir() and (p / "export.yaml").is_file():
        return p, None
    run_dir = runs.run_dir_of(p)
    return export_for(p, outputs.eval_dir(runs.name(run_dir), "mujoco", "exports")), p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legs")
    ap.add_argument("--arms")
    ap.add_argument("--plant", default="isaac", choices=["isaac", "robocasa"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=2, help="Render every n-th policy step (50 fps at 100 Hz).")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not (a.legs or a.arms):
        ap.error("--legs and/or --arms")

    import cv2
    import imageio
    import mujoco
    import torch

    from evals.mujoco import runner as R
    from evals.mujoco.policy import combined_settings
    from evals.mujoco.sim import qmul, qrot

    legs_export, legs_ckpt = resolve(a.legs) if a.legs else (None, None)
    arms_export, arms_ckpt = resolve(a.arms) if a.arms else (None, None)
    settings = combined_settings(legs_export, arms_export)
    legs, arms = legs_export is not None, arms_export is not None
    sim = R.make_sim(settings, legs_export, arms_export, a.plant, pinned=not legs, hold_zero=legs and not arms)
    model, d = sim.m, sim.d
    model.vis.global_.offwidth, model.vis.global_.offheight = TILE_W, TILE_H
    rng = np.random.default_rng(a.seed)
    sim.reset(rng)
    R.reset_actors(sim)
    sim.pelvis_goals = legs and arms
    r = settings.get("command_ranges") or {}
    full = tuple(float(max(abs(v[0]), abs(v[1]))) for v in (r.get("lin_vel_x", [-0.55, 0.55]),
                                                            r.get("lin_vel_y", [-0.54, 0.54]),
                                                            r.get("ang_vel_z", [-0.97, 0.97])))
    segs = program.segments(legs, full)
    goal_at = set(int(round(t / sim.policy_dt)) for t in program.goal_times(segs, legs)) if arms else set()
    tables, goal_rng = (R.goal_table(settings), np.random.default_rng(a.seed + 1)) if arms else (None, None)
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    weight = float(model.body_mass.sum() * -model.opt.gravity[2])

    def hook_point():
        return d.xpos[torso] + d.xmat[torso].reshape(3, 3) @ HOOK

    anchor0 = None
    if legs:  # hung: the default pose, the feet HANG_LIFT off the ground, the hook's anchor where it is
        d.qpos[sim.qadr], d.qvel[:] = sim.default, 0.0
        d.qpos[3:7], d.qpos[0:3] = (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.2)
        mujoco.mj_forward(model, d)
        sole = min(d.xpos[f][2] for f in sim.feet) + R.C.SOLE_Z
        d.qpos[2] += program.HANG_LIFT - sole
        mujoco.mj_forward(model, d)
        sim.applied = sim.default.copy()
        sim.ik_q = d.qpos[sim.qadr][sim.arm_idx].copy()
        sim._prev_root = (d.qpos[0:3].copy(), d.qpos[3:7].copy())
        anchor0 = hook_point().copy()

    def rope(t):
        """Training's rope force for this step; returns (share carried, anchor) or (0, None)."""
        d.qfrc_applied[:] = 0.0
        share = program.rope_share(t) if legs else 0.0
        if share <= 0.0:
            return 0.0, None
        anchor = anchor0.copy()
        anchor[2] -= program.anchor_drop(t)
        pos = hook_point()
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, torso, v6, 0)
        vel = v6[3:6] + np.cross(v6[0:3], pos - d.xpos[torso])
        f = np.zeros(3)
        f[2] = max(share * (weight + ROPE_KZ * (anchor[2] - pos[2]) - ROPE_CZ * vel[2]), 0.0)
        f[:2] = share * (ROPE_KXY * (anchor[:2] - pos[:2]) - ROPE_CXY * vel[:2])
        mujoco.mj_applyFT(model, d, f, np.zeros(3), pos, torso, d.qfrc_applied)
        return f[2] / weight, anchor

    def connector(scn, kind, width, p0, p1, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
        mujoco.mjv_connector(g, kind, width, np.asarray(p0, float), np.asarray(p1, float))
        scn.ngeom += 1

    def arrow(scn, origin, v, rgba):
        dv = np.r_[v[:2], 0.0] * M_PER_MPS
        if np.linalg.norm(dv) >= 0.02:
            connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, 0.03, origin, origin + dv, rgba)

    def arc(scn, center, yaw, wz, radius, rgba):
        sweep = wz * RAD_PER_RADPS
        if abs(sweep) < 0.05:
            return
        n = max(int(abs(sweep) / 0.1), 2)
        pts = [center + radius * np.array([math.cos(yaw + sweep * i / n), math.sin(yaw + sweep * i / n), 0.0])
               for i in range(n + 1)]
        for p0, p1 in zip(pts[:-2], pts[1:-1]):
            connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, p0, p1, rgba)
        connector(scn, mujoco.mjtGeom.mjGEOM_ARROW, 0.03, pts[-2], pts[-1], rgba)

    def frame_geom(scn, pose, length, width, alpha=1.0):
        for axis, rgba in enumerate(AXIS_RGBA):
            tip = pose[:3] + qrot(pose[3:], np.eye(3)[axis]) * length
            connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, width, pose[:3], tip, rgba * [1, 1, 1, alpha])

    def text(img, lines, color=(255, 255, 255), bottom=False):
        for i, line in enumerate(lines):
            y = (TILE_H - 16 - 24 * (len(lines) - 1 - i)) if bottom else 26 + 24 * i
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)

    renderer = mujoco.Renderer(model, TILE_H, TILE_W)
    cams = []
    for az, el in ((135.0, -15.0), (90.0, -8.0)):
        c = mujoco.MjvCamera()
        c.type, c.trackbodyid, c.distance, c.azimuth, c.elevation = mujoco.mjtCamera.mjCAMERA_TRACKING, sim.pelvis, 3.2, az, el
        cams.append(c)
    owner = legs_ckpt or arms_ckpt
    if a.out:
        out = Path(a.out)
    elif owner is not None:
        tag = f"video_{runs.step_of(owner)}" + (f"_with_{runs.short(runs.run_dir_of(arms_ckpt))}_{runs.step_of(arms_ckpt)}"
                                               if legs_ckpt and arms_ckpt else "") + ("" if a.plant == "isaac" else f"_{a.plant}")
        out = outputs.eval_dir(runs.name(runs.run_dir_of(owner)), "mujoco", "legs" if legs else "arms") / f"{tag}.mp4"
    else:
        ap.error("--out is needed for an export without its run")
    out.parent.mkdir(parents=True, exist_ok=True)
    dt = sim.policy_dt
    writer = imageio.get_writer(str(out), fps=round(1 / (dt * a.every)), codec="libx264", quality=8, macro_block_size=1)
    torch.set_num_threads(4)
    obs = sim.observations()
    steps, falls, down, goal_n = int(round(program.duration(segs) / dt)), 0, 0, 0
    mode = ("legs " + ("+ arms policies" if arms else "policy, arms held at 0 rad") if legs
            else "arms policy, pelvis pinned")
    for k in range(steps):
        t = k * dt
        seg = program.at(segs, t)
        sim.vel_cmd = np.array(seg.cmd)
        if k in goal_at:
            sim.arm_mode, sim.believed = True, R.draw_goal(tables, goal_rng)
            goal_n += 1
            obs = sim.observations()
        carried, anchor = rope(t)
        obs = sim.step(obs)
        if legs and carried <= 0.0 and R.fell(sim):
            down += 1
            if down > int(1.0 / dt):  # a second on the ground, then back up (on its feet; the program goes on)
                falls, down = falls + 1, 0
                sim.reset(rng)
                R.reset_actors(sim)
                obs = sim.observations()
        if k % a.every:
            continue
        pos, q = sim.root()
        vw = d.qvel[0:3]
        fwd = qrot(q, np.array([1.0, 0.0, 0.0]))
        yaw = math.atan2(fwd[1], fwd[0])
        meas = R.yaw_frame_velocity(sim)
        c, s = math.cos(yaw), math.sin(yaw)
        tiles = []
        for i, cam in enumerate(cams):
            renderer.update_scene(d, cam)
            scn = renderer.scene
            ground = np.array([pos[0], pos[1], ARROW_Z])
            if legs:
                arrow(scn, ground, np.array([c * seg.cmd[0] - s * seg.cmd[1], s * seg.cmd[0] + c * seg.cmd[1]]), CMD_RGBA)
                arrow(scn, ground, vw, ACT_RGBA)
                arc(scn, ground, yaw, seg.cmd[2], 0.55, CMD_RGBA)
                arc(scn, ground, yaw, meas[2], 0.70, ACT_RGBA)
            if arms and sim.arm_mode:
                for b, w in zip(sim.believed, sim.wrists):
                    frame_geom(scn, np.r_[pos + qrot(q, b[:3]), qmul(q, b[3:])], 0.15, 0.012)
                    frame_geom(scn, np.r_[d.xpos[w], d.xquat[w]], 0.09, 0.005, alpha=0.7)
            if anchor is not None:
                hook = hook_point()
                connector(scn, mujoco.mjtGeom.mjGEOM_CYLINDER, 0.012, hook, [anchor[0], anchor[1], ROPE_TOP], ROPE_RGBA)
                connector(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.03, hook - [0, 0, 0.01], hook + [0, 0, 0.01], HOOK_RGBA)
            img = renderer.render().copy()
            if i == 0:
                lines = [f"MuJoCo ({a.plant} plant), {mode}   t = {t:5.1f} s   {seg.name}"]
                if legs:
                    lines += [f"cmd  vx {seg.cmd[0]:+.2f}  vy {seg.cmd[1]:+.2f}  wz {seg.cmd[2]:+.2f}",
                              f"meas vx {meas[0]:+.2f}  vy {meas[1]:+.2f}  wz {meas[2]:+.2f}   falls {falls}"]
                    if carried > 0:
                        lines.append(f"hoist carries {carried:.0%} of the weight")
                if arms and sim.arm_mode:
                    err, rot = R.wrist_errors(sim)
                    lines.append(f"arm goal {goal_n}: wrist error L {1000 * err[0]:5.1f} mm {math.degrees(rot[0]):4.1f} deg,"
                                 f"  R {1000 * err[1]:5.1f} mm {math.degrees(rot[1]):4.1f} deg")
                text(img, lines)
                legend = (["green: commanded, orange: actual (arrow velocity, arc turn rate)"] if legs else []) + \
                         (["arm goal: thick RGB frame; wrist: thin (x red, y green, z blue)"] if arms else [])
                text(img, legend, (230, 230, 230), bottom=True)
            else:
                text(img, ["side view"])
            tiles.append(img)
        writer.append_data(np.concatenate(tiles, axis=1))
    writer.close()
    print(f"[evals] wrote {out}: {program.duration(segs):.0f} s, falls {falls}")
    if shutil.which("ffmpeg"):
        small = out.with_name(out.stem + "_small.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-vf", "scale=960:-2", "-crf", "28",
                        str(small)], check=False)
        print(f"[evals] wrote {small}")


if __name__ == "__main__":
    main()
