"""Orthographic side views of the H1-2 for the workspace figure, with an alpha mask and a metric pixel scale.

    MUJOCO_GL=egl python paper/reach/tools/render_robot.py --pose standing --out figures/render_standing.png
    MUJOCO_GL=egl python paper/reach/tools/render_robot.py --trace results/<label>/isaac/traces.npz \
        --goal_id b003_d50_e00 --out figures/render_crouch.png

The camera looks along +y at the robot's right side, so the robot faces +x = right in the image; orthographic, so
world (x, z) maps linearly to pixels and the workspace contours (metres) overlay the image exactly. Writes the PNG
(RGBA, background transparent) and <out>.json with the pixel scale and the world coordinates of the image corners.

Poses: --pose standing (the task's default joint angles, soles on the ground) or a recorded Isaac state
(root pose + joint positions by name) from an eval_isaac.py traces.npz: --goal_id picks a traced trial and
--frame which step (default: the last step the trial was upright).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REACH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REACH))
from reachlib import common as C  # noqa: E402

SCENE = C.DEFAULT_ASSETS / "mujoco_assets" / "h1_2_magpie.xml"
DEFAULT_Q = {"hip_pitch": -0.3, "knee": 0.5, "ankle_pitch": -0.2}


def load_model():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    m.vis.global_.orthographic = 1
    m.vis.global_.offwidth, m.vis.global_.offheight = 2400, 2400
    m.vis.quality.shadowsize = 0
    m.vis.headlight.ambient[:] = (0.22, 0.22, 0.22)
    m.vis.headlight.diffuse[:] = (0.62, 0.62, 0.62)
    m.vis.headlight.specular[:] = (0.05, 0.05, 0.05)
    # a mid-grey robot: the real robot's near-black shows no shading under the overlays
    vis = m.geom_group == 1
    m.geom_rgba[vis] = (0.58, 0.60, 0.63, 1.0)
    m.geom_matid[vis] = -1
    return m


def set_pose(m, d, root_pos, root_quat, joints: dict[str, float]):
    d.qpos[:] = 0.0
    d.qpos[0:3] = root_pos
    d.qpos[3:7] = root_quat
    for name, val in joints.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = val
    mujoco.mj_forward(m, d)


def standing_pose(m, d):
    joints = {f"{s}_{k}_joint": v for s in ("left", "right") for k, v in DEFAULT_Q.items()}
    set_pose(m, d, [0, 0, C.STANDING_PELVIS_HEIGHT], [1, 0, 0, 0], joints)
    # soles on the ground: lowest ankle_roll_link origin + SOLE_Z = 0
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")]
    dz = min(d.xpos[f][2] for f in feet) + C.SOLE_Z
    set_pose(m, d, [0, 0, C.STANDING_PELVIS_HEIGHT - dz], [1, 0, 0, 0], joints)
    return joints


def render(m, d, center_xz=(0.25, 0.95), extent_z=2.1, width=900, height=1200):
    """RGBA image (H, W, 4) of the current pose and its pixel mapping {px_per_m, x0, z0} (world at pixel 0, 0)."""
    m.vis.global_.fovy = extent_z
    r = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (center_xz[0], 0.0, center_xz[1])
    cam.distance = 4.0
    cam.azimuth = 90.0
    cam.elevation = 0.0
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1            # the visual meshes (group 1); collision geoms and the floor stay hidden
    opt.sitegroup[:] = 0
    r.update_scene(d, camera=cam, scene_option=opt)
    rgb = r.render().copy()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam, scene_option=opt)
    seg = r.render()
    r.close()
    alpha = (seg[..., 0] >= 0).astype(np.uint8) * 255
    px_per_m = height / extent_z
    mapping = {"px_per_m": px_per_m, "x_left": center_xz[0] - width / 2 / px_per_m,
               "z_top": center_xz[1] + height / 2 / px_per_m, "width": width, "height": height}
    return np.dstack([rgb, alpha]), mapping


def yaw_of(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def frame_in_standing(state_t, state_0):
    """A recorded state (root pos 3, quat 4, joints...) expressed in the standing frame of state_0: the first frame's
    x, y and yaw removed, heights kept (Isaac's flat ground is at z = 0)."""
    p0, q0 = state_0[:3], state_0[3:7]
    yaw0 = yaw_of(q0)
    c, s_ = np.cos(-yaw0), np.sin(-yaw0)
    d = state_t[:3] - p0
    pos = np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1], state_t[2]])
    qyaw_inv = np.array([np.cos(-yaw0 / 2), 0, 0, np.sin(-yaw0 / 2)])
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, qyaw_inv, np.asarray(state_t[3:7], float))
    return pos, quat


def render_trace_frames(trace_npz: Path, goal_id: str, frames: list[int], center=(0.25, 0.95), extent=2.1,
                        width=600, height=800):
    """[(RGBA image, mapping)] for the given steps of a traced trial, all in the goal's standing frame."""
    z = np.load(trace_npz, allow_pickle=False)
    names = [str(n) for n in z["joint_names"]]
    state = z[f"trace_{goal_id}_state"]
    m = load_model()
    d = mujoco.MjData(m)
    out = []
    for t in frames:
        t = min(t, len(state) - 1)
        pos, quat = frame_in_standing(state[t], state[0])
        set_pose(m, d, pos, quat, dict(zip(names, state[t, 7:].tolist())))
        out.append(render(m, d, center_xz=center, extent_z=extent, width=width, height=height))
    return out


def trace_pose(path: Path, goal_id: str, frame: int | None):
    z = np.load(path, allow_pickle=False)
    names = [str(n) for n in z["joint_names"]]
    state = z[f"trace_{goal_id}_state"]                      # (T, 7 + J): root pos, root quat wxyz, joint pos
    alive = z[f"trace_{goal_id}_alive"]
    t = frame if frame is not None else int(np.nonzero(alive)[0][-1])
    s = state[t]
    return s[:3], s[3:7], dict(zip(names, s[7:].tolist())), t


def hero_label() -> str | None:
    """conditions.yaml's hero when it has Isaac results, else the listed policy with the highest lambda."""
    conds = [c for c in C.load_conditions() if (C.RESULTS_DIR / c["label"] / "isaac" / "trials.csv").is_file()]
    for c in conds:
        if c.get("hero"):
            return c["label"]
    best = None
    for c in conds:
        lam = C.read_reward_share(Path(c["run"])) if c.get("run") else None
        if best is None or (lam or 0) > best[0]:
            best = ((lam or 0), c["label"])
    return best[1] if best else None


def pick_crouch():
    """(traces.npz, goal_id, frame) of the lowest traced goal a lambda >= 0.99 full policy reached in Isaac, at the
    step its 1 s hold completed."""
    import csv
    hero = hero_label()
    best = None
    for path in sorted(C.RESULTS_DIR.glob("*/isaac/trials.csv")):
        if path.parent.parent.name.startswith("_"):
            continue
        traces = path.parent / "traces.npz"
        if not traces.is_file():
            continue
        names = set(np.load(traces).files)
        for r in csv.DictReader(open(path)):
            if path.parent.parent.name != hero or r["success"] not in ("1", "True") or r["seed"] != "0":
                continue
            if f"trace_{r['goal_id']}_state" not in names:
                continue
            h = float(r["target_min_height"])
            if best is None or h < best[0]:
                best = (h, traces, r["goal_id"], int(round(float(r["t_success"]) / C.POLICY_DT)) - 1)
    if best is None:
        raise SystemExit(f"[render_robot] no reached traced goal of the hero condition ({hero}) yet")
    print(f"[render_robot] crouch pose: {best[2]} (lower target {best[0]:.2f} m) from {best[1]}")
    return best[1], best[2], best[3]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pose", choices=["standing"], default=None)
    p.add_argument("--trace", default=None)
    p.add_argument("--goal_id", default=None)
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--auto", action="store_true",
                   help="crouch pose: the lowest goal the hero condition (conditions.yaml) reached among its traced trials")
    p.add_argument("--out", required=True)
    p.add_argument("--center", default="0.25,0.95", help="world x,z at the image centre (m), robot-relative")
    p.add_argument("--extent", type=float, default=2.1, help="image height in metres")
    a = p.parse_args()
    m = load_model()
    d = mujoco.MjData(m)
    info = {}
    if a.auto:
        a.trace, a.goal_id, a.frame = pick_crouch()
    if a.trace:
        _, _, joints, t = trace_pose(Path(a.trace), a.goal_id, a.frame)
        zz = np.load(a.trace, allow_pickle=False)
        state = zz[f"trace_{a.goal_id}_state"]
        pos, quat = frame_in_standing(state[t], state[0])
        set_pose(m, d, pos, quat, joints)
        info = {"trace": str(a.trace), "goal_id": a.goal_id, "frame": t, "pelvis_h": float(pos[2])}
    else:
        standing_pose(m, d)
        info = {"pose": "standing"}
    cx, cz = (float(v) for v in a.center.split(","))
    img, mapping = render(m, d, center_xz=(cx, cz), extent_z=a.extent)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out)
    pelvis = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
    shoulders = {s: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_shoulder_pitch_link")].tolist()
                 for s in ("left", "right")}
    out.with_suffix(".json").write_text(json.dumps({**mapping, **info, "pelvis": pelvis.tolist(),
                                                    "shoulders": shoulders}, indent=1))
    print(f"[render_robot] {out} ({mapping['width']}x{mapping['height']} px, {mapping['px_per_m']:.1f} px/m)")


if __name__ == "__main__":
    main()
