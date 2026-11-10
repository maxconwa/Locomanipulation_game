"""Build the AMP discriminator's reference states (mdp.amp_style) from Unitree's LAFAN1 retargeting to the H1-2.

The clips (huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset, h1_2/*.csv; the unitreerobotics original
is no longer public) are 30 fps rows of the root position, its quaternion (x, y, z, w) and the 27 joints in Unitree's
order, kinematic only. The walk, run and sprint clips, each also mirrored left to right (LAFAN1's capture room bends
every walk into curves), are resampled to the policy's 100 Hz (cubic splines on the root position and the joints, a
rotation spline on the root orientation) and every frame is reduced to the state mdp.amp_style reads in the env
(mdp.amp.AMP_STATE_DIM): the leg joints and their velocities, gravity in the pelvis frame, the pelvis's linear and
angular velocity in its own frame, both ankle-roll links relative to the pelvis in its heading (yaw-only) frame, their
heights above the ground and their velocities in the heading frame, by forward kinematics on CL_Assets' h1_2 MJCF
(whose legs are the retargeting URDF's). The ground is each clip's: the median over its upright frames of the lower
foot's sole. Kept frames: upright (the pelvis at least 0.80 m above the lower foot's ground, tilted less than 20
degrees) and turning no faster than the command's 0.97 rad/s (the yaw rate's mean over a 0.70 s gait cycle).

    python scripts/build_amp_reference.py <h1_2 csv dir> [--out <npz>]

Writes states (N, 47) and next_valid (N,): frame i and i + 1 are consecutive kept frames of one clip; and for
reference state initialisation (mdp.reset_from_reference) each frame's pelvis height above the clip's ground
(root_height) and its orientation with the heading removed (root_quat, w x y z). LAFAN1 is
CC BY-NC-ND 4.0: the file is for research use and is not redistributed.
"""

import argparse
import glob
import os

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation, RotationSpline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF = os.path.join(REPO, "third_party", "CL_Assets", "mujoco_assets", "h1_2_magpie.xml")
OUT = os.path.join(REPO, "source", "locomanipulation_game", "locomanipulation_game", "data", "amp_lafan1_h1_2.npz")
SRC_FPS, FPS = 30.0, 100.0
KINDS = ("walk", "run", "sprint")
JOINTS = ["left_hip_yaw", "left_hip_pitch", "left_hip_roll", "left_knee", "left_ankle_pitch", "left_ankle_roll",
          "right_hip_yaw", "right_hip_pitch", "right_hip_roll", "right_knee", "right_ankle_pitch", "right_ankle_roll"]
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]
SOLE = 0.045       # the ankle-roll link above the ground, sport mode's stance (pelvis 0.9798 m)
MIN_HEIGHT = 0.80  # m, pelvis above the lower foot's ground
MAX_TILT = np.radians(20.0)
MAX_YAW_RATE = 0.97  # rad/s, the command's limit (LocoManipMarlEnvCfg's ang_vel_z range)
GAIT_WINDOW = 70     # 0.70 s at 100 Hz: the tracking terms' cycle mean
# mirroring a row left to right: the leg and arm blocks swap, the roll and yaw joints change sign
_SIDE = ["hip_yaw", "hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll"]
_ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
_ORDER = ([f"left_{j}" for j in _SIDE] + [f"right_{j}" for j in _SIDE] + ["torso"]
          + [f"left_{j}" for j in _ARM] + [f"right_{j}" for j in _ARM])
_MIRROR_SRC = [_ORDER.index(n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")) for n in _ORDER]
_MIRROR_SIGN = np.array([-1.0 if ("roll" in n or "yaw" in n or n == "torso") else 1.0 for n in _ORDER])
# mdp.amp.AMP_STATE_NAMES
STATE_NAMES = ([f"q_{j}" for j in JOINTS] + [f"qd_{j}" for j in JOINTS] + ["grav_x", "grav_y", "grav_z"]
               + ["v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]
               + [f"{s}_foot_{a}" for s in ("left", "right") for a in "xyz"]
               + ["left_foot_height", "right_foot_height"]
               + [f"{s}_foot_v{a}" for s in ("left", "right") for a in "xyz"])


def mirror(raw: np.ndarray) -> np.ndarray:
    """The rows reflected in the x-z plane: y, the quaternion's x and z, the sides swapped."""
    out = raw.copy()
    out[:, 1] *= -1.0
    out[:, 3] *= -1.0  # qx
    out[:, 5] *= -1.0  # qz
    out[:, 7:] = raw[:, 7:][:, _MIRROR_SRC] * _MIRROR_SIGN
    return out


def clip_states(raw: np.ndarray, m: mujoco.MjModel, d: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                                np.ndarray]:
    t_src = np.arange(len(raw)) / SRC_FPS
    t = np.arange(0.0, t_src[-1], 1.0 / FPS)
    pos_s, q_s = CubicSpline(t_src, raw[:, :3]), CubicSpline(t_src, raw[:, 7:19])
    rot_s = RotationSpline(t_src, Rotation.from_quat(raw[:, 3:7]))  # scalar-last
    pos, q, qd = pos_s(t), q_s(t), q_s(t, 1)
    rot = rot_s(t)
    # the spline's rate is the body-frame angular velocity (log(R(t)^T R(t + h)) / h)
    w_b = rot_s(t, 1)
    v_b = rot.inv().apply(pos_s(t, 1))
    grav_b = rot.inv().apply(np.array([0.0, 0.0, -1.0]))
    yaw = rot.as_euler("ZYX")[:, 0]
    heading_inv = Rotation.from_euler("Z", -yaw)
    qadr = [m.jnt_qposadr[m.joint(j + "_joint").id] for j in JOINTS]
    feet_ids = [m.body(f).id for f in FEET]
    quat_wxyz = rot.as_quat()[:, [3, 0, 1, 2]]
    feet_w = np.zeros((len(t), 2, 3))
    for i in range(len(t)):
        d.qpos[:] = 0.0
        d.qpos[0:3], d.qpos[3:7] = pos[i], quat_wxyz[i]
        d.qpos[qadr] = q[i]
        mujoco.mj_kinematics(m, d)
        feet_w[i] = d.xpos[feet_ids]
    rel = feet_w - pos[:, None, :]
    feet_h = np.stack([heading_inv.apply(rel[:, k]) for k in range(2)], axis=1).reshape(len(t), 6)
    lower_sole = feet_w[:, :, 2].min(axis=1) - SOLE
    height = pos[:, 2] - lower_sole
    tilt = np.arccos(np.clip(-grav_b[:, 2], -1.0, 1.0))
    upright = (height >= MIN_HEIGHT) & (tilt <= MAX_TILT)
    ground = np.median(lower_sole[upright]) if upright.any() else 0.0
    feet_z = feet_w[:, :, 2] - ground
    feet_v = np.gradient(feet_w, axis=0) * FPS
    feet_v = np.stack([heading_inv.apply(feet_v[:, k]) for k in range(2)], axis=1).reshape(len(t), 6)
    states = np.concatenate([q, qd, grav_b, v_b, w_b, feet_h, feet_z, feet_v], axis=1)
    yaw_rate = uniform_filter1d(np.gradient(np.unwrap(yaw)) * FPS, GAIT_WINDOW, mode="nearest")
    keep = upright & (np.abs(yaw_rate) <= MAX_YAW_RATE)
    # the pelvis's height above the ground and its tilt (the orientation with the heading taken out), w x y z
    root = np.concatenate([(pos[:, 2] - ground)[:, None], (heading_inv * rot).as_quat()[:, [3, 0, 1, 2]]], axis=1)
    return states, keep, feet_w, root


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv_dir")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    paths = sorted(p for p in glob.glob(os.path.join(args.csv_dir, "*.csv"))
                   if os.path.basename(p).split("_")[0].rstrip("0123456789") in KINDS)
    if not paths:
        raise SystemExit(f"no {'/'.join(KINDS)} clips in {args.csv_dir}")
    states, next_valid, roots, names = [], [], [], []
    for path in paths:
        raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 34:
            raise ValueError(f"{path}: expected 34 columns, got {raw.shape}")
        name = os.path.basename(path)[:-4]
        s, keep, feet_w, root = clip_states(raw, m, d)
        s_m, keep_m, feet_m, root_m = clip_states(mirror(raw), m, d)
        # the mirror's left foot is the original's right foot reflected
        err = np.abs(feet_m[:, 0] - feet_w[:, 1] * [1.0, -1.0, 1.0]).max()
        if err > 1e-6:
            raise RuntimeError(f"{name}: the mirrored feet are off by {err:.2e} m")
        for label, st, kp, rt in ((name, s, keep, root), (name + "_mirror", s_m, keep_m, root_m)):
            valid = np.zeros(len(st), bool)
            valid[:-1] = kp[:-1] & kp[1:]
            states.append(st[kp])
            roots.append(rt[kp])
            # indices shift when frames are dropped: a pair survives only when both of its frames do, and frame i + 1
            # of the kept array is then the original's i + 1
            next_valid.append(valid[kp])
            names.append(label)
            print(f"{label:31s} {len(st) / FPS:6.1f} s, kept {kp.mean():4.0%}, pairs {valid.sum():6d}")
    states = np.concatenate(states).astype(np.float32)
    next_valid = np.concatenate(next_valid)
    roots = np.concatenate(roots).astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, states=states, next_valid=next_valid, state_names=np.array(STATE_NAMES),
                        root_height=roots[:, 0], root_quat=roots[:, 1:], clips=np.array(names), fps=FPS)
    print(f"wrote {args.out}: {len(states)} states ({len(states) / FPS / 60:.1f} min), {next_valid.sum()} pairs")


if __name__ == "__main__":
    main()
