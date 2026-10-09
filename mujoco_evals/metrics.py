"""Per-trial reach, posture and balance metrics from a per-step log; the Isaac and MuJoCo evaluators share them.

A step log covers one arm goal (T policy steps of POLICY_DT) for N trials, numpy arrays shaped (T, N, ...):

    pos_err (T,N,2) m, rot_err (T,N,2) rad    each wrist against its true world target
    alive (T,N) bool                          False from the step a trial fell
    root_pos (T,N,3), root_quat (T,N,4) wxyz  pelvis pose, world
    root_ang_vel_b (T,N,3) rad/s              pelvis angular velocity, pelvis frame
    ground (T,N) m                            ground height under the pelvis
    com (T,N,3), com_vel (T,N,3)              whole-body centre of mass and its velocity, world
    feet_pos (T,N,2,3), feet_quat (T,N,2,4)   ankle_roll_link frames, world (left, right)
    feet_vel (T,N,2,3), feet_force (T,N,2)    their linear velocity and contact normal force (N)
    wrist_pos (T,N,2,3)                       wrist_yaw_link origins, world
    torque_ratio (T,N,4)                      peak |tau| / effort limit over hip, knee, ankle, arm joints
    cmd_drift (T,N) m                         believed (odometry) minus true target, mean of both wrists

plus pelvis_h_start (N,), the pelvis height when the goal starts.

Definitions (all used in the paper and the page):
    success           both wrists inside POS_TOL and ROT_TOL for HOLD_S without a break, before the goal ends
    closest           the least, over the goal, of the mean of both wrists' position errors (the curriculum's measure)
    support polygon   convex hull of the sole rectangles of the feet carrying more than CONTACT_FORCE_N
    com margin        signed distance of the CoM's ground projection to the support polygon's edge, + inside;
                      report it over the hold (last second): its minimum over a goal is set by steps, where the
                      static margin to the single supporting foot is negative by construction
    dcm margin        the same for the divergent component of motion xi = c + c_dot / omega, omega = sqrt(g / z_c)
    foot slip         horizontal speed of a foot carrying more than STANCE_FORCE_N (its maximum includes touchdown)
    steps             lift-offs: a foot going from more than CONTACT_FORCE_N to none
    ee jitter         RMS distance of a wrist from its own mean position over the goal's last second (mm)
"""

from __future__ import annotations

import math

import numpy as np

from .common import (CONTACT_FORCE_N, HOLD_S, POLICY_DT, POS_TOL, ROT_TOL, SOLE_X, SOLE_Y, SOLE_Z,
                     STANCE_FORCE_N)

# looser and stricter versions of the success test: (position m, orientation rad) on the best 1 s window
TOL_PAIRS = {"success_2cm": (0.02, 0.20), "success_3cm": (0.03, 0.25), "success_8cm": (0.08, 0.50),
             "success_10cm": (0.10, 0.60)}

G = 9.81


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v (..., 3) by unit quaternions q (..., 4), w first."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def pitch_deg(q: np.ndarray) -> np.ndarray:
    """Pitch (about the body y axis, + nose down = leaning forward) of wxyz quaternions, degrees."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    s = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    return np.degrees(np.arcsin(s))


def tilt_deg(q: np.ndarray) -> np.ndarray:
    """Angle between the body z axis and the world vertical, degrees."""
    zb = quat_rotate(q, np.broadcast_to(np.array([0.0, 0.0, 1.0]), q.shape[:-1] + (3,)))
    return np.degrees(np.arccos(np.clip(zb[..., 2], -1.0, 1.0)))


def sole_corners(feet_pos: np.ndarray, feet_quat: np.ndarray) -> np.ndarray:
    """(..., 2, 3) foot frames -> (..., 8, 2) sole corner positions in the ground plane (left 0-3, right 4-7)."""
    local = np.array([[x, y, SOLE_Z] for x in SOLE_X for y in SOLE_Y])                  # (4, 3)
    q = np.repeat(feet_quat[..., None, :], 4, axis=-2)                                  # (..., 2, 4, 4)
    c = quat_rotate(q, np.broadcast_to(local, q.shape[:-1] + (3,))) + feet_pos[..., None, :]
    return c[..., :2].reshape(*feet_pos.shape[:-2], 8, 2)


def polygon_margin(point: np.ndarray, corners: np.ndarray, valid: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Signed distance (m, + inside) of point (M, 2) to the convex hull of the valid corners (M, 8, 2).

    A directed corner pair i -> j is a hull edge when every valid corner lies on or left of it; the margin is the
    least signed distance to the hull edges, which is the distance to the boundary for a point inside and a lower
    bound on minus the distance outside. NaN with fewer than three valid corners.
    """
    m = len(point)
    out = np.full(m, np.nan)
    ii, jj = np.where(~np.eye(8, dtype=bool))
    for s in range(0, m, chunk):
        p, c, v = point[s:s + chunk], corners[s:s + chunk], valid[s:s + chunk]
        a, b = c[:, ii], c[:, jj]                                        # (k, 56, 2)
        e = b - a
        rel = c[:, None, :, :] - a[:, :, None, :]                        # (k, 56, 8, 2)
        cross = e[:, :, None, 0] * rel[..., 1] - e[:, :, None, 1] * rel[..., 0]
        length = np.linalg.norm(e, axis=-1)
        ok_corner = np.where(v[:, None, :], cross >= -1e-9 * np.maximum(length[..., None], 1e-9), True)
        edge = ok_corner.all(-1) & v[:, ii] & v[:, jj] & (length > 1e-6)
        q = p[:, None, :] - a
        dist = (e[..., 0] * q[..., 1] - e[..., 1] * q[..., 0]) / np.maximum(length, 1e-9)
        dist = np.where(edge, dist, np.inf)
        res = dist.min(-1)
        enough = v.sum(-1) >= 3
        out[s:s + chunk] = np.where(enough & np.isfinite(res), res, np.nan)
    return out


def hold_success(within: np.ndarray, alive: np.ndarray, hold_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """(success (N,), t_success (N,) s, NaN if never): the end of the first unbroken run of hold_steps."""
    t_n = within.shape[0]
    run = np.zeros(within.shape[1], dtype=int)
    done = np.full(within.shape[1], np.nan)
    for t in range(t_n):
        ok = within[t] & alive[t]
        run = np.where(ok, run + 1, 0)
        newly = (run >= hold_steps) & np.isnan(done)
        done[newly] = (t + 1) * POLICY_DT
    return ~np.isnan(done), done


def _last_window(x: np.ndarray, alive: np.ndarray, steps: int) -> np.ndarray:
    """Mean over the last `steps` alive steps of each trial (x (T,N) or (T,N,k) -> (N,) or (N,k)); NaN if none."""
    t_n = x.shape[0]
    last_alive = np.where(alive.any(0), t_n - 1 - np.argmax(alive[::-1], axis=0), -1)    # (N,)
    idx = np.arange(t_n)[:, None]
    w = (idx <= last_alive[None]) & (idx > last_alive[None] - steps) & alive
    w = w.astype(float)
    if x.ndim == 3:
        w = w[..., None]
    num = (np.nan_to_num(x) * w).sum(0)
    den = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def trial_metrics(log: dict, pelvis_h_start: np.ndarray, standing_height: float, series: dict | None = None) -> list[dict]:
    """One dict per trial with the TRIAL_COLUMNS metric fields. If series is a dict, it receives the per-step
    (T, N) arrays behind them: pelvis_h, tilt_deg, com_margin, dcm_margin, err_mean."""
    alive = log["alive"].astype(bool)
    t_n, n = alive.shape
    hold = int(round(HOLD_S / POLICY_DT))
    last = int(round(1.0 / POLICY_DT))
    pos, rot = log["pos_err"], log["rot_err"]

    within_each = (pos < POS_TOL) & (rot < ROT_TOL)                     # (T,N,2)
    success, t_success = hold_success(within_each.all(-1), alive, hold)
    success_l, _ = hold_success(within_each[..., 0], alive, hold)
    success_r, _ = hold_success(within_each[..., 1], alive, hold)
    mean_err = np.where(alive, pos.mean(-1), np.inf)
    closest = mean_err.min(0)

    # 1 s hold errors: over every 1 s window of upright steps, the worst position (rotation) error of either wrist;
    # the best window's value. Success at a position tolerance tau alone is hold_err_pos < tau.
    worst_p = np.where(alive, pos.max(-1), np.inf)
    worst_r = np.where(alive, rot.max(-1), np.inf)
    win_p = np.lib.stride_tricks.sliding_window_view(worst_p, hold, axis=0).max(-1)       # (T-hold+1, N)
    win_r = np.lib.stride_tricks.sliding_window_view(worst_r, hold, axis=0).max(-1)
    hold_err_pos, hold_err_rot = win_p.min(0), win_r.min(0)
    # strict success at other tolerance pairs (both wrists, both errors, the same window)
    tol_pairs = TOL_PAIRS
    tol_success = {k: ((win_p < tp) & (win_r < tr)).any(0) for k, (tp, tr) in tol_pairs.items()}

    fell = ~alive.all(0)
    t_fall = np.where(fell, np.argmin(alive, axis=0) * POLICY_DT, np.nan)
    last_idx = np.where(alive.any(0), t_n - 1 - np.argmax(alive[::-1], axis=0), 0)
    ar = np.arange(n)
    err_final = np.where(fell[:, None], np.nan, pos[last_idx, ar])
    rot_final = np.where(fell[:, None], np.nan, rot[last_idx, ar])

    h = log["root_pos"][..., 2] - log["ground"]
    h_alive = np.where(alive, h, np.nan)
    h_last = _last_window(h, alive, last)
    tilt = np.where(alive, tilt_deg(log["root_quat"]), np.nan)
    pitch_last = _last_window(pitch_deg(log["root_quat"]), alive, last)
    angvel = np.where(alive[..., None], log["root_ang_vel_b"][..., :2], np.nan)

    # support polygon, CoM and DCM margins
    corners = sole_corners(log["feet_pos"], log["feet_quat"]).reshape(t_n * n, 8, 2)
    contact = log["feet_force"] > CONTACT_FORCE_N                      # (T,N,2)
    valid = np.repeat(contact, 4, axis=-1).reshape(t_n * n, 8)
    com, comv = log["com"], log["com_vel"]
    zc = np.maximum(com[..., 2] - log["ground"], 0.05)
    omega = np.sqrt(G / zc)
    xi = com[..., :2] + comv[..., :2] / omega[..., None]
    com_m = polygon_margin(com[..., :2].reshape(-1, 2), corners, valid).reshape(t_n, n)
    dcm_m = polygon_margin(xi.reshape(-1, 2), corners, valid).reshape(t_n, n)
    com_m, dcm_m = np.where(alive, com_m, np.nan), np.where(alive, dcm_m, np.nan)

    # feet: slip while planted, lift-offs
    planted = log["feet_force"] > STANCE_FORCE_N
    slip = np.where(planted & alive[..., None], np.linalg.norm(log["feet_vel"][..., :2], axis=-1), np.nan)
    liftoff = (contact[:-1] & ~contact[1:] & alive[1:, :, None]).sum((0, 2))

    # end effector steadiness over the last second
    wp = log["wrist_pos"]
    w_alive = alive[..., None, None]
    jit, spd = [], []
    for arm in range(2):
        mean_p = _last_window(wp[:, :, arm], alive, last)               # (N,3)
        dev = np.linalg.norm(wp[:, :, arm] - mean_p[None], axis=-1) ** 2
        jit.append(1000.0 * np.sqrt(_last_window(dev, alive, last)))
    vel = np.diff(wp, axis=0) / POLICY_DT                               # (T-1,N,2,3)
    sp2 = (np.linalg.norm(vel, axis=-1) ** 2).mean(-1)                  # (T-1,N)
    spd = np.sqrt(_last_window(sp2, alive[1:], last))

    tq = np.where(alive[..., None], log["torque_ratio"], np.nan)
    if series is not None:
        series.update({"pelvis_h": h_alive, "tilt_deg": tilt, "com_margin": com_m, "dcm_margin": dcm_m,
                       "err_mean": np.where(alive, pos.mean(-1), np.nan)})
    rows = []
    with np.errstate(invalid="ignore"):
        for i in range(n):
            rows.append({
                "success": bool(success[i]), "success_l": bool(success_l[i]), "success_r": bool(success_r[i]),
                "t_success": float(t_success[i]), "closest": float(closest[i]),
                "hold_err_pos": float(hold_err_pos[i]), "hold_err_rot": float(hold_err_rot[i]),
                **{k: bool(v[i]) for k, v in tol_success.items()},
                "err_final_l": float(err_final[i, 0]), "err_final_r": float(err_final[i, 1]),
                "rot_final_l": float(rot_final[i, 0]), "rot_final_r": float(rot_final[i, 1]),
                "err_last1s": float(_last_window(pos[:, i:i + 1].mean(-1), alive[:, i:i + 1], last)[0]),
                "rot_last1s": float(_last_window(rot[:, i:i + 1].mean(-1), alive[:, i:i + 1], last)[0]),
                "pelvis_h_start": float(pelvis_h_start[i]), "pelvis_h_min": float(np.nanmin(h_alive[:, i])),
                "pelvis_h_last1s": float(h_last[i]), "pelvis_drop_last1s": float(pelvis_h_start[i] - h_last[i]),
                "fell": bool(fell[i]), "t_fall": float(t_fall[i]),
                "tilt_max_deg": float(np.nanmax(tilt[:, i])),
                "angvel_rms": float(np.sqrt(np.nanmean(angvel[:, i] ** 2))),
                "com_margin_min": float(np.nanmin(com_m[:, i])) if np.isfinite(com_m[:, i]).any() else math.nan,
                "com_margin_last1s": float(_last_window(com_m[:, i:i + 1], alive[:, i:i + 1], last)[0]),
                "dcm_margin_min": float(np.nanmin(dcm_m[:, i])) if np.isfinite(dcm_m[:, i]).any() else math.nan,
                "dcm_margin_last1s": float(_last_window(dcm_m[:, i:i + 1], alive[:, i:i + 1], last)[0]),
                "foot_slip_max": float(np.nanmax(slip[:, i])) if np.isfinite(slip[:, i]).any() else 0.0,
                "steps": int(liftoff[i]),
                "ee_jitter_l": float(jit[0][i]), "ee_jitter_r": float(jit[1][i]), "ee_speed_rms": float(spd[i]),
                "torque_hip_max": float(np.nanmax(tq[:, i, 0])), "torque_knee_max": float(np.nanmax(tq[:, i, 1])),
                "torque_ankle_max": float(np.nanmax(tq[:, i, 2])), "torque_arm_max": float(np.nanmax(tq[:, i, 3])),
                "cmd_drift_end": float(log["cmd_drift"][last_idx[i], i]),
                "pitch_last1s_deg": float(pitch_last[i]),
            })
    return rows
