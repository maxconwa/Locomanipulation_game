"""The H1-2 Sport Mode Walking Report's metric code, copied here so the scorecards need nothing outside the repo.

Copied from the report's analysis (~/Downloads/HumanoidBags_extracted/analysis, 2026-10-19): fit_resp, lagged_ref and
tracking from metrics.py, stepping from gait.py, lpf from fit_cmd.py, unchanged except that the stride window (their
fixed 70 samples, sport mode's 0.70 s stride) is a parameter, and that nothing runs or reads files at import.
evals/deploy/test_sportmode_metrics.py checks that this copy still reproduces the report on the real bags.

Every series is sampled at 100 Hz (DT). Velocities are (vx, vy, wz): the pelvis velocity in the yaw frame and the yaw
rate; commands the same three.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter

DT = 0.01
REPORT_STRIDE = 70  # samples: sport mode's 0.70 s stride, the report's averaging window

# (key in tracking(), label, sport mode's value as the report gives it, target, "ge" or "le"); the order the
# scorecards print them in (direction and turn sense first)
TARGETS = [
    ("cos_v_mean", "Direction cos(commanded, actual) xy velocity, mean", "0.96 (creep removed; 0.65 raw)", 0.95, "ge"),
    ("turn_sign_agree", "Turn sense correct, commanded |wz| > 0.1 rad/s", 0.994, 0.99, "ge"),
    ("lin_err_mean", "Linear velocity error, mean (m/s)", 0.147, 0.15, "le"),
    ("ang_err_mean", "Angular velocity error, mean (rad/s)", 0.050, 0.05, "le"),
    ("speed_ratio_median", "Linear speed achieved / commanded, median", 0.93, 0.90, "ge"),
    ("turn_ratio_median", "Turn rate achieved / commanded, median", 0.91, 0.90, "ge"),
]

# Sport mode's gait and posture, from the report (results/metrics.json "sport") and the foot-lift analysis
# (logs/analysis/foot_lift/results.md: feet_metrics.sport() on the same bags). Units as the scorecard prints them.
SPORT_REF = {
    "stride_T_median": 0.700,       # s
    "stride_T_iqr": 0.01,           # s
    "roll_sd_deg": 0.83,
    "pitch_sd_deg": 0.75,
    "clearance_median_cm": 5.95,    # swing ankle over stance ankle at the knee peak, pelvis frame, levelled
    "clearance_p10_cm": 5.34,
    "stride_dz_median_cm": 6.3,     # per-stride max |left - right ankle height|, levelled
    "stride_dz_p90_cm": 6.8,
    "knee_swing_median": 0.40,      # rad, per stride window
    "knee_peak_median": 1.02,       # rad
    "steps_per_s": 2.78,            # knee peaks while stepping
}


def lpf(x, tau):
    """First-order low-pass with time constant tau (s); tau 0 passes x through."""
    a = np.exp(-DT / tau) if tau > 0 else 0.0
    return lfilter([1 - a], [1, -a], x)


def stepping(q: np.ndarray) -> np.ndarray:
    """(T,) bool: stepping, i.e. a knee swinging (rolling sd over 0.5 s above 0.03 rad). q: (T, >=10) joint angles
    in LowCmd order (knees at 3 and 9), or (T, 2) knees (left, right)."""
    k = (q[:, [3, 9]] if q.shape[1] > 2 else q).astype(float)
    m = uniform_filter1d(k, 50, axis=0)
    sd = np.sqrt(np.maximum(uniform_filter1d(k**2, 50, axis=0) - m**2, 0))
    return sd.max(1) > 0.03


def fit_resp(cmds, vels, masks, stride: int = REPORT_STRIDE) -> dict:
    """Per axis: v = b + K LPF_tau(cmd(t - d)), all stride-averaged, fitted on the masked samples."""
    out = {}
    for a, nm in enumerate(("vx", "vy", "wz")):
        best = None
        v = np.concatenate([uniform_filter1d(x[:, a], stride) for x in vels])
        m = np.concatenate(masks)
        for tau in np.r_[0.01, np.arange(0.05, 2.01, 0.05)]:
            for d in range(0, 61, 5):
                f = np.concatenate([uniform_filter1d(lpf(np.r_[np.zeros(d), c[:len(c) - d, a]], tau), stride)
                                    for c in cmds])
                X = np.stack([np.ones(m.sum()), f[m]], 1)
                k, *_ = np.linalg.lstsq(X, v[m], rcond=None)
                r = v[m] - X @ k
                if best is None or (r**2).sum() < best[0]:
                    best = ((r**2).sum(), dict(
                        tau=round(float(tau), 2), delay=d * DT, bias=float(k[0]), gain=float(k[1]),
                        r2=float(1 - (r**2).sum() / ((v[m] - v[m].mean())**2).sum()),
                        resid_rmse=float(np.sqrt((r**2).mean()))))
        out[nm] = best[1]
    return out


def lagged_ref(cmds, resp: dict, stride: int = REPORT_STRIDE) -> list:
    """The command after the controller's own best-fit lag and smoothing (no gain or bias), stride-averaged."""
    out = []
    for c in cmds:
        cols = []
        for a, nm in enumerate(("vx", "vy", "wz")):
            p = resp[nm]
            d = int(round(p["delay"] / DT))
            cols.append(uniform_filter1d(lpf(np.r_[np.zeros(d), c[:len(c) - d, a]], p["tau"]), stride))
        out.append(np.stack(cols, 1))
    return out


def tracking(ref, vel, mask, stride: int = REPORT_STRIDE, vthr: float = 0.1, wthr: float = 0.1) -> dict:
    """The tracking numbers of the stride-averaged velocity against ref, on the masked samples."""
    ref = np.concatenate(ref)
    vel = np.concatenate([uniform_filter1d(v, stride, axis=0) for v in vel])
    m = np.concatenate(mask)
    e = vel - ref
    lin = np.linalg.norm(e[m, :2], axis=1)
    ang = np.abs(e[m, 2])
    rv, vv = ref[m, :2], vel[m, :2]
    mv = np.linalg.norm(rv, axis=1) > vthr
    cos = (rv[mv] * vv[mv]).sum(1) / (np.linalg.norm(rv[mv], axis=1) * np.linalg.norm(vv[mv], axis=1) + 1e-9)
    mw = np.abs(ref[m, 2]) > wthr
    sign = np.sign(ref[m, 2][mw]) == np.sign(vel[m, 2][mw])
    spd_ratio = np.linalg.norm(vv[mv], axis=1) / np.linalg.norm(rv[mv], axis=1)
    wr = vel[m, 2][mw] / ref[m, 2][mw]
    return dict(lin_err_mean=float(lin.mean()), lin_err_p95=float(np.percentile(lin, 95)),
                ang_err_mean=float(ang.mean()), ang_err_p95=float(np.percentile(ang, 95)),
                cos_v_mean=float(cos.mean()), cos_v_p10=float(np.percentile(cos, 10)), n_cos_s=float(mv.sum() * DT),
                turn_sign_agree=float(sign.mean()), n_turn_s=float(mw.sum() * DT),
                speed_ratio_median=float(np.median(spd_ratio)), turn_ratio_median=float(np.median(wr)))
