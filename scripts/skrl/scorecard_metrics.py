"""The sport-mode scorecard's metrics: the H1-2 Sport Mode Walking Report's tracking numbers, computed by the report's
own code, plus its stride and torso-tilt statistics. Numpy and scipy only; eval_scorecard.py feeds it simulated
runs and test_scorecard_sportmode.py the real sport-mode bags.

The tracking metrics are fit_resp, lagged_ref and tracking from the report's analysis (metrics.py in
--analysis_dir, default ~/Downloads/HumanoidBags_extracted/analysis or $SPORTMODE_ANALYSIS_DIR): per axis the best-fit
gain, delay and smoothing time constant of the velocity's response to the command; then, on time spent stepping
(gait.stepping, the knee-swing detector), the velocity averaged over one stride against the command delayed and
smoothed by that response (the lag-allowed command, without the gain or bias). Every series is sampled at 100 Hz
(their DT). Their stride window is 70 samples, sport mode's 0.70 s stride; with_stride swaps in another policy's.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy.signal import find_peaks

DT = 0.01
DEFAULT_ANALYSIS_DIR = os.environ.get(
    "SPORTMODE_ANALYSIS_DIR", os.path.expanduser("~/Downloads/HumanoidBags_extracted/analysis"))
REPORT_STRIDE = 70  # samples: the stride window metrics.py averages over

# (key in tracking(), label, sport mode's value as the report gives it, target, "ge" or "le")
TARGETS = [
    ("turn_sign_agree", "Turn sense correct, commanded |wz| > 0.1 rad/s", 0.994, 0.99, "ge"),
    ("speed_ratio_median", "Linear speed achieved / commanded, median", 0.93, 0.90, "ge"),
    ("turn_ratio_median", "Turn rate achieved / commanded, median", 0.91, 0.90, "ge"),
    ("cos_v_mean", "Direction cos(commanded, actual) xy velocity, mean", 0.96, 0.95, "ge"),
    ("ang_err_mean", "Angular velocity error, mean (rad/s)", 0.050, 0.05, "le"),
    ("lin_err_mean", "Linear velocity error, mean (m/s)", 0.147, 0.15, "le"),
]


def load_analysis(analysis_dir: str = DEFAULT_ANALYSIS_DIR) -> SimpleNamespace:
    """The report's metrics, gait and fit_cmd modules. They read the analysis dir from sys.argv[1] at import (and
    metrics.py lists the bags' arrays there), so sys.argv is swapped while they load."""
    analysis_dir = os.path.abspath(os.path.expanduser(analysis_dir))
    if not os.path.isfile(os.path.join(analysis_dir, "metrics.py")):
        raise FileNotFoundError(f"no metrics.py in {analysis_dir}: pass --analysis_dir (the sport-mode bag analysis)")
    saved = sys.argv[:]
    sys.argv = [saved[0] if saved else "", analysis_dir]
    sys.path.insert(0, analysis_dir)
    try:
        modules = {name: importlib.import_module(name) for name in ("fit_cmd", "gait", "metrics")}
    finally:
        sys.argv = saved
        sys.path.remove(analysis_dir)
    for name, module in modules.items():
        if os.path.dirname(os.path.abspath(module.__file__)) != analysis_dir:
            raise ImportError(f"imported {name} from {module.__file__}, not {analysis_dir}")
    return SimpleNamespace(dir=analysis_dir, **modules)


@contextlib.contextmanager
def with_stride(a: SimpleNamespace, samples: int):
    """Inside, metrics.py averages over `samples` where it averages over its 70-sample stride."""
    original = a.metrics.uniform_filter1d
    if samples != REPORT_STRIDE:
        a.metrics.uniform_filter1d = lambda x, size, *args, **kw: original(
            x, samples if size == REPORT_STRIDE else size, *args, **kw)
    try:
        yield
    finally:
        a.metrics.uniform_filter1d = original


def tracking_safe(a: SimpleNamespace, ref, vel, mask) -> dict:
    """metrics.tracking, with NaN for what a subset has no samples for (no turn, no walk)."""
    try:
        with np.errstate(all="ignore"):
            return a.metrics.tracking(ref, vel, mask)
    except (IndexError, ValueError):
        out = {}
        m = np.concatenate(mask)
        if m.sum() == 0:
            return {"n_samples": 0}
        # fall back per quantity: each needs its own subset non-empty
        r, v = np.concatenate(ref), np.concatenate([a.metrics.uniform_filter1d(x, REPORT_STRIDE, axis=0) for x in vel])
        e = (v - r)[m]
        out["lin_err_mean"] = float(np.linalg.norm(e[:, :2], axis=1).mean())
        out["ang_err_mean"] = float(np.abs(e[:, 2]).mean())
        return out


def score(a: SimpleNamespace, cmds, vels, masks, stride_samples: int = REPORT_STRIDE, subsets: dict | None = None,
          creep_removed: bool = False) -> dict:
    """The report's tracking numbers for one controller. cmds, vels: per series (T, 3) arrays (vx, vy, wz; velocity in
    the yaw frame), masks: per series (T,) bool (stepping and valid). subsets: name -> per-series masks, scored against
    the same fitted response. creep_removed: also score with the fitted vx bias removed from the velocity (the
    report's diagnostic for sport mode's forward creep)."""
    with with_stride(a, stride_samples):
        response = a.metrics.fit_resp(cmds, vels, masks)
        ref = a.metrics.lagged_ref(cmds, response)
        out = {
            "stride_samples": stride_samples,
            "response": response,
            "tracking_vs_lagged": tracking_safe(a, ref, vels, masks),
            "tracking_vs_command": tracking_safe(
                a, [a.metrics.uniform_filter1d(c, stride_samples, axis=0) for c in cmds], vels, masks),
        }
        if creep_removed:
            bias = np.array([response["vx"]["bias"], 0.0, 0.0])
            out["tracking_creep_removed"] = tracking_safe(a, ref, [v - bias for v in vels], masks)
        for name, sub in (subsets or {}).items():
            out.setdefault("subsets", {})[name] = tracking_safe(a, ref, vels, [m & s for m, s in zip(masks, sub)])
    return out


def stride_stats(knee_series, masks) -> dict:
    """Stride period from each knee's swing peaks while stepping, as the report's gait_stats: peaks of prominence
    0.1 rad at least 0.25 s apart, periods under 2 s. knee_series: per series (T, 2) knee angles (left, right)."""
    periods = []
    for q, w in zip(knee_series, masks):
        t = np.arange(len(q)) * DT
        for j in range(2):
            pk, _ = find_peaks(q[:, j], prominence=0.1, distance=25)
            pk = pk[w[pk]]
            d = np.diff(t[pk])
            periods += d[d < 2].tolist()
    T = np.array(periods)
    if len(T) == 0:
        return {"stride_T_median": float("nan"), "stride_T_iqr": float("nan"), "n_strides": 0}
    return {"stride_T_median": float(np.median(T)), "stride_T_iqr": float(np.subtract(*np.percentile(T, [75, 25]))),
            "n_strides": int(len(T))}


def tilt_stats(rp_series, masks) -> dict:
    """Torso roll and pitch sd while stepping (deg), each series' mean removed, as the report's body_stats.
    rp_series: per series (T, 2) torso roll, pitch (rad)."""
    rp = [x[m] - x[m].mean(0) for x, m in zip(rp_series, masks) if m.sum() >= 100]
    if not rp:
        return {"roll_sd_deg": float("nan"), "pitch_sd_deg": float("nan")}
    rp = np.concatenate(rp)
    return {"roll_sd_deg": float(np.degrees(rp[:, 0].std())), "pitch_sd_deg": float(np.degrees(rp[:, 1].std()))}


def commanded(cmd: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Samples with a non-zero command, as the report counts them: |vx| or |vy| or |wz| over 0.05."""
    return (np.abs(cmd[:, :2]).max(1) > threshold) | (np.abs(cmd[:, 2]) > threshold)


def verdict(value: float, target: float, sense: str) -> bool:
    return bool(np.isfinite(value) and (value >= target if sense == "ge" else value <= target))
