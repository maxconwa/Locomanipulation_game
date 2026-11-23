"""The scorecard's metric code (evals/common/sportmode.py, copied from the report's analysis, and legs_metrics.py) on
the real sport-mode bags reproduces the H1-2 Sport Mode Walking Report's sport-mode numbers: the tracking rows exactly
as its results/*.json, and every row within the report's rounding. Reads the bags' grids and the report's results from
the analysis dir (default ~/Downloads/HumanoidBags_extracted/analysis or $SPORTMODE_ANALYSIS_DIR; it imports none of
its code); no simulator.

    python -m evals.deploy.test_sportmode_metrics [--analysis_dir <dir>]      (or: pytest evals/deploy/test_sportmode_metrics.py)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # the repo root
from evals.common import legs_metrics as sm  # noqa: E402
from evals.common import sportmode  # noqa: E402

DEFAULT_ANALYSIS_DIR = os.environ.get(
    "SPORTMODE_ANALYSIS_DIR", os.path.expanduser("~/Downloads/HumanoidBags_extracted/analysis"))

# the report's sport-mode column, as printed (value, decimals it is rounded to)
REPORT = {
    "turn_sign_agree": (0.994, 3),
    "speed_ratio_median": (0.93, 2),
    "turn_ratio_median": (0.91, 2),
    "cos_v_mean": (0.65, 2),
    "ang_err_mean": (0.050, 3),
    "lin_err_mean": (0.147, 3),
    "lin_err_p95": (0.221, 3),
    "ang_err_p95": (0.122, 3),
}
REPORT_CREEP_REMOVED_COS = (0.96, 2)
REPORT_STRIDE = {"stride_T_median": (0.700, 3), "stride_T_iqr": (0.01, 2), "n_strides": (1178, 0)}
REPORT_TILT = {"roll_sd_deg": (0.83, 2), "pitch_sd_deg": (0.75, 2)}


def sport_runs(analysis_dir: str) -> list:
    """The eight sport-mode bags' 100 Hz grids, as the report's metrics.py lists them (SPORT)."""
    paths = sorted(glob.glob(os.path.join(analysis_dir, "data", "*_bag_*_grid.npz")))
    if not paths:
        raise FileNotFoundError(f"no data/*_bag_*_grid.npz in {analysis_dir}: pass --analysis_dir")
    return [(os.path.basename(p)[:-9], np.load(p)) for p in paths]


def sport_cmd(g, fit: dict) -> np.ndarray:
    """Stick to commanded body velocity: the report's per-axis stick gain only (metrics.py sport_cmd)."""
    st = g["st"].astype(float)
    return np.stack([fit["vx"]["gain"] * st[:, 1], fit["vy"]["gain"] * st[:, 0], fit["wz"]["gain"] * st[:, 2]], 1)


def sport_scorecard(analysis_dir: str = DEFAULT_ANALYSIS_DIR) -> dict:
    runs = sport_runs(analysis_dir)
    fit = json.load(open(os.path.join(analysis_dir, "data", "cmdfit.json")))
    cmds = [sport_cmd(g, fit) for _, g in runs]
    vels = [np.stack([g["v"][:, 0], g["v"][:, 1], g["wz"]], 1) for _, g in runs]
    masks = [sportmode.stepping(g["q"]) for _, g in runs]
    out = sm.score(cmds, vels, masks, sm.REPORT_STRIDE, creep_removed=True)
    out["gait"] = sm.stride_stats([g["q"][:, [3, 9]].astype(float) for _, g in runs], masks)
    out["tilt"] = sm.tilt_stats([g["imu_rpy"][:, :2].astype(float) for _, g in runs],
                                [w & (g["kp"][:, 0] > 0) for w, (_, g) in zip(masks, runs)])
    out["feet"] = sm.foot_stats([g["q"][:, :12].astype(float) for _, g in runs],
                                [g["imu_rpy"][:, :2].astype(float) for _, g in runs],
                                [w & (g["kp"][:, 0] > 0) for w, (_, g) in zip(masks, runs)])
    out["bags"] = [name for name, _ in runs]
    return out


def check(out: dict, analysis_dir: str) -> list[str]:
    """Mismatches against the report (empty: it reproduces)."""
    bad = []

    def near(name, value, expected, decimals):
        if round(float(value), decimals) != round(expected, decimals):
            bad.append(f"{name}: {value:.4f}, the report has {expected}")

    lag = out["tracking_vs_lagged"]
    for key, (expected, decimals) in REPORT.items():
        near(key, lag[key], expected, decimals)
    near("cos_v_mean (creep removed)", out["tracking_creep_removed"]["cos_v_mean"], *REPORT_CREEP_REMOVED_COS)
    for key, (expected, decimals) in REPORT_STRIDE.items():
        near(key, out["gait"][key], expected, decimals)
    for key, (expected, decimals) in REPORT_TILT.items():
        near(key, out["tilt"][key], expected, decimals)
    # the feet: sportmode.SPORT_REF's values (results/metrics.json gait, logs/analysis/foot_lift/results.md)
    for key in ("clearance_median_cm", "clearance_p10_cm", "stride_dz_median_cm", "stride_dz_p90_cm",
                "knee_swing_median", "knee_peak_median", "steps_per_s"):
        near(key, out["feet"][key], sportmode.SPORT_REF[key], 1 if key.endswith("_cm") else 2)
    # and exactly the report's own result files
    results = os.path.join(analysis_dir, "results")
    saved = json.load(open(os.path.join(results, "metrics.json")))["sport"]
    extras = json.load(open(os.path.join(results, "extras.json")))
    for key, value in saved["tracking_vs_lagged"].items():
        if not np.isclose(lag[key], value, rtol=1e-9, atol=1e-12):
            bad.append(f"tracking_vs_lagged {key}: {lag[key]!r} != results/metrics.json {value!r}")
    for axis, fit in saved["response"].items():
        for key, value in fit.items():
            if not np.isclose(out["response"][axis][key], value, rtol=1e-9, atol=1e-12):
                bad.append(f"response {axis} {key}: {out['response'][axis][key]!r} != {value!r}")
    for key, value in extras["sport_tracking_creep_removed"].items():
        if not np.isclose(out["tracking_creep_removed"][key], value, rtol=1e-9, atol=1e-12):
            bad.append(f"creep removed {key}: {out['tracking_creep_removed'][key]!r} != results/extras.json {value!r}")
    return bad


def test_sport_mode_bags_reproduce_report():
    out = sport_scorecard()
    bad = check(out, DEFAULT_ANALYSIS_DIR)
    assert not bad, "\n".join(bad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis_dir", default=DEFAULT_ANALYSIS_DIR)
    args = parser.parse_args()
    out = sport_scorecard(args.analysis_dir)
    lag = out["tracking_vs_lagged"]
    print(f"bags: {', '.join(out['bags'])}")
    for key, label, value, target, sense in sm.TARGETS:
        print(f"  {label:55s} {lag[key]:.4f}  (report {REPORT[key][0]})")
    print(f"  cos with the creep removed: {out['tracking_creep_removed']['cos_v_mean']:.4f}  (report 0.96)")
    print(f"  stride {out['gait']['stride_T_median']:.3f} s, IQR {out['gait']['stride_T_iqr']:.3f} s,"
          f" {out['gait']['n_strides']} strides; torso roll / pitch sd {out['tilt']['roll_sd_deg']:.2f} /"
          f" {out['tilt']['pitch_sd_deg']:.2f} deg")
    print("  feet: " + ", ".join(f"{k} {v:.3f}" for k, v in out["feet"].items()))
    for axis, fit in out["response"].items():
        print(f"  {axis}: gain {fit['gain']:.3f}, delay {fit['delay']:.2f} s, tau {fit['tau']:.2f} s, R2 {fit['r2']:.3f}")
    bad = check(out, args.analysis_dir)
    print("\n".join(["FAIL"] + bad) if bad else "PASS: reproduces the report's sport-mode numbers")
    sys.exit(1 if bad else 0)
