"""The scorecard's metrics (scorecard_metrics.py) on the real sport-mode bags reproduce the H1-2 Sport Mode Walking
Report's sport-mode numbers: the tracking rows exactly as its results/*.json, and every row within the report's
rounding. Needs the analysis dir (scorecard_metrics.DEFAULT_ANALYSIS_DIR); no simulator.

    python scripts/skrl/test_scorecard_sportmode.py [--analysis_dir <dir>]      (or: pytest scripts/skrl/test_scorecard_sportmode.py)
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scorecard_metrics as sm  # noqa: E402

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


def sport_scorecard(analysis_dir: str = sm.DEFAULT_ANALYSIS_DIR) -> dict:
    a = sm.load_analysis(analysis_dir)
    m = a.metrics
    runs = m.SPORT
    cmds = [m.sport_cmd(g) for _, g in runs]
    vels = [m.meas(g) for _, g in runs]
    masks = [a.gait.stepping(g) for _, g in runs]
    out = sm.score(a, cmds, vels, masks, sm.REPORT_STRIDE, creep_removed=True)
    out["gait"] = sm.stride_stats([g["q"][:, [3, 9]].astype(float) for _, g in runs], masks)
    out["tilt"] = sm.tilt_stats([g["imu_rpy"][:, :2].astype(float) for _, g in runs],
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
    bad = check(out, sm.DEFAULT_ANALYSIS_DIR)
    assert not bad, "\n".join(bad)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis_dir", default=sm.DEFAULT_ANALYSIS_DIR)
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
    for axis, fit in out["response"].items():
        print(f"  {axis}: gain {fit['gain']:.3f}, delay {fit['delay']:.2f} s, tau {fit['tau']:.2f} s, R2 {fit['r2']:.3f}")
    bad = check(out, args.analysis_dir)
    print("\n".join(["FAIL"] + bad) if bad else "PASS: reproduces the report's sport-mode numbers")
    sys.exit(1 if bad else 0)
