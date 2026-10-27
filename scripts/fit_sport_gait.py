"""Fit sport mode's leg-joint motion as a function of the gait phase and the velocity, from the sport-mode bags'
100 Hz tables, for the legs' sport_gait reward (mdp.sport_gait_exp).

    python scripts/fit_sport_gait.py <analysis_dir> [--out <npz>]

<analysis_dir> is the bag analysis (its data/<bag>_grid.npz tables, build.py's: q in Unitree motor order, v the
levelled body-frame velocity, wz the yaw rate); the policy bags (policy_*) are left out. Each joint's offset from
that bag's standing pose is fitted as sum_k f_k(v) * b_k(phase): f = (1, vx, vy, wz, |vx|, |vy|, |wz|), the
velocity a 0.70 s window's mean, and b = (1, cos, sin) of 1-3 times the phase. The phase is the gait clock's
(legs_rewards.leg_phase, the left leg's): each left-knee peak is mid-swing, STANCE_THRESHOLD + half the swing, and
the phase runs linearly between peaks. Prints each joint's R^2, fitted and on each bag left out.
"""

import argparse
import glob
import os

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

STANCE_THRESHOLD = 0.55  # legs_rewards.STANCE_THRESHOLD
MID_SWING = STANCE_THRESHOLD + (1.0 - STANCE_THRESHOLD) / 2
HARMONICS = 3
WINDOW = 70  # 0.70 s at 100 Hz: one stride
STRIDE_RANGE = (0.55, 0.85)  # s, the knee-peak intervals kept
LEGS = 12  # the leg joints, first in Unitree's order (as LOWER_JOINT_NAMES)
NAMES = [f"{s} {j}" for s in ("L", "R") for j in ("hip yaw", "hip pitch", "hip roll", "knee", "ankle pitch", "ankle roll")]


def features(v: np.ndarray) -> np.ndarray:
    """(n, 7): 1, vx, vy, wz and their magnitudes."""
    return np.concatenate([np.ones((len(v), 1)), v, np.abs(v)], axis=1)


def basis(phase: np.ndarray) -> np.ndarray:
    """(n, 1 + 2 HARMONICS): 1, cos and sin of each harmonic of the phase."""
    cols = [np.ones_like(phase)]
    for h in range(1, HARMONICS + 1):
        cols += [np.cos(2 * np.pi * h * phase), np.sin(2 * np.pi * h * phase)]
    return np.stack(cols, axis=1)


def design(v: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """(n, 7 * (1 + 2 HARMONICS)): every feature times every basis function, feature-major."""
    return (features(v)[:, :, None] * basis(phase)[:, None, :]).reshape(len(v), -1)


def bag_samples(path: str):
    """A bag's stepping samples: (v (n, 3), phase (n,), dq (n, 12) from the standing pose)."""
    g = np.load(path)
    t, q = g["t"], g["q"][:, :LEGS].astype(float)
    knees = q[:, [3, 9]]
    mean = uniform_filter1d(knees, 50, axis=0)
    sd = np.sqrt(np.maximum(uniform_filter1d(knees**2, 50, axis=0) - mean**2, 0)).max(1)
    stepping = sd > 0.03
    still = uniform_filter1d(stepping.astype(float), WINDOW) < 0.01
    stand = np.median(q[still], axis=0) if still.sum() > 100 else None
    peaks, _ = find_peaks(q[:, 3], prominence=0.15, distance=30)
    peaks = peaks[stepping[peaks]]
    v = np.concatenate([g["v"][:, :2], g["wz"][:, None]], axis=1).astype(float)
    v = uniform_filter1d(v, WINDOW, axis=0)
    idx, phase = [], []
    for a, b in zip(peaks[:-1], peaks[1:]):
        if not STRIDE_RANGE[0] <= t[b] - t[a] <= STRIDE_RANGE[1]:
            continue
        idx.append(np.arange(a, b))
        phase.append((MID_SWING + (t[a:b] - t[a]) / (t[b] - t[a])) % 1.0)
    idx = np.concatenate(idx)
    return v[idx], np.concatenate(phase), q[idx], stand


def fit(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    return np.linalg.solve(X.T @ X + ridge * len(X) * np.eye(X.shape[1]), X.T @ Y)


def r2(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    return 1.0 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis_dir")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "source", "locomanipulation_game",
                                                      "locomanipulation_game", "data", "sport_gait_reference.npz"))
    args = parser.parse_args()
    paths = [p for p in sorted(glob.glob(os.path.join(args.analysis_dir, "data", "*_grid.npz")))
             if not os.path.basename(p).startswith("policy")]
    bags = {os.path.basename(p)[:-9]: bag_samples(p) for p in paths}
    stands = [s for *_, s in bags.values() if s is not None]
    common_stand = np.median(stands, axis=0)
    data = {}
    for name, (v, phase, q, stand) in bags.items():
        data[name] = (design(v, phase), q - (stand if stand is not None else common_stand), v)
        print(f"{name}: {len(v) / 100:.0f} s stepping, stand pose {'own' if stand is not None else 'common'}")
    X = np.concatenate([d[0] for d in data.values()])
    Y = np.concatenate([d[1] for d in data.values()])
    V = np.concatenate([d[2] for d in data.values()])
    coef = fit(X, Y)
    fitted = r2(Y, X @ coef)
    held = []
    for name in data:
        rest = [k for k in data if k != name]
        c = fit(np.concatenate([data[k][0] for k in rest]), np.concatenate([data[k][1] for k in rest]))
        held.append(((data[name][1] - data[name][0] @ c) ** 2).sum(0))
    held_r2 = 1.0 - np.sum(held, axis=0) / ((Y - Y.mean(0)) ** 2).sum(0)
    resid = np.sqrt(((Y - X @ coef) ** 2).mean(0))
    print(f"\n{len(Y) / 100:.0f} s of stepping from {len(data)} bags; velocity range (p1, p99):"
          f" vx {np.percentile(V[:, 0], [1, 99]).round(2)}, vy {np.percentile(V[:, 1], [1, 99]).round(2)},"
          f" wz {np.percentile(V[:, 2], [1, 99]).round(2)}")
    print(f"{'joint':14s}{'R2 fit':>8s}{'R2 held-out':>12s}{'resid RMS':>11s}{'swing RMS':>11s}")
    for j in range(LEGS):
        print(f"{NAMES[j]:14s}{fitted[j]:8.2f}{held_r2[j]:12.2f}{resid[j]:11.3f}{Y[:, j].std():11.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, coef=coef.astype(np.float32), harmonics=HARMONICS, residual_rms=resid.astype(np.float32),
             v_low=np.percentile(V, 1, axis=0).astype(np.float32), v_high=np.percentile(V, 99, axis=0).astype(np.float32),
             bags=np.array(list(data)))
    print(f"wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
