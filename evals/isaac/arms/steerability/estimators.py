"""The metrics, from runs of the schedules (design doc: Metric definitions). Pure numpy; units are SI (m, rad) here
and converted to mm and degrees only in the report."""

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from . import se3
from .runner import RunResult
from .schedules import RepeatPlan, StepPlan


def _take(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """values (..., 6) at each element's own axis (...)."""
    return np.take_along_axis(values, axis[..., None], -1)[..., 0]


def step_responses(plan: StepPlan, result: RunResult, window_s: float = 0.5, flat_frac: float = 0.1,
                   flat_floor: tuple[float, float] = (2e-5, 2e-4)) -> dict[str, np.ndarray]:
    """Per trial item (M, N, arms): the settled poses as deltas from the base (y0 before, plus, mid, minus, end), the
    responses along the stepped axis (resp_plus: + step from the base; resp_minus: - step, sign flipped, so both are
    positive when correct), the hysteresis gap (mid, back from +, less end, back from -; along the axis), the central
    difference column (plus - minus) / 2 magnitude (6, NaN for shams), and flat: every segment of the trial settled
    (its window's drift under flat_frac of the step, or flat_floor (m, rad) for small steps and shams)."""
    y = se3.relative(plan.bases[None], result.settled)  # (S, N, arms, 6)
    M = len(plan.axis)
    first = 1 + 4 * np.arange(M)
    y0, plus, mid, minus, end = (y[first + j] for j in (-1, 0, 1, 2, 3))
    k, mag = plan.axis, plan.magnitude
    column = np.where((mag > 0)[..., None], (plus - minus) / np.where(mag > 0, 2 * mag, 1.0)[..., None], np.nan)
    rot = k >= 3
    allowed = np.maximum(flat_frac * mag, np.where(rot, flat_floor[1], flat_floor[0]))
    flat = np.ones_like(mag, dtype=bool)
    for j in range(4):
        drift = np.where(rot, result.drift[first + j][..., 1], result.drift[first + j][..., 0])
        flat &= drift * window_s <= allowed
    return {"y0": y0, "plus": plus, "mid": mid, "minus": minus, "end": end,
            "resp_plus": _take(plus - y0, k), "resp_minus": -_take(minus - mid, k), "gap": _take(mid - end, k),
            "column": column, "flat": flat}


def local_jacobians(plan: StepPlan, resp: dict[str, np.ndarray], require_flat: bool = True) -> np.ndarray:
    """(N, arms, scales, 6, 6): per instance, arm and scale, the mean central-difference columns (rows: response
    axes, columns: stepped axes); NaN where a column has no settled trial."""
    S = len(plan.scales)
    N, arms = plan.bases.shape[:2]
    total = np.zeros((N, arms, S, 6, 6))  # (..., stepped axis, response)
    count = np.zeros((N, arms, S, 6))
    ok = plan.scale >= 0
    if require_flat:
        ok &= resp["flat"]
    m, n, a = np.nonzero(ok)
    s, k = plan.scale[m, n, a], plan.axis[m, n, a]
    np.add.at(total, (n, a, s, k), resp["column"][m, n, a])
    np.add.at(count, (n, a, s, k), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cols = total / count[..., None]
    return np.swapaxes(cols, -1, -2)


def jacobian_metrics(J: np.ndarray) -> dict[str, np.ndarray]:
    """Per 6 x 6 Jacobian (...): each 3 x 3 block's gains (diagonal), distance from identity (Frobenius), condition
    number and coupling (off-diagonal over diagonal, Frobenius), and the cross blocks (m per rad, rad per m)."""
    out = {}
    finite = np.isfinite(J).all(axis=(-1, -2))
    safe = np.where(finite[..., None, None], J, np.eye(6))
    for name, block in (("trans", safe[..., :3, :3]), ("rot", safe[..., 3:, 3:])):
        diag = np.diagonal(block, axis1=-2, axis2=-1)
        off = block - diag[..., None] * np.eye(3)
        values = {"gain": diag,
                  "identity_dev": np.linalg.norm(block - np.eye(3), axis=(-2, -1)),
                  "cond": np.linalg.cond(block),
                  "coupling": np.linalg.norm(off, axis=(-2, -1)) / np.linalg.norm(diag, axis=-1)}
        for key, value in values.items():
            mask = finite if value.ndim == finite.ndim else finite[..., None]
            out[f"{name}_{key}"] = np.where(mask, value, np.nan)
    out["rot_to_trans"] = np.where(finite[..., None, None], J[..., :3, 3:], np.nan)
    out["trans_to_rot"] = np.where(finite[..., None, None], J[..., 3:, :3], np.nan)
    return out


def repeatability(plan: RepeatPlan, result: RunResult) -> dict[str, np.ndarray]:
    """Spread of the base's settled pose over the K approaches (N, arms): per-axis standard deviation (6), the RMS
    distance from the mean position (m) and the RMS rotation from the mean (rad)."""
    at_base = result.settled[1::2]  # (K, N, arms, 7)
    mean = se3.mean_pose(at_base)
    dev = se3.relative(mean[None], at_base)  # (K, N, arms, 6)
    return {"std": dev.std(axis=0, ddof=1),
            "radius_rms": np.sqrt((np.linalg.norm(dev[..., :3], axis=-1) ** 2).mean(axis=0)),
            "rot_rms": np.sqrt((np.linalg.norm(dev[..., 3:], axis=-1) ** 2).mean(axis=0))}


def resolution(responses: list[np.ndarray], sham: np.ndarray, magnitudes: np.ndarray, sign_frac: float = 0.9,
               z: float = 2.0) -> float:
    """The smallest step magnitude from which on (it and every larger one) the responses along the stepped axis are
    correctly signed at least sign_frac of the time and their mean is more than z standard errors from the shams'.
    responses[i]: the pooled responses (sign-corrected, so positive is right) at magnitudes[i], ascending; inf when
    none qualifies."""
    sham = sham[np.isfinite(sham)]
    passed = []
    for r in responses:
        r = r[np.isfinite(r)]
        if len(r) < 2 or len(sham) < 2:
            passed.append(False)
            continue
        se = np.sqrt(r.var(ddof=1) / len(r) + sham.var(ddof=1) / len(sham))
        passed.append((r > 0).mean() >= sign_frac and r.mean() - sham.mean() > z * max(se, 1e-15))
    for i in range(len(passed)):
        if all(passed[i:]):
            return float(magnitudes[i])
    return float("inf")


def field_increments(plan: StepPlan, result: RunResult, resp: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Open-loop bias b(c) = relative(c, settled) at the base before each translation step and at the stepped
    command: per item (M, N, arms), the spacing h (m; 0 for shams) and the bias change (6); NaN off translation axes."""
    M = len(plan.axis)
    first = 1 + 4 * np.arange(M)
    seg_poses = np.stack([s.poses for s in plan.segments])
    bias = se3.relative(seg_poses, result.settled)  # (S, N, arms, 6)
    delta = bias[first] - bias[first - 1]
    trans = plan.axis < 3
    return {"h": np.where(trans, plan.magnitude, np.nan), "delta": np.where(trans[..., None], delta, np.nan)}


def variogram(h: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per distinct spacing (ascending): gamma(h) = half the mean squared bias change, position (m^2) and rotation
    (rad^2) apart. h (P,), delta (P, 6); NaNs dropped."""
    keep = np.isfinite(h) & np.isfinite(delta).all(-1)
    h, delta = h[keep], delta[keep]
    levels = np.unique(h)
    g_pos = np.array([0.5 * (np.linalg.norm(delta[h == v, :3], axis=-1) ** 2).mean() for v in levels])
    g_rot = np.array([0.5 * (np.linalg.norm(delta[h == v, 3:], axis=-1) ** 2).mean() for v in levels])
    return levels, g_pos, g_rot


def validity_radius(levels: np.ndarray, gamma: np.ndarray, tol: float) -> float:
    """The largest spacing up to which (every spacing at or below it) the RMS bias change, the h = 0 nugget (the
    repeatability, from the shams) taken out, stays under tol. 0 when even the smallest step exceeds it."""
    nugget = gamma[levels == 0].mean() if (levels == 0).any() else 0.0
    change = np.sqrt(np.maximum(2 * (gamma - nugget), 0.0))
    radius = 0.0
    for v, c in zip(levels, change):
        if v == 0:
            continue
        if c >= tol:
            break
        radius = float(v)
    return radius


def step_dynamics(trace: np.ndarray, dt: float, band: np.ndarray | float, window_s: float = 0.5
                  ) -> dict[str, np.ndarray]:
    """From traces (T, P) of the stepped axis (each the achieved along the axis, from a segment's start): the step
    (final, the last window's mean, less the first sample), the settling time (s after which the trace stays within
    band of the final value; NaN if it never does), the overshoot (largest excursion past the final value over the
    step's size) and the 10-90% rise time (s)."""
    T = len(trace)
    final = trace[-max(2, int(round(window_s / dt))):].mean(axis=0)
    start = trace[0]
    step = final - start
    band = np.broadcast_to(band, final.shape)
    outside = np.abs(trace - final) > band
    last_out = np.where(outside.any(axis=0), T - 1 - np.argmax(outside[::-1], axis=0), -1)
    settle = np.where(last_out < T - 1, (last_out + 1) * dt, np.nan)
    sign = np.sign(step)
    with np.errstate(invalid="ignore", divide="ignore"):
        overshoot = np.maximum((sign * (trace - final)).max(axis=0), 0.0) / np.abs(step)
        frac = sign * (trace - start) / np.abs(step)
    t10 = np.argmax(frac >= 0.1, axis=0)
    t90 = np.argmax(frac >= 0.9, axis=0)
    rise = np.where((frac >= 0.9).any(axis=0), (t90 - t10) * dt, np.nan)
    return {"step": step, "settle_time": settle, "overshoot": overshoot, "rise_time": rise}


def predicted_floor(J: np.ndarray, gain: float, sensor_cov: np.ndarray, rep_cov: np.ndarray) -> np.ndarray:
    """The steady error covariance (6 x 6) of a look-then-move loop c <- c + gain (target - measured), from
    e_{k+1} = (I - gain J) e_k - gain J n_sensor + n_rep (design doc, metric 7): the sensor's noise reaches the wrist
    through the command, so through J too. inf when the loop diverges."""
    A = np.eye(6) - gain * J
    if np.max(np.abs(np.linalg.eigvals(A))) >= 1.0:
        return np.full((6, 6), np.inf)
    return solve_discrete_lyapunov(A, gain ** 2 * J @ sensor_cov @ J.T + rep_cov)
