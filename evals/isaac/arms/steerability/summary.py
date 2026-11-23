"""Per-region summaries of a system's benchmark run: medians with 95% bootstrap intervals over base poses (each
(instance, arm) is one base), in mm and degrees."""

import numpy as np

from . import estimators as est

MM, DEG = 1e-3, np.pi / 180


def regions(margin: np.ndarray, sigma_min: np.ndarray, offtable: np.ndarray) -> np.ndarray:
    """(N, arms) region names from each base's seed posture (design doc, Protocol): off-table first, then near
    singular (the pool's lowest 10% of the weighted Jacobian's smallest singular value), near limits (a joint within
    0.15 rad of its limit), interior (every joint 0.3 rad inside, singular value above the pool's median), else other.
    The rarer region wins a tie, in that order."""
    out = np.full(margin.shape, "other", dtype=object)
    out[(margin > 0.3) & (sigma_min > np.median(sigma_min))] = "interior"
    out[margin < 0.15] = "near_limit"
    out[sigma_min < np.quantile(sigma_min, 0.1)] = "near_singular"
    out[offtable] = "offtable"
    return out


def bootstrap_median(values: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float, float]:
    """(median, low, high) of the finite values, the 95% percentile interval over n resamples."""
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    if len(v) == 1:
        return (float(v[0]),) * 3
    meds = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return float(np.median(v)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def per_base_steps(plan, result, resp) -> dict[str, np.ndarray]:
    """Per base (N, arms) and scale: the step-schedule metrics in mm and degrees."""
    J = est.local_jacobians(plan, resp)  # (N, arms, S, 6, 6)
    m = est.jacobian_metrics(J)
    S = len(plan.scales)
    gap = np.full(plan.bases.shape[:2] + (S, 2), np.nan)  # translation mm, rotation deg: mean |gap| per base
    for s in range(S):
        for kind, sel, unit in ((0, plan.axis < 3, MM), (1, plan.axis >= 3, DEG)):
            mask = sel & (plan.scale == s) & resp["flat"]
            g = np.where(mask, np.abs(resp["gap"]), np.nan)
            with np.errstate(invalid="ignore"):
                gap[..., s, kind] = np.nanmean(g, axis=0) / unit
    return {"trans_gain": m["trans_gain"].mean(-1), "rot_gain": m["rot_gain"].mean(-1),
            "trans_identity_dev": m["trans_identity_dev"], "rot_identity_dev": m["rot_identity_dev"],
            "trans_cond": m["trans_cond"], "rot_cond": m["rot_cond"],
            "trans_coupling": m["trans_coupling"], "rot_coupling": m["rot_coupling"],
            "rot_to_trans_mm_per_deg": np.abs(m["rot_to_trans"]).max(axis=(-1, -2)) * DEG / MM,
            "trans_gap_mm": gap[..., 0], "rot_gap_deg": gap[..., 1], "flat_share": resp["flat"].mean(0)}


def pooled_resolution(plan, resp, mask: np.ndarray) -> tuple[float, float]:
    """(translation mm, rotation deg) resolution over the bases in mask (N, arms), responses pooled."""
    out = []
    for sel, col, unit in ((plan.axis < 3, 0, MM), (plan.axis >= 3, 1, DEG)):
        keep = sel & mask[None] & resp["flat"]
        pooled = [np.concatenate([resp["resp_plus"][keep & (plan.scale == s)], resp["resp_minus"][keep & (plan.scale == s)]])
                  for s in range(len(plan.scales))]
        sham = np.concatenate([resp["resp_plus"][keep & (plan.scale < 0)], resp["resp_minus"][keep & (plan.scale < 0)]])
        out.append(est.resolution(pooled, sham, plan.scales[:, col]) / unit)
    return tuple(out)


def pooled_validity(plan, result, resp, mask: np.ndarray, tol: float = 0.5 * MM) -> float:
    """The bias field's validity radius (mm) over the bases in mask: the largest step over which the open-loop bias
    changes by under tol (RMS, the shams' nugget taken out)."""
    inc = est.field_increments(plan, result, resp)
    keep = mask[None] & resp["flat"] & np.isfinite(inc["h"])
    levels, g_pos, _ = est.variogram(inc["h"][keep], inc["delta"][keep])
    return est.validity_radius(levels, g_pos, tol) / MM


def table(rows: dict[str, dict[str, tuple[float, float, float]]], fmt: str = "{:.3g}") -> str:
    """Markdown: a row per metric, a column per region, each cell 'median [low, high]'."""
    regions_ = sorted({r for row in rows.values() for r in row})
    lines = ["| Metric | " + " | ".join(regions_) + " |", "| --- |" + " --- |" * len(regions_)]
    for name, row in rows.items():
        cells = []
        for r in regions_:
            v = row.get(r)
            if v is None or not np.isfinite(v[0]):
                cells.append("")
            elif v[1] == v[2]:
                cells.append(fmt.format(v[0]))
            else:
                cells.append(f"{fmt.format(v[0])} [{fmt.format(v[1])}, {fmt.format(v[2])}]")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
